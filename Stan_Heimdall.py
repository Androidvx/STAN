# -*- coding: utf-8 -*-
"""
=============================================================================
STAN (System for Tactical Analysis of Documents) - STAN version 2.0
=============================================================================
Copyright (c) 2026, André Luiz Oliveira da Silva
ORCID: https://orcid.org/0000-0003-4768-959X
Lattes: https://lattes.cnpq.br/773185247965855

This software is provided for academic and research purposes.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to use, copy, and modify, 
provided that the above copyright notice and this permission notice appear in 
all copies or substantial portions of the Software. 

If you use this software in academic research, YOU MUST CITE THE AUTHOR.
=============================================================================
"""
import tkinter as tk
from tkinter import filedialog, messagebox, BooleanVar
from PIL import Image, ImageTk
import pandas as pd
import os
import gzip
import json
from datetime import datetime
import threading
import sys
import re
import subprocess
import html
import fitz  # PyMuPDF
import concurrent.futures
import gc
import pyarrow.parquet as pq
import uuid
import time
import glob

# --- LIBRARY AUTO-INSTALLATION ---
try:
    import matplotlib.pyplot as plt
    import pyarrow
    import customtkinter as ctk
    import tiktoken
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "pyarrow", "fastparquet", "customtkinter", "tiktoken"])
    import site
    from importlib import reload
    reload(site)
    import matplotlib.pyplot as plt
    import pyarrow
    import customtkinter as ctk
    import tiktoken

# Tesseract OCR Verification
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

def configure_tesseract():
    if not PYTESSERACT_AVAILABLE: return False
    
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.abspath(os.path.dirname(__file__))
        
    caminho_portatil = os.path.join(base_dir, "Tesseract-OCR", "tesseract.exe")
    
    paths = [
        caminho_portatil,
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    ]
    
    for c in paths:
        if os.path.exists(c):
            pytesseract.pytesseract.tesseract_cmd = c
            if c == caminho_portatil:
                os.environ["TESSDATA_PREFIX"] = os.path.dirname(c)
            return True
    return False

# =====================================================================
# GLOBAL SEARCH FUNCTIONS AND BOOLEAN LOGIC
# =====================================================================
def global_extract_year(text):
    if not text or str(text).lower() == "unknown date": return "unknown date"
    match = re.search(r'\b(19|20)\d{2}\b', str(text))
    return match.group(0) if match else "unknown date"

def global_generate_term_regex(term):
    term = term.strip()
    if not term: return ""
    if term.startswith('"') and term.endswith('"'): return r'\b' + re.escape(term.strip('"')) + r'\b'
    elif term.endswith('*'): return r'\b' + re.escape(term.rstrip('*')) + r'\w*'
    else: return r'\b' + re.escape(term) + r'\b'

def match_query(text, query):
    if not query: return True
    text = str(text)
    and_groups = [g.strip() for g in re.split(r'\sAND\s', query, flags=re.IGNORECASE) if g.strip()]
    for group in and_groups:
        not_parts = [p.strip() for p in re.split(r'\sNOT\s', group, flags=re.IGNORECASE)]
        pos = not_parts[0]
        neg = not_parts[1:]
        
        if pos:
            or_terms = [t.strip() for t in re.split(r',|\sOR\s', pos, flags=re.IGNORECASE) if t.strip()]
            if or_terms:
                regex_or = '|'.join(global_generate_term_regex(t) for t in or_terms)
                if not re.search(regex_or, text, re.IGNORECASE):
                    return False
        for n in neg:
            if n and re.search(global_generate_term_regex(n), text, re.IGNORECASE):
                return False
    return True

# =====================================================================
# WORKER-SIDE EXPORT FUNCTIONS
# =====================================================================
def sanitize_text(text): 
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', str(text)) if text is not None else ""

def extract_snippets(text, query_string, window=1500):
    text = sanitize_text(text)
    if not query_string or not text: return text[:4000] + "\n\n*(Text truncated for AI context limits)*"
    try:
        positive_query = re.split(r'\sNOT\s', query_string, flags=re.IGNORECASE)[0]
        terms = [t.strip() for t in re.split(r',|\sOR\s', positive_query, flags=re.IGNORECASE) if t.strip()]
        patterns = [global_generate_term_regex(t) for t in terms if t]
        if not patterns: return text[:4000] + "\n\n*(Text truncated for AI context limits)*"
        
        snippets = []
        for p in patterns:
            matches = list(re.finditer(p, text, flags=re.IGNORECASE))[:5]
            for match in matches:
                start = max(0, match.start() - window)
                end = min(len(text), match.end() + window)
                snippet = text[start:end]
                snippet = re.sub(f'({p})', r'**\1**', snippet, flags=re.IGNORECASE)
                snippets.append(f"[...] {snippet.strip()} [...]")
                
        if snippets: return "\n\n---\n*Context Shift*\n---\n\n".join(snippets)
        else: return text[:4000] + "\n\n*(Text truncated for AI context limits)*"
    except:
        return text[:4000] + "\n\n*(Text truncated for AI context limits)*"

def flush_worker_data(docs, output_folder, query_string, term_clean, timestamp, worker_id, part_num):
    if not docs: return
    
    base_name = f"report_{term_clean}_{timestamp}_{worker_id}_p{part_num}"
    base_path = os.path.join(output_folder, base_name)
    
    # 1. Write JSONL (GZipped)
    df = pd.DataFrame(docs)
    if 'mentioned_countries' in df.columns: 
        df['mentioned_countries'] = df['mentioned_countries'].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    with gzip.open(base_path + ".jsonl.gz", 'wt', encoding='utf-8') as z: 
        z.write(df.to_json(orient='records', lines=True, force_ascii=False) + "\n")

    # 2. Write HTML with EXPLICIT DIV MARKER and Zotero RIS Export
    css = """
    <style>
        body{font-family:sans-serif;line-height:1.6;background:#f4f4f4;color:#333;margin:0;padding:20px}
        .container{max-width:1000px;margin:0 auto;background:#fff;padding:20px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1)}
        .doc{background:#fafafa;padding:15px;margin-bottom:20px;border-left:4px solid #1f538d; position: relative;}
        .ocr{background:#eaeaea;padding:10px;max-height:400px;overflow-y:auto;white-space:pre-wrap;font-size:0.9em}
        .nav{padding:20px;background:#1f538d;color:white;text-align:center;margin-bottom:20px;border-radius:5px;}
        .btn{display:inline-block;padding:10px 20px;margin:5px 10px;background:#ffffff;color:#1f538d;text-decoration:none;border-radius:4px;font-weight:bold;transition:background 0.3s, transform 0.1s; cursor:pointer; border:none;}
        .btn:hover{background:#e0e0e0; transform: translateY(-1px);}
        .export-panel{position: sticky; top: 0; background: #fff; padding: 15px; border-bottom: 3px solid #1f538d; z-index: 1000; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 10px rgba(0,0,0,0.1);}
        .counter-badge{background:#b22222; color:white; padding:4px 12px; border-radius:15px; font-weight:bold; margin-left: 10px;}
        mark {background-color: #ffeb3b; color: black; font-weight: bold;}
    </style>
    """

    js_export = f"""
    <script>
    const STORAGE_KEY = 'stan_export_{timestamp}'; 
    
    function getStore() {{
        return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}');
    }}
    function saveStore(store) {{
        localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
        updateUI();
    }}
    function updateUI() {{
        const store = getStore();
        const count = Object.keys(store).length;
        document.getElementById('counter').innerText = count + ' selected';
        document.querySelectorAll('.doc-check').forEach(cb => {{
            cb.checked = !!store[cb.value];
        }});
    }}
    function onCheck(cb) {{
        let store = getStore();
        if (cb.checked) {{
            store[cb.value] = JSON.parse(cb.getAttribute('data-meta'));
        }} else {{
            delete store[cb.value];
        }}
        saveStore(store);
    }}
    function togglePage(master) {{
        document.querySelectorAll('.doc-check').forEach(cb => {{
            cb.checked = master.checked;
            onCheck(cb);
        }});
    }}
    function exportToZotero() {{
        const store = getStore();
        const items = Object.values(store);
        if (items.length === 0) {{ alert('Please select at least one document.'); return; }}
        
        let ris = "";
        items.forEach(d => {{
            ris += "TY  - RPRT\\r\\n";
            ris += "TI  - " + d.title + "\\r\\n";
            
            let authorStr = d.author || "Unknown";
            let authors = authorStr.split(',');
            authors.forEach(author => {{
                let cleanAuthor = author.trim();
                if (cleanAuthor !== "") {{
                    ris += "AU  - " + cleanAuthor + "\\r\\n";
                }}
            }});
            
            if (d.year && d.year !== "unknown date") {{
                ris += "PY  - " + d.year + "\\r\\n";
            }}
            if (d.full_date && d.full_date.trim() !== "") {{
                ris += "DA  - " + d.full_date + "\\r\\n";
            }}
            if (d.collection && d.collection.trim() !== "") {{
                ris += "PB  - " + d.collection + "\\r\\n"; 
            }}
            if (d.id && d.id.trim() !== "") {{
                ris += "RN  - " + d.id + "\\r\\n"; 
                ris += "M3  - " + d.id + "\\r\\n"; 
            }}
            
            ris += "UR  - " + d.url + "\\r\\n";
            ris += "ER  - \\r\\n\\r\\n";
        }});

        const blob = new Blob([ris], {{type: 'application/x-research-info-systems;charset=utf-8'}});
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'stan_references.ris';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }}
    
    window.addEventListener('DOMContentLoaded', updateUI);
    window.addEventListener('storage', updateUI);
    </script>
    """

    with open(base_path + ".html", 'w', encoding='utf-8') as f:
        f.write(f"<!DOCTYPE html><html><head><meta charset='UTF-8'>{css}{js_export}</head><body><div class='container'>")
        
        f.write(f"""
        <div class='export-panel'>
            <div>
                <input type='checkbox' id='master-check' onclick='togglePage(this)' style='transform:scale(1.5); cursor:pointer;'> 
                <label for='master-check' style='margin-left:10px; font-weight:bold; cursor:pointer;'>Select All on Page</label>
                <span class='counter-badge' id='counter'>0 selected</span>
            </div>
            <button class='btn' onclick='exportToZotero()' style='background:#27ae60; color:white;'>⬇ Export to Zotero (.RIS)</button>
        </div>
        """)

        f.write("<div class='nav'>")
        f.write(f"<h2 style='margin-top:0;'>Stan Report — {term_clean}</h2>")
        f.write(f"<p>Batch generated by Worker: {worker_id}</p>")
        f.write("<div id='STAN_NAV_PANEL'></div>")
        f.write("</div><hr>")

        for d in docs:
            doc_id = str(d.get('id', '')).strip()
            title = str(d.get('title', '')).replace('"', '&quot;')
            author = str(d.get('author', '')).replace('"', '&quot;')
            full_date = str(d.get('document_date', '')).strip()
            year = global_extract_year(full_date)
            coll = str(d.get('collection', '')).replace('"', '&quot;')
            url = f"https://www.industrydocuments.ucsf.edu/docs/{doc_id}"
            
            meta_payload = json.dumps({
                "id": doc_id, 
                "title": title, 
                "author": author, 
                "full_date": full_date, 
                "year": year, 
                "collection": coll, 
                "url": url
            }).replace("'", "&#39;")
            
            full_text = sanitize_text(d.get('ocr_text', ''))
            escaped_text = html.escape(full_text[:50000])

            if query_string:
                try:
                    pos_query = re.split(r'\sNOT\s', query_string, flags=re.IGNORECASE)[0]
                    terms = [t.strip() for t in re.split(r',|\sOR\s', pos_query, flags=re.IGNORECASE) if t.strip()]
                    patterns = [global_generate_term_regex(t) for t in terms if t]
                    for p in patterns:
                        if p: escaped_text = re.sub(f'({p})', r'<mark>\1</mark>', escaped_text, flags=re.IGNORECASE)
                except: pass

            f.write(f"""
            <div class='doc'>
                <div style='display:flex; align-items:start;'>
                    <input type='checkbox' class='doc-check' value='{doc_id}' data-meta='{meta_payload}' 
                           onclick='onCheck(this)'
                           style='margin-right:15px; margin-top:5px; transform:scale(1.5); cursor:pointer;'>
                    <div>
                        <h4 style='margin:0;'>{title}</h4>
                        <p style='font-size:0.85em; color:#555;'>
                            ID: <strong>{doc_id}</strong> | Date: {full_date} | Institution: {coll}<br>
                            Type: {d.get('type', '')} <br>
                            Link: <a href='{url}' target='_blank'>Open Document</a>
                        </p>
                    </div>
                </div>
                <div class='ocr'>{escaped_text}</div>
            </div>
            """)
        f.write("</div></body></html>")

    with open(base_path + ".md", 'w', encoding='utf-8') as f_md, open(base_path + ".txt", 'w', encoding='utf-8') as f_txt:
        header = f"# Stan Search Report\n- Worker ID: {worker_id} | Part: {part_num}\n---\n"
        f_md.write(header); f_txt.write(header)
        
        for d in docs:
            doc_id = d.get('id', 'Unknown_ID')
            title = d.get('title', '')
            year = global_extract_year(d.get('document_date'))
            link = f"https://www.industrydocuments.ucsf.edu/docs/{doc_id}"
            
            f_txt.write(f"\nID: {doc_id} | Title: {title} | Year: {year}\nLink: {link}\nText: {sanitize_text(d.get('ocr_text', ''))[:5000]}\n{'-'*40}\n")
            
            f_md.write(f"## Document ID: {doc_id}\n- **Title:** {title}\n- **Year:** {year}\n- **Link:** {link}\n\n")
            context_snippet = extract_snippets(d.get('ocr_text', ''), query_string)
            f_md.write(f"### Relevant Context:\n{context_snippet}\n\n---\n")

def worker_process_file(file_path, query_string, start_year, end_year,
                                  use_ocr, include_unknown, selected_types,
                                  country_regex_pattern, all_countries_pattern,
                                  COLUMNS, SEARCH_COL, pytesseract_available,
                                  output_folder, search_timestamp):
    
    f_country_re = re.compile(country_regex_pattern, re.IGNORECASE) if country_regex_pattern else None
    all_countries_re = re.compile(all_countries_pattern, re.IGNORECASE) if all_countries_pattern else None
    if use_ocr: configure_tesseract()

    local_docs_counter = 0
    worker_id = uuid.uuid4().hex[:6]
    part_num = 1
    
    LIMIT_BYTES = 45 * 1024 * 1024  
    LIMIT_WORDS = 400000            
    
    current_docs = []
    current_bytes = 0
    current_words = 0
    term_clean = re.sub(r'[\\/*?:"<>|]', "", query_string)[:30] if query_string else "general"

    try:
        if file_path.lower().endswith((".csv", ".parquet")):
            if file_path.lower().endswith(".csv"):
                try: df_iter = pd.read_csv(file_path, sep="|", dtype=str, on_bad_lines='skip', chunksize=2000, encoding='utf-8')
                except: df_iter = pd.read_csv(file_path, sep="|", dtype=str, on_bad_lines='skip', chunksize=2000, encoding='latin-1')
            else:
                def parquet_chunk_generator():
                    parquet_file = pq.ParquetFile(file_path)
                    for batch in parquet_file.iter_batches(batch_size=2000):
                        yield batch.to_pandas().astype(str)
                df_iter = parquet_chunk_generator()

            for chunk in df_iter:
                chunk.fillna("", inplace=True)
                if SEARCH_COL not in chunk.columns: continue
                
                df_f = chunk.copy()
                
                if query_string:
                    if 'title' in df_f.columns:
                        search_target = df_f['title'].astype(str) + " \n " + df_f[SEARCH_COL].astype(str)
                    else:
                        search_target = df_f[SEARCH_COL].astype(str)
                        
                    mask = search_target.apply(lambda x: match_query(x, query_string))
                    df_f = df_f[mask]

                if not df_f.empty and "ALL" not in selected_types and 'type' in df_f.columns:
                    df_f = df_f[df_f['type'].astype(str).isin(selected_types)]
                
                if not df_f.empty and f_country_re:
                    df_f = df_f[df_f[SEARCH_COL].apply(lambda x: bool(f_country_re.search(str(x))))]

                if not df_f.empty:
                    year_mask = pd.Series(False, index=df_f.index)
                    if 'document_date' in df_f.columns:
                        ext_years = df_f['document_date'].apply(global_extract_year)
                        if include_unknown: year_mask |= (ext_years == 'unknown date')
                        num_years = pd.to_numeric(ext_years, errors='coerce')
                        r_m = pd.Series(True, index=df_f.index)
                        if start_year: r_m &= (num_years >= start_year)
                        if end_year: r_m &= (num_years <= end_year)
                        year_mask |= r_m.fillna(False)
                        df_f = df_f[year_mask]
                    elif include_unknown: pass
                    else: df_f = df_f.iloc[0:0]

                if not df_f.empty:
                    if all_countries_re:
                        df_f['mentioned_countries'] = df_f[SEARCH_COL].apply(lambda x: sorted(list(set(f.strip() for f in all_countries_re.findall(str(x)) if f.strip()))))
                    else: df_f['mentioned_countries'] = [[] for _ in range(len(df_f))]
                    
                    for _, row in df_f.iterrows():
                        doc_dict = {col: row.get(col, '') for col in COLUMNS}
                        doc_size = len(str(doc_dict).encode('utf-8'))
                        doc_text = str(row.get(SEARCH_COL, ''))
                        doc_word_count = len(doc_text.split())
                        
                        current_docs.append(doc_dict)
                        current_bytes += doc_size
                        current_words += doc_word_count
                        local_docs_counter += 1
                        
                        if current_bytes >= LIMIT_BYTES or current_words >= LIMIT_WORDS:
                            flush_worker_data(current_docs, output_folder, query_string, term_clean, search_timestamp, worker_id, part_num)
                            part_num += 1
                            current_docs.clear()
                            current_bytes = 0
                            current_words = 0
                            gc.collect()
                
                del df_f; del chunk; gc.collect()

        elif file_path.lower().endswith((".pdf", ".jpg", ".png", ".jpeg", ".tiff")):
            text = ""
            if file_path.lower().endswith(".pdf"):
                with fitz.open(file_path) as doc:
                    for page in doc: 
                        t = page.get_text()
                        if use_ocr and pytesseract_available and len(t.strip()) < 10:
                            pix = page.get_pixmap(dpi=300); img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            t = pytesseract.image_to_string(img, lang='por+eng+chi_sim')
                        text += t + "\n"
            elif use_ocr and pytesseract_available:
                text = pytesseract.image_to_string(Image.open(file_path), lang='por+eng+chi_sim')

            pdf_title = os.path.basename(file_path)
            search_target = pdf_title + " \n " + text
            
            if search_target and match_query(search_target, query_string):
                m_c = []
                if all_countries_re: m_c = sorted(list(set(f.strip() for f in all_countries_re.findall(text) if f.strip())))
                doc_data = {"id": pdf_title, "title": pdf_title, "document_date": global_extract_year(text), "ocr_text": text[:50000], "type": "Digital/Scanned PDF", "collection": "Local Extraction", "mentioned_countries": m_c}
                
                doc_word_count = len(text.split())
                
                current_docs.append(doc_data)
                local_docs_counter += 1
                current_bytes += len(str(doc_data).encode('utf-8'))
                current_words += doc_word_count
                
                if current_bytes >= LIMIT_BYTES or current_words >= LIMIT_WORDS:
                    flush_worker_data(current_docs, output_folder, query_string, term_clean, search_timestamp, worker_id, part_num)
                    part_num += 1
                    current_docs.clear()
                    current_bytes = 0
                    current_words = 0

        if current_docs:
            flush_worker_data(current_docs, output_folder, query_string, term_clean, search_timestamp, worker_id, part_num)
            current_docs.clear()
            gc.collect()

    except Exception as e: print(f"Error processing: {e}")
    
    return (local_docs_counter, file_path)

# =====================================================================
# INFORMATIVE TEXTS
# =====================================================================
ABOUT_TEXT = """STAN: A SCRIPT FOR DATA CURATION, METHODOLOGICAL DESCRIPTION

Introduction
This document describes the Python script "Stan", named as a tribute for UCSF researcher Stanton Glantz and acronym for System for Tactical Analysis of Documents, developed by André Luiz Oliveira da Silva (https://orcid.org/0000-0003-4768-959X), to automate querying, filtering, and extracting a corpus.

Data Ingestion and Processing Mechanism
* CSV and Parquet File Processing — Leverages pandas and pyarrow. 
* Parallel Processing Architecture — Distributes the reading and regex matching across available CPU cores.
* STAN version 2.0 Mode — Integrates Tesseract OCR to extract embedded text from obfuscated documents.

Output Generation
* Compressed JSON files (.jsonl.gz).
* AI-Optimized Markdown Exports chunked safely.
* Human-Readable HTML Reports with RIS Zotero Export."""

QUICK_GUIDE_TEXT = """HOW TO USE STAN:

1. Search Terms: Enter your keywords. (Click the 'Instructions' button to see how to use advanced operators like AND, OR, NOT).
2. Select Folders: Click 'Select Folders' to choose your target directory. Note: All subfolders inside it will be automatically scanned.
3. Filters (Optional): 
   - Year Filter: Set a Start and End year. Check 'Incl. Unknown' to keep documents with missing dates in the results.
   - Apply specific Document Types or Country filters.
4. OCR (Optional): Check 'STAN version 2.0 OCR' to extract text from images and scanned PDFs (this makes the process slower).
5. Run: Click 'START SEARCH' and choose where the reports will be saved.
6. AI Packaging: Once finished, click '📦 Build AI Packs' to automatically merge the files for AI ingestion.

---
WHY PARQUET?
We recommend converting your CSV databases to Parquet using the 'Convert CSVs to Parquet' button. 
Parquet is a highly compressed format that makes searches extremely fast and drastically reduces RAM consumption."""

CONTACT_TEXT = """Developer: 
André Luiz Oliveira da Silva
Email: andre.sp.ensp@gmail.com
CV Lattes: https://lattes.cnpq.br/773185247965855
ORCID: https://orcid.org/0000-0003-4768-959X"""

INSTRUCTIONS_TEXT = """Use the operators below to refine your search:

AND (in all caps) -> Both terms
OR (or comma ,) -> Either term
NOT (in all caps) -> Excludes the following term
" " (quotation marks) -> Exact phrase
* (asterisk at the end) -> Starts with...

---
OCR MODE:
If checked, Stan will try to read text inside IMAGES and scanned PDFs.
This makes the search MUCH SLOWER."""

# =====================================================================
# MAIN CLASS
# =====================================================================
class IDLSearchApp:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("Stan — IDL Assistant")
        self.root.geometry("1100x950")
        self.root.minsize(900, 800)
        
        self.SEARCH_COLUMN = "ocr_text" 
        self.COLUMNS = ["id", "bates", "title", "author", "recipient", "document_date", "ocr_text", "type", "collection", "mentioned_countries"] 

        self.folders = []
        self.status_text = tk.StringVar(value="Ready")
        self.counter_text = tk.StringVar(value="Found: 0")
        self.type_filter_label_var = tk.StringVar(value="Selection: All")
        self.country_filter_label_var = tk.StringVar(value="Countries: All")
        self.use_ocr = tk.BooleanVar(value=False)
        self.include_unknown_date_var = tk.BooleanVar(value=True) 

        self.cancel_event = threading.Event()
        self.tesseract_installed = configure_tesseract()
        self.search_start_time = 0 
        
        self.selected_doc_types = ["ALL"] 
        self.raw_doc_types = sorted(["abstract", "advertisement", "advertising copy", "affidavit", "agenda", "agreement", "appendix", "audio", "bibliography", "binder cover", "blank", "blank form", "blank page", "book", "brief", "budget", "budget review", "calendar", "cartons", "catalog", "certificate", "chart", "cigarette packages", "computer printout", "conference", "contract agreement", "cover sheet", "deposition", "detailed billing", "diagram", "diary", "document", "drawing", "email", "email attachment", "envelope", "excerpt", "expense", "fax", "file folder", "file sheet", "filled in form", "financial", "footnote", "form", "full", "graph", "graphic", "handwritten", "index", "invoice", "lab notebook", "label", "legal", "legal memo", "letter", "list", "magazine article", "manual", "marketing", "medical", "meeting materials", "meeting notes", "memo", "minutes", "motion", "newsletter", "newspaper", "note", "notes", "notice", "oral testimony", "order", "organizational chart", "other", "outline", "packaging", "pamphlet", "patent", "patent abstract", "patent application", "photograph", "pleading", "presentation", "press release", "procedures manual", "promotional material", "proposal", "publication", "purchase order", "questionnaire", "receipt", "regulation", "report", "resolution", "resume", "routing slip", "science", "scientific", "scientific publication", "scientific research proposal", "shipping document", "shipping/receiving", "smoke/tobacco analysis", "specification", "speech", "spreadsheet", "subpoena", "survey questionnaire", "tab page", "tab sheet", "table", "telephone message", "telex", "testimony", "text", "transcript", "trial transcript", "trip", "video"])
        
        self.selected_countries = ["ALL"] 
        self.raw_countries = ["Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei Darussalam", "Bulgaria", "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic", "Chad", "Chile", "China", "Colombia", "Comoros", "Congo", "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czechia", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kuwait", "Kyrgyzstan", "Lao People's Democratic Republic", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Micronesia", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Macedonia", "Norway", "Oman", "Pakistan", "Palau", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Republic of Korea", "Republic of Moldova", "Romania", "Russian Federation", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", "Syrian Arab Republic", "Tajikistan", "Thailand", "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "United States of America", "Uruguay", "Uzbekistan", "Vanuatu", "Venezuela", "Viet Nam", "Yemen", "Zambia", "Zimbabwe"]

        self._build_interface()

    def _build_interface(self):
        self.main = ctk.CTkFrame(self.root, fg_color="transparent")
        self.main.pack(fill="both", expand=True, padx=30, pady=20)
        self.main.grid_columnconfigure(0, weight=1)

        try:
            img = Image.open(resource_path("Stan_logo.png"))
            w, h = img.size; target_w = 200; target_h = int(h * (target_w / w))
            self.stan_photo = ctk.CTkImage(light_image=img, dark_image=img, size=(target_w, target_h))
            ctk.CTkLabel(self.main, image=self.stan_photo, text="").grid(row=0, column=0, pady=(10, 20), sticky="n")
        except:
            ctk.CTkLabel(self.main, text="[ STAN ]", font=("Arial", 30, "bold")).grid(row=0, column=0, pady=20)

        nav_frame = ctk.CTkFrame(self.main, fg_color="transparent")
        nav_frame.grid(row=1, column=0, pady=(0, 20))
        ctk.CTkButton(nav_frame, text="Instructions", command=lambda: messagebox.showinfo("Instructions", INSTRUCTIONS_TEXT), fg_color="#f2f2f2", text_color="black", border_width=1, width=110).pack(side="left", padx=5)
        ctk.CTkButton(nav_frame, text="Quick Guide", command=lambda: messagebox.showinfo("Quick Guide", QUICK_GUIDE_TEXT), fg_color="#f2f2f2", text_color="black", border_width=1, width=140).pack(side="left", padx=5)
        ctk.CTkButton(nav_frame, text="About / Info", command=lambda: messagebox.showinfo("About", ABOUT_TEXT), fg_color="#f2f2f2", text_color="black", border_width=1, width=110).pack(side="left", padx=5)
        ctk.CTkButton(nav_frame, text="Contact", command=lambda: messagebox.showinfo("Contact", CONTACT_TEXT), fg_color="#f2f2f2", text_color="black", border_width=1, width=110).pack(side="left", padx=5)

        input_frame = ctk.CTkFrame(self.main, fg_color="white", border_width=1, border_color="#dbdbdb")
        input_frame.grid(row=2, column=0, sticky="ew", padx=10)
        input_frame.columnconfigure(1, weight=1)

        ctk.CTkLabel(input_frame, text="Search Terms:").grid(row=0, column=0, padx=15, pady=(15, 0), sticky="w")
        self.entry_term = ctk.CTkEntry(input_frame, height=35, placeholder_text="e.g. tobacco AND brazil")
        self.entry_term.grid(row=0, column=1, padx=15, sticky="ew", pady=(15, 10))
        
        self.btn_add_folder = ctk.CTkButton(input_frame, text="Select Folders", command=self._select_folders, fg_color="#1f538d")
        self.btn_add_folder.grid(row=1, column=0, columnspan=2, pady=10)
        
        self.text_folders = ctk.CTkTextbox(input_frame, height=60, border_width=1)
        self.text_folders.grid(row=2, column=0, columnspan=2, padx=15, pady=(0, 15), sticky="ew")

        filters_f = ctk.CTkFrame(self.main, fg_color="#f9f9f9", border_width=1, border_color="#e0e0e0")
        filters_f.grid(row=3, column=0, sticky="ew", pady=20, padx=10)
        
        r1 = ctk.CTkFrame(filters_f, fg_color="transparent"); r1.pack(fill="x", padx=15, pady=10)
        ctk.CTkLabel(r1, text="Year Filter:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        self.entry_start_year = ctk.CTkEntry(r1, width=60, placeholder_text="Start"); self.entry_start_year.pack(side="left", padx=5)
        self.entry_end_year = ctk.CTkEntry(r1, width=60, placeholder_text="End"); self.entry_end_year.pack(side="left", padx=5)
        ctk.CTkCheckBox(r1, text="Incl. Unknown", variable=self.include_unknown_date_var).pack(side="left", padx=15)
        self.check_ocr = ctk.CTkCheckBox(r1, text="STAN version 2.0 OCR", variable=self.use_ocr, text_color="#b22222", font=ctk.CTkFont(weight="bold"))
        self.check_ocr.pack(side="right")

        r2 = ctk.CTkFrame(filters_f, fg_color="transparent"); r2.pack(fill="x", padx=15, pady=5)
        self.btn_f_type = ctk.CTkButton(r2, text="Document Type", command=self._open_type_selector, width=150, fg_color="#e0e0e0", text_color="black")
        self.btn_f_type.pack(side="left", padx=5)
        ctk.CTkLabel(r2, textvariable=self.type_filter_label_var).pack(side="left", padx=10)

        r3 = ctk.CTkFrame(filters_f, fg_color="transparent"); r3.pack(fill="x", padx=15, pady=5)
        self.btn_f_country = ctk.CTkButton(r3, text="Filter by Country", command=self._open_country_selector, width=150, fg_color="#e0e0e0", text_color="black")
        self.btn_f_country.pack(side="left", padx=5)
        ctk.CTkLabel(r3, textvariable=self.country_filter_label_var).pack(side="left", padx=10)

        actions = ctk.CTkFrame(self.main, fg_color="transparent"); actions.grid(row=4, column=0, pady=10)
        ctk.CTkButton(actions, text="Clear All", command=self._clear, fg_color="transparent", border_width=1, text_color="black", width=100).pack(side="left", padx=5)
        self.btn_tl = ctk.CTkButton(actions, text="Generate Timeline", command=self._generate_timeline, fg_color="#2b5d8c", width=140)
        self.btn_tl.pack(side="left", padx=5)
        self.btn_search = ctk.CTkButton(actions, text="START SEARCH", command=self._start_search, fg_color="#b22222", font=ctk.CTkFont(size=14, weight="bold"), height=40, width=180)
        self.btn_search.pack(side="left", padx=15)
        
        self.btn_ai_pack = ctk.CTkButton(actions, text="📦 Build AI Packs (JSON/MD)", command=self._build_ai_packs, state="disabled", fg_color="#e67e22", text_color="white", width=180)
        self.btn_ai_pack.pack(side="left", padx=5)

        self.btn_concat_html = ctk.CTkButton(actions, text="📄 Concat HTML", command=self._trigger_html_concat, state="disabled", fg_color="#27ae60", text_color="white", width=140)
        self.btn_concat_html.pack(side="left", padx=5)
        
        self.btn_cancel = ctk.CTkButton(actions, text="Cancel", command=self._cancel_search, state="disabled", fg_color="gray", width=100)
        self.btn_cancel.pack(side="left", padx=5)
        
        self.status_frame = ctk.CTkFrame(self.main, fg_color="transparent")
        self.status_frame.grid(row=5, column=0, sticky="ew")
        self.status_frame.columnconfigure(0, weight=1)
        self.status_frame.columnconfigure(1, weight=3)
        
        self.progress_bar = ctk.CTkProgressBar(self.status_frame)
        self.progress_bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        self.progress_bar.set(0)
        
        ctk.CTkLabel(self.status_frame, textvariable=self.counter_text, font=ctk.CTkFont(weight="bold")).grid(row=1, column=0, sticky="w")
        ctk.CTkLabel(self.status_frame, textvariable=self.status_text, font=ctk.CTkFont(slant="italic")).grid(row=1, column=1, sticky="e")
        self.status_frame.grid_remove()

        ctk.CTkLabel(self.root, text="By Andre", font=ctk.CTkFont(size=10, slant="italic"), text_color="#777777").place(relx=1.0, rely=1.0, x=-20, y=-10, anchor="se")
        self.btn_conv = ctk.CTkButton(self.root, text="Convert CSVs to Parquet", command=self._popup_parquet_conversion, fg_color="#e67e22", height=40)
        self.btn_conv.place(relx=1.0, rely=1.0, x=-20, y=-35, anchor="se")

        self.widgets_to_toggle = [self.btn_add_folder, self.btn_search, self.entry_term, self.btn_conv, self.btn_f_type, self.btn_f_country, self.btn_tl, self.entry_start_year, self.entry_end_year, self.check_ocr]

    def _open_type_selector(self):
        win = ctk.CTkToplevel(self.root); win.title("Doc Types"); win.geometry("450x600"); win.grab_set()
        frame = ctk.CTkFrame(win); frame.pack(fill="both", expand=True, padx=10, pady=10)
        listbox = tk.Listbox(frame, selectmode="multiple", font=("Segoe UI", 11)); listbox.pack(fill="both", expand=True, side="left")
        scroll = tk.Scrollbar(frame); scroll.pack(side="right", fill="y"); listbox.config(yscrollcommand=scroll.set); scroll.config(command=listbox.yview)
        for t in ["All"] + self.raw_doc_types: listbox.insert("end", t)
        def confirm():
            sels = [listbox.get(i) for i in listbox.curselection()]
            self.selected_doc_types = ["ALL"] if "All" in sels or not sels else sels
            self.type_filter_label_var.set(f"Selection: {'All' if 'ALL' in self.selected_doc_types else len(self.selected_doc_types)}")
            win.destroy()
        ctk.CTkButton(win, text="Confirm", command=confirm).pack(pady=10)

    def _open_country_selector(self):
        win = ctk.CTkToplevel(self.root); win.title("Countries"); win.geometry("450x600"); win.grab_set()
        frame = ctk.CTkFrame(win); frame.pack(fill="both", expand=True, padx=10, pady=10)
        listbox = tk.Listbox(frame, selectmode="multiple", font=("Segoe UI", 11)); listbox.pack(fill="both", expand=True, side="left")
        scroll = tk.Scrollbar(frame); scroll.pack(side="right", fill="y"); listbox.config(yscrollcommand=scroll.set); scroll.config(command=listbox.yview)
        for c in ["All"] + self.raw_countries: listbox.insert("end", c)
        def confirm():
            sels = [listbox.get(i) for i in listbox.curselection()]
            self.selected_countries = ["ALL"] if "All" in sels or not sels else sels
            self.country_filter_label_var.set(f"Countries: {'All' if 'ALL' in self.selected_countries else len(self.selected_countries)}")
            win.destroy()
        ctk.CTkButton(win, text="Confirm", command=confirm).pack(pady=10)

    def _select_folders(self):
        p = filedialog.askdirectory()
        if p and p not in self.folders: self.folders.append(p); self.text_folders.insert("end", f"{p}\n")

    def _clear(self):
        self.entry_term.delete(0, "end"); self.text_folders.delete("0.0", "end"); self.folders = []
        self.entry_start_year.delete(0, "end"); self.entry_end_year.delete(0, "end")
        self.selected_doc_types = ["ALL"]; self.selected_countries = ["ALL"]
        self.type_filter_label_var.set("Selection: All"); self.country_filter_label_var.set("Countries: All")

    def _popup_parquet_conversion(self):
        source = filedialog.askdirectory(title="Select the ROOT folder for CSV files")
        if not source: return
        win = ctk.CTkToplevel(self.root); win.title("Parquet Configuration"); win.geometry("600x300"); win.grab_set()
        ctk.CTkLabel(win, text="Save Parquet structure to:", font=("Arial", 15, "bold")).pack(pady=20)
        option = tk.StringVar(value="original")
        ctk.CTkRadioButton(win, text="Original directories (Recommended)", variable=option, value="original").pack(pady=10)
        ctk.CTkRadioButton(win, text="New location (Preserves structure)", variable=option, value="new").pack(pady=10)
        def start():
            choice = option.get(); win.destroy(); dest = None
            if choice == "new":
                dest = filedialog.askdirectory()
                if not dest: return
            threading.Thread(target=self._execute_parquet_conversion, args=(source, choice, dest), daemon=True).start()
        ctk.CTkButton(win, text="Convert Now", command=start, fg_color="#e67e22").pack(pady=20)

    def _execute_parquet_conversion(self, source, choice, dest):
        for w in self.widgets_to_toggle: w.configure(state="disabled")
        csv_files = [os.path.join(r, f) for r, _, fs in os.walk(source) for f in fs if f.endswith('.csv')]
        if not csv_files: self.root.after(0, lambda: [w.configure(state="normal") for w in self.widgets_to_toggle]); return
        
        self.root.after(0, self.status_frame.grid); self.root.after(0, self.progress_bar.set, 0)
        created_folders = set()
        
        for i, csv in enumerate(csv_files):
            fname = os.path.basename(csv)
            if len(fname) > 45: fname = fname[:42] + "..." 
            self.root.after(0, self.status_text.set, f"Converting: {fname} ({i+1}/{len(csv_files)})")
            
            csv_dir = os.path.dirname(csv)
            dir_dest = csv_dir + "_parquet" if choice == "original" else os.path.join(dest, os.path.relpath(csv_dir, source) + "_parquet")
            os.makedirs(dir_dest, exist_ok=True); created_folders.add(dir_dest)
            try: pd.read_csv(csv, sep='|', on_bad_lines='skip', dtype=str).to_parquet(os.path.join(dir_dest, os.path.basename(csv).replace('.csv', '.parquet')))
            except: pass
            self.root.after(0, self.progress_bar.set, (i+1)/len(csv_files))
            
        self.root.after(0, lambda: [w.configure(state="normal") for w in self.widgets_to_toggle])
        self.root.after(0, self.status_frame.grid_remove)
        self.root.after(0, messagebox.showinfo, "Success", f"Created {len(created_folders)} '_parquet' folders.")

    def _generate_timeline(self):
        path = filedialog.askopenfilename(filetypes=[("GZIP JSON Lines", "*.jsonl.gz")])
        if not path: return
        try:
            data = []
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                for line in f: data.append(json.loads(line))
            df = pd.DataFrame(data)
            df['Year'] = df['document_date'].apply(global_extract_year)
            df = df[df['Year'] != 'unknown date']
            df['Year'] = pd.to_numeric(df['Year'])
            counts = df['Year'].value_counts().sort_index()
            plt.figure(figsize=(10, 5)); plt.plot(counts.index, counts.values, marker='o', color='#b22222', linewidth=2)
            plt.title("Document Volume Over Time"); plt.xlabel("Year"); plt.ylabel("Number of Documents"); plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(path.replace('.jsonl.gz', '_timeline.png'), dpi=300); plt.show()
        except Exception as e: messagebox.showerror("Error", str(e))

    def _cancel_search(self):
        self.cancel_event.set()
        self.status_text.set("Canceling... Please wait for workers to stop.")
        self.btn_cancel.configure(state="disabled")

    def _start_search(self):
        if not self.folders: messagebox.showwarning("Error", "Select folders first."); return
        self.output_folder = filedialog.askdirectory(title="Report Destination Folder")
        if not self.output_folder: return
        
        self.search_start_time = time.time()
        self.cancel_event.clear()
        
        for w in self.widgets_to_toggle: w.configure(state="disabled")
        self.btn_ai_pack.configure(state="disabled")
        self.btn_concat_html.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        
        self.status_frame.grid()
        self.search_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.total_docs_found = 0
        
        threading.Thread(target=self._execute_parallel, daemon=True).start()

    def _execute_parallel(self):
        files = [os.path.join(r, f) for p in self.folders for r, _, fs in os.walk(p) for f in fs if f.lower().endswith((".csv", ".parquet", ".pdf"))]
        total = len(files)
        if total == 0: self.root.after(0, self._finalize); return
        
        max_workers = max(1, (os.cpu_count() or 4) - 1)
        
        q = self.entry_term.get()
        start_y = int(self.entry_start_year.get()) if self.entry_start_year.get().isdigit() else None
        end_y = int(self.entry_end_year.get()) if self.entry_end_year.get().isdigit() else None
        c_pat = r'\b(?:' + '|'.join(map(re.escape, self.selected_countries)) + r')\b' if "ALL" not in self.selected_countries else None
        all_c_pat = r'\b(?:' + '|'.join(map(re.escape, self.raw_countries)) + r')\b'

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(worker_process_file, c, q, start_y, end_y, self.use_ocr.get(), self.include_unknown_date_var.get(), self.selected_doc_types, c_pat, all_c_pat, self.COLUMNS, self.SEARCH_COLUMN, PYTESSERACT_AVAILABLE, self.output_folder, self.search_timestamp): c for c in files}
            
            for i, f in enumerate(concurrent.futures.as_completed(futures)):
                if self.cancel_event.is_set(): 
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                
                c_name = futures[f]
                fname = os.path.basename(c_name)
                if len(fname) > 45: fname = fname[:42] + "..."
                
                try:
                    res = f.result()
                    if res:
                        count, _ = res
                        self.total_docs_found += count
                        self.root.after(0, self.counter_text.set, f"Found: {self.total_docs_found}")
                            
                except Exception as e: 
                    pass
                
                self.root.after(0, self.status_text.set, f"File: {fname} ({i+1}/{total})")
                self.root.after(0, self.progress_bar.set, (i+1)/total)

        self.root.after(0, self._finalize)

    def _open_folder(self, path):
        try:
            if sys.platform == 'win32': os.startfile(os.path.realpath(path))
            elif sys.platform == 'darwin': subprocess.Popen(['open', path])
            else: subprocess.Popen(['xdg-open', path])
        except Exception as e: print(f"Error opening folder: {e}")

    # --- FULLY SECURED PAGINATION AND INDEX GENERATION ---
    def _finalize(self):
        self.status_text.set("Organizing pages and generating Index...")
        self.btn_cancel.configure(state="disabled")
        threading.Thread(target=self._stitch_pagination, daemon=True).start()

    def _stitch_pagination(self):
        if hasattr(self, 'output_folder') and self.output_folder and os.path.exists(self.output_folder):
            
            time.sleep(3.5)
            
            try:
                html_files = [f for f in os.listdir(self.output_folder) if f.endswith('.html') and self.search_timestamp in f and not f.startswith("INDEX_")]
                
                def get_part_num(f_name):
                    m = re.search(r'_p(\d+)\.html$', f_name)
                    return int(m.group(1)) if m else 0
                
                html_files.sort(key=lambda x: (x.split('_p')[0], get_part_num(x)))
                total_pages = len(html_files)
                index_filename = f"INDEX_{self.search_timestamp}.html"
                
                for i, filename in enumerate(html_files):
                    filepath = os.path.join(self.output_folder, filename)
                    prev_link = html_files[i-1] if i > 0 else None
                    next_link = html_files[i+1] if i < total_pages - 1 else None
                    
                    nav_html = f"<div class='nav-buttons' style='margin-top:10px;'><p><strong>Page {i+1} of {total_pages}</strong></p>"
                    if prev_link: nav_html += f"<a href='{prev_link}' class='btn'>&laquo; Previous</a> "
                    nav_html += f"<a href='{index_filename}' class='btn' style='background:#e67e22; color:white;'>Back to Index</a> "
                    if next_link: nav_html += f"<a href='{next_link}' class='btn'>Next &raquo;</a>"
                    if total_pages == 1: nav_html += "<br><span style='opacity:0.6; font-size: 0.9em; font-weight:bold;'>Single Page (End of results)</span>"
                    nav_html += "</div>"
                    
                    for tentativa in range(6):
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                content = f.read()
                            marker = "<div id='STAN_NAV_PANEL'></div>"
                            if marker in content:
                                pieces = content.split(marker)
                                new_content = pieces[0] + nav_html + pieces[1]
                                with open(filepath, 'w', encoding='utf-8') as f_out:
                                    f_out.write(new_content)
                            break 
                        except Exception:
                            time.sleep(1.0)
                            
                duration_sec = time.time() - self.search_start_time
                mins, secs = divmod(int(duration_sec), 60)
                time_str = f"{mins} min {secs} sec" if mins > 0 else f"{secs} sec"

                if total_pages > 0:
                    termo = self.entry_term.get() or "None (Blank search / All)"
                    start_y = self.entry_start_year.get() or "No limit"
                    end_y = self.entry_end_year.get() or "No limit"
                    inc_unk = "Yes" if self.include_unknown_date_var.get() else "No"
                    ocr_status = "Enabled (Image reading via STAN version 2.0 mode)" if self.use_ocr.get() else "Disabled (Native text only)"
                    tipos_str = "All types" if "ALL" in self.selected_doc_types else str(len(self.selected_doc_types)) + " specific types"
                    paises_str = "All countries" if "ALL" in self.selected_countries else str(len(self.selected_countries)) + " specific countries"
                    folders_li = "".join([f"<li>{os.path.normpath(f)}</li>" for f in self.folders])
                    index_path = os.path.join(self.output_folder, index_filename)
                    
                    index_html_content = f"""<!DOCTYPE html><html><head><meta charset='UTF-8'><title>Methodological Index - STAN</title>
<style>
    body{{font-family:sans-serif;padding:20px;background:#f4f4f4;color:#333;}} 
    .container{{max-width:900px;margin:auto;background:white;padding:30px;border-radius:8px;box-shadow:0 2px 5px rgba(0,0,0,0.1);}} 
    a{{display:inline-block;padding:10px 15px;margin:5px;background:#1f538d;color:white;text-decoration:none;border-radius:4px;font-weight:bold;}} 
    a:hover{{background:#123254;transform:translateY(-1px);}} 
    h1{{color:#1f538d;text-align:center;border-bottom:2px solid #1f538d;padding-bottom:10px;}}
    .metadata-box {{background:#fafafa; border:1px solid #ddd; padding:15px; border-radius:5px; margin-bottom:20px;}}
    .metadata-box h3 {{margin-top:0; color:#b22222;}}
    ul {{margin-top:5px; margin-bottom:5px; padding-left:20px;}}
    .links-grid {{display:flex; flex-wrap:wrap; justify-content:center; gap:10px; margin-top:20px;}}
</style>
</head>
<body>
<div class='container'>
    <h1>STAN Extraction Report</h1>
    <div class='metadata-box'>
        <h3>Analysis Summary</h3>
        <ul>
            <li><strong>Searched Term(s):</strong> <span style='color:#b22222; font-weight:bold;'>{termo}</span></li>
            <li><strong>Documents Found (IDs):</strong> {self.total_docs_found} documents</li>
            <li><strong>Generated Reports:</strong> {total_pages} page(s)</li>
            <li><strong>Processing Time:</strong> {time_str}</li>
            <li><strong>Extraction Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
        </ul>
    </div>
    <div class='metadata-box'>
        <h3>Methodological Parameters (Applied Filters)</h3>
        <ul>
            <li><strong>Year Filter:</strong> {start_y} to {end_y} (Include unknown dates: {inc_unk})</li>
            <li><strong>Document Filter:</strong> {tipos_str}</li>
            <li><strong>Country Filter:</strong> {paises_str}</li>
            <li><strong>OCR Engine:</strong> {ocr_status}</li>
        </ul>
    </div>
    <div class='metadata-box'>
        <h3>Scanned Databases (Directories)</h3>
        <ul style='font-size: 0.9em; font-family: monospace;'>{folders_li}</ul>
    </div>
    <h3 style='text-align:center; margin-top:30px;'>Results Navigation</h3>
    <div class='links-grid'>
"""
                    for i, html_file in enumerate(html_files):
                        index_html_content += f"<a href='{html_file}'>Page {i+1}</a>"
                    index_html_content += "</div></div></body></html>"
                    
                    try:
                        with open(index_path, 'w', encoding='utf-8') as f: f.write(index_html_content)
                    except: pass

            except Exception as e:
                self.root.after(0, lambda err=e: messagebox.showwarning("Pagination Warning", f"Could not stitch the links: {err}"))

        self.root.after(0, self._finish_ui)

    def _finish_ui(self):
        self.status_frame.grid_remove()
        for w in self.widgets_to_toggle: w.configure(state="normal")
        
        if hasattr(self, 'output_folder') and self.output_folder and os.path.exists(self.output_folder):
            self.btn_ai_pack.configure(state="normal")
            self.btn_concat_html.configure(state="normal")
        else:
            self.btn_ai_pack.configure(state="disabled")
            self.btn_concat_html.configure(state="disabled")
        
        if self.cancel_event.is_set():
            messagebox.showinfo("Canceled", "The search was safely canceled.")
        else:
            msg = "Search finished!\n\nAn 'INDEX' file was also created in your destination folder.\n\nYou can now click 'Build AI Packs' to automatically concatenate the AI ingestion files."
            messagebox.showinfo("Done", msg)

    # =====================================================================
    # CONCATENAÇÃO HTML INDEPENDENTE E SEGURA (NEW)
    # =====================================================================
    def _trigger_html_concat(self):
        initial = getattr(self, 'output_folder', os.getcwd())
        target_dir = filedialog.askdirectory(title="Select Folder with HTML files to concatenate", initialdir=initial)
        if not target_dir: return
        
        self.status_frame.grid()
        self.progress_bar.set(0)
        self.status_text.set("Concatenating HTML files...")
        
        for w in self.widgets_to_toggle + [self.btn_ai_pack, self.btn_concat_html, self.btn_cancel]: 
            w.configure(state="disabled")
            
        threading.Thread(target=self._execute_html_concat, args=(target_dir,), daemon=True).start()

    def _execute_html_concat(self, target_dir):
        try:
            html_files = sorted([f for f in glob.glob(os.path.join(target_dir, "*.html")) if "INDEX" not in os.path.basename(f) and "Merged_HTML" not in os.path.basename(f)])
            if not html_files:
                self.root.after(0, messagebox.showinfo, "Info", "No HTML files found to concatenate in the selected folder.")
                return

            out_dir = os.path.join(target_dir, "Concatenated_HTML")
            os.makedirs(out_dir, exist_ok=True)
            
            base_nome = "Stan_Merged_HTML"
            max_html_size = 100 * 1024 * 1024  # 100 MB de limite dinâmico
            
            part_html = 1
            curr_html_size = 0
            
            out_html_path = os.path.join(out_dir, f"{base_nome}_Part{part_html}.html")
            out_html = open(out_html_path, 'w', encoding='utf-8')
            
            first_file_header = ""
            for fpath in html_files:
                with open(fpath, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    if '<hr>' in content:
                        first_file_header = content.split('<hr>', 1)[0] + '<hr>\n'
                        break
            
            if not first_file_header:
                first_file_header = "<!DOCTYPE html><html><head><meta charset='UTF-8'></head><body><div class='container'>\n"

            out_html.write(first_file_header)
            
            for i, fpath in enumerate(html_files):
                try:
                    with open(fpath, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                except Exception:
                    continue
                    
                hr_split = content.split('<hr>', 1)
                body_content = hr_split[1] if len(hr_split) > 1 else content
                body_content = body_content.replace('</div></body></html>', '')
                
                docs = body_content.split("<div class='doc'>")
                
                for doc in docs:
                    if not doc.strip() or 'STAN_NAV_PANEL' in doc: continue
                    doc_str = "<div class='doc'>" + doc
                    doc_bytes = doc_str.encode('utf-8')
                    doc_size = len(doc_bytes)
                    
                    if curr_html_size + doc_size > max_html_size and curr_html_size > 0:
                        out_html.write("</div></body></html>")
                        out_html.close()
                        
                        part_html += 1
                        out_html_path = os.path.join(out_dir, f"{base_nome}_Part{part_html}.html")
                        out_html = open(out_html_path, 'w', encoding='utf-8')
                        
                        out_html.write(first_file_header)
                        out_html.write("<div style='background:#ffeb3b; padding:10px; margin-bottom:15px; font-weight:bold; text-align:center;'>Continuação dos resultados (Parte "+str(part_html)+").</div>\n")
                        curr_html_size = 0
                        
                    out_html.write(doc_str)
                    curr_html_size += doc_size
                    
                self.root.after(0, self.progress_bar.set, (i+1)/len(html_files))
            
            if curr_html_size > 0:
                out_html.write("</div></body></html>")
                out_html.close()

            self.root.after(0, self.status_text.set, "HTML concatenation finished!")
            self.root.after(0, messagebox.showinfo, "Success", f"Concatenated HTML files were created at:\n{out_dir}")
            self.root.after(0, self._open_folder, out_dir)

        except Exception as e:
            self.root.after(0, messagebox.showerror, "Concatenation Error", f"An error occurred:\n{e}")
        finally:
            self.root.after(0, self.status_frame.grid_remove)
            self.root.after(0, lambda: [w.configure(state="normal") for w in self.widgets_to_toggle + [self.btn_ai_pack, self.btn_concat_html]])


    # =====================================================================
    # MOTOR DE PÓS-PROCESSAMENTO DESACOPLADO (CONCATENAÇÃO APENAS IA)
    # =====================================================================
    def _build_ai_packs(self):
        if not hasattr(self, 'output_folder') or not os.path.exists(self.output_folder):
            return
        
        self.status_frame.grid()
        self.progress_bar.set(0)
        self.status_text.set("Building AI Packs... Compressing and stitching.")
        
        for w in self.widgets_to_toggle + [self.btn_ai_pack, self.btn_concat_html, self.btn_cancel]: 
            w.configure(state="disabled")
            
        threading.Thread(target=self._execute_ai_packaging, daemon=True).start()

    def _execute_ai_packaging(self):
        try:
            import tiktoken
            try: enc = tiktoken.encoding_for_model("gpt-4")
            except: enc = tiktoken.get_encoding("cl100k_base")
            
            ai_folder = os.path.join(self.output_folder, "AI_Ready_Packages")
            os.makedirs(ai_folder, exist_ok=True)
            
            base_nome = "Stan_AI_Pack"
            max_md_words = 400000
            max_json_mb = 48
            max_json_tokens = 2000000
            max_json_bytes = max_json_mb * 1024 * 1024
            
            # --- 1. Concatenar Markdown (Para NotebookLM) ---
            md_files = sorted(glob.glob(os.path.join(self.output_folder, "*.md")))
            if md_files:
                self.root.after(0, self.status_text.set, "Stitching Markdown files for NotebookLM...")
                part_md = 1
                curr_words = 0
                out_md_path = os.path.join(ai_folder, f"{base_nome}_MD_Part{part_md}.md")
                out_md = open(out_md_path, 'w', encoding='utf-8')
                
                for i, fpath in enumerate(md_files):
                    with open(fpath, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        words = len(content.split())
                        
                        if curr_words + words > max_md_words and curr_words > 0:
                            out_md.close()
                            part_md += 1
                            out_md_path = os.path.join(ai_folder, f"{base_nome}_MD_Part{part_md}.md")
                            out_md = open(out_md_path, 'w', encoding='utf-8')
                            curr_words = 0
                            
                        out_md.write(content + "\n\n")
                        curr_words += words
                    self.root.after(0, self.progress_bar.set, (i/len(md_files))*0.5)
                out_md.close()

            # --- 2. Concatenar JSONL.GZ (Para ChatGPT) - COM SEGURANÇA DE STACK ---
            gz_files = sorted(glob.glob(os.path.join(self.output_folder, "*.jsonl.gz")))
            if gz_files:
                self.root.after(0, self.status_text.set, "Stitching and Re-Compressing JSONL for ChatGPT...")
                part_gz = 1
                curr_tokens = 0
                curr_bytes = 0
                out_gz_path = os.path.join(ai_folder, f"{base_nome}_JSON_Part{part_gz}.jsonl.gz")
                out_gz = gzip.open(out_gz_path, 'wt', encoding='utf-8')
                
                for i, fpath in enumerate(gz_files):
                    try:
                        with gzip.open(fpath, 'rt', encoding='utf-8', errors='ignore') as infile:
                            content = infile.read()
                    except: continue
                    
                    content_chunks = [content[i:i+50000] for i in range(0, len(content), 50000)]
                    tokens = sum(len(enc.encode(chunk)) for chunk in content_chunks)
                    
                    b_len = len(content.encode('utf-8'))
                    
                    if (curr_tokens + tokens > max_json_tokens) or (curr_bytes + b_len > max_json_bytes):
                        if curr_tokens > 0:
                            out_gz.close()
                            part_gz += 1
                            out_gz_path = os.path.join(ai_folder, f"{base_nome}_JSON_Part{part_gz}.jsonl.gz")
                            out_gz = gzip.open(out_gz_path, 'wt', encoding='utf-8')
                            curr_tokens = 0
                            curr_bytes = 0
                            
                    out_gz.write(content) 
                    curr_tokens += tokens
                    curr_bytes += b_len
                    self.root.after(0, self.progress_bar.set, 0.5 + (i/len(gz_files))*0.5)
                out_gz.close()

            self.root.after(0, self.status_text.set, "AI Packs generated successfully!")
            self.root.after(0, messagebox.showinfo, "Success", f"AI Ready Packages were successfully created inside:\n{ai_folder}")
            self.root.after(0, self._open_folder, ai_folder)

        except Exception as e:
            self.root.after(0, messagebox.showerror, "Packaging Error", f"An error occurred while building the AI packs:\n{e}")
        finally:
            self.root.after(0, self.status_frame.grid_remove)
            self.root.after(0, lambda: [w.configure(state="normal") for w in self.widgets_to_toggle + [self.btn_ai_pack, self.btn_concat_html]])

if __name__ == "__main__":
    ctk.set_appearance_mode("Light")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    app = IDLSearchApp(root)
    
    def on_closing():
        app.cancel_event.set()
        root.destroy()
        os._exit(0)
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()