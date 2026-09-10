# STAN
Offline, decentralized data-curation and text-mining pipeline designed to extract targeted research corpora from the UCSF Industry Documents Library for AI/NLP workflows.
# STAN: System for Tactical Analysis of Documents (v2.1)

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0003--4768--959X-green.svg)](https://orcid.org/0000-0003-4768-959X)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **STAN** is a software tool for data curation and textual mining, named in tribute to UCSF researcher **Stanton Glantz**. Developed by **André Luiz Oliveira da Silva** (ORCID: [0000-0003-4768-959X](https://orcid.org/0000-0003-4768-959X)).

---

## 📌 Overview

**STAN** automates the querying, filtering, and extraction of targeted research-ready corpora from large-scale documentary collections—specifically the **UCSF Industry Documents Library (IDL)**. 

Unlike broad web scrapers, STAN operates as a **strictly offline, decentralized desktop pipeline** that processes raw datasets directly on local storage (HDD/SSD/LAN). It converts heterogeneous, raw IDL data into structured outputs optimized for Artificial Intelligence (AI), Large Language Models (LLMs), and Natural Language Processing (NLP) pipelines (e.g., RAG, NotebookLM, ChatGPT).

---

## 🚀 Key Features

* **Desktop GUI (CustomTkinter):** Fully parameterized graphical interface—no code modification required.
* **Advanced Boolean Retrieval:** Supports `AND`, `OR`, `NOT`, exact phrases (`"..."`), and wildcard suffix matching (`term*`).
* **Multimodal Data Ingestion:**
  * Chunked processing of legacy **CSV datasets** via `pandas`.
  * Page-by-page text extraction from **PDFs** using `PyMuPDF` (`fitz`).
* **Heimdall Mode (OCR):**
  * Integrated multi-language OCR module based on localized `Tesseract OCR` and `Pillow (PIL)`.
  * Supports English, Portuguese, and Simplified Chinese for non-searchable PDFs and images (`.jpg`, `.png`, `.tiff`).
* **Multiprocessing Architecture:** Concurrent text matching and OCR routines powered by `ProcessPoolExecutor`.
* **AI & RAG Preparation:**
  * **Dual-trigger batching** (45 MB physical limit OR 400,000-word ceiling) preventing RAM overflow.
  * Exports machine-readable newline-delimited compressed JSON Lines (`.jsonl.gz`) and structured Markdown (`.md`).
  * **AI Packs Engine:** Integrated `tiktoken` consolidation into super-packages up to 2,000,000 tokens for immediate LLM context ingestion.
* **Academic Reference & Human Reporting:**
  * Paginated, interactive `.html` reports with source hyperlinking.
  * Direct `.RIS` export panel compatible with reference managers like **Zotero**.
  * Dynamic HTML stitching capped at 100 MB chunks to avoid browser crashes.
  * Automated timeline generation (`.png`) for corpus volume visualization.

---

## 🧭 Workflow & Usage

1. **Download Source Data:** Obtain official datasets from the [UCSF Box Repository](https://ucsf.app.box.com/v/IDL-DataSets/folder/78644252849) and place them in a local folder.
2. **Search Terms:** Input Boolean queries and wildcard patterns in the main search bar.
3. **Set Filters:**
   * **Temporal:** Specify Start and End years (toggle *Incl. Unknown* if needed).
   * **Document Type:** Filter native categories (e.g., Reports, Letters, Memorandums).
   * **Geographic:** Filter records matching World Health Organization (WHO) member states.
4. **Configure OCR (Optional):** Check `Heimdall mode (OCR)` to force extraction on scanned materials.
5. **Execute:** Click `START SEARCH` and select the output destination directory.
6. **Post-Processing:**
   * `📦 Build AI Packs (JSON/MD)`: Bundles token-optimized packages into `AI_Ready_Packages`.
   * `📄 Concat HTML`: Merges reports into continuous reading files.
   * `Generate Timeline`: Visualizes document frequency distributions over time.

---

## ⚙️ Technical Architecture & Precision: STAN vs. UCSF Portal

Searches executed in STAN may return different document counts compared to the UCSF live web portal due to methodological design:

| Feature | UCSF Online Portal | STAN (v2.1) |
| :--- | :--- | :--- |
| **Search Paradigm** | Dragnet / Broad Discovery | Scalpel / Analytical Precision |
| **Target Scope** | Text body + extensive archivist metadata | Core document body and title only |
| **Matching Logic** | Fuzzy matching & semantic search | Strict Regex (`\b` word boundaries) + manual wildcards |
| **Source State** | Dynamic, cloud-updated live database | Static, local snapshot |
| **Hardware Throttling** | Cloud-based indexing | Local RAM limits (e.g., 50k character safety buffer on OCR pages) |

---

## 🛡️ About the Name "Heimdall"

In Norse mythology, **Heimdall** is the vigilant guardian endowed with extraordinary perception across different realms. In STAN, *Heimdall Mode* extends textual visibility across heterogeneous, low-text, or deliberately obfuscated document types.

---

## 👤 Author & Citation

**André Luiz Oliveira da Silva**  
* Brazilian HEalth Regulatory Agency (ANVISA)/ Regulatory & Tobacco Control Research  
* ORCID: [0000-0003-4768-959X](https://orcid.org/0000-0003-4768-959X)

If you use STAN in your academic research, please cite this repository and the accompanying methodological documentation.
