# EOWR Document Intelligence Pipeline

A RAG (Retrieval-Augmented Generation) system that lets you ask questions 
about End-of-Well Report documents and get cited answers grounded in your 
actual well data.

---

## Folder structure

```
eowr_agent/
│
├── schema.py              Shared data contracts (typed models for extraction + classification results)
├── main.py                Entry point: extract + classify a single file
│
├── extractors/            Turns raw files into clean Markdown
│   ├── extractor_factory.py    Auto-detects file type, routes to the right extractor
│   ├── csv_extractor.py
│   ├── xlsx_extractor.py       Handles merged cells, report-style layouts, multi-sheet
│   ├── docx_extractor.py
│   └── pdf_extractor.py        Handles table detection, text deduplication, truncation
│
├── generators/            One-time setup scripts (run before using the app)
│   ├── generate_labels_from_folders.py   Scans data/ → labels.xlsx
│   ├── build_training_data.py            labels.xlsx + data/ → training_data.csv
│   ├── train_classifier.py               training_data.csv → classifier_model.pkl
│   ├── predict_category.py               Classify a new unlabeled file
│   ├── classifier_model.pkl              Trained TF-IDF + Logistic Regression model
│   ├── training_data.csv                 Extracted content + ground-truth labels
│   └── labels.xlsx                       filename → category mapping
│
├── retrieval/             Query-time logic (runs when the app answers a question)
│   ├── chunker.py                Split markdown into chunks for embedding
│   ├── search.py                 Store chunks in ChromaDB + hybrid BM25/vector retrieval
│   ├── answer_query.py           Detect question category + retrieve + Gemini cited answer
│   ├── bulk_load_documents.py    Load all 188 documents into the database (run once)
│   └── vector_db/                ChromaDB data (auto-created when bulk_load runs)
│
├── app/
│   └── streamlit_app.py          The web UI
│
├── data/                  Your 188 real documents, organized by category subfolder
│                          (not in this repo — copy your own data/ here)
│
└── tests/                 Test scripts and small sample files
```

---

## Setup

### 1. Install dependencies

```bash
pip install pandas openpyxl python-docx pdfplumber pymupdf pydantic \
            scikit-learn chromadb sentence-transformers rank-bm25 \
            google-genai streamlit tabulate
```

### 2. Set your Gemini API key

Get a free key from https://aistudio.google.com/apikey

```powershell
$env:GEMINI_API_KEY = "your-key-here"
```

---

## Usage

### One-time setup: load all documents

```powershell
cd D:\SwarmLens\EOWR\eowr_agent\retrieval
python bulk_load_documents.py "D:\SwarmLens\EOWR\eowr_agent\data"
```

This embeds all 188 documents into ChromaDB and builds the BM25 index.

### Run the app

```powershell
cd D:\SwarmLens\EOWR\eowr_agent
streamlit run app/streamlit_app.py
```

### Ask a question directly from the terminal (no UI)

```powershell
cd D:\SwarmLens\EOWR\eowr_agent\retrieval
python answer_query.py "What NPT incidents happened during drilling?"
```

### Classify a new document

```powershell
cd D:\SwarmLens\EOWR\eowr_agent\generators
python predict_category.py "..\data\some_new_file.xlsx"
```

---

## How it works

### Offline (run once per document)

```
Raw file → Extract to Markdown → Classify (TF-IDF model) → Chunk → Embed → ChromaDB + BM25
```

### Online (every question)

```
Question → Gemini detects category → Hybrid search (BM25 + vector + RRF) 
        → Top chunks → Gemini generates cited answer → Streamlit displays
```

### Key design decisions

| Decision | Why |
|----------|-----|
| Markdown as extraction output | Preserves headings, tables, key-value structure in one human+LLM-readable format |
| TF-IDF + Logistic Regression classifier | 97% accuracy on full documents; right tool for long text with rich vocabulary |
| Gemini for question category detection | Classifier tested poorly on short questions (6/10); LLM understands intent from sparse text |
| BM25 + vector hybrid search | Vector search misses exact identifiers (well names, API numbers); BM25 covers that gap |
| Reciprocal Rank Fusion (RRF) | BM25 and vector scores are on different scales; RRF combines by rank position, not raw score |
| Citations enforced in prompt | Prevents LLM from blending retrieved facts with general knowledge |
| class_weight=balanced in classifier | Corrects for drilling (82 docs) dominating smaller categories (5 docs each) |

---

## Notes

- `data/` is NOT included — copy your real 188-document folder structure here
- The classifier was trained on 188 labeled documents across 13 categories
- The embedding model (~80MB) downloads automatically on first use and is cached locally
- Each question costs 2 Gemini API calls (category detection + answer generation)
