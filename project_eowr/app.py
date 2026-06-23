"""
app.py — Streamlit UI for the EOWR Document Intelligence system.

HOW TO RUN:
  cd D:\SwarmLens\eowr_agent_final\project_eowr
  python -m streamlit run app.py
"""

import os
import sys
from pathlib import Path

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / '.env')
except ImportError:
    pass

# Path setup — works whether app.py is at project root or inside app/
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_parent  = os.path.dirname(THIS_DIR)
PROJECT_ROOT = _parent if os.path.exists(os.path.join(_parent, 'retrieval')) else THIS_DIR

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'extractors'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'generators'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'retrieval'))

import streamlit as st
from search import build_bm25_index, collection_stats
from answer_query import answer, CATEGORIES

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EOWR Document Intelligence",
    page_icon="🏗️",
    layout="wide",
)

st.title("🏗️ WellMind - EOWR ")
st.caption("Ask any question about your well documents. Select a category to filter the search, then get a cited answer from your actual documents.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    category = st.selectbox(
        "Document Category",
        options=["All Categories"] + CATEGORIES,
        help="Select the category your question relates to. This filters the search to only relevant documents, improving answer quality.",
    )

    n_results = st.slider(
        "Chunks to retrieve", min_value=3, max_value=10, value=5,
        help="More chunks = more context for Gemini, but also more tokens.",
    )

    st.divider()
    st.subheader("📊 Database Stats")
    try:
        stats = collection_stats()
        st.metric("Total chunks indexed", stats['total_chunks'])
        if stats['by_category']:
            st.write("**By category:**")
            for cat, count in sorted(stats['by_category'].items()):
                st.write(f"  `{cat}` — {count}")
    except Exception:
        st.warning("Could not load stats. Run bulk_load_documents.py first.")

# ── BM25 index ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building search index...")
def init_search_index():
    return build_bm25_index()

n_indexed = init_search_index()
if n_indexed == 0:
    st.warning("⚠️ No documents in database. Run `bulk_load_documents.py` first.")
    st.stop()

# ── Question input ────────────────────────────────────────────────────────────
question = st.text_input(
    "Ask a question:",
    placeholder="e.g. What NPT incidents happened during drilling?",
)

if not question.strip():
    st.info("Type a question above and select a category to get started.")
    st.stop()

# ── API key check ─────────────────────────────────────────────────────────────
if not os.environ.get('GEMINI_API_KEY'):
    st.error("GEMINI_API_KEY is not set. Add it to your .env file.")
    st.stop()

# ── Run pipeline ──────────────────────────────────────────────────────────────
selected_category = None if category == "All Categories" else category

with st.spinner("Searching documents and generating answer..."):
    try:
        result = answer(question, category=selected_category, n_results=n_results)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# ── Display results ───────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Answer")
    if result['category']:
        st.markdown(f"**Category filter:** `{result['category']}`")
    else:
        st.markdown("**Category filter:** `all categories`")
    st.markdown(result['answer'])

with col2:
    st.subheader("Sources")
    st.caption(f"{result['chunks_used']} chunks retrieved")
    for s in result['sources']:
        with st.expander(f"[{s['n']}] {s['filename']}"):
            st.write(f"**Section:** {s['heading']}")
            st.write(f"**Category:** `{s['category']}`")