"""
search.py — stores and retrieves document chunks using hybrid BM25 + vector search.

TWO RESPONSIBILITIES:
  1. STORAGE  — embed chunks and store in ChromaDB
                (called by bulk_load_documents.py)
  2. RETRIEVAL — given a question + optional category, return the most
                relevant chunks using BM25 + vector merged via RRF
"""

import os
#os.environ['ANONYMIZED_TELEMETRY'] = 'False' # It's a privacy/security setting

import re
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(THIS_DIR, 'vector_db')
COLLECTION = 'eowr_documents'
RRF_K      = 60    # smoothing constant from the original RRF paper

# internal private variables
_embedder       = None
_client         = None
_collection     = None
_bm25_index     = None
_bm25_chunk_ids = None
_bm25_lookup    = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        print("Loading embedding model (one-time, ~80MB download if not cached)...")
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client     = chromadb.PersistentClient(path=DB_PATH)
        _collection = _client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"})
    return _collection


# ── Storage ──────────────────────────────────────────────────────────────────

def add_chunks(chunks: list[dict]) -> int:
    """Embeds chunks (from chunker.py) and upserts them into ChromaDB.
    Uses upsert so re-running the same file overwrites old chunks."""
    valid = [c for c in chunks if c['text'].strip()]
    if not valid:
        return 0
    embeddings = _get_embedder().encode([c['text'] for c in valid]).tolist()
    _get_collection().upsert(
        ids=[f"{c['filename']}::{c['heading']}::{c['chunk_index']}" for c in valid],
        embeddings=embeddings,
        documents=[c['text'] for c in valid],
        metadatas=[{'filename': c['filename'], 'category': c['category'],
                    'heading':  c['heading'],  'chunk_index': c['chunk_index'],
                    'quality':  c['quality']}  for c in valid],
    )
    return len(valid)


def collection_stats() -> dict:
    """How many chunks are stored, broken down by category."""
    count = _get_collection().count()
    if count == 0:
        return {'total_chunks': 0, 'by_category': {}}
    by_cat = {}
    for m in _get_collection().get(include=['metadatas'])['metadatas']:
        by_cat[m['category']] = by_cat.get(m['category'], 0) + 1
    return {'total_chunks': count, 'by_category': by_cat}


# ── BM25 index ───────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer that preserves hyphenated identifiers."""
    return re.findall(r'[a-z0-9][a-z0-9\-/]*[a-z0-9]|[a-z0-9]', text.lower())


def build_bm25_index() -> int:
    """Builds an in-memory BM25 index from all chunks in ChromaDB.
    Must be called once after loading documents, before using hybrid_search()."""
    global _bm25_index, _bm25_chunk_ids, _bm25_lookup
    data = _get_collection().get(include=['documents', 'metadatas'])
    if not data['ids']:
        _bm25_index = None
        _bm25_chunk_ids = []
        _bm25_lookup = {}
        return 0
    _bm25_chunk_ids = data['ids']
    _bm25_index     = BM25Okapi([_tokenize(d) for d in data['documents']])
    _bm25_lookup    = {
        cid: {'text':     data['documents'][i],
              'filename': data['metadatas'][i]['filename'],
              'category': data['metadatas'][i]['category'],
              'heading':  data['metadatas'][i]['heading']}
        for i, cid in enumerate(_bm25_chunk_ids)
    }
    return len(_bm25_chunk_ids)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def hybrid_search(query: str, category: str = None, n_results: int = 5) -> list[dict]:
    """
    Returns the top n_results chunks most relevant to query.
    Combines BM25 and vector search using Reciprocal Rank Fusion.
    category: if given, only searches within that one category.
    """
    if not _bm25_index:
        build_bm25_index()

    # BM25 leg — exact keyword matching
    bm25_scores  = _bm25_index.get_scores(_tokenize(query))
    bm25_ranked  = sorted(
        [(cid, bm25_scores[i]) for i, cid in enumerate(_bm25_chunk_ids)
         if not category or _bm25_lookup[cid]['category'] == category],
        key=lambda x: x[1], reverse=True
    )[:20]

    # Vector leg — semantic similarity
    emb = _get_embedder().encode([query]).tolist()
    vres = _get_collection().query(
        query_embeddings=emb, n_results=20,
        where={'category': category} if category else None
    )
    vector_ranked = vres['ids'][0] if vres['ids'] and vres['ids'][0] else []

    # Reciprocal Rank Fusion — merges both ranked lists by position
    rrf = {}
    for rank, (cid, _) in enumerate(bm25_ranked):
        rrf[cid] = rrf.get(cid, 0) + 1.0 / (RRF_K + rank + 1)
    for rank, cid in enumerate(vector_ranked):
        rrf[cid] = rrf.get(cid, 0) + 1.0 / (RRF_K + rank + 1)

    top = sorted(rrf, key=lambda c: rrf[c], reverse=True)[:n_results]
    return [
        {**_bm25_lookup[cid], 'rrf_score': round(rrf[cid], 5)}
        for cid in top if cid in _bm25_lookup
    ]
