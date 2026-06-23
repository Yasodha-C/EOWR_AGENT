"""
answer_query.py — retrieves relevant chunks and generates a cited answer.

WHAT IT DOES:
  1. Takes a question + category (passed by the user from the UI dropdown)
  2. Retrieves the most relevant chunks using hybrid search (search.py)
  3. Asks Gemini to answer USING ONLY those chunks, with citation markers
  4. Returns the answer + sources list for display in the UI

DESIGN DECISION — category passed by user, not auto-detected:
  An earlier version used a second Gemini call to auto-detect the category
  from the question. This was removed because:
    - It doubled the API calls per question (2 instead of 1)
    - It caused quota exhaustion during testing
    - The Streamlit UI already has a category dropdown — user intent is clear
  The user selects the category alongside their question. If they select
  'All categories', the search runs unfiltered across all 3,155 chunks.

WHY CITATIONS:
  Without forcing citations, an LLM can mix retrieved facts with its own
  general knowledge — the user can't tell which is which. Every factual
  claim is tagged [1], [2], etc. matching a specific source document.

REQUIRES: GEMINI_API_KEY environment variable to be set.
"""

import os
import re
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / '.env')
except ImportError:
    pass

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from search import hybrid_search, build_bm25_index

CATEGORIES = [
    'bha_bit', 'casing_cementing', 'completion_production', 'cost_afe',
    'directional', 'drilling', 'geology', 'hse_well_control',
    'lessons_learned', 'mud_logging', 'npt_incidents',
    'pressure_tests', 'well_summary',
]

ANSWER_SYSTEM_PROMPT = """You answer questions using ONLY the numbered source excerpts provided.

Rules:
- Every factual claim must end with a citation marker like [1] or [2].
- If a sentence uses multiple sources, cite all: [1][3].
- If the sources don't contain enough to answer, say so — never guess or use outside knowledge.
- Keep the answer concise and directly address the question."""

_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set.\n"
                "Set it in your .env file or with: $env:GEMINI_API_KEY='your-key'"
            )
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _gemini_call(prompt: str, max_retries: int = 3) -> str:
    """Calls Gemini with automatic retry on rate-limit errors."""
    from google.genai import errors as genai_errors
    client = _get_gemini_client()

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            return response.text
        except genai_errors.ClientError as e:
            is_rate_limit = '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e)
            if not is_rate_limit or attempt == max_retries - 1:
                raise
            match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)s", str(e))
            wait = int(match.group(1)) if match else (2 ** attempt) * 5
            print(f"Rate limited — waiting {wait}s ({attempt+1}/{max_retries})...")
            time.sleep(wait)


def answer(question: str, category: str = None, n_results: int = 5) -> dict:
    """
    Retrieves relevant chunks and asks Gemini to answer with citations.

    question : the user's question
    category : one of the 13 category strings, or None to search all categories
    n_results: how many chunks to retrieve and send to Gemini
    """
    chunks = hybrid_search(question, category=category, n_results=n_results)

    if not chunks:
        return {
            'category': category,
            'answer': "No relevant documents found"
                      + (f" in category '{category}'." if category else "."),
            'sources': [],
            'chunks_used': 0,
        }

    context = "\n\n".join(
        f"[{i}] Source: {c['filename']} / {c['heading']}\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )

    sources = [
        {'n': i, 'filename': c['filename'],
         'heading': c['heading'], 'category': c['category']}
        for i, c in enumerate(chunks, 1)
    ]

    prompt = f"""{ANSWER_SYSTEM_PROMPT}

Sources:
{context}

Question: {question}

Answer using only the sources above, with citation markers."""

    llm_answer = _gemini_call(prompt)

    return {
        'category': category,
        'answer': llm_answer,
        'sources': sources,
        'chunks_used': len(chunks),
    }


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python answer_query.py "your question" <category>')
        print('       python answer_query.py "your question" none')
        print()
        print('Categories:', ', '.join(CATEGORIES))
        sys.exit(0)

    question = sys.argv[1]
    category = None if sys.argv[2].lower() == 'none' else sys.argv[2]

    build_bm25_index()
    result = answer(question, category=category)

    print(f"CATEGORY : {result['category'] or 'all'}")
    print(f"CHUNKS   : {result['chunks_used']}")
    print()
    print("ANSWER:")
    print(result['answer'])
    print()
    print("SOURCES:")
    for s in result['sources']:
        print(f"  [{s['n']}] {s['filename']} — {s['heading']}")