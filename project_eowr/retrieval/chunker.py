"""
Splits extracted markdown into chunks suitable for embedding.

STRATEGY (hybrid — structure first, size as a safety net):
  1. Split the document by ## headings — each section becomes a candidate chunk.
     This respects the document's natural structure: tables, key-value blocks,
     and related content under one heading stay together.
  2. If a section is small enough (<= MAX_CHUNK_CHARS), keep it as ONE chunk.
  3. If a section is too large (some 'drilling' docs are huge), split it
     further by paragraph, with a small overlap between pieces so context
     isn't lost at the cut point.

Each chunk carries metadata: source filename, category, section heading,
and a chunk index — so downstream retrieval can filter and trace back
to the original document.

USAGE:
  from chunker import chunk_document
  chunks = chunk_document(result)   # result = a ClassifiedDocument from main.py
"""

import re
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from schema import ClassifiedDocument


# Roughly 2000 characters ≈ 400-500 tokens, safely under most embedding
# model limits (512 tokens is the common ceiling, e.g. all-MiniLM-L6-v2)
MAX_CHUNK_CHARS = 2000

# When a section must be split further, this much text repeats at the
# start of the next piece, so context isn't lost at the cut point
OVERLAP_CHARS = 200


def split_by_headings(markdown: str) -> list[dict]:
    """
    Splits markdown into sections at each '## heading' boundary.
    Returns a list of {'heading': str, 'content': str} dicts.
    Content BEFORE the first heading (if any) gets heading='(intro)'.
    """
    # Split on lines that start with '##' (but not '###' going deeper —
    # we treat ## as the section boundary; ### stays inside its parent section)
    pattern = re.compile(r'^## (.+)$', re.MULTILINE)

    matches = list(pattern.finditer(markdown))

    if not matches:
        # No headings at all — treat the whole document as one section
        return [{'heading': '(full document)', 'content': markdown.strip()}]

    sections = []

    # Content before the first heading
    if matches[0].start() > 0:
        intro = markdown[:matches[0].start()].strip()
        if intro:
            sections.append({'heading': '(intro)', 'content': intro})

    # Each heading's content runs until the next heading (or end of doc)
    for i, match in enumerate(matches):
        heading_text = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        content = markdown[start:end].strip()
        sections.append({'heading': heading_text, 'content': content})

    return sections


def split_large_section(content: str, max_chars: int, overlap: int) -> list[str]:
    """
    Splits an oversized section into smaller pieces by paragraph,
    with overlap between consecutive pieces.
    """
    if len(content) <= max_chars:
        return [content]

    paragraphs = [p for p in content.split('\n\n') if p.strip()]

    pieces = []
    current = ''

    for para in paragraphs:
        # If adding this paragraph would exceed the limit, save current piece
        if current and len(current) + len(para) + 2 > max_chars:
            pieces.append(current.strip())
            # Start next piece with overlap from the end of the previous one
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + '\n\n' + para
        else:
            current = current + '\n\n' + para if current else para

    if current.strip():
        pieces.append(current.strip())

    # Edge case: a single paragraph itself longer than max_chars
    # (e.g. one giant table with no blank lines) — hard split with overlap
    final_pieces = []
    for piece in pieces:
        if len(piece) <= max_chars:
            final_pieces.append(piece)
        else:
            start = 0
            while start < len(piece):
                end = start + max_chars
                final_pieces.append(piece[start:end])
                start = end - overlap if end - overlap > start else end

    return final_pieces


def chunk_document(result: ClassifiedDocument) -> list[dict]:
    """
    Main entry point. Takes a ClassifiedDocument (from main.py's process())
    and returns a list of chunk dicts, each ready for embedding.

    Each chunk dict:
        {
            'text': str,              # the actual chunk content to embed
            'filename': str,
            'category': str,
            'heading': str,           # which ## section this came from
            'chunk_index': int,       # position within this section (0 if not split further)
            'quality': str,           # extraction quality, passed through
        }
    """
    markdown = result.extraction.markdown
    sections = split_by_headings(markdown)

    chunks = []
    for section in sections:
        pieces = split_large_section(section['content'], MAX_CHUNK_CHARS, OVERLAP_CHARS)

        for idx, piece in enumerate(pieces):
            if not piece.strip():
                continue
            chunks.append({
                'text': piece,
                'filename': result.extraction.file,
                'category': result.category,
                'heading': section['heading'],
                'chunk_index': idx,
                'quality': result.extraction.quality,
            })

    return chunks


if __name__ == '__main__':
    import sys
    import os

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(THIS_DIR)   # retrieval/ -> up one level to project root
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'extractors'))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'generators'))

    from main import process

    if len(sys.argv) < 2:
        print("Usage: python chunker.py <path_to_file>")
        sys.exit(0)

    result = process(sys.argv[1])
    chunks = chunk_document(result)

    print(f"File     : {result.extraction.file}")
    print(f"Category : {result.category}")
    print(f"Total chunks: {len(chunks)}\n")

    for i, c in enumerate(chunks):
        print(f"--- Chunk {i+1} ---")
        print(f"  heading     : {c['heading']}")
        print(f"  chunk_index : {c['chunk_index']}")
        print(f"  length      : {len(c['text'])} chars")
        print(f"  preview     : {c['text'][:100]!r}")
        print()
