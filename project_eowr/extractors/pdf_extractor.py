"""
Robust PDF Extractor
====================
Converts any PDF to clean structured Markdown.
"""

import fitz
import pdfplumber
import re
from pathlib import Path
from collections import defaultdict


# ─────────────────────────────────────────────────────────────
# PART 1 — PyMuPDF word extraction helpers
# ─────────────────────────────────────────────────────────────

def get_clean_words(page):
    """Deduplicated, non-clipped words from a PyMuPDF page."""
    seen, result = set(), []
    for w in page.get_text("words"):
        x0, y0, x1, y1, text, *_ = w
        if x0 < 0:                                   # skip clipped
            continue
        key = (round(x0), round(y0), text.strip())
        if key in seen:                              # skip duplicate
            continue
        seen.add(key)
        result.append(w)
    return result


def group_lines(words, y_tol=3):
    """Group words by Y proximity, each line sorted left→right."""
    if not words:
        return []
    buckets = defaultdict(list)
    for w in words:
        buckets[round(w[1] / y_tol) * y_tol].append(w)
    return [sorted(v, key=lambda w: w[0]) for _, v in sorted(buckets.items())]


def text_in_bbox(fitz_page, bbox, at_right_edge=False):
    x0, y0, x1, y1 = bbox
    x1_read = x1 + 60 if at_right_edge else x1
    words = [w for w in fitz_page.get_text("words")
             if x0 - 2 <= w[0] <= x1_read + 2
             and y0 - 2 <= w[1] <= y1 + 2
             and w[0] >= 0]
    return ' '.join(w[4] for w in sorted(words, key=lambda w: w[0])).strip()


# ─────────────────────────────────────────────────────────────
# PART 2 — Table type detection
# ─────────────────────────────────────────────────────────────

def is_kv_header(row):
    if not row or len(row) < 4:
        return False
    return [str(c).strip().lower() if c else '' for c in row][:4] == \
           ['parameter', 'value', 'parameter', 'value']


def none_ratio(table):
    total = sum(len(r) for r in table)
    empty = sum(1 for r in table for c in r if not (c and str(c).strip()))
    return empty / total if total else 1.0


def has_midword_splits(table):
    """True if cells are split mid-word (Fix 5 detection)."""
    for row in table:
        cells = [str(c).strip() if c else '' for c in row]
        for i in range(len(cells) - 1):
            a, b = cells[i], cells[i + 1]
            if (a and b and len(a) > 5
                    and a[-1] not in ' .!?,;:|)'
                    and b[0].islower()):
                return True
    return False


def is_single_col(table):
    """All data is in column 0; all other columns are empty."""
    for row in table:
        for cell in (row[1:] if len(row) > 1 else []):
            if cell and str(cell).strip():
                return False
    return True


# ─────────────────────────────────────────────────────────────
# PART 3 — Table renderers
# ─────────────────────────────────────────────────────────────

def kv4_to_md(plumber_table_obj, fitz_page):
    lines = []
    table_x1 = plumber_table_obj.bbox[2]
    for row in plumber_table_obj.rows:
        cells = []
        for cell_bbox in row.cells:
            if cell_bbox:
                at_edge = (table_x1 - cell_bbox[2]) < 10
                cells.append(text_in_bbox(fitz_page, cell_bbox, at_right_edge=at_edge))
            else:
                cells.append('')
        if is_kv_header(cells):
            continue
        if len(cells) >= 2 and cells[0] and cells[1]:
            lines.append(f"**{cells[0]}** {cells[1]}")
        if len(cells) >= 4 and cells[2] and cells[3]:
            lines.append(f"**{cells[2]}** {cells[3]}")
    return '\n'.join(lines)


def single_col_to_md(table):
    lines = []
    for row in table:
        t = str(row[0]).strip() if row and row[0] else ''
        if t:
            lines.append(t)
    return '\n'.join(lines)


def cert_to_md(table):
    """Fix 5: join cells WITHOUT space to fix mid-word splits."""
    lines = []
    for row in table:
        cells = [str(c).strip() if c else '' for c in row]
        non_empty = [c for c in cells if c]
        if not non_empty:
            continue
        if any('_' in c or c == 'Date:' for c in non_empty):
            lines.append('  '.join(non_empty))   # signature row
        else:
            lines.append(''.join(non_empty))      # text row — no space
    return '\n\n'.join(lines)


def generic_to_md(table):
    cleaned = [[str(c).strip() if c else '' for c in row] for row in table]
    cleaned = [r for r in cleaned if any(c for c in r)]
    if not cleaned:
        return ''
    n = max(len(r) for r in cleaned)
    pad = lambda r: r + [''] * (n - len(r))
    hdr  = pad(cleaned[0])
    rows = [pad(r) for r in cleaned[1:]]
    lines = ['| ' + ' | '.join(hdr) + ' |',
             '| ' + ' | '.join(['---'] * n) + ' |']
    for r in rows:
        lines.append('| ' + ' | '.join(r) + ' |')
    return '\n'.join(lines)


def words_in_bbox_to_md(fitz_page, bbox, x_split=250):
    """
    Fix 3: reconstruct text from PyMuPDF words when pdfplumber table is garbled.
    Groups lines by Y, splits at x_split for left/right columns.
    """
    bx0, by0, bx1, by1 = bbox
    words = [w for w in get_clean_words(fitz_page)
             if bx0 <= w[0] <= bx1 + 60 and by0 <= w[1] <= by1]
    lines = group_lines(words)
    parts = []
    for lw in lines:
        left  = ' '.join(w[4] for w in lw if w[0] < x_split).strip()
        right = ' '.join(w[4] for w in lw if w[0] >= x_split).strip()
        if not right:
            parts.append(left)
        elif not left:
            parts.append(right)
        else:
            parts.append(f"**{left}** {right}")
    return '\n'.join(parts)


def render_table(plumber_table_obj, fitz_page):
    """Choose the correct renderer for a table."""
    table = plumber_table_obj.extract()
    bbox  = plumber_table_obj.bbox

    if not table:
        return ''

    # Single-col block FIRST (before none_ratio check, since empty col2 inflates ratio)
    if is_single_col(table):
        return single_col_to_md(table)

    # Fix 3: garbled table → PyMuPDF fallback
    if none_ratio(table) > 0.40:
        return words_in_bbox_to_md(fitz_page, bbox)

    first = table[0] if table else []

    # Fix 4: 4-col kv table → use cell bboxes + PyMuPDF for full text
    if is_kv_header(first):
        return kv4_to_md(plumber_table_obj, fitz_page)

    # Fix 5: cert block with mid-word splits
    if has_midword_splits(table):
        return cert_to_md(table)

    return generic_to_md(table)


# ─────────────────────────────────────────────────────────────
# PART 4 — Line classifier
# ─────────────────────────────────────────────────────────────

def classify(text):
    t = text.strip()
    if not t:
        return 'empty'
    letters = re.sub(r'[^a-zA-Z]', '', t)
    if (letters and letters == letters.upper()
            and len(letters) > 3 and len(t.split()) <= 12):
        return 'heading'
    if re.match(r'^\d+[\.\:]', t) and len(t.split()) <= 8:
        return 'heading'
    return 'text'


# ─────────────────────────────────────────────────────────────
# PART 5 — Per-page extraction
# ─────────────────────────────────────────────────────────────

def extract_page(fitz_page, plumber_page):
    plumber_objs = plumber_page.find_tables()
    table_bboxes = [t.bbox for t in plumber_objs]

    def in_table(x, y):
        for bx0, by0, bx1, by1 in table_bboxes:
            if bx0 <= x <= bx1 and by0 <= y <= by1:
                return True
        return False

    words = get_clean_words(fitz_page)
    lines = group_lines(words)
    parts = []
    tables_done = [False] * len(plumber_objs)
    table_tops  = sorted(
        [(bbox[1], i) for i, bbox in enumerate(table_bboxes)],
        key=lambda x: x[0]
    )
    tp = 0

    for lw in lines:
        ly, lx = lw[0][1], lw[0][0]

        # Insert tables whose top Y has been passed
        while tp < len(table_tops):
            ty, ti = table_tops[tp]
            if ty <= ly:
                if not tables_done[ti]:
                    tables_done[ti] = True
                    parts.append(render_table(plumber_objs[ti], fitz_page))
                tp += 1
            else:
                break

        if in_table(lx, ly):    # Fix 6: suppress duplicated raw text
            continue

        lt = ' '.join(w[4] for w in lw).strip()
        if not lt:
            continue

        kind = classify(lt)
        if kind == 'heading':
            parts.append(f"\n## {lt}\n")
        else:
            parts.append(lt)

    # Flush remaining tables
    while tp < len(table_tops):
        _, ti = table_tops[tp]
        if not tables_done[ti]:
            tables_done[ti] = True
            parts.append(render_table(plumber_objs[ti], fitz_page))
        tp += 1

    return '\n\n'.join(p for p in parts if p and p.strip())


# ─────────────────────────────────────────────────────────────
# PART 6 — Main entry point
# ─────────────────────────────────────────────────────────────

def extract_pdf_to_markdown(file_path: str) -> dict:
    path = Path(file_path)
    warnings, page_mds = [], []

    try:
        fitz_doc    = fitz.open(file_path)
        plumber_doc = pdfplumber.open(file_path)
        total = len(fitz_doc)

        for i in range(total):
            try:
                md = extract_page(fitz_doc[i], plumber_doc.pages[i])
                if md.strip():
                    page_mds.append(f"<!-- Page {i+1} -->\n{md}")
                else:
                    warnings.append(f"Page {i+1}: no content extracted")
            except Exception as e:
                warnings.append(f"Page {i+1}: {e}")

        fitz_doc.close()
        plumber_doc.close()

        full_md = '\n\n---\n\n'.join(page_mds)
        quality = ('failed'  if not full_md.strip() else
                   'partial' if len(warnings) > total / 2 else
                   'full')

        return {'file': path.name, 'pages': total,
                'markdown': full_md, 'quality': quality, 'warnings': warnings}

    except Exception as e:
        return {'file': path.name, 'pages': 0, 'markdown': '',
                'quality': 'failed', 'warnings': [f'Cannot open file: {e}']}


# ─────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    result = extract_pdf_to_markdown("data/pressure_tests/DSEC-AD-1X_BOP_Function_and_Pressure_Tests.pdf")
    print(f"File    : {result['file']}")
    print(f"Pages   : {result['pages']}")
    print(f"Quality : {result['quality']}")
    if result['warnings']:
        print(f"Warnings: {result['warnings']}")
    print()
    print('=' * 60)
    print(result['markdown'])
