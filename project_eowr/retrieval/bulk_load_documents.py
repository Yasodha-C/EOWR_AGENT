"""
bulk_load_documents.py — embeds all 188 labeled documents into ChromaDB
and builds the BM25 index so the app can search them immediately.

RUN THIS ONCE (or whenever you add new documents to data/).

USAGE:
  python bulk_load_documents.py <data path>
HOW IT WORKS:
  Reads labels.xlsx (in generators/) to get each file's correct category.
  Extracts the file, splits into chunks, embeds, and stores in ChromaDB.
  Uses the GROUND-TRUTH label from labels.xlsx — NOT the trained classifier
  (which would be redundant for already-labeled documents).
"""

import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

import sys
import pandas as pd

THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
GENERATORS_DIR = os.path.join(PROJECT_ROOT, 'generators')
EXTRACTORS_DIR = os.path.join(PROJECT_ROOT, 'extractors')

for path in (PROJECT_ROOT, THIS_DIR, EXTRACTORS_DIR, GENERATORS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from extractor_factory import extract
from schema import ClassifiedDocument
from chunker import chunk_document
from search import add_chunks, collection_stats, build_bm25_index


def main():
    if len(sys.argv) < 2:
        print('Usage: python bulk_load_documents.py "<path_to_data_folder>"')
        sys.exit(0)

    data_folder = os.path.abspath(sys.argv[1])
    labels_path = os.path.join(GENERATORS_DIR, 'labels.xlsx')

    if not os.path.exists(labels_path):
        print(f"ERROR: labels.xlsx not found at {labels_path}")
        sys.exit(1)
    if not os.path.exists(data_folder):
        print(f"ERROR: data folder not found: {data_folder}")
        sys.exit(1)

    df = pd.read_excel(labels_path)
    print(f"Loaded {len(df)} filename-label pairs")
    print(f"Data folder: {data_folder}\n")

    total_chunks = 0
    succeeded = 0
    failed = []

    for _, row in df.iterrows():
        filepath = str(row['filepath']).strip()
        label    = str(row['label']).strip()
        full_path = os.path.join(data_folder, filepath)

        if not os.path.exists(full_path):
            print(f"  [MISSING] {filepath}")
            failed.append((filepath, 'file not found'))
            continue

        try:
            extraction = extract(full_path)
            if extraction.quality == 'failed':
                print(f"  [FAILED]  {filepath}")
                failed.append((filepath, str(extraction.warnings)))
                continue

            classified = ClassifiedDocument(
                extraction=extraction, category=label,
                confidence=1.0, low_confidence=False)

            chunks = chunk_document(classified)
            added  = add_chunks(chunks)
            total_chunks += added
            succeeded += 1

            status = '[OK]' if extraction.quality == 'full' else '[PARTIAL]'
            print(f"  {status} {filepath}  ({added} chunks, label={label})")

        except Exception as e:
            print(f"  [ERROR]   {filepath} -- {e}")
            failed.append((filepath, str(e)))

    print(f"\n{'='*60}")
    print(f"Done. {succeeded}/{len(df)} files loaded, {total_chunks} total chunks.")
    if failed:
        print(f"{len(failed)} failed:")
        for fp, reason in failed:
            print(f"  - {fp}: {reason}")

    print("\nBuilding BM25 index...")
    n = build_bm25_index()
    print(f"BM25 index built: {n} chunks indexed")

    print("\nFinal stats:")
    s = collection_stats()
    print(f"  Total chunks: {s['total_chunks']}")
    for cat, count in sorted(s['by_category'].items()):
        print(f"    {cat:<25} {count}")

    print(f"\nDone. Run the Streamlit app: streamlit run app/streamlit_app.py")


if __name__ == '__main__':
    main()
