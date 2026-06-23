"""
The full pipeline for ONE document: extract -> classify -> return typed result.
USAGE:
  from process_document import process

  result = process("path/to/some_file.pdf")
  print(result.category)              # 'npt_incidents'
  print(result.confidence)            # 0.82
  print(result.extraction.markdown)   # the actual extracted content
"""

import sys
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = THIS_DIR   # main.py sits AT the project root, not inside a subfolder
EXTRACTORS_DIR = os.path.join(PROJECT_ROOT, 'extractors')
GENERATORS_DIR = os.path.join(PROJECT_ROOT, 'generators')

for path in (THIS_DIR, EXTRACTORS_DIR, GENERATORS_DIR, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from extractor_factory import extract
from predict_category import predict
from schema import ClassifiedDocument, Quality


def process(file_path: str) -> ClassifiedDocument:
    """
    Runs the full pipeline on one file: extract, then classify.

    If extraction fails (quality='failed'), classification is SKIPPED —
    there's no point predicting a category for empty/broken content.
    In that case category is set to 'unknown' with 0 confidence, so
    downstream code can still rely on getting a ClassifiedDocument back
    (never None, never a crash) and just check confidence==0 to filter
    these out.
    """
    extraction_result = extract(file_path)

    if extraction_result.quality == Quality.FAILED:
        return ClassifiedDocument(
            extraction=extraction_result,
            category='unknown',
            confidence=0.0,
            low_confidence=True,
        )

    prediction = predict(extraction_result.markdown)

    return ClassifiedDocument(
        extraction=extraction_result,
        category=prediction['label'],
        confidence=prediction['confidence'],
        low_confidence=prediction['low_confidence'],
    )


if __name__ == '__main__':
    import sys as _sys
    files = _sys.argv[1:]
    if not files:
        print("Usage: python main.py <path_to_file> [more files...]")
        print("Example: python main.py \"data/npt_incidents/DSEC-AD-1X_NPT_Log.xlsx\"")
        _sys.exit(0)

    for f in files:
        if not os.path.exists(f):
            print(f"Skipping (not found): {f}")
            continue
        result = process(f)
        print(f"File       : {result.extraction.file}")
        print(f"Category   : {result.category}")
        print(f"Confidence : {result.confidence:.2f}")
        if result.low_confidence:
            print(f"Note       : low confidence — consider manual review")
        print('-' * 50)
