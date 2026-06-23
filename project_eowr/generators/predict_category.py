"""
Predicts the category for a document using the trained classifier
(TF-IDF + Logistic Regression) only.

WHY NO FALLBACK:
  An earlier version of this script used a local embedding-similarity
  fallback for low-confidence predictions, on the theory that a second
  opinion would help on uncertain cases. Real testing on this project's
  data showed the opposite: across a representative sample of 39 real
  documents, the classifier alone was correct on 38 (97%) — including
  every case where the embedding fallback had previously been checked
  and found wrong. The fallback was overriding more correct answers
  than it was fixing actual mistakes, so it was removed. The confidence
  score is still returned so low-confidence predictions can be flagged
  for manual review if desired — it just no longer triggers a second,
  weaker guess.

USAGE:
  from predict_category import predict
  result = predict("some markdown content here...")
  print(result)
  # {'label': 'npt_incidents', 'confidence': 0.82}
"""

import pickle
import os
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(THIS_DIR, 'classifier_model.pkl')

# Below this confidence, the prediction is still returned but can be
# treated as "low confidence — consider manual review" by the caller.
# This is informational only — it no longer changes which label is returned.
LOW_CONFIDENCE_THRESHOLD = 0.30


_model_cache = None


def _load_classifier():
    global _model_cache
    if _model_cache is None:
        with open(MODEL_PATH, 'rb') as f:
            _model_cache = pickle.load(f)
    return _model_cache


def predict_with_classifier(markdown_text: str):
    """Returns (label, confidence) from the trained classifier."""
    bundle = _load_classifier()
    vectorizer = bundle['vectorizer']
    model = bundle['model']

    X = vectorizer.transform([markdown_text])
    probabilities = model.predict_proba(X)[0]

    best_idx = np.argmax(probabilities)
    best_label = model.classes_[best_idx]
    best_confidence = probabilities[best_idx]

    return best_label, float(best_confidence)


def predict(markdown_text: str) -> dict:
    """
    Main entry point. Always trusts the classifier's top prediction.
    Flags low-confidence results for awareness, without overriding them.
    """
    label, confidence = predict_with_classifier(markdown_text)

    return {
        'label': label,
        'confidence': round(confidence, 3),
        'low_confidence': confidence < LOW_CONFIDENCE_THRESHOLD,
    }


if __name__ == '__main__':
    import sys as _sys

    if len(_sys.argv) < 2:
        print("Usage: python predict_category.py <path_to_file>")
        print('Example: python predict_category.py "..\\data\\drilling\\DSEC-AD-1X_DDR_Day_15.xlsx"')
        _sys.exit(0)

    file_path = _sys.argv[1]

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        _sys.exit(1)

    # Make extractor_factory.py importable, whether this script sits inside
    # eowr_agent/extractors/ OR a sibling folder like eowr_agent/generators/
    PROJECT_ROOT = os.path.dirname(THIS_DIR)
    EXTRACTORS_DIR = os.path.join(PROJECT_ROOT, 'extractors')
    for path in (THIS_DIR, EXTRACTORS_DIR, PROJECT_ROOT):
        if path not in _sys.path:
            _sys.path.insert(0, path)

    from extractor_factory import extract

    extraction = extract(file_path)

    print(f"File    : {extraction.file}")
    print(f"Quality : {extraction.quality}")

    if extraction.quality == 'failed':
        print(f"Warnings: {extraction.warnings}")
        print("Cannot classify — extraction failed.")
        _sys.exit(1)

    result = predict(extraction.markdown)

    print()
    print(f"Predicted category : {result['label']}")
    print(f"Confidence         : {result['confidence']}")
    if result['low_confidence']:
        print(f"Note: confidence is below {LOW_CONFIDENCE_THRESHOLD} — consider manual review")
