"""
Trains a TF-IDF + Logistic Regression classifier on ALL labeled documents.

Since several categories have only 5-7 examples, we don't hold out a test
set (too few examples per category to test reliably). Instead we train on
everything, and rely on a CONFIDENCE THRESHOLD at prediction time — if the
model isn't confident, we fall back to embedding similarity instead of
trusting a shaky guess.

INPUT:
  training_data.csv with columns: filename, label, markdown, quality

OUTPUT:
  classifier_model.pkl   — the trained model (vectorizer + classifier bundled)

USAGE:
  python train_classifier.py
"""

import pandas as pd
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_PATH = os.path.join(THIS_DIR, 'training_data.csv')
MODEL_OUTPUT_PATH  = os.path.join(THIS_DIR, 'classifier_model.pkl')


def main():
    df = pd.read_csv(TRAINING_DATA_PATH)
    print(f"Loaded {len(df)} rows from {TRAINING_DATA_PATH}\n")

    # Only train on good-quality extractions
    before = len(df)
    df = df[df['quality'] == 'full'].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with quality != 'full'")

    # Drop rows with empty/missing markdown
    df = df[df['markdown'].notna() & (df['markdown'].str.strip() != '')].reset_index(drop=True)
    print(f"Training on ALL {len(df)} documents across {df['label'].nunique()} labels\n")

    print("Label distribution:")
    print(df['label'].value_counts())
    print()

    X = df['markdown']
    y = df['label']

    # ── TF-IDF: convert text to number vectors ──
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=1                # keep min_df=1 since we have very few docs per category already
    )
    X_vec = vectorizer.fit_transform(X)

    # ── Train on EVERYTHING — no held-out test set ──
    # class_weight='balanced' corrects for 'drilling' having 82 examples
    # while others have as few as 5.
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_vec, y)

    # ── Sanity check: how well does it fit the data it just learned from? ──
    # NOTE: this is NOT a real accuracy measure (it's testing on the same
    # data it trained on, so it will look artificially good). It only tells
    # us the model is technically working, not how it'll perform on new files.
    train_predictions = model.predict(X_vec)
    train_accuracy = (train_predictions == y).mean()
    print(f"Training-set fit (NOT real-world accuracy): {train_accuracy:.0%}")
    print("This number is expected to look high — it's not a true accuracy measure.\n")

    # ── Save model + vectorizer + the label list together ──
    with open(MODEL_OUTPUT_PATH, 'wb') as f:
        pickle.dump({
            'vectorizer': vectorizer,
            'model': model,
            'labels': sorted(y.unique().tolist())
        }, f)

    print(f"Model saved to: {MODEL_OUTPUT_PATH}")
    print("\nNext step: use predict_category.py to classify new documents,")
    print("with a confidence threshold to decide when to trust this model")
    print("vs falling back to embedding similarity.")


if __name__ == '__main__':
    main()
