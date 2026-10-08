"""
MODEL 1: TF-IDF + Logistic Regression

Why this model: a strong, fast, highly interpretable classical baseline for
text classification. TF-IDF turns each text segment into a vector of weighted
word/n-gram frequencies; Logistic Regression then learns a linear decision
boundary between the effectiveness classes. It is a standard first baseline
against which more complex models should be compared.

Run directly with:
    python src/train_model_1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import common

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODEL_KEY = "tfidf_logreg"


def main() -> None:
    print("\n" + "#" * 70)
    print(f"# TRAINING: {config.MODEL_DISPLAY_NAMES[MODEL_KEY]}")
    print("#" * 70)

    common.set_seed()
    train_df, test_df = common.load_train_test_split()
    labels = common.load_label_classes()

    X_train_text = train_df["clean_text"].astype(str)
    X_test_text = test_df["clean_text"].astype(str)
    y_train = train_df["effectiveness_label_id"]
    y_test_ids = test_df["effectiveness_label_id"]
    # Map back to class NAMES for metrics, so results are human-readable and
    # directly comparable across all models regardless of internal encoding.
    id_to_label = {i: c for i, c in enumerate(labels)}
    y_test = [id_to_label[i] for i in y_test_ids]

    # TF-IDF is fit ONLY on the training text, never on test text, to avoid
    # leaking test-set vocabulary/statistics into the features.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),       # unigrams + bigrams capture short argumentative phrases
        max_features=20000,
        min_df=1,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # helps with class imbalance (requirement mentions this is likely)
        random_state=config.RANDOM_SEED,
    )

    with common.Timer() as train_timer:
        model.fit(X_train, y_train)
    print(f"[INFO] Training took {train_timer.seconds:.3f}s")

    with common.Timer() as pred_timer:
        y_pred_ids = model.predict(X_test)
    y_pred = [id_to_label[i] for i in y_pred_ids]
    print(f"[INFO] Prediction on {len(y_test)} test rows took {pred_timer.seconds:.4f}s")

    metrics = common.compute_metrics(y_test, y_pred, labels)
    common.save_model_results(MODEL_KEY, metrics, train_timer.seconds, pred_timer.seconds)

    # Save artifacts needed to reload this exact model later (by the app).
    joblib.dump(vectorizer, config.model_path(MODEL_KEY, "vectorizer.joblib"))
    joblib.dump(model, config.model_path(MODEL_KEY, "model.joblib"))
    print(f"[INFO] Saved model artifacts to {config.MODELS_SAVED_DIR / MODEL_KEY}")


if __name__ == "__main__":
    main()
