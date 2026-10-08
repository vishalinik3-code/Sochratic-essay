"""
MODEL 3: TF-IDF + Random Forest

Why this model: a non-linear, tree-based ensemble that can capture feature
interactions TF-IDF + linear models cannot, and serves as a useful contrast
to Models 1 and 2 to see whether non-linearity helps on this dataset.

Run directly with:
    python src/train_model_3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import common

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

MODEL_KEY = "tfidf_rf"


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
    id_to_label = {i: c for i, c in enumerate(labels)}
    y_test = [id_to_label[i] for i in y_test_ids]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=1, sublinear_tf=True)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
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

    joblib.dump(vectorizer, config.model_path(MODEL_KEY, "vectorizer.joblib"))
    joblib.dump(model, config.model_path(MODEL_KEY, "model.joblib"))
    print(f"[INFO] Saved model artifacts to {config.MODELS_SAVED_DIR / MODEL_KEY}")


if __name__ == "__main__":
    main()
