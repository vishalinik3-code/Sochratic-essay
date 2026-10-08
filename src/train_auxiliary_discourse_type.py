"""
Auxiliary classifier: predicts discourse_type (Lead/Position/Claim/Evidence/...).

This is INTENTIONALLY separate from, and much simpler than, the effectiveness-model
comparison, which focuses entirely on the PRIMARY target
(discourse_effectiveness), as requested. Duplicating all architectures for
a second target would substantially complicate the project for a secondary,
"if practical" requirement - so instead, a single fast, reasonable TF-IDF +
Logistic Regression classifier is trained here, just so the Streamlit app
can still show a predicted discourse type alongside the main effectiveness
prediction.

This script is skipped automatically (with a clear message) if the dataset
has no discourse-type column.

Run directly with:
    python src/train_auxiliary_discourse_type.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import common

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODEL_KEY = "aux_discourse_type"


def main() -> None:
    print("\n" + "#" * 70)
    print("# TRAINING AUXILIARY CLASSIFIER: discourse_type")
    print("#" * 70)

    if not config.FULL_PROCESSED_CSV.is_file():
        raise FileNotFoundError(
            "data/processed/full_preprocessed.csv was not found. Run "
            "'python src/data_preprocessing.py' first."
        )

    df = pd.read_csv(config.FULL_PROCESSED_CSV)
    type_col = None
    for candidate in config.TYPE_COLUMN_ALTERNATIVES:
        if candidate in df.columns:
            type_col = candidate
            break

    if type_col is None:
        print("[INFO] No discourse-type column found in the dataset. "
              "Skipping the auxiliary discourse-type classifier - the app will "
              "simply not show a predicted discourse type.")
        return

    common.set_seed()
    from sklearn.model_selection import train_test_split

    df = df[df[type_col].notnull()]
    df = common.filter_rare_classes(df, type_col, min_count=2)
    train_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=df[type_col]
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=1, sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_df["clean_text"].astype(str))
    X_test = vectorizer.transform(test_df["clean_text"].astype(str))

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.RANDOM_SEED)

    with common.Timer() as train_timer:
        model.fit(X_train, train_df[type_col])
    with common.Timer() as pred_timer:
        y_pred = model.predict(X_test)

    labels = sorted(df[type_col].unique().tolist())
    metrics = common.compute_metrics(test_df[type_col].tolist(), y_pred.tolist(), labels)
    common.save_model_results(MODEL_KEY, metrics, train_timer.seconds, pred_timer.seconds,
                               extra_info={"note": "Auxiliary classifier for discourse_type, not part of the effectiveness-model comparison."})

    joblib.dump(vectorizer, config.model_path(MODEL_KEY, "vectorizer.joblib"))
    joblib.dump(model, config.model_path(MODEL_KEY, "model.joblib"))
    common.save_json({"labels": labels}, config.model_path(MODEL_KEY, "labels.json"))
    print(f"[INFO] Saved auxiliary discourse-type classifier to {config.MODELS_SAVED_DIR / MODEL_KEY}")


if __name__ == "__main__":
    main()
