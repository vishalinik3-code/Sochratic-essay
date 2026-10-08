"""
MODEL 2: TF-IDF + Linear SVM

Why this model: Linear Support Vector Machines are well known to perform very
strongly on high-dimensional sparse text features like TF-IDF, often matching
or beating more complex models on moderate-sized text classification tasks,
while remaining fast to train.

Run directly with:
    python src/train_model_2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import common

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

MODEL_KEY = "tfidf_svm"


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

    # Plain LinearSVC has no predict_proba. We wrap it in CalibratedClassifierCV
    # so the app can show a confidence score (the UI requirement asks for
    # "Confidence/probability if the selected model supports it" - this makes
    # SVM support it too, fairly, without changing its decision boundary much).
    base_svm = LinearSVC(class_weight="balanced", random_state=config.RANDOM_SEED, max_iter=5000)
    model = CalibratedClassifierCV(base_svm, cv=3)

    with common.Timer() as train_timer:
        model.fit(X_train, y_train)
    print(f"[INFO] Training took {train_timer.seconds:.3f}s")

    with common.Timer() as pred_timer:
        y_pred_ids = model.predict(X_test)
    y_pred = [id_to_label[i] for i in y_pred_ids]
    print(f"[INFO] Prediction on {len(y_test)} test rows took {pred_timer.seconds:.4f}s")

    metrics = common.compute_metrics(y_test, y_pred, labels)
    common.save_model_results(MODEL_KEY, metrics, train_timer.seconds, pred_timer.seconds,
                               extra_info={"note": "LinearSVC wrapped in CalibratedClassifierCV to provide predict_proba."})

    joblib.dump(vectorizer, config.model_path(MODEL_KEY, "vectorizer.joblib"))
    joblib.dump(model, config.model_path(MODEL_KEY, "model.joblib"))
    print(f"[INFO] Saved model artifacts to {config.MODELS_SAVED_DIR / MODEL_KEY}")


if __name__ == "__main__":
    main()
