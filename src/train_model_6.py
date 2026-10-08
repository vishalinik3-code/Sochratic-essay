"""
MODEL 6: Pretrained Sentence Transformer embeddings + MLP classifier

Why this model: instead of learning a text representation from scratch (as
Models 4/5 do) or using word-frequency features (Models 1-3), this model uses
a small, pretrained sentence-transformer to turn each text segment into a
dense embedding that already captures general semantic meaning learned from
a huge amount of text. A simple MLP (Multi-Layer Perceptron) classifier is
then trained on top of these frozen embeddings.

We deliberately use a LIGHTWEIGHT pretrained model ("all-MiniLM-L6-v2",
~80MB) rather than a large transformer, per the project requirements.

Note on text input: this model uses the ORIGINAL text (not the lowercased,
punctuation-stripped "clean_text" used by Models 1-5). Pretrained sentence
embeddings are trained on natural text and generally work better when
capitalisation and punctuation are preserved - stripping them would throw
away information the pretrained model knows how to use. This is a deliberate,
documented difference, not an inconsistency.

Requires: sentence-transformers (and, transitively, PyTorch). If either is
not installed, this script prints a clear error and exits without crashing
the rest of the pipeline.

Run directly with:
    python src/train_model_6.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import common

import joblib
import numpy as np

MODEL_KEY = "sbert_mlp"


def require_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer
    except Exception as exc:
        raise ImportError(
            "The 'sentence-transformers' package (and PyTorch, which it depends on) is "
            "required for this model but is not installed. Install it with:\n"
            "    pip install sentence-transformers\n"
            f"(Original error: {exc})"
        ) from exc


def main() -> None:
    print("\n" + "#" * 70)
    print(f"# TRAINING: {config.MODEL_DISPLAY_NAMES[MODEL_KEY]}")
    print("#" * 70)

    SentenceTransformer = require_sentence_transformers()
    from sklearn.neural_network import MLPClassifier

    common.set_seed()
    train_df, test_df = common.load_train_test_split()
    labels = common.load_label_classes()
    id_to_label = {i: c for i, c in enumerate(labels)}

    text_col = "original_text" if "original_text" in train_df.columns else "clean_text"
    train_texts = train_df[text_col].astype(str).tolist()
    test_texts = test_df[text_col].astype(str).tolist()
    y_train = train_df["effectiveness_label_id"].to_numpy()
    y_test_ids = test_df["effectiveness_label_id"].to_numpy()
    y_test = [id_to_label[i] for i in y_test_ids]

    print(f"[INFO] Loading pretrained sentence-transformer: {config.SENTENCE_TRANSFORMER_MODEL} "
          f"(first run downloads it, ~80MB; later runs use the local cache)")
    embedder = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)

    with common.Timer() as train_timer:
        print("[INFO] Encoding training texts into embeddings...")
        X_train = embedder.encode(train_texts, show_progress_bar=True, batch_size=32)
        mlp = MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=500,
            random_state=config.RANDOM_SEED,
            early_stopping=True,
        )
        print("[INFO] Training MLP classifier on embeddings...")
        mlp.fit(X_train, y_train)
    print(f"[INFO] Training (embedding + MLP fit) took {train_timer.seconds:.3f}s")

    with common.Timer() as pred_timer:
        X_test = embedder.encode(test_texts, show_progress_bar=False, batch_size=32)
        y_pred_ids = mlp.predict(X_test)
    y_pred = [id_to_label[i] for i in y_pred_ids]
    print(f"[INFO] Prediction (embedding + MLP predict) on {len(y_test)} rows took {pred_timer.seconds:.4f}s")

    metrics = common.compute_metrics(y_test, y_pred, labels)
    common.save_model_results(
        MODEL_KEY, metrics, train_timer.seconds, pred_timer.seconds,
        extra_info={
            "sentence_transformer_model": config.SENTENCE_TRANSFORMER_MODEL,
            "text_column_used": text_col,
            "embedding_dim": int(X_train.shape[1]),
        },
    )

    joblib.dump(mlp, config.model_path(MODEL_KEY, "mlp_classifier.joblib"))
    common.save_json(
        {"sentence_transformer_model": config.SENTENCE_TRANSFORMER_MODEL, "text_column_used": text_col},
        config.model_path(MODEL_KEY, "embedding_config.json"),
    )
    print(f"[INFO] Saved model artifacts to {config.MODELS_SAVED_DIR / MODEL_KEY}")


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        print(f"\n[SKIPPING MODEL 6] {exc}\n")
        sys.exit(1)
