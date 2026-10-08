"""
Reads results/model_comparison.csv (built by compare_models.py), picks the
model with the HIGHEST Macro F1 (the primary selection metric for this
multiclass, potentially imbalanced task), and saves that decision to
models/results/best_model.json.

The best model is never hard-coded here - it is always read from the actual
saved metrics produced by training. Running this script twice on the same
results will always pick the same (correct) winner.

Also records, for each model type, exactly which saved artifact files the
Streamlit app needs to reload it for live prediction.

Run directly with:
    python src/select_best_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import common

import pandas as pd

# Maps a model_key to the "model_type" the app's prediction.py module uses to
# know HOW to reload it (which artifact files to open, with which library).
MODEL_TYPE_BY_KEY = {
    "tfidf_logreg": "sklearn_tfidf",
    "tfidf_svm": "sklearn_tfidf",
    "tfidf_rf": "sklearn_tfidf",
    "embed_lstm": "torch_lstm",
    "embed_bilstm": "torch_lstm",
    "sbert_mlp": "sentence_transformer_mlp",
}


def main() -> None:
    if not config.MODEL_COMPARISON_CSV.is_file():
        raise FileNotFoundError(
            "results/model_comparison.csv was not found. Run 'python src/compare_models.py' first "
            "(or run the full pipeline with 'python run_all.py')."
        )

    df = pd.read_csv(config.MODEL_COMPARISON_CSV)
    if df.empty:
        raise ValueError("results/model_comparison.csv is empty - no models to select from.")

    # The comparison table is already sorted by Macro F1 descending, but we
    # re-sort here defensively in case the CSV was edited or regenerated.
    best_row = df.sort_values("Macro F1", ascending=False).iloc[0]
    best_key = best_row["model_key"]
    model_type = MODEL_TYPE_BY_KEY.get(best_key, "unknown")

    best_model_info = {
        "best_model": best_row["Model"],
        "model_key": best_key,
        "model_type": model_type,
        "metric": "macro_f1",
        "score": float(best_row["Macro F1"]),
        "accuracy": float(best_row["Accuracy"]),
        "weighted_f1": float(best_row["Weighted F1"]),
        "artifacts_dir": str((config.MODELS_SAVED_DIR / best_key).relative_to(config.PROJECT_ROOT)),
    }
    common.save_json(best_model_info, config.BEST_MODEL_JSON)

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)
    print(f"Model: {best_model_info['best_model']}")
    print(f"Macro F1: {best_model_info['score']:.4f}")
    print("=" * 70)
    print(f"\n[SUCCESS] Saved selection to: {config.BEST_MODEL_JSON}\n")


if __name__ == "__main__":
    main()
