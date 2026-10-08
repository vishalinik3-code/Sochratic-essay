"""
Reads the saved results of every trained model (models/results/*.json),
builds a single comparison table sorted by Macro F1 (the main selection
metric for this multiclass, potentially imbalanced task), prints it, and
saves it to results/model_comparison.csv.

This does NOT decide or save the "best" model - that is select_best_model.py.
This script only builds the comparison table.

Run directly with:
    python src/compare_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import common

import pandas as pd


def build_comparison_table() -> pd.DataFrame:
    # Only the registered effectiveness models (never the auxiliary discourse_type
    # classifier, which is a separate, simpler model and must not be ranked
    # alongside the effectiveness models being compared for the primary task).
    result_files = [
        config.result_json_path(key)
        for key in config.MODEL_DISPLAY_NAMES
        if config.result_json_path(key).is_file()
    ]
    if not result_files:
        raise FileNotFoundError(
            "No trained model results were found in models/results/. "
            "Train at least one model first (e.g. python src/train_model_1.py)."
        )

    rows = []
    for path in result_files:
        record = common.load_json(path)
        rows.append({
            "Model": record["model_name"],
            "model_key": record["model_key"],
            "Accuracy": record["accuracy"],
            "Precision (macro)": record["precision_macro"],
            "Recall (macro)": record["recall_macro"],
            "Macro F1": record["macro_f1"],
            "Weighted F1": record["weighted_f1"],
            "Training Time (s)": record.get("training_time_seconds"),
            "Prediction Time (s)": record.get("prediction_time_seconds"),
        })

    df = pd.DataFrame(rows).sort_values("Macro F1", ascending=False).reset_index(drop=True)
    return df


def print_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    name_width = max(len(n) for n in df["Model"]) + 2
    print(f"{'Model':<{name_width}}Macro F1")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"{row['Model']:<{name_width}}{row['Macro F1']:.4f}")
    print("=" * 70)


def main() -> None:
    config.ensure_dirs()
    df = build_comparison_table()
    print_table(df)
    df.to_csv(config.MODEL_COMPARISON_CSV, index=False)
    print(f"\n[SUCCESS] Full comparison table saved to: {config.MODEL_COMPARISON_CSV}\n")


if __name__ == "__main__":
    main()
