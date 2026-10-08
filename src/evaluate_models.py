"""
Prints a consolidated evaluation summary for every model that has already
been trained (i.e. every JSON file found in models/results/, except
best_model.json, which is written later by select_best_model.py).

This does NOT retrain anything - it only reads the results each
train_model_X.py script already saved. Useful to quickly re-check results
without re-running training.

Run directly with:
    python src/evaluate_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import common


def main() -> None:
    config.ensure_dirs()
    # Only the official 6 comparison models (not the auxiliary discourse_type
    # classifier, which is trained and reported separately).
    result_files = [
        config.result_json_path(key)
        for key in config.MODEL_DISPLAY_NAMES
        if config.result_json_path(key).is_file()
    ]

    if not result_files:
        print("[INFO] No trained models found yet in models/results/. "
              "Run the train_model_*.py scripts (or run_all.py) first.")
        return

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY (read from saved results, nothing retrained)")
    print("=" * 70)

    for path in result_files:
        record = common.load_json(path)
        print(f"\nModel: {record['model_name']}  (key: {record['model_key']})")
        print(f"  Accuracy        : {record['accuracy']}")
        print(f"  Precision (macro): {record['precision_macro']}")
        print(f"  Recall (macro)   : {record['recall_macro']}")
        print(f"  Macro F1         : {record['macro_f1']}")
        print(f"  Weighted F1      : {record['weighted_f1']}")
        if record.get("training_time_seconds") is not None:
            print(f"  Training time    : {record['training_time_seconds']:.3f}s")
        if record.get("prediction_time_seconds") is not None:
            print(f"  Prediction time  : {record['prediction_time_seconds']:.4f}s")

    print("\n" + "=" * 70)
    print(f"Evaluated {len(result_files)} model(s). "
          f"Run 'python src/compare_models.py' for a sorted comparison table.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
