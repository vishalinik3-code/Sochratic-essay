"""
Runs the complete training and comparison pipeline with one command:

    python run_all.py

Steps:
    1. Check the dataset exists.
    2. Preprocess the data (one shared train/test split for all models).
    3-8. Train Models 1-6.
    10. Train the auxiliary discourse_type classifier (used by the app).
    11. Evaluate all models (prints a summary from saved results).
    12. Build the model comparison table (results/model_comparison.csv).
    13. Select the best model by Macro F1 (models/results/best_model.json).
    14. Print the final ranking.

Each step runs as its own subprocess. If an optional model fails because a
dependency is unavailable or training fails, a clear warning is printed and the
pipeline CONTINUES with the remaining steps, rather than stopping completely -
the comparison and best-model selection still work correctly with whichever
models did succeed (at least 1 model must succeed for selection to work).

This script must be run from the project root (the folder containing this
file) - see README.md for exact commands.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

# (step title, script path, is_required)
# is_required=True means: if this step fails, stop the whole pipeline (there is
# no point continuing without it). is_required=False means: print a clear
# warning and move on to the next step.
STEPS = [
    ("Checking dataset & preprocessing data", SRC_DIR / "data_preprocessing.py", True),
    ("Training Model 1: TF-IDF + Logistic Regression", SRC_DIR / "train_model_1.py", True),
    ("Training Model 2: TF-IDF + Linear SVM", SRC_DIR / "train_model_2.py", True),
    ("Training Model 3: TF-IDF + Random Forest", SRC_DIR / "train_model_3.py", True),
    ("Training Model 4: Word Embedding + LSTM", SRC_DIR / "train_model_4.py", False),
    ("Training Model 5: Word Embedding + BiLSTM", SRC_DIR / "train_model_5.py", False),
    ("Training Model 6: Sentence Transformer + MLP", SRC_DIR / "train_model_6.py", False),
    ("Training auxiliary discourse_type classifier", SRC_DIR / "train_auxiliary_discourse_type.py", False),
    ("Evaluating all trained models", SRC_DIR / "evaluate_models.py", True),
    ("Building model comparison table", SRC_DIR / "compare_models.py", True),
    ("Selecting the best model (by Macro F1)", SRC_DIR / "select_best_model.py", True),
]


def run_step(title: str, script: Path, required: bool) -> bool:
    print("\n" + "=" * 70)
    print(f"STEP: {title}")
    print("=" * 70)

    if not script.is_file():
        print(f"[ERROR] Script not found: {script}")
        return False

    start = time.perf_counter()
    result = subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT))
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        if required:
            print(f"\n[FATAL] Required step '{title}' failed (exit code {result.returncode}) "
                  f"after {elapsed:.1f}s. Stopping the pipeline.")
        else:
            print(f"\n[WARNING] Optional step '{title}' failed (exit code {result.returncode}) "
                  f"after {elapsed:.1f}s. This usually means a required library "
                  f"(PyTorch or sentence-transformers) is missing or could not install. "
                  f"Continuing with the remaining steps - this model will simply be "
                  f"missing from the comparison.")
        return result.returncode == 0

    print(f"\n[OK] '{title}' completed in {elapsed:.1f}s.")
    return True


def main() -> None:
    print("\n" + "#" * 70)
    print("# RUNNING THE FULL ARGUMENT MINING PIPELINE")
    print("#" * 70)

    failed_optional = []
    for title, script, required in STEPS:
        success = run_step(title, script, required)
        if not success:
            if required:
                sys.exit(1)
            failed_optional.append(title)

    best_model_json = PROJECT_ROOT / "models" / "results" / "best_model.json"
    comparison_csv = PROJECT_ROOT / "results" / "model_comparison.csv"

    print("\n" + "#" * 70)
    print("# PIPELINE COMPLETE")
    print("#" * 70)
    if failed_optional:
        print(f"\n[NOTE] {len(failed_optional)} optional model(s) could not be trained and were skipped:")
        for title in failed_optional:
            print(f"  - {title}")
        print("See the warnings above for the exact missing dependency. The comparison "
              "and best-model selection below are still valid and based on real results "
              "from the models that DID train successfully.")

    print(f"\nComparison table: {comparison_csv}")
    print(f"Best model file : {best_model_json}")
    print("\nNext step: launch the app with:")
    print("    python -m streamlit run app/app.py")
    print()


if __name__ == "__main__":
    main()
