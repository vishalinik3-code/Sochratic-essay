"""
Shared helpers used by every training script: reproducible seeding, timing,
metric computation, and saving results in a consistent format so that
compare_models.py can read them all the same way.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import config


# =====================================================================================
# REPRODUCIBILITY
# =====================================================================================
def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Sets the random seed for every library that might be used, so that
    results are reproducible across runs. Safe to call even if some optional
    libraries (torch) are not installed.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # torch may be missing, or only partially installed - seeding it is a
        # nice-to-have, never something that should crash the whole pipeline.
        pass


# =====================================================================================
# TIMING
# =====================================================================================
class Timer:
    """Simple context manager: `with Timer() as t: ...` then read t.seconds."""

    def __enter__(self):
        self._start = time.perf_counter()
        self.seconds = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.seconds = round(time.perf_counter() - self._start, 4)
        return False


# =====================================================================================
# JSON HELPERS
# =====================================================================================
def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    # Allows numpy types to be saved to JSON without crashing.
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


# =====================================================================================
# METRICS
# =====================================================================================
def compute_metrics(y_true: Sequence, y_pred: Sequence, labels: List[str]) -> Dict:
    """
    Computes every metric the project requires for one model's predictions on
    the SAME test set. `labels` should be the fixed, ordered list of class
    names used consistently across all models (so confusion matrices and
    per-class reports line up when compared).
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    accuracy = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report_text = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    return {
        "accuracy": round(float(accuracy), 4),
        "precision_macro": round(float(macro_p), 4),
        "recall_macro": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "classification_report": report_text,
    }


def save_confusion_matrix_plot(cm: List[List[int]], labels: List[str], model_key: str) -> Path:
    """Saves a confusion matrix heatmap as a PNG. Falls back to skipping the
    plot (with a warning) if matplotlib is not installed - this never crashes
    the whole training run just because a plotting library is missing."""
    path = config.CONFUSION_MATRICES_DIR / f"{model_key}_confusion_matrix.png"
    try:
        import matplotlib

        matplotlib.use("Agg")  # no GUI needed, works on any machine/server
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 2, 1.2 * len(labels) + 2))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"Confusion Matrix: {config.MODEL_DISPLAY_NAMES.get(model_key, model_key)}")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                        color="white" if cm[i][j] > (max(max(r) for r in cm) / 2) else "black")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except ImportError:
        print(f"[WARNING] matplotlib is not installed, skipping confusion matrix plot for '{model_key}'. "
              f"The numeric confusion matrix was still saved in the JSON/CSV results.")
    return path


def save_model_results(
    model_key: str,
    metrics: Dict,
    training_time_seconds: Optional[float],
    prediction_time_seconds: Optional[float],
    extra_info: Optional[Dict] = None,
) -> Path:
    """
    Saves one model's full results to models/results/<model_key>.json, a plain
    text classification report to results/classification_reports/, a
    confusion matrix CSV to results/confusion_matrices/, and a PNG plot
    (if matplotlib is available). Returns the path to the main JSON file.
    """
    config.ensure_dirs()
    display_name = config.MODEL_DISPLAY_NAMES.get(
        model_key, config.AUXILIARY_MODEL_DISPLAY_NAMES.get(model_key, model_key)
    )

    record = {
        "model_key": model_key,
        "model_name": display_name,
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "training_time_seconds": training_time_seconds,
        "prediction_time_seconds": prediction_time_seconds,
        "labels": metrics["labels"],
    }
    if extra_info:
        record["extra_info"] = extra_info

    save_json(record, config.result_json_path(model_key))

    # Classification report (plain text, easy for a beginner to open and read).
    report_path = config.CLASSIFICATION_REPORTS_DIR / f"{model_key}_classification_report.txt"
    report_path.write_text(
        f"Model: {display_name}\n"
        f"{'=' * 60}\n{metrics['classification_report']}\n",
        encoding="utf-8",
    )

    # Confusion matrix as a readable CSV (rows/columns labelled).
    cm_df = pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"])
    cm_csv_path = config.CONFUSION_MATRICES_DIR / f"{model_key}_confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)

    save_confusion_matrix_plot(metrics["confusion_matrix"], metrics["labels"], model_key)

    print(f"\n[RESULTS] {display_name}")
    print(f"  Accuracy      : {metrics['accuracy']}")
    print(f"  Precision(M)  : {metrics['precision_macro']}")
    print(f"  Recall(M)     : {metrics['recall_macro']}")
    print(f"  Macro F1      : {metrics['macro_f1']}")
    print(f"  Weighted F1   : {metrics['weighted_f1']}")
    if training_time_seconds is not None:
        print(f"  Training time : {training_time_seconds:.2f}s")
    if prediction_time_seconds is not None:
        print(f"  Prediction time: {prediction_time_seconds:.4f}s")

    return config.result_json_path(model_key)


# =====================================================================================
# DEVICE DETECTION (for the deep-learning models)
# =====================================================================================
def get_torch_device():
    """Returns ('cuda', torch.device) if a GPU is available, else ('cpu', torch.device).
    GPU is never required - this function always returns something usable."""
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.device("cuda")
    return "cpu", torch.device("cpu")


# =====================================================================================
# LOADING THE SHARED, ALREADY-SPLIT DATA
# =====================================================================================
def load_train_test_split():
    """
    Loads the train/test CSVs produced by data_preprocessing.py. Every model
    script calls this SAME function so that all models are evaluated on the
    exact same test set, split the exact same way - required for a fair
    comparison and to avoid data leakage.
    """
    if not config.TRAIN_SPLIT_CSV.is_file() or not config.TEST_SPLIT_CSV.is_file():
        raise FileNotFoundError(
            "Processed train/test files were not found. Please run "
            "'python src/data_preprocessing.py' (or 'python run_all.py') first."
        )
    train_df = pd.read_csv(config.TRAIN_SPLIT_CSV)
    test_df = pd.read_csv(config.TEST_SPLIT_CSV)
    return train_df, test_df


def load_label_classes() -> List[str]:
    """Returns the fixed, ordered list of class names for the primary target
    (discourse_effectiveness), shared by all models so confusion matrices and
    reports are directly comparable."""
    info = load_json(config.LABEL_ENCODER_JSON)
    return info["classes"]


def filter_rare_classes(df: pd.DataFrame, label_col: str, min_count: int = 2) -> pd.DataFrame:
    """
    Stratified train/test splitting requires at least `min_count` examples of
    every class. Removes rows belonging to any class with fewer examples than
    that, printing a clear warning naming which classes (and how many rows)
    were dropped, so this is never a silent surprise. This is a data-quality
    safeguard, not a modelling choice - classes this rare cannot be reliably
    evaluated with a train/test split anyway.
    """
    counts = df[label_col].value_counts()
    rare_classes = counts[counts < min_count].index.tolist()
    if not rare_classes:
        return df
    print(f"[WARNING] Dropping {len(rare_classes)} class(es) in '{label_col}' with fewer than "
          f"{min_count} example(s) (too few to stratify-split): {rare_classes}")
    return df[~df[label_col].isin(rare_classes)].reset_index(drop=True)


# =====================================================================================
# SIMPLE WORD-LEVEL VOCABULARY (used by the LSTM / BiLSTM models)
# =====================================================================================
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocab(texts: Sequence[str], max_vocab_size: int) -> Dict[str, int]:
    """
    Builds a word -> integer-id vocabulary from the given texts ONLY (which
    must be the TRAINING texts - never the test texts, to avoid leaking any
    information about the test set into the model). Index 0 is reserved for
    padding, index 1 for unknown/out-of-vocabulary words.
    """
    from collections import Counter

    counter: Counter = Counter()
    for text in texts:
        counter.update(str(text).split())

    most_common = counter.most_common(max_vocab_size - 2)  # reserve PAD, UNK
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, _ in most_common:
        vocab[word] = len(vocab)
    return vocab


def encode_texts(texts: Sequence[str], vocab: Dict[str, int], max_length: int) -> np.ndarray:
    """Converts a list of texts into a fixed-length (max_length) array of
    integer word-ids, padding short texts with PAD and truncating long ones.
    Words not seen during training become UNK - this is the expected and
    correct way to handle unseen words at test/inference time."""
    pad_id = vocab[PAD_TOKEN]
    unk_id = vocab[UNK_TOKEN]
    encoded = np.full((len(texts), max_length), pad_id, dtype=np.int64)
    for row, text in enumerate(texts):
        word_ids = [vocab.get(w, unk_id) for w in str(text).split()[:max_length]]
        encoded[row, : len(word_ids)] = word_ids
    return encoded
