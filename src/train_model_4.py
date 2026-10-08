"""
MODEL 4: Word Embedding + LSTM

Why this model: unlike TF-IDF, which ignores word order, an LSTM (Long
Short-Term Memory network) reads the text sequentially and can, in principle,
learn patterns that depend on word order and sentence structure - potentially
useful for judging whether an argument is well-constructed.

The word embeddings here are learned FROM SCRATCH on this dataset (trained
together with the LSTM), not pretrained - "Word Embedding" in the project
spec refers to this learned embedding layer, as opposed to Model 6, which
uses a pretrained sentence-transformer.

Requires PyTorch. If PyTorch is not installed, this script prints a clear
error and exits without crashing the rest of the pipeline (run_all.py will
skip it and continue with the other models).

Run directly with:
    python src/train_model_4.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from _torch_lstm_shared import train_with_early_stopping, require_torch

MODEL_KEY = "embed_lstm"


def main() -> None:
    print("\n" + "#" * 70)
    print(f"# TRAINING: {config.MODEL_DISPLAY_NAMES[MODEL_KEY]}")
    print("#" * 70)
    require_torch()  # fail fast with a clear message if torch is unavailable
    train_with_early_stopping(MODEL_KEY, bidirectional=False)


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        # A clean, short message instead of a long traceback - this is an
        # expected, recoverable situation (missing optional dependency), not
        # a bug, so run_all.py can skip this model and continue with the rest.
        print(f"\n[SKIPPING MODEL 4] {exc}\n")
        sys.exit(1)
