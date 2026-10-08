"""
MODEL 5: Word Embedding + BiLSTM

Why this model: a Bidirectional LSTM reads the text both forwards and
backwards, so each word's representation has context from both directions.
This often helps classification tasks where meaning depends on what comes
both before AND after a word (e.g. negation changing the meaning of a later
claim). This is directly comparable to Model 4, since it is identical except
for the `bidirectional=True` setting.

Requires PyTorch. If PyTorch is not installed, this script prints a clear
error and exits without crashing the rest of the pipeline.

Run directly with:
    python src/train_model_5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from _torch_lstm_shared import train_with_early_stopping, require_torch

MODEL_KEY = "embed_bilstm"


def main() -> None:
    print("\n" + "#" * 70)
    print(f"# TRAINING: {config.MODEL_DISPLAY_NAMES[MODEL_KEY]}")
    print("#" * 70)
    require_torch()
    train_with_early_stopping(MODEL_KEY, bidirectional=True)


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        print(f"\n[SKIPPING MODEL 5] {exc}\n")
        sys.exit(1)
