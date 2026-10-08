"""
Shared PyTorch training logic for Model 4 (LSTM) and Model 5 (BiLSTM).

Both models share: a word-embedding layer learned from scratch on this
dataset (not a pretrained embedding - that is what distinguishes them from
Model 6, which uses pretrained sentence embeddings), an LSTM or BiLSTM
encoder, and a dense (Linear) output layer, trained with early stopping on
an internal validation split carved OUT of the training data only (so the
official test set is never touched during training or model selection).

This file is imported by train_model_4.py and train_model_5.py - it is not
run directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import common

import numpy as np


def require_torch():
    """Imports torch with a clear, friendly error message if it is missing
    or broken, instead of a raw ImportError/AttributeError traceback."""
    try:
        import torch
        import torch.nn as nn

        _ = torch.zeros(1)  # sanity check: a broken/partial install can import but fail on use
        return torch, nn
    except Exception as exc:
        raise ImportError(
            "PyTorch ('torch') is required for this model but is not installed or is "
            "broken in this environment. Install it with:\n"
            "    pip install torch\n"
            f"(Original error: {exc})"
        ) from exc


class LSTMClassifier:
    """Thin wrapper so train_model_4.py / train_model_5.py stay almost identical,
    differing only in bidirectional=True/False."""

    def __init__(self, vocab_size: int, num_classes: int, bidirectional: bool):
        torch, nn = require_torch()
        self.torch = torch

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM, padding_idx=0)
                self.lstm = nn.LSTM(
                    input_size=config.EMBEDDING_DIM,
                    hidden_size=config.LSTM_HIDDEN_DIM,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
                out_dim = config.LSTM_HIDDEN_DIM * (2 if bidirectional else 1)
                self.dropout = nn.Dropout(0.3)
                self.fc = nn.Linear(out_dim, num_classes)

            def forward(self, x):
                embedded = self.embedding(x)
                _, (hidden, _) = self.lstm(embedded)
                if bidirectional:
                    # concatenate the final forward and backward hidden states
                    final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
                else:
                    final_hidden = hidden[-1]
                return self.fc(self.dropout(final_hidden))

        self.net = _Net()

    def parameters(self):
        return self.net.parameters()


def train_with_early_stopping(
    model_key: str,
    bidirectional: bool,
    max_epochs: int = 30,
    patience: int = 4,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
):
    """
    Full training routine shared by Model 4 and Model 5:
      1. Build a vocabulary from the TRAINING text only.
      2. Carve an internal validation split out of the training data (never
         touching the official test set) to decide when to stop training.
      3. Train with early stopping on validation macro-F1.
      4. Evaluate once, at the end, on the official test set.
      5. Save results + the trained model + vocabulary.
    """
    torch, nn = require_torch()
    from sklearn.model_selection import train_test_split

    common.set_seed()
    device_name, device = common.get_torch_device()
    print(f"[INFO] Using device: {device_name} (GPU is optional - CPU works fine, just slower)")

    train_df, test_df = common.load_train_test_split()
    labels = common.load_label_classes()
    id_to_label = {i: c for i, c in enumerate(labels)}
    num_classes = len(labels)

    # Internal validation split, carved out of TRAIN only.
    inner_train_df, inner_val_df = train_test_split(
        train_df, test_size=0.15, random_state=config.RANDOM_SEED,
        stratify=train_df["effectiveness_label_id"],
    )

    vocab = common.build_vocab(inner_train_df["clean_text"].astype(str), config.MAX_VOCAB_SIZE)
    print(f"[INFO] Vocabulary size: {len(vocab)} words")

    X_train = common.encode_texts(inner_train_df["clean_text"].astype(str), vocab, config.MAX_SEQUENCE_LENGTH)
    X_val = common.encode_texts(inner_val_df["clean_text"].astype(str), vocab, config.MAX_SEQUENCE_LENGTH)
    X_test = common.encode_texts(test_df["clean_text"].astype(str), vocab, config.MAX_SEQUENCE_LENGTH)
    y_train = inner_train_df["effectiveness_label_id"].to_numpy()
    y_val = inner_val_df["effectiveness_label_id"].to_numpy()
    y_test_ids = test_df["effectiveness_label_id"].to_numpy()
    y_test = [id_to_label[i] for i in y_test_ids]

    def to_tensor_dataset(X, y):
        return torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)
        )

    train_loader = torch.utils.data.DataLoader(to_tensor_dataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(to_tensor_dataset(X_val, y_val), batch_size=batch_size)

    wrapper = LSTMClassifier(len(vocab), num_classes, bidirectional=bidirectional)
    net = wrapper.net.to(device)

    # Class weights to help with class imbalance (matches the classical models' class_weight="balanced").
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_counts[class_counts == 0] = 1.0
    class_weights = torch.tensor((class_counts.sum() / class_counts), dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    from sklearn.metrics import f1_score

    best_val_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0

    with common.Timer() as train_timer:
        for epoch in range(1, max_epochs + 1):
            net.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_loss = total_loss / len(train_loader.dataset)

            # Validation (on the INTERNAL validation split, not the test set).
            net.eval()
            val_preds, val_true = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    logits = net(xb)
                    val_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                    val_true.extend(yb.numpy().tolist())
            val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
            print(f"[EPOCH {epoch}/{max_epochs}] train_loss={avg_loss:.4f}  val_macro_f1={val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"[INFO] Early stopping triggered after {epoch} epochs "
                          f"(no improvement for {patience} epochs). Best val macro F1: {best_val_f1:.4f}")
                    break

    if best_state is not None:
        net.load_state_dict(best_state)

    # Final, one-time evaluation on the official, untouched test set.
    net.eval()
    with common.Timer() as pred_timer:
        with torch.no_grad():
            test_logits = net(torch.tensor(X_test, dtype=torch.long).to(device))
            y_pred_ids = test_logits.argmax(dim=1).cpu().numpy().tolist()
    y_pred = [id_to_label[i] for i in y_pred_ids]

    metrics = common.compute_metrics(y_test, y_pred, labels)
    common.save_model_results(
        model_key, metrics, train_timer.seconds, pred_timer.seconds,
        extra_info={
            "device": device_name,
            "vocab_size": len(vocab),
            "best_internal_val_macro_f1": round(float(best_val_f1), 4),
            "max_sequence_length": config.MAX_SEQUENCE_LENGTH,
        },
    )

    # Save artifacts needed to reload this model for inference.
    torch.save(net.state_dict(), config.model_path(model_key, "model_state.pt"))
    common.save_json(
        {"vocab": vocab, "bidirectional": bidirectional, "num_classes": num_classes},
        config.model_path(model_key, "vocab_and_config.json"),
    )
    print(f"[INFO] Saved model artifacts to {config.MODELS_SAVED_DIR / model_key}")
