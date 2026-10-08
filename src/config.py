"""
Central configuration for the whole project: paths, column names, random seed.

This is the ONLY file you should normally need to edit, and only if your
dataset uses different column names than the Feedback Prize dataset.
Everything else in the project uses relative paths computed from this file,
so the project works no matter where you place the project folder, as long
as you run scripts from the project root (see README.md).
"""
from __future__ import annotations

import os
from pathlib import Path

# =====================================================================================
# PATHS (relative to the project root - the folder that contains this "src" folder)
# =====================================================================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODELS_SAVED_DIR = PROJECT_ROOT / "models" / "saved"
MODELS_RESULTS_DIR = PROJECT_ROOT / "models" / "results"

RESULTS_DIR = PROJECT_ROOT / "results"
CLASSIFICATION_REPORTS_DIR = RESULTS_DIR / "classification_reports"
CONFUSION_MATRICES_DIR = RESULTS_DIR / "confusion_matrices"
MODEL_COMPARISON_CSV = RESULTS_DIR / "model_comparison.csv"
BEST_MODEL_JSON = MODELS_RESULTS_DIR / "best_model.json"

TRAIN_SPLIT_CSV = DATA_PROCESSED_DIR / "train.csv"
TEST_SPLIT_CSV = DATA_PROCESSED_DIR / "test.csv"
SPLIT_INFO_JSON = DATA_PROCESSED_DIR / "split_info.json"
LABEL_ENCODER_JSON = DATA_PROCESSED_DIR / "label_encoder.json"
FULL_PROCESSED_CSV = DATA_PROCESSED_DIR / "full_preprocessed.csv"

ALL_DIRS = [
    DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_SAVED_DIR, MODELS_RESULTS_DIR,
    RESULTS_DIR, CLASSIFICATION_REPORTS_DIR, CONFUSION_MATRICES_DIR,
    PROJECT_ROOT / "notebooks",
]


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# =====================================================================================
# DATASET COLUMN NAMES
# =====================================================================================
# These match the Kaggle "Feedback Prize - Predicting Effective Arguments" dataset.
# If your dataset uses different column names, either rename your columns to match,
# or change the values below (data_preprocessing.py will also try a few common
# alternatives automatically and tell you clearly if it cannot find a match).
TEXT_COLUMN = "discourse_text"
TYPE_COLUMN = "discourse_type"          # secondary target (auxiliary classifier)
EFFECTIVENESS_COLUMN = "discourse_effectiveness"  # PRIMARY target for model comparison

# Reasonable alternative column names data_preprocessing.py will also recognise,
# in case your copy of the dataset was renamed or came from a slightly different
# export. The first match found is used.
TEXT_COLUMN_ALTERNATIVES = [TEXT_COLUMN, "text", "essay_text", "discourse", "argument_text"]
TYPE_COLUMN_ALTERNATIVES = [TYPE_COLUMN, "type", "argument_type", "label_type"]
EFFECTIVENESS_COLUMN_ALTERNATIVES = [
    EFFECTIVENESS_COLUMN, "effectiveness", "quality", "discourse_effectivness", "label",
]

# =====================================================================================
# REPRODUCIBILITY
# =====================================================================================
RANDOM_SEED = 42
TEST_SIZE = 0.2  # 20% held out for testing, stratified on the primary target

# =====================================================================================
# MODEL REGISTRY
# =====================================================================================
# Internal key -> human-readable display name. Every script in src/ refers to a
# model only by its key, so renaming a display name here updates it everywhere
# (comparison table, best_model.json, the Streamlit app) consistently.
MODEL_DISPLAY_NAMES = {
    "tfidf_logreg": "TF-IDF + Logistic Regression",
    "tfidf_svm": "TF-IDF + Linear SVM",
    "tfidf_rf": "TF-IDF + Random Forest",
    "embed_lstm": "Word Embedding + LSTM",
    "embed_bilstm": "Word Embedding + BiLSTM",
    "sbert_mlp": "Sentence Transformer + MLP",
}

# Display name for the auxiliary discourse_type classifier. Kept in a SEPARATE
# dict (not merged into MODEL_DISPLAY_NAMES above) so that compare_models.py /
# evaluate_models.py never accidentally include it in the effectiveness
# model comparison.
AUXILIARY_MODEL_DISPLAY_NAMES = {
    "aux_discourse_type": "Auxiliary: TF-IDF + Logistic Regression (discourse_type)",
}

# Lightweight pretrained sentence-transformer model used by Model 6.
# "all-MiniLM-L6-v2" is a small (~80MB), fast, widely-used general-purpose model -
# a deliberately lightweight choice rather than a large transformer.
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Text-length limits used by the LSTM/BiLSTM tokenizer (in tokens/words).
MAX_SEQUENCE_LENGTH = 128
EMBEDDING_DIM = 100
LSTM_HIDDEN_DIM = 64
MAX_VOCAB_SIZE = 20000


def model_path(model_key: str, filename: str) -> Path:
    """Standard location to save an artifact file for a given model key."""
    folder = MODELS_SAVED_DIR / model_key
    folder.mkdir(parents=True, exist_ok=True)
    return folder / filename


def result_json_path(model_key: str) -> Path:
    return MODELS_RESULTS_DIR / f"{model_key}.json"
