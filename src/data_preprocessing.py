"""
Data preprocessing for the AI-Powered Argument Mining project.

This script is the FOUNDATION that all effectiveness models build on. It:
    1. Finds and loads the dataset from data/raw/ (auto-detects the CSV file;
       extracts it automatically if it is inside a .zip).
    2. Checks data quality (missing values, duplicates).
    3. Cleans the text (lowercase, remove URLs, remove special characters,
       normalize whitespace).
    4. Encodes the primary target label (discourse_effectiveness) as integers.
    5. Creates ONE stratified train/test split and saves it to
       data/processed/train.csv and data/processed/test.csv.

Every model script loads the SAME train.csv / test.csv produced here. This guarantees
all models are compared
on identical data (no data leakage, no unfair advantage to any model).

Run directly with:
    python src/data_preprocessing.py
(or indirectly via `python run_all.py`, which calls this first).
"""
from __future__ import annotations

import os
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import common

import pandas as pd


# =====================================================================================
# STEP 0: FIND THE DATASET (auto-detect CSV, auto-extract ZIP, no renaming needed)
# =====================================================================================
def _is_safe_member_name(name: str) -> bool:
    normalized = os.path.normpath(name)
    return not normalized.startswith("..") and not os.path.isabs(normalized)


def _extract_zip_files(raw_folder: Path) -> int:
    zip_files = [f for f in os.listdir(raw_folder) if f.lower().endswith(".zip")]
    extracted = 0
    for zip_name in zip_files:
        zip_path = raw_folder / zip_name
        try:
            with zipfile.ZipFile(zip_path, "r") as archive:
                for member in archive.infolist():
                    if member.is_dir() or not member.filename.lower().endswith(".csv"):
                        continue
                    if not _is_safe_member_name(member.filename):
                        continue
                    target_name = os.path.basename(member.filename)
                    if not target_name:
                        continue
                    target_path = raw_folder / target_name
                    if target_path.exists():
                        continue
                    with archive.open(member) as src, open(target_path, "wb") as dst:
                        dst.write(src.read())
                    print(f"[INFO] Extracted '{target_name}' from '{zip_name}'.")
                    extracted += 1
        except zipfile.BadZipFile:
            print(f"[WARNING] '{zip_name}' is not a valid ZIP file, skipping.")
    return extracted


def find_dataset_csv() -> Path:
    """Finds the dataset CSV in data/raw/. Auto-extracts ZIPs. If several CSVs
    exist, prefers one literally named 'train.csv' (matching the suggested
    project layout); otherwise uses the first one found and tells you so."""
    raw_folder = config.DATA_RAW_DIR
    raw_folder.mkdir(parents=True, exist_ok=True)

    _extract_zip_files(raw_folder)

    csv_files = sorted(f for f in os.listdir(raw_folder) if f.lower().endswith(".csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV dataset was found in '{raw_folder}'.\n"
            f"Please place your dataset (a .csv file, or a .zip containing one) "
            f"directly inside that folder and run this script again."
        )

    if "train.csv" in csv_files:
        chosen = "train.csv"
    elif len(csv_files) == 1:
        chosen = csv_files[0]
    else:
        chosen = csv_files[0]
        print(f"[WARNING] Multiple CSV files found in {raw_folder}: {csv_files}. "
              f"Using '{chosen}'. To use a different one, remove the others or "
              f"rename your preferred file to 'train.csv'.")

    print(f"[INFO] Using dataset file: {raw_folder / chosen}")
    return raw_folder / chosen


# =====================================================================================
# STEP 1: LOAD
# =====================================================================================
def load_raw_dataset(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1")
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"The file '{csv_path}' is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"The file '{csv_path}' could not be parsed as CSV: {exc}") from exc

    print(f"[INFO] Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    print(f"[INFO] Columns found: {list(df.columns)}")
    return df


def _resolve_column(df: pd.DataFrame, alternatives, purpose: str, required: bool) -> str | None:
    for name in alternatives:
        if name in df.columns:
            return name
    # Case-insensitive fallback.
    lowered = {c.lower(): c for c in df.columns}
    for name in alternatives:
        if name.lower() in lowered:
            return lowered[name.lower()]
    if required:
        raise ValueError(
            f"Could not find a {purpose} column. Looked for any of: {alternatives}.\n"
            f"Columns actually in your dataset: {list(df.columns)}.\n"
            f"Please edit src/config.py and set the matching *_COLUMN variable to the "
            f"correct name from your dataset."
        )
    print(f"[WARNING] Could not find a {purpose} column (looked for {alternatives}). "
          f"Continuing without it.")
    return None


# =====================================================================================
# STEP 2 & 3: DATA QUALITY + CLEANING
# =====================================================================================
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_ALLOWED_CHARS_PATTERN = re.compile(r"[^a-zA-Z0-9\s\.\,\!\?\'\-]")
_EXTRA_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(raw_text) -> str:
    """Lowercase, remove URLs, remove special characters (keeping letters,
    digits, basic punctuation), normalize whitespace. Negation words ('not',
    'no', 'never') and numbers are preserved deliberately - they matter for
    judging argument quality."""
    if raw_text is None or (isinstance(raw_text, float)):
        return ""
    text = str(raw_text).lower()
    text = _URL_PATTERN.sub(" ", text)
    text = _ALLOWED_CHARS_PATTERN.sub(" ", text)
    text = _EXTRA_WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def check_and_clean(df: pd.DataFrame, text_col: str, type_col: str | None, eff_col: str) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("DATA QUALITY CHECK")
    print("=" * 70)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    duplicate_count = df.duplicated().sum()
    print(f"\nFully duplicate rows: {duplicate_count}")
    print(f"\nDistribution of '{eff_col}':")
    print(df[eff_col].value_counts(dropna=False))

    print("\n" + "=" * 70)
    print("CLEANING")
    print("=" * 70)
    df_clean = df.drop_duplicates().copy()
    print(f"Removed {duplicate_count} duplicate row(s).")

    before = len(df_clean)
    df_clean = df_clean[df_clean[text_col].notnull()]
    df_clean = df_clean[df_clean[text_col].astype(str).str.strip() != ""]
    print(f"Removed {before - len(df_clean)} row(s) with missing/empty text.")

    # The PRIMARY target must be present - a row with no effectiveness label
    # cannot be used for model comparison (there is no ground truth to
    # evaluate against), so (unlike Day 1's gentler handling) these rows are
    # removed here, specifically for building the supervised train/test split.
    before = len(df_clean)
    df_clean = df_clean[df_clean[eff_col].notnull()]
    print(f"Removed {before - len(df_clean)} row(s) with missing '{eff_col}' (cannot train/evaluate without it).")

    if type_col is not None:
        missing_type = df_clean[type_col].isnull().sum()
        if missing_type > 0:
            df_clean[type_col] = df_clean[type_col].fillna("UNLABELED")
            print(f"Found {missing_type} row(s) with missing '{type_col}'; kept and marked 'UNLABELED'.")

    df_clean["original_text"] = df_clean[text_col]
    df_clean["clean_text"] = df_clean[text_col].apply(clean_text)
    df_clean = df_clean[df_clean["clean_text"].str.len() > 0]
    df_clean = df_clean.reset_index(drop=True)
    print(f"\nFinal cleaned dataset: {df_clean.shape[0]} rows.")
    return df_clean


# =====================================================================================
# STEP 4: LABEL ENCODING
# =====================================================================================
def encode_labels(df: pd.DataFrame, eff_col: str) -> tuple[pd.DataFrame, list[str]]:
    classes = sorted(df[eff_col].unique().tolist())
    class_to_id = {c: i for i, c in enumerate(classes)}
    df = df.copy()
    df["effectiveness_label_id"] = df[eff_col].map(class_to_id)
    common.save_json(
        {"classes": classes, "class_to_id": class_to_id},
        config.LABEL_ENCODER_JSON,
    )
    print(f"\n[INFO] Encoded {len(classes)} classes for '{eff_col}': {classes}")
    return df, classes


# =====================================================================================
# STEP 5: STRATIFIED TRAIN/TEST SPLIT (the ONE split every model will reuse)
# =====================================================================================
def make_split(df: pd.DataFrame, eff_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    common.set_seed(config.RANDOM_SEED)
    df = common.filter_rare_classes(df, eff_col, min_count=2)
    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=df[eff_col],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    common.save_json(
        {
            "random_seed": config.RANDOM_SEED,
            "test_size": config.TEST_SIZE,
            "stratify_column": eff_col,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
        },
        config.SPLIT_INFO_JSON,
    )
    print(f"\n[INFO] Train rows: {len(train_df)} | Test rows: {len(test_df)} "
          f"(stratified on '{eff_col}', seed={config.RANDOM_SEED})")
    return train_df, test_df


# =====================================================================================
# MAIN
# =====================================================================================
def main() -> None:
    print("\n" + "#" * 70)
    print("# DATA PREPROCESSING")
    print("#" * 70)

    config.ensure_dirs()
    csv_path = find_dataset_csv()
    df = load_raw_dataset(csv_path)

    text_col = _resolve_column(df, config.TEXT_COLUMN_ALTERNATIVES, "text", required=True)
    type_col = _resolve_column(df, config.TYPE_COLUMN_ALTERNATIVES, "discourse type", required=False)
    eff_col = _resolve_column(df, config.EFFECTIVENESS_COLUMN_ALTERNATIVES, "effectiveness", required=True)

    df_clean = check_and_clean(df, text_col, type_col, eff_col)
    df_encoded, classes = encode_labels(df_clean, eff_col)

    # Save the full cleaned+encoded dataset for reference/debugging.
    df_encoded.to_csv(config.FULL_PROCESSED_CSV, index=False)

    train_df, test_df = make_split(df_encoded, eff_col)
    class_counts = train_df[eff_col].value_counts()
    target_count = class_counts.max()
    balanced_parts = []

    for class_name, class_df in train_df.groupby(eff_col):
        balanced_class = class_df.sample(
            n=target_count,
            replace=len(class_df) < target_count,
            random_state=config.RANDOM_SEED,
        )
        balanced_parts.append(balanced_class)

    train_df = pd.concat(balanced_parts, ignore_index=True)
    train_df = train_df.sample(
        frac=1,
        random_state=config.RANDOM_SEED,
    ).reset_index(drop=True)

    print("\nBalanced training counts:")
    print(train_df[eff_col].value_counts())
    
    train_df.to_csv(config.TRAIN_SPLIT_CSV, index=False)
    test_df.to_csv(config.TEST_SPLIT_CSV, index=False)

    print(f"\n[SUCCESS] Saved:")
    print(f"  {config.FULL_PROCESSED_CSV}")
    print(f"  {config.TRAIN_SPLIT_CSV}")
    print(f"  {config.TEST_SPLIT_CSV}")
    print(f"  {config.LABEL_ENCODER_JSON}")
    print(f"  {config.SPLIT_INFO_JSON}")
    print("\n" + "#" * 70)
    print("# DATA PREPROCESSING COMPLETE")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
