# Argument Mining and Socratic Feedback

A Streamlit application that classifies short argumentative text segments by
their effectiveness and discourse type, then offers reflective questions to
help the writer improve the argument.

## What it does

- Predicts **discourse effectiveness** as `Ineffective`, `Adequate`, or
  `Effective`.
- Predicts **discourse type** (for example, Claim or Evidence) when the
  auxiliary classifier is available.
- Shows the selected model and its confidence scores.
- Provides Socratic feedback questions rather than rewriting the argument.
- Compares six effectiveness models using Accuracy, macro Precision, macro
  Recall, Macro F1, Weighted F1, and timing metrics. The best available model
  is selected by **Macro F1**.

## Models compared

1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVM
3. TF-IDF + Random Forest
4. Word Embedding + LSTM
5. Word Embedding + BiLSTM
6. Sentence Transformer + MLP

A separate TF-IDF + Logistic Regression model predicts discourse type. It is
not included in the effectiveness-model ranking.

## Project layout

```text
app/app.py                    Streamlit user interface
src/data_preprocessing.py     Dataset cleaning, stratified split, oversampling
src/train_model_1.py ...      Model training scripts
src/train_auxiliary_discourse_type.py
src/compare_models.py         Build the comparison table
src/select_best_model.py      Select the best model by Macro F1
src/prediction.py             Load saved artifacts and predict
src/socratic_feedback.py      Reflective feedback questions
run_all.py                    Run preprocessing, training, and comparison
data/raw/                     Put the input dataset here (not committed)
data/processed/               Generated splits (not committed)
models/saved/                 Saved model artifacts
models/results/               Per-model results and best_model.json
results/                      Comparison table, reports, and confusion matrices
```

## Requirements

- Python 3.10 or newer
- Windows, macOS, or Linux
- Internet access the first time Sentence Transformers downloads its pretrained
  model
- A GPU is optional; PyTorch can train the LSTM models on CPU

## Run locally

From the project root, create and activate a virtual environment, install the
dependencies, then start the app:

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_all.py
python -m streamlit run app/app.py
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py
python -m streamlit run app/app.py
```

The training pipeline can take a long time, especially on CPU. After it
finishes, Streamlit prints a local URL (usually `http://localhost:8501`). To
launch the app later without retraining, run only:

```bash
python -m streamlit run app/app.py
```

That requires the saved model and `models/results/best_model.json` to be
present.

## Dataset

The project expects a labeled CSV in `data/raw/`. The default column names are:

| Column | Purpose |
|---|---|
| `discourse_text` | Text supplied to the classifiers |
| `discourse_type` | Secondary target for the auxiliary classifier |
| `discourse_effectiveness` | Main three-class target |

The pipeline cleans the text, makes a stratified 80/20 train/test split, then
randomly oversamples classes in the **training split only**. The test split
remains at its original class distribution for evaluation. Column aliases and
other settings are in `src/config.py`.

Obtain the dataset from its authorized source and check its license before
using or sharing it. Dataset CSVs and generated split CSVs are excluded from
Git by `.gitignore`.

## Results and artifacts

- `results/model_comparison.csv`: model comparison
- `results/classification_reports/`: per-model classification reports
- `results/confusion_matrices/`: confusion matrices as CSV and PNG
- `models/results/best_model.json`: model chosen for the app
- `models/saved/`: trained model files

The Random Forest model artifact is larger than GitHub's per-file upload
limit, so it is excluded from this repository. Re-run training after adding the
dataset to regenerate excluded artifacts. Do not commit credentials, private
data, or `.env` files.

## Notes

- Results depend on the dataset, split, and available dependencies.
- Oversampling the training data does not guarantee higher accuracy; compare
  macro F1 and per-class results on the unchanged test set.
- This project is an educational prototype. Review model predictions before
  using them to make consequential decisions about writers.
