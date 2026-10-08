# The Effectiveness Models, Explained Simply

This project trains multiple approaches on the **same** train/test split of
the dataset, to predict `discourse_effectiveness` (Ineffective / Adequate /
Effective), and picks whichever one actually performs best - rather than
assuming in advance which approach "should" win.

---

### Model 1: TF-IDF + Logistic Regression
**How it works:** Turns each text segment into a vector of weighted word/
phrase frequencies (TF-IDF), then draws a straight-line decision boundary
between the three effectiveness classes.
**Why include it:** The standard, fast, highly interpretable baseline that
any new approach should be compared against.

### Model 2: TF-IDF + Linear SVM
**How it works:** Same TF-IDF features as Model 1, but uses a Support Vector
Machine, which tries to find the boundary that separates the classes with the
widest possible margin.
**Why include it:** SVMs are well known to perform very strongly on exactly
this kind of high-dimensional, sparse text data.

### Model 3: TF-IDF + Random Forest
**How it works:** Same TF-IDF features again, but uses many decision trees
voting together, which can capture non-linear combinations of words that a
straight-line model (Models 1/2) cannot.
**Why include it:** Tests whether non-linear modelling helps on this dataset,
as a contrast to the two linear models above.

### Model 4: Word Embedding + LSTM
**How it works:** Each word is converted to a numeric vector ("embedding")
that the model learns from scratch during training (not a pretrained
embedding), then an LSTM (a type of neural network built for sequences) reads
the text word by word, remembering context as it goes.
**Why include it:** Unlike TF-IDF, which ignores word order completely, an
LSTM can in principle learn that word ORDER matters (e.g. "not effective" vs
"effective not").

### Model 5: Word Embedding + BiLSTM
**How it works:** Identical to Model 4, except the LSTM reads the text in
BOTH directions (start-to-end and end-to-start) and combines both readings.
**Why include it:** Directly comparable to Model 4 - isolates whether reading
in both directions actually helps on this specific dataset, or whether it's
unnecessary extra complexity.

### Model 6: Sentence Transformer + MLP
**How it works:** Uses a small, pretrained "sentence-transformer" model
(`all-MiniLM-L6-v2`, ~80MB) that already understands general English meaning
from being trained on huge amounts of text elsewhere, to convert each segment
into a single meaning-vector. A simple MLP (small neural network) classifier
is then trained on top of these ready-made vectors.
**Why include it:** Tests whether general-purpose pretrained language
understanding beats representations learned only from this one dataset
(Models 4/5) or simple word-frequency counts (Models 1-3).

---

## How the comparison is kept fair

- All effectiveness models are evaluated on the **same untouched test set**
  produced by `data_preprocessing.py`.
- TF-IDF vectorizers (Models 1-3) and the LSTM/BiLSTM vocabulary (Models 4-5)
  are built **only from the training data** - never from the test data - to
  avoid data leakage.
- Model 6's pretrained embeddings are frozen.
- The same random seed is used everywhere for reproducibility.

## Why Macro F1 is the deciding metric

This is a **multiclass** problem (3 effectiveness classes), and real-world
argument datasets are often **imbalanced** (e.g. far more "Adequate" examples
than "Ineffective" ones). Plain accuracy can look good on an imbalanced
dataset just by favouring the majority class. **Macro F1** averages the F1
score of EACH class equally, so a model that ignores a minority class is
correctly penalised. This is why Macro F1 - not accuracy - is used to select
the final best model.

## No model is assumed to win

The comparison table and the best-model selection are always computed from
the actual saved results of whichever models you trained. Nothing in this
project hard-codes "BiLSTM is best" or "Sentence Transformers are best" - if
a simple TF-IDF model happens to score the highest Macro F1 on your dataset,
that is exactly what gets selected and used in the final app.
