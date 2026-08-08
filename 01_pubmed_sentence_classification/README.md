# PubMed Abstract Sentence Classification

Feed-forward PyTorch baseline for classifying sentences in PubMed RCT abstracts.

## Data

Source: Kaggle PubMed 20k RCT dataset.

Expected local layout:

```text
data/raw/PubMed_20k_RCT/train.csv
data/raw/PubMed_20k_RCT/dev.csv
data/raw/PubMed_20k_RCT/test.csv
```

## Run

From the repository root:

```powershell
python -m pip install -e .
python 01_pubmed_sentence_classification/scripts/train.py
```

The checkpoint is written to `01_pubmed_sentence_classification/models/`.

