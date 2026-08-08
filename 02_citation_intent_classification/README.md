# Citation Intent Classification

PyTorch baselines for SciCite citation intent classification. The refactored package includes:

- `CitationClassifier`: full citation context only.
- `ImprovedCitationClassifier`: full context, section title, local left/right context, and citation position.

## Data

Source: Kaggle SciCite citation intent dataset.

Expected local layout:

```text
data/raw/train.csv
data/raw/validation.csv
data/raw/test.csv
```

## Run

From the repository root:

```powershell
python -m pip install -e .
python 02_citation_intent_classification/scripts/train.py
```

