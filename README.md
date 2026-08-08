# Scientific NLP From Scratch

This repository contains three compact NLP projects built from notebooks and refactored into reusable Python modules:

- `01_pubmed_sentence_classification`: PubMed RCT abstract sentence classification from Kaggle.
- `02_citation_intent_classification`: SciCite citation intent classification from Kaggle.
- `03_toxhabits_ner_bilstm`: Spanish ToxHabits NER with CharCNN + BiLSTM + CRF from locally downloaded BRAT annotations.

The original notebooks are kept unchanged. Each project also includes a cleaner notebook copy for quick reading and a `src/` package for reusable code.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m spacy download es_core_news_sm
```

## Data

Datasets are intentionally ignored by git. Put the files under each project's `data/` directory as described in that project's `data/README.md`.

## Projects

Each project has:

- `README.md` with dataset notes and run commands.
- `src/<package_name>/` with data, model, training, and evaluation code.
- `scripts/train.py` as a simple command-line entry point.
- `notebooks/*_clean.ipynb` as a polished companion notebook.

