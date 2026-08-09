# ToxHabits NER with CharCNN + BiLSTM + CRF

Spanish named entity recognition for toxic habit mentions: `Drug`, `Alcohol`, `Tobacco`, and `Cannabis`.

Developed pipeline covers:

- BRAT `.ann/.txt` loading.
- spaCy tokenization with medical abbreviation sentence-boundary fixes.
- BIO tagging and sentence-preserving chunking.
- Multi-label document-level train/validation/test split.
- CharCNN + BiLSTM encoder with a linear-chain CRF.
- Exact entity-level micro-F1, class F1, BIO validity, and error analysis utilities with graphical interpretation.

## Data

This dataset was downloaded from https://zenodo.org/records/17566029. Expected local layout:

```text
data/ToxNER/ToxHabits(ToxNER)_Train_ANNFiles/train_annotations
data/ToxNER/ToxHabits(ToxNER)_Test_ANNFiles/test_good
```

## Run

From the repository root:

```powershell
python -m pip install -e .
python -m spacy download es_core_news_sm
python 03_toxhabits_ner_bilstm/scripts/train.py
```

