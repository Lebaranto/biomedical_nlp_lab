from collections import Counter
from pathlib import Path
import re

import pandas as pd
import torch
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


LABELS = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
NUM_PATTERN = re.compile(r"^[+-]?\d+(?:[\.,]\d+)?(?:[eE][+-]?\d+)?$")


def load_splits(data_dir):
    data_dir = Path(data_dir)
    return {
        "train": pd.read_csv(data_dir / "train.csv"),
        "validation": pd.read_csv(data_dir / "dev.csv"),
        "test": pd.read_csv(data_dir / "test.csv"),
    }


def tokenize_text(text):
    tokens = word_tokenize(str(text).lower())
    return ["<num>" if NUM_PATTERN.fullmatch(token) else token for token in tokens]


def preprocess_dataframe(data, text_column="abstract_text"):
    data = data.copy()
    data["tokens"] = data[text_column].apply(tokenize_text)
    return data


def build_vocab(token_sequences, min_frequency=2):
    counts = Counter(token for sequence in token_sequences for token in sequence)
    vocabulary = {
        token for token, count in counts.items()
        if count >= min_frequency
    }
    vocabulary.update({"<pad>", "<unk>", "<num>"})
    return {token: idx for idx, token in enumerate(sorted(vocabulary))}


def replace_oov(tokens, word2idx):
    return [token if token in word2idx else "<unk>" for token in tokens]


def encode_labels(data, label2idx, label_column="target"):
    data = data.copy()
    data["label_id"] = data[label_column].map(label2idx)
    return data[["tokens", "label_id"]]


def prepare_splits(splits, min_frequency=2):
    processed = {
        name: preprocess_dataframe(frame)
        for name, frame in splits.items()
    }
    word2idx = build_vocab(processed["train"]["tokens"], min_frequency=min_frequency)

    for name in processed:
        processed[name]["tokens"] = processed[name]["tokens"].apply(
            lambda tokens: replace_oov(tokens, word2idx)
        )

    labels = [label for label in LABELS if label in set(processed["train"]["target"])]
    if not labels:
        labels = sorted(processed["train"]["target"].unique().tolist())

    label2idx = {label: idx for idx, label in enumerate(labels)}
    encoded = {
        name: encode_labels(frame, label2idx)
        for name, frame in processed.items()
    }
    return encoded, word2idx, label2idx


class PubMedDataset(Dataset):
    def __init__(self, data, word2idx):
        self.data = data.reset_index(drop=True)
        self.word2idx = word2idx
        self.unk_idx = word2idx["<unk>"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        word_ids = [self.word2idx.get(token, self.unk_idx) for token in row["tokens"]]
        return (
            torch.tensor(word_ids, dtype=torch.long),
            torch.tensor(row["label_id"], dtype=torch.long),
        )


def make_collate_fn(pad_idx, max_length):
    def collate_fn(batch):
        inputs, labels = zip(*batch)
        inputs = [item[:max_length] for item in inputs]
        inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
        labels = torch.stack(labels)
        mask = inputs != pad_idx
        return inputs, labels, mask

    return collate_fn


def create_dataloader(data, word2idx, batch_size=64, max_length=55, shuffle=False):
    return DataLoader(
        PubMedDataset(data, word2idx),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate_fn(word2idx["<pad>"], max_length),
    )

