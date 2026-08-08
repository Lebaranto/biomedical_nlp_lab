from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .preprocessing import add_cite_token, preprocess_text


LABEL_NAMES = ["BACKGROUND", "METHODS", "RESULTS"]


def load_splits(data_dir):
    data_dir = Path(data_dir)
    return {
        "train": pd.read_csv(data_dir / "train.csv"),
        "validation": pd.read_csv(data_dir / "validation.csv"),
        "test": pd.read_csv(data_dir / "test.csv"),
    }


def build_vocab(frames, min_frequency=2):
    counts = Counter(
        token
        for frame in frames
        for column in ["tokenized_string", "tokenized_section"]
        for sequence in frame[column]
        for token in sequence
    )
    vocabulary = {token for token, count in counts.items() if count >= min_frequency}
    vocabulary.update({"<pad>", "<unk>", "<cite>", "<num>", "<unknown_section>"})
    return {token: idx for idx, token in enumerate(sorted(vocabulary))}


def encode_labels(frames):
    train_labels = frames["train"]["label"]
    if pd.api.types.is_integer_dtype(train_labels):
        label2idx = {int(label): int(label) for label in sorted(train_labels.unique())}
        idx2label = {idx: LABEL_NAMES[idx] if idx < len(LABEL_NAMES) else str(idx) for idx in label2idx.values()}
    else:
        labels = sorted(train_labels.unique().tolist())
        label2idx = {label: idx for idx, label in enumerate(labels)}
        idx2label = {idx: label for label, idx in label2idx.items()}

    encoded = {}
    for name, frame in frames.items():
        frame = frame.copy()
        frame["label_id"] = frame["label"].map(label2idx).astype(int)
        encoded[name] = frame
    return encoded, label2idx, idx2label


def replace_oov(tokens, word2idx):
    return [token if token in word2idx else "<unk>" for token in tokens]


def prepare_splits(splits, min_frequency=2):
    processed = {}
    for name, frame in splits.items():
        frame = frame.copy()
        frame["sectionName"] = frame["sectionName"].fillna("<unknown_section>")
        frame = add_cite_token(frame)
        frame = preprocess_text(frame, source_column="string", save_column="tokenized_string")
        frame = preprocess_text(frame, source_column="sectionName", save_column="tokenized_section")
        processed[name] = frame

    word2idx = build_vocab([processed["train"]], min_frequency=min_frequency)
    for name, frame in processed.items():
        frame = frame.copy()
        frame["tokenized_string"] = frame["tokenized_string"].apply(lambda tokens: replace_oov(tokens, word2idx))
        frame["tokenized_section"] = frame["tokenized_section"].apply(lambda tokens: replace_oov(tokens, word2idx))
        processed[name] = frame

    encoded, label2idx, idx2label = encode_labels(processed)
    return encoded, word2idx, label2idx, idx2label


def tokens_to_idx(tokens, word2idx):
    unk_idx = word2idx["<unk>"]
    return [word2idx.get(token, unk_idx) for token in tokens]


class CitationDataset(Dataset):
    def __init__(self, data, word2idx, x_var="tokenized_string"):
        self.data = data.reset_index(drop=True)
        self.word2idx = word2idx
        self.x_var = x_var

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return (
            torch.tensor(tokens_to_idx(row[self.x_var], self.word2idx), dtype=torch.long),
            torch.tensor(row["label_id"], dtype=torch.long),
        )


class CitationDatasetAdvanced(Dataset):
    def __init__(self, data, word2idx):
        self.data = data.reset_index(drop=True)
        self.word2idx = word2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens = row["tokenized_string"]
        cite_idx = tokens.index("<cite>") if "<cite>" in tokens else max(0, len(tokens) // 2)
        source_length = max(len(str(row["string"])), 1)
        return (
            torch.tensor(tokens_to_idx(tokens, self.word2idx), dtype=torch.long),
            torch.tensor(tokens_to_idx(row["tokenized_section"], self.word2idx), dtype=torch.long),
            torch.tensor(tokens_to_idx(tokens[:cite_idx], self.word2idx), dtype=torch.long),
            torch.tensor(tokens_to_idx(tokens[cite_idx + 1 :], self.word2idx), dtype=torch.long),
            torch.tensor(row["citeStart"] / source_length, dtype=torch.float32),
            torch.tensor(row["label_id"], dtype=torch.long),
        )


def make_collate_fn(pad_idx, max_length):
    def collate_fn(batch):
        sentences, labels = zip(*batch)
        sentences = [sentence[:max_length] for sentence in sentences]
        sentences = pad_sequence(sentences, batch_first=True, padding_value=pad_idx)
        labels = torch.stack(labels)
        return sentences, labels, sentences != pad_idx

    return collate_fn


def create_dataloader(data, word2idx, batch_size=32, max_length=54, shuffle=False):
    return DataLoader(
        CitationDataset(data, word2idx),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate_fn(word2idx["<pad>"], max_length),
    )


def make_advanced_collate_fn(pad_idx, max_length, context_window):
    def collate_fn(batch):
        sentences, sections, left_contexts, right_contexts, positions, labels = zip(*batch)
        empty = lambda: torch.tensor([pad_idx], dtype=torch.long)
        sentences = [sentence[:max_length] for sentence in sentences]
        sections = [section[:max_length] if len(section) else empty() for section in sections]
        left_contexts = [context[-context_window:] if len(context) else empty() for context in left_contexts]
        right_contexts = [context[:context_window] if len(context) else empty() for context in right_contexts]

        sentences = pad_sequence(sentences, batch_first=True, padding_value=pad_idx)
        sections = pad_sequence(sections, batch_first=True, padding_value=pad_idx)
        left_contexts = pad_sequence(left_contexts, batch_first=True, padding_value=pad_idx)
        right_contexts = pad_sequence(right_contexts, batch_first=True, padding_value=pad_idx)
        positions = torch.stack(positions).unsqueeze(-1)
        labels = torch.stack(labels)

        return {
            "sentences": sentences,
            "sections": sections,
            "left_contexts": left_contexts,
            "right_contexts": right_contexts,
            "positions": positions,
            "labels": labels,
            "sentence_mask": sentences != pad_idx,
            "section_mask": sections != pad_idx,
            "left_context_mask": left_contexts != pad_idx,
            "right_context_mask": right_contexts != pad_idx,
        }

    return collate_fn


def create_dataloader_advanced(data, word2idx, batch_size=32, max_length=54, context_window=15, shuffle=False):
    return DataLoader(
        CitationDatasetAdvanced(data, word2idx),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_advanced_collate_fn(word2idx["<pad>"], max_length, context_window),
    )

