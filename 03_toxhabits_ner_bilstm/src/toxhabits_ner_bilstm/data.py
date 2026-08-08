from pathlib import Path

import pandas as pd
import spacy
from spacy.language import Language

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
except ImportError:
    MultilabelStratifiedShuffleSplit = None


ENTITY_LABELS = ["Drug", "Alcohol", "Tobacco", "Cannabis"]
ABBREVIATIONS = {"comp", "comps", "dr", "dra", "mg", "ml"}


if not Language.has_factory("fix_medical_sentence_boundaries"):
    @Language.component("fix_medical_sentence_boundaries")
    def fix_medical_sentence_boundaries(doc):
        for i, token in enumerate(doc[:-1]):
            if token.text.endswith(".") and token.text[:-1].lower() in ABBREVIATIONS:
                doc[i + 1].is_sent_start = False
            if token.text == "." and i > 0 and doc[i - 1].lower_ in ABBREVIATIONS:
                doc[i + 1].is_sent_start = False
        return doc


def create_spanish_nlp(model_name="es_core_news_sm"):
    nlp = spacy.load(model_name)
    if "fix_medical_sentence_boundaries" not in nlp.pipe_names:
        nlp.add_pipe("fix_medical_sentence_boundaries", before="parser")
    return nlp


def list_filenames(annotation_dir):
    annotation_dir = Path(annotation_dir)
    return sorted(path.stem for path in annotation_dir.glob("*.txt"))


def extract_sample(annotation_dir, filename):
    annotation_dir = Path(annotation_dir)
    text = (annotation_dir / f"{filename}.txt").read_text(encoding="utf-8")
    annotations = []
    ann_path = annotation_dir / f"{filename}.ann"
    if not ann_path.exists():
        return text, annotations

    for line in ann_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        annotation_id, span_info, annotation_text = line.split("\t", maxsplit=2)
        label, start, end = span_info.split()
        annotations.append(
            {
                "id": annotation_id,
                "label": label,
                "start": int(start),
                "end": int(end),
                "text": annotation_text,
            }
        )
    return text, annotations


def tokenize_with_offsets(nlp, text):
    doc = nlp(text)
    return [(token.text, token.idx, token.idx + len(token.text)) for token in doc]


def build_bio_sample(annotation_dir, filename, nlp):
    text, annotations = extract_sample(annotation_dir, filename)
    tokenized_text = tokenize_with_offsets(nlp, text)
    bio_tags = ["O"] * len(tokenized_text)

    for annotation in annotations:
        label = annotation["label"]
        start = annotation["start"]
        end = annotation["end"]
        first_token = True

        for i, (_, token_start, token_end) in enumerate(tokenized_text):
            if token_start >= start and token_end <= end:
                bio_tags[i] = f"B-{label}" if first_token else f"I-{label}"
                first_token = False

    tokens = [token[0] for token in tokenized_text]
    return filename, tokens, bio_tags


def build_bio_chunks(annotation_dir, filename, nlp, max_tokens=256):
    _, tokens, bio_tags = build_bio_sample(annotation_dir, filename, nlp)
    text, _ = extract_sample(annotation_dir, filename)
    doc = nlp(text)
    tokenized_text = tokenize_with_offsets(nlp, text)

    if len(tokens) != len(tokenized_text) or len(tokens) != len(bio_tags):
        raise ValueError(f"Token/BIO length mismatch in {filename}")

    sentence_data = []
    token_ptr = 0

    for sent in doc.sents:
        sent_tokens = []
        sent_offsets = []
        sent_tags = []

        while token_ptr < len(tokenized_text):
            token_text, token_start, token_end = tokenized_text[token_ptr]
            if token_start >= sent.end_char:
                break
            if token_start >= sent.start_char and token_end <= sent.end_char:
                sent_tokens.append(token_text)
                sent_offsets.append((token_start, token_end))
                sent_tags.append(bio_tags[token_ptr])
            token_ptr += 1

        if sent_tokens:
            sentence_data.append(
                {
                    "start": sent.start_char,
                    "end": sent.end_char,
                    "tokens": sent_tokens,
                    "offsets": sent_offsets,
                    "bio_tags": sent_tags,
                }
            )

    chunks = []
    current_sentences = []
    current_len = 0

    for sent in sentence_data:
        sent_len = len(sent["tokens"])
        if current_sentences and current_len + sent_len > max_tokens:
            chunks.append(_merge_sentences(filename, text, current_sentences, len(chunks)))
            current_sentences = []
            current_len = 0

        current_sentences.append(sent)
        current_len += sent_len

    if current_sentences:
        chunks.append(_merge_sentences(filename, text, current_sentences, len(chunks)))

    return chunks


def _merge_sentences(filename, text, sentences, chunk_id):
    chunk_start = sentences[0]["start"]
    chunk_end = sentences[-1]["end"]
    chunk_tokens = []
    chunk_offsets = []
    chunk_tags = []

    for sent in sentences:
        chunk_tokens.extend(sent["tokens"])
        chunk_offsets.extend(sent["offsets"])
        chunk_tags.extend(sent["bio_tags"])

    return {
        "filename": filename,
        "chunk_id": chunk_id,
        "text": text[chunk_start:chunk_end],
        "start": chunk_start,
        "end": chunk_end,
        "tokens": chunk_tokens,
        "offsets": chunk_offsets,
        "bio_tags": chunk_tags,
    }


def build_doc_label_matrix(annotation_dir, filenames):
    rows = []
    for filename in filenames:
        _, annotations = extract_sample(annotation_dir, filename)
        present_labels = {label: 0 for label in ENTITY_LABELS}
        for ann in annotations:
            if ann["label"] in present_labels:
                present_labels[ann["label"]] = 1
        rows.append({"filename": filename, **present_labels})
    return pd.DataFrame(rows)


def multilabel_train_val_test_split(annotation_dir, filenames, test_size=0.1, val_size=0.1, random_state=42):
    if MultilabelStratifiedShuffleSplit is None:
        raise ImportError(
            "Install iterative-stratification to use multilabel_train_val_test_split: "
            "python -m pip install iterative-stratification"
        )

    df = build_doc_label_matrix(annotation_dir, filenames)
    x = df["filename"].values
    y = df[ENTITY_LABELS].values

    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(splitter.split(x, y))

    x_train_val = x[train_val_idx]
    y_train_val = y[train_val_idx]
    val_relative_size = val_size / (1 - test_size)

    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_relative_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(x_train_val, y_train_val))

    return x_train_val[train_idx].tolist(), x_train_val[val_idx].tolist(), x[test_idx].tolist()


def build_vocabularies(chunks):
    tokens = {token for doc_chunks in chunks for chunk in doc_chunks for token in chunk["tokens"]}
    tags = {tag for doc_chunks in chunks for chunk in doc_chunks for tag in chunk["bio_tags"]}
    chars = {char for token in tokens for char in token}

    tokens.update({"<pad>", "<unk>"})
    chars.update({"<pad>", "<unk>"})
    tags.add("O")

    word2idx = {token: idx for idx, token in enumerate(sorted(tokens))}
    char2idx = {char: idx for idx, char in enumerate(sorted(chars))}
    tag2idx = {tag: idx for idx, tag in enumerate(sorted(tags))}
    return word2idx, char2idx, tag2idx
