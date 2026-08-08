from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score


def extract_entities_bio(tags):
    entities = []
    start = None
    current_label = None

    for i, tag in enumerate(tags):
        if tag == "O":
            if current_label is not None:
                entities.append((start, i, current_label))
                start = None
                current_label = None
            continue

        prefix, label = tag.split("-", maxsplit=1)
        if prefix == "B" or current_label is None or current_label != label:
            if current_label is not None:
                entities.append((start, i, current_label))
            start = i
            current_label = label

    if current_label is not None:
        entities.append((start, len(tags), current_label))
    return entities


def count_invalid_bio_transitions(tags):
    invalid = 0
    total_non_o = 0
    previous_tag = "O"

    for tag in tags:
        if tag == "O":
            previous_tag = tag
            continue
        total_non_o += 1
        prefix, label = tag.split("-", maxsplit=1)
        if prefix == "I":
            if previous_tag == "O":
                invalid += 1
            else:
                _, previous_label = previous_tag.split("-", maxsplit=1)
                invalid += int(previous_label != label)
        previous_tag = tag

    return invalid, total_non_o


def compute_entity_metrics(all_gold_tags, all_pred_tags, entity_labels):
    tp = Counter()
    fp = Counter()
    fn = Counter()
    invalid = 0
    non_o = 0

    for gold_tags, pred_tags in zip(all_gold_tags, all_pred_tags):
        gold_entities = set(extract_entities_bio(gold_tags))
        pred_entities = set(extract_entities_bio(pred_tags))
        bad, total = count_invalid_bio_transitions(pred_tags)
        invalid += bad
        non_o += total

        for label in entity_labels:
            gold_label_entities = {entity for entity in gold_entities if entity[2] == label}
            pred_label_entities = {entity for entity in pred_entities if entity[2] == label}
            tp[label] += len(gold_label_entities & pred_label_entities)
            fp[label] += len(pred_label_entities - gold_label_entities)
            fn[label] += len(gold_label_entities - pred_label_entities)

    per_class = {}
    f1_values = []
    for label in entity_labels:
        precision = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp[label], "fp": fp[label], "fn": fn[label]}
        f1_values.append(f1)

    micro_tp = sum(tp.values())
    micro_fp = sum(fp.values())
    micro_fn = sum(fn.values())
    micro_precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall else 0.0

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "invalid_bio_rate": invalid / non_o if non_o else 0.0,
        "per_class": per_class,
    }


def compute_token_accuracy(all_gold_tags, all_pred_tags):
    gold = [tag for sequence in all_gold_tags for tag in sequence]
    pred = [tag for sequence in all_pred_tags for tag in sequence]
    return accuracy_score(gold, pred) if gold else 0.0

