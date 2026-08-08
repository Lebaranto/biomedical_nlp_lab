from collections import Counter

import pandas as pd


def extract_entities_with_text(tokens, offsets, tags):
    entities = []
    start = None
    label = None

    for i, tag in enumerate(tags + ["O"]):
        if tag == "O":
            if label is not None:
                entity_offsets = offsets[start:i]
                entity_tokens = tokens[start:i]
                entities.append(
                    {
                        "start_token": start,
                        "end_token": i,
                        "label": label,
                        "tokens": entity_tokens,
                        "text": " ".join(entity_tokens),
                        "start_char": entity_offsets[0][0],
                        "end_char": entity_offsets[-1][1],
                        "length_tokens": len(entity_tokens),
                    }
                )
                start = None
                label = None
            continue

        prefix, current_label = tag.split("-", maxsplit=1)
        if prefix == "B" or label is None or label != current_label:
            if label is not None:
                entity_offsets = offsets[start:i]
                entity_tokens = tokens[start:i]
                entities.append(
                    {
                        "start_token": start,
                        "end_token": i,
                        "label": label,
                        "tokens": entity_tokens,
                        "text": " ".join(entity_tokens),
                        "start_char": entity_offsets[0][0],
                        "end_char": entity_offsets[-1][1],
                        "length_tokens": len(entity_tokens),
                    }
                )
            start = i
            label = current_label

    return entities


def get_predictions_with_metadata(model, dataloader, device, idx2tag):
    import torch

    model.eval()
    rows = []
    with torch.no_grad():
        for batch in dataloader:
            device_batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            best_paths, _ = model(
                word_ids=device_batch["word_ids"],
                char_ids=device_batch["char_ids"],
                character_lengths=device_batch["character_lengths"],
                lengths=device_batch["lengths"],
                token_mask=device_batch["token_mask"],
            )
            gold_ids = batch["tag_ids"]
            masks = batch["token_mask"]
            for i, path in enumerate(best_paths):
                seq_len = int(masks[i].sum().item())
                rows.append(
                    {
                        "document_id": batch["document_ids"][i],
                        "chunk_id": batch["chunk_ids"][i],
                        "tokens": batch["tokens"][i],
                        "offsets": batch["offsets"][i],
                        "gold_tags": [idx2tag[idx] for idx in gold_ids[i, :seq_len].tolist()],
                        "pred_tags": [idx2tag[idx] for idx in path[:seq_len]],
                    }
                )
    return rows


def build_seen_entity_sets(rows):
    seen_surface_forms = set()
    seen_label_surface_forms = set()
    for row in rows:
        for entity in extract_entities_with_text(row["tokens"], row["offsets"], row["gold_tags"]):
            text = entity["text"].lower()
            seen_surface_forms.add(text)
            seen_label_surface_forms.add((entity["label"], text))
    return seen_surface_forms, seen_label_surface_forms


def analyze_errors(rows, seen_surface_forms, entity_labels):
    false_positives = []
    false_negatives = []
    boundary_errors = []
    label_confusions = []
    confusion_counter = Counter()
    counters = {name: Counter() for name in ["tp", "fp", "fn", "one_tp", "one_fp", "one_fn", "multi_tp", "multi_fp", "multi_fn", "seen_tp", "seen_fp", "seen_fn", "oov_tp", "oov_fp", "oov_fn"]}

    for row in rows:
        gold_entities = extract_entities_with_text(row["tokens"], row["offsets"], row["gold_tags"])
        pred_entities = extract_entities_with_text(row["tokens"], row["offsets"], row["pred_tags"])
        gold_by_exact = {(e["start_token"], e["end_token"], e["label"]): e for e in gold_entities}
        pred_by_exact = {(e["start_token"], e["end_token"], e["label"]): e for e in pred_entities}
        matched = set(gold_by_exact) & set(pred_by_exact)

        for key in matched:
            entity = gold_by_exact[key]
            label = entity["label"]
            counters["tp"][label] += 1
            counters["one_tp" if entity["length_tokens"] == 1 else "multi_tp"][label] += 1
            counters["seen_tp" if entity["text"].lower() in seen_surface_forms else "oov_tp"][label] += 1

        unmatched_gold = [gold_by_exact[key] for key in set(gold_by_exact) - matched]
        unmatched_pred = [pred_by_exact[key] for key in set(pred_by_exact) - matched]
        used_gold = set()
        used_pred = set()

        for gi, gold in enumerate(unmatched_gold):
            for pi, pred in enumerate(unmatched_pred):
                if pi in used_pred:
                    continue
                same_span = gold["start_token"] == pred["start_token"] and gold["end_token"] == pred["end_token"]
                overlap = not (gold["end_token"] <= pred["start_token"] or pred["end_token"] <= gold["start_token"])
                if same_span and gold["label"] != pred["label"]:
                    label_confusions.append({**_base_error(row), "text": gold["text"], "gold_label": gold["label"], "pred_label": pred["label"]})
                    confusion_counter[(gold["label"], pred["label"])] += 1
                    used_gold.add(gi)
                    used_pred.add(pi)
                    break
                if overlap:
                    boundary_errors.append({**_base_error(row), "gold_text": gold["text"], "pred_text": pred["text"], "gold_label": gold["label"], "pred_label": pred["label"], "gold_span": (gold["start_token"], gold["end_token"]), "pred_span": (pred["start_token"], pred["end_token"])})
                    used_gold.add(gi)
                    used_pred.add(pi)
                    break

        for gi, gold in enumerate(unmatched_gold):
            if gi in used_gold:
                continue
            _add_miss(counters, "fn", gold, seen_surface_forms)
            false_negatives.append({**_base_error(row), "text": gold["text"], "label": gold["label"], "start_token": gold["start_token"], "end_token": gold["end_token"]})

        for pi, pred in enumerate(unmatched_pred):
            if pi in used_pred:
                continue
            _add_miss(counters, "fp", pred, seen_surface_forms)
            false_positives.append({**_base_error(row), "text": pred["text"], "label": pred["label"], "start_token": pred["start_token"], "end_token": pred["end_token"]})

    summary = [_summary_row(label, counters) for label in entity_labels]
    confusion_df = pd.DataFrame(0, index=entity_labels, columns=entity_labels, dtype=int)
    for (gold_label, pred_label), count in confusion_counter.items():
        confusion_df.loc[gold_label, pred_label] = count

    return {
        "summary_df": pd.DataFrame(summary),
        "false_positives_df": pd.DataFrame(false_positives),
        "false_negatives_df": pd.DataFrame(false_negatives),
        "boundary_errors_df": pd.DataFrame(boundary_errors),
        "label_confusions_df": pd.DataFrame(label_confusions),
        "confusion_df": confusion_df,
    }


def _base_error(row):
    return {"document_id": row["document_id"], "chunk_id": row["chunk_id"]}


def _add_miss(counters, miss_type, entity, seen_surface_forms):
    label = entity["label"]
    counters[miss_type][label] += 1
    counters[f"{'one' if entity['length_tokens'] == 1 else 'multi'}_{miss_type}"][label] += 1
    counters[f"{'seen' if entity['text'].lower() in seen_surface_forms else 'oov'}_{miss_type}"][label] += 1


def _score(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _summary_row(label, counters):
    precision, recall, f1 = _score(counters["tp"][label], counters["fp"][label], counters["fn"][label])
    _, _, one_word_f1 = _score(counters["one_tp"][label], counters["one_fp"][label], counters["one_fn"][label])
    _, _, multi_word_f1 = _score(counters["multi_tp"][label], counters["multi_fp"][label], counters["multi_fn"][label])
    _, _, seen_f1 = _score(counters["seen_tp"][label], counters["seen_fp"][label], counters["seen_fn"][label])
    _, _, oov_f1 = _score(counters["oov_tp"][label], counters["oov_fp"][label], counters["oov_fn"][label])
    return {
        "label": label,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "one_word_f1": one_word_f1,
        "multi_word_f1": multi_word_f1,
        "seen_f1": seen_f1,
        "oov_f1": oov_f1,
        "tp": counters["tp"][label],
        "fp": counters["fp"][label],
        "fn": counters["fn"][label],
    }

