from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from toxhabits_ner_bilstm import (
    ENTITY_LABELS,
    ToxHabitsDataset,
    ToxHabitsNERWithCRF,
    build_bio_chunks,
    build_vocabularies,
    create_collate_fn,
    create_spanish_nlp,
    fit,
    list_filenames,
    multilabel_train_val_test_split,
)


def main():
    project_dir = Path(__file__).resolve().parents[1]
    annotation_dir = project_dir / "data" / "ToxNER" / "ToxHabits(ToxNER)_Train_ANNFiles" / "train_annotations"
    model_path = project_dir / "models" / "tox_habits_ner_crf_model_base.pt"

    nlp = create_spanish_nlp()
    filenames = list_filenames(annotation_dir)
    train_files, val_files, _ = multilabel_train_val_test_split(annotation_dir, filenames)

    train_chunks = [build_bio_chunks(annotation_dir, filename, nlp, max_tokens=256) for filename in train_files]
    val_chunks = [build_bio_chunks(annotation_dir, filename, nlp, max_tokens=256) for filename in val_files]
    word2idx, char2idx, tag2idx = build_vocabularies(train_chunks)
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}

    collate_fn = create_collate_fn(word2idx["<pad>"], char2idx["<pad>"], tag2idx["O"])
    train_loader = DataLoader(ToxHabitsDataset(train_chunks, word2idx, char2idx, tag2idx), batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(ToxHabitsDataset(val_chunks, word2idx, char2idx, tag2idx), batch_size=16, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ToxHabitsNERWithCRF(
        vocab_size=len(word2idx),
        word_pad_idx=word2idx["<pad>"],
        char_vocab_size=len(char2idx),
        char_pad_idx=char2idx["<pad>"],
        num_tags=len(tag2idx),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    history, best_f1 = fit(model, train_loader, val_loader, optimizer, device, idx2tag, ENTITY_LABELS, epochs=30)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "word2idx": word2idx,
            "char2idx": char2idx,
            "tag2idx": tag2idx,
            "history": history,
            "best_validation_micro_f1": best_f1,
        },
        model_path,
    )


if __name__ == "__main__":
    main()

