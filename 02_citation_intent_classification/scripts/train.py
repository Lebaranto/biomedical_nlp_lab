from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from citation_intent_classification import (
    ImprovedCitationClassifier,
    create_dataloader_advanced,
    load_splits,
    prepare_splits,
    train_advanced,
)


def main():
    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data" / "raw"
    checkpoint_path = project_dir / "models" / "citation_intent_advanced.pt"

    splits = load_splits(data_dir)
    datasets, word2idx, label2idx, _ = prepare_splits(splits)
    train_loader = create_dataloader_advanced(datasets["train"], word2idx, shuffle=True)
    validation_loader = create_dataloader_advanced(datasets["validation"], word2idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedCitationClassifier(
        vocab_size=len(word2idx),
        num_classes=len(label2idx),
        pad_idx=word2idx["<pad>"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    train_advanced(model, train_loader, validation_loader, criterion, optimizer, device, epochs=15, checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    main()

