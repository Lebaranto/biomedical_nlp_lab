from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score


def move_batch(batch, device):
    return tuple(item.to(device) for item in batch)


def train_epoch(model, loader, criterion, optimizer, device, gradient_clip=None):
    model.train()
    total_loss = 0.0

    for batch in loader:
        inputs, labels, mask = move_batch(batch, device)
        optimizer.zero_grad()
        logits = model(inputs, mask)
        loss = criterion(logits, labels)
        loss.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    labels_all = []
    predictions_all = []

    with torch.no_grad():
        for batch in loader:
            inputs, labels, mask = move_batch(batch, device)
            logits = model(inputs, mask)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())

    return {
        "loss": total_loss / total_examples,
        "accuracy": accuracy_score(labels_all, predictions_all),
        "macro_f1": f1_score(labels_all, predictions_all, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels_all, predictions_all, average="weighted", zero_division=0),
        "labels": labels_all,
        "predictions": predictions_all,
    }


def train(model, train_loader, validation_loader, criterion, optimizer, device, epochs, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        validation_metrics = evaluate(model, validation_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **validation_metrics})

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_macro_f1={validation_metrics['macro_f1']:.4f}"
        )

        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            torch.save(model.state_dict(), checkpoint_path)

    return history

