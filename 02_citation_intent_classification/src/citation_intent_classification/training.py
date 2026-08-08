from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score


def evaluate_predictions(labels, predictions, loss, examples):
    return {
        "loss": loss / examples,
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "labels": labels,
        "predictions": predictions,
    }


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, labels, mask in loader:
        inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs, mask), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    examples = 0
    labels_all = []
    predictions_all = []
    with torch.no_grad():
        for inputs, labels, mask in loader:
            inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
            logits = model(inputs, mask)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            examples += labels.size(0)
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())
    return evaluate_predictions(labels_all, predictions_all, total_loss, examples)


def train(model, train_loader, validation_loader, criterion, optimizer, device, epochs, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, validation_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_macro_f1={metrics['macro_f1']:.4f}")
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            torch.save(model.state_dict(), checkpoint_path)
    return history


def move_advanced_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def advanced_logits(model, batch):
    return model(
        batch["sentences"],
        batch["sections"],
        batch["left_contexts"],
        batch["right_contexts"],
        batch["positions"],
        batch["sentence_mask"],
        batch["section_mask"],
        batch["left_context_mask"],
        batch["right_context_mask"],
    )


def train_epoch_advanced(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = move_advanced_batch(batch, device)
        optimizer.zero_grad()
        loss = criterion(advanced_logits(model, batch), batch["labels"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_advanced(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    examples = 0
    labels_all = []
    predictions_all = []
    with torch.no_grad():
        for batch in loader:
            batch = move_advanced_batch(batch, device)
            logits = advanced_logits(model, batch)
            loss = criterion(logits, batch["labels"])
            predictions = logits.argmax(dim=1)
            total_loss += loss.item() * batch["labels"].size(0)
            examples += batch["labels"].size(0)
            labels_all.extend(batch["labels"].cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())
    return evaluate_predictions(labels_all, predictions_all, total_loss, examples)


def train_advanced(model, train_loader, validation_loader, criterion, optimizer, device, epochs, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch_advanced(model, train_loader, criterion, optimizer, device)
        metrics = evaluate_advanced(model, validation_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_macro_f1={metrics['macro_f1']:.4f}")
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            torch.save(model.state_dict(), checkpoint_path)
    return history

