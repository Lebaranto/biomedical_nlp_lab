import torch

from .metrics import compute_entity_metrics, compute_token_accuracy


def move_batch_to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def train_one_epoch(model, dataloader, optimizer, device, gradient_clip=5.0, crf_reduction="token_mean"):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        loss = model(
            word_ids=batch["word_ids"],
            char_ids=batch["char_ids"],
            character_lengths=batch["character_lengths"],
            lengths=batch["lengths"],
            token_mask=batch["token_mask"],
            tags=batch["tag_ids"],
            reduction=crf_reduction,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, idx2tag, entity_labels, crf_reduction="token_mean"):
    model.eval()
    total_loss = 0.0
    all_gold_tags = []
    all_pred_tags = []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            loss = model(
                word_ids=batch["word_ids"],
                char_ids=batch["char_ids"],
                character_lengths=batch["character_lengths"],
                lengths=batch["lengths"],
                token_mask=batch["token_mask"],
                tags=batch["tag_ids"],
                reduction=crf_reduction,
            )
            total_loss += loss.item()
            best_paths, _ = model(
                word_ids=batch["word_ids"],
                char_ids=batch["char_ids"],
                character_lengths=batch["character_lengths"],
                lengths=batch["lengths"],
                token_mask=batch["token_mask"],
            )

            masks = batch["token_mask"].detach().cpu()
            gold_ids = batch["tag_ids"].detach().cpu()
            for i, path in enumerate(best_paths):
                seq_len = int(masks[i].sum().item())
                all_gold_tags.append([idx2tag[idx] for idx in gold_ids[i, :seq_len].tolist()])
                all_pred_tags.append([idx2tag[idx] for idx in path[:seq_len]])

    entity_metrics = compute_entity_metrics(all_gold_tags, all_pred_tags, entity_labels)
    return {
        "loss": total_loss / len(dataloader),
        "token_accuracy": compute_token_accuracy(all_gold_tags, all_pred_tags),
        **entity_metrics,
    }


def fit(model, train_loader, validation_loader, optimizer, device, idx2tag, entity_labels, epochs=30, patience=5, gradient_clip=5.0, crf_reduction="token_mean"):
    best_f1 = -1.0
    best_state_dict = None
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, gradient_clip, crf_reduction)
        metrics = evaluate(model, validation_loader, device, idx2tag, entity_labels, crf_reduction)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_micro_f1={metrics['micro_f1']:.4f}")

        if metrics["micro_f1"] > best_f1:
            best_f1 = metrics["micro_f1"]
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)

    return history, best_f1

