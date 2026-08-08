import torch
import torch.nn as nn


class AbstractSentenceClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        pad_idx,
        embedding_dim=128,
        hidden_dim=256,
        dropout=0.3,
        freeze_embeddings=False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding.weight.requires_grad = not freeze_embeddings
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs, mask):
        embeddings = self.embedding(inputs)
        pooled = (embeddings * mask.unsqueeze(-1)).sum(dim=1)
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return self.classifier(pooled / lengths)

