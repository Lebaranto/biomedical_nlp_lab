import torch
import torch.nn as nn


class CitationClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, pad_idx, embedding_dim=100, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def masked_mean_pooling(self, inputs, mask):
        embeddings = self.embedding(inputs)
        pooled = (embeddings * mask.unsqueeze(-1)).sum(dim=1)
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return pooled / lengths

    def forward(self, inputs, mask):
        return self.classifier(self.masked_mean_pooling(inputs, mask))


class ImprovedCitationClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, pad_idx, embedding_dim=256, hidden_dim=512, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 4 + 1, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def masked_mean_pooling(self, inputs, mask):
        embeddings = self.embedding(inputs)
        pooled = (embeddings * mask.unsqueeze(-1)).sum(dim=1)
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return pooled / lengths

    def forward(
        self,
        sentences,
        sections,
        left_contexts,
        right_contexts,
        positions,
        sentence_mask,
        section_mask,
        left_context_mask,
        right_context_mask,
    ):
        combined = torch.cat(
            [
                self.masked_mean_pooling(sentences, sentence_mask),
                self.masked_mean_pooling(sections, section_mask),
                self.masked_mean_pooling(left_contexts, left_context_mask),
                self.masked_mean_pooling(right_contexts, right_context_mask),
                positions,
            ],
            dim=1,
        )
        return self.classifier(combined)

