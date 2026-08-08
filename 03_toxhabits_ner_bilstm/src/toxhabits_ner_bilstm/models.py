import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNN(nn.Module):
    def __init__(self, char_vocab_size, char_pad_idx, char_embedding_dim=30, num_filters=30, kernel_sizes=(3, 4, 5), dropout=0.2):
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=char_pad_idx)
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(char_embedding_dim, num_filters, kernel_size=kernel_size)
                for kernel_size in self.kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = num_filters * len(self.kernel_sizes)

    def forward(self, char_ids, character_lengths):
        batch_size, sequence_length, max_char_length = char_ids.shape
        flat_char_ids = char_ids.reshape(batch_size * sequence_length, max_char_length)
        flat_lengths = character_lengths.reshape(batch_size * sequence_length)
        embedded = self.char_embedding(flat_char_ids).transpose(1, 2)

        largest_kernel = max(self.kernel_sizes)
        if max_char_length < largest_kernel:
            embedded = F.pad(embedded, pad=(0, largest_kernel - max_char_length), value=0.0)

        pooled_features = []
        for convolution, kernel_size in zip(self.convolutions, self.kernel_sizes):
            output = F.relu(convolution(embedded))
            windows = output.size(-1)
            valid_counts = (flat_lengths - kernel_size + 1).clamp(min=1, max=windows)
            positions = torch.arange(windows, device=char_ids.device).unsqueeze(0)
            valid_mask = positions < valid_counts.unsqueeze(1)
            output = output.masked_fill(~valid_mask.unsqueeze(1), torch.finfo(output.dtype).min)
            pooled = output.max(dim=-1).values
            pooled = pooled.masked_fill((flat_lengths == 0).unsqueeze(1), 0.0)
            pooled_features.append(pooled)

        features = torch.cat(pooled_features, dim=-1)
        features = self.dropout(features)
        return features.reshape(batch_size, sequence_length, self.output_dim)


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.2):
        super().__init__()
        self.output_dim = hidden_dim * 2
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_features, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(token_features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.bilstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return self.dropout(output)


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        word_pad_idx,
        char_vocab_size,
        char_pad_idx,
        word_embedding_dim=100,
        char_embedding_dim=30,
        char_num_filters=30,
        char_kernel_sizes=(3, 4, 5),
        lstm_hidden_dim=128,
        lstm_num_layers=1,
        dropout=0.2,
    ):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=word_pad_idx)
        self.char_cnn = CharCNN(char_vocab_size, char_pad_idx, char_embedding_dim, char_num_filters, char_kernel_sizes, dropout)
        self.bilstm = BiLSTM(word_embedding_dim + self.char_cnn.output_dim, lstm_hidden_dim, lstm_num_layers, dropout)
        self.input_dropout = nn.Dropout(dropout)

    def forward(self, word_ids, char_ids, character_lengths, lengths):
        word_features = self.word_embedding(word_ids)
        char_features = self.char_cnn(char_ids, character_lengths)
        token_features = torch.cat([word_features, char_features], dim=-1)
        return self.bilstm(self.input_dropout(token_features), lengths)


class ToxHabitsNERModel(nn.Module):
    def __init__(self, vocab_size, word_pad_idx, char_vocab_size, char_pad_idx, num_tags, **encoder_kwargs):
        super().__init__()
        self.encoder = Encoder(vocab_size, word_pad_idx, char_vocab_size, char_pad_idx, **encoder_kwargs)
        self.classifier = nn.Linear(self.encoder.bilstm.output_dim, num_tags)

    def forward(self, word_ids, char_ids, character_lengths, lengths):
        return self.classifier(self.encoder(word_ids, char_ids, character_lengths, lengths))


class LinearChainCRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(self, emissions, tags, mask, reduction="mean"):
        nll = self._compute_log_partition(emissions, mask) - self._compute_gold_score(emissions, tags, mask)
        if reduction == "none":
            return nll
        if reduction == "sum":
            return nll.sum()
        if reduction == "mean":
            return nll.mean()
        if reduction == "token_mean":
            return nll.sum() / mask.sum()
        raise ValueError(f"Unknown reduction: {reduction}")

    def _compute_gold_score(self, emissions, tags, mask):
        batch_size, sequence_length, _ = emissions.shape
        safe_tags = tags.masked_fill(~mask, 0)
        first_tags = safe_tags[:, 0]
        score = self.start_transitions[first_tags]
        score = score + emissions[torch.arange(batch_size, device=emissions.device), 0, first_tags]

        for timestep in range(1, sequence_length):
            previous_tags = safe_tags[:, timestep - 1]
            current_tags = safe_tags[:, timestep]
            transition_score = self.transitions[previous_tags, current_tags]
            emission_score = emissions[torch.arange(batch_size, device=emissions.device), timestep, current_tags]
            score = score + (transition_score + emission_score) * mask[:, timestep]

        last_indices = mask.long().sum(dim=1) - 1
        last_tags = safe_tags.gather(dim=1, index=last_indices.unsqueeze(1)).squeeze(1)
        return score + self.end_transitions[last_tags]

    def _compute_log_partition(self, emissions, mask):
        alpha = self.start_transitions + emissions[:, 0]
        for timestep in range(1, emissions.size(1)):
            scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, timestep].unsqueeze(1)
            next_alpha = torch.logsumexp(scores, dim=1)
            alpha = torch.where(mask[:, timestep].unsqueeze(1), next_alpha, alpha)
        return torch.logsumexp(alpha + self.end_transitions, dim=1)

    @torch.no_grad()
    def decode(self, emissions, mask):
        batch_size, sequence_length, _ = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        history = []

        for timestep in range(1, sequence_length):
            scores = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_scores, best_previous_tags = scores.max(dim=1)
            best_scores = best_scores + emissions[:, timestep]
            score = torch.where(mask[:, timestep].unsqueeze(1), best_scores, score)
            history.append(best_previous_tags)

        score = score + self.end_transitions
        best_final_scores, best_final_tags = score.max(dim=1)
        sequence_lengths = mask.long().sum(dim=1)
        best_paths = []

        for batch_index in range(batch_size):
            sequence_length = int(sequence_lengths[batch_index].item())
            current_tag = int(best_final_tags[batch_index].item())
            path = [current_tag]
            for backpointers in reversed(history[: sequence_length - 1]):
                current_tag = int(backpointers[batch_index, current_tag].item())
                path.append(current_tag)
            path.reverse()
            best_paths.append(path)

        return best_paths, best_final_scores


class ToxHabitsNERWithCRF(nn.Module):
    def __init__(self, vocab_size, word_pad_idx, char_vocab_size, char_pad_idx, num_tags, **model_kwargs):
        super().__init__()
        self.ner_model = ToxHabitsNERModel(vocab_size, word_pad_idx, char_vocab_size, char_pad_idx, num_tags, **model_kwargs)
        self.crf = LinearChainCRF(num_tags)

    def forward(self, word_ids, char_ids, character_lengths, lengths, token_mask, tags=None, reduction="mean"):
        emissions = self.ner_model(word_ids, char_ids, character_lengths, lengths)
        if tags is not None:
            return self.crf(emissions=emissions, tags=tags, mask=token_mask, reduction=reduction)
        return self.crf.decode(emissions=emissions, mask=token_mask)

