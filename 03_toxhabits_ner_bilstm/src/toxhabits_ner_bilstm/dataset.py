import torch
from torch.utils.data import Dataset


class ToxHabitsDataset(Dataset):
    def __init__(self, data, word2idx, char2idx, tag2idx):
        self.word2idx = word2idx
        self.char2idx = char2idx
        self.tag2idx = tag2idx
        self.word_unk_idx = word2idx["<unk>"]
        self.char_unk_idx = char2idx["<unk>"]
        self.samples = []

        for document_index, document_chunks in enumerate(data):
            for default_chunk_id, chunk in enumerate(document_chunks):
                tokens = chunk["tokens"]
                offsets = chunk["offsets"]
                bio_tags = chunk["bio_tags"]
                if len(tokens) != len(offsets) or len(tokens) != len(bio_tags):
                    raise ValueError(f"Invalid chunk in {chunk['filename']}")
                if not tokens:
                    continue
                self.samples.append(
                    {
                        "document_id": chunk["filename"],
                        "document_index": document_index,
                        "chunk_id": chunk.get("chunk_id", default_chunk_id),
                        "tokens": tokens,
                        "offsets": offsets,
                        "bio_tags": bio_tags,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = sample["tokens"]
        bio_tags = sample["bio_tags"]
        return {
            "word_ids": torch.tensor(
                [self.word2idx.get(token, self.word_unk_idx) for token in tokens],
                dtype=torch.long,
            ),
            "char_ids": [
                torch.tensor([self.char2idx.get(char, self.char_unk_idx) for char in token], dtype=torch.long)
                for token in tokens
            ],
            "tag_ids": torch.tensor([self.tag2idx[tag] for tag in bio_tags], dtype=torch.long),
            "length": len(tokens),
            "document_id": sample["document_id"],
            "chunk_id": sample["chunk_id"],
            "tokens": tokens,
            "offsets": sample["offsets"],
        }


def create_collate_fn(word_pad_idx, char_pad_idx, tag_pad_idx):
    def collate_fn(batch):
        batch_size = len(batch)
        lengths = torch.tensor([sample["length"] for sample in batch], dtype=torch.long)
        max_sequence_length = int(lengths.max().item())
        max_character_length = max(len(chars) for sample in batch for chars in sample["char_ids"])

        word_ids = torch.full((batch_size, max_sequence_length), word_pad_idx, dtype=torch.long)
        tag_ids = torch.full((batch_size, max_sequence_length), tag_pad_idx, dtype=torch.long)
        char_ids = torch.full((batch_size, max_sequence_length, max_character_length), char_pad_idx, dtype=torch.long)
        token_mask = torch.zeros((batch_size, max_sequence_length), dtype=torch.bool)
        character_mask = torch.zeros((batch_size, max_sequence_length, max_character_length), dtype=torch.bool)
        character_lengths = torch.zeros((batch_size, max_sequence_length), dtype=torch.long)

        document_ids = []
        chunk_ids = []
        tokens = []
        offsets = []

        for batch_index, sample in enumerate(batch):
            sequence_length = sample["length"]
            word_ids[batch_index, :sequence_length] = sample["word_ids"]
            tag_ids[batch_index, :sequence_length] = sample["tag_ids"]
            token_mask[batch_index, :sequence_length] = True

            for token_index, token_char_ids in enumerate(sample["char_ids"]):
                char_length = len(token_char_ids)
                char_ids[batch_index, token_index, :char_length] = token_char_ids
                character_mask[batch_index, token_index, :char_length] = True
                character_lengths[batch_index, token_index] = char_length

            document_ids.append(sample["document_id"])
            chunk_ids.append(sample["chunk_id"])
            tokens.append(sample["tokens"])
            offsets.append(sample["offsets"])

        return {
            "word_ids": word_ids,
            "char_ids": char_ids,
            "tag_ids": tag_ids,
            "token_mask": token_mask,
            "character_mask": character_mask,
            "character_lengths": character_lengths,
            "lengths": lengths,
            "document_ids": document_ids,
            "chunk_ids": chunk_ids,
            "tokens": tokens,
            "offsets": offsets,
        }

    return collate_fn

