import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, tokenizer):
        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_lines = [line.strip() for line in f]
        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.tgt_lines = [line.strip() for line in f]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src = self.tokenizer.encode(self.src_lines[idx])
        tgt = self.tokenizer.encode(self.tgt_lines[idx])
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def collate_fn(batch):
    """
    Collate function to pad sequences in a batch."""
    src_batch, tgt_batch = zip(*batch)
    src_pad = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_pad = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_pad, tgt_pad
