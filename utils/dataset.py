import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, tokenizer, max_len=100):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        with open(src_file, 'r', encoding='utf-8') as f:
            src_lines = [line.strip() for line in f]
        with open(tgt_file, 'r', encoding='utf-8') as f:
            tgt_lines = [line.strip() for line in f]
        
        assert len(src_lines) == len(tgt_lines)
        
        # word limit for token count
        max_words = int(max_len * 0.7)  # 100 tokens → ~70 words
        
        # Filter together to maintain alignment
        self.src_lines = []
        self.tgt_lines = []
        
        for src, tgt in zip(src_lines, tgt_lines):
            if len(src.split()) <= max_words and len(tgt.split()) <= max_words:
                self.src_lines.append(src)
                self.tgt_lines.append(tgt)
        
        print(f"Kept {len(self.src_lines):,}/{len(src_lines):,} pairs")
        
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
