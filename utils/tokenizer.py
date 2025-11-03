import sentencepiece as spm

class Tokenizer:
    def __init__(self, model_path="../data/wmt_bpe.model"):
        self.tokenizer = spm.SentencePieceProcessor(model_file=model_path)

        # Define special tokens
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

    def encode(self, text, add_bos=True, add_eos=True):
        ids = self.tokenizer.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def vocab_size(self):
        return self.tokenizer.get_piece_size()
