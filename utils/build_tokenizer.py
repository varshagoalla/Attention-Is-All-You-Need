import os
from datasets import load_dataset
import sentencepiece as spm
import re

print(f"Loading WMT14 En–De dataset from Hugging Face")
dataset = load_dataset("wmt14", "de-en")

# Make sure output dir exists
os.makedirs("data", exist_ok=True)

en_path = "data/train.en"
de_path = "data/train.de"

def clean_line(text: str) -> str:
    # Normalize line endings and collapse multiple spaces
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)   # replace tabs, multiple spaces, etc.
    return text.strip()                # remove leading/trailing spaces


with open(en_path, "w", encoding="utf-8") as f_en, open(de_path, "w", encoding="utf-8") as f_de:
    kept, skipped = 0, 0
    for item in dataset["train"]:
        en = clean_line(item["translation"].get("en", ""))
        de = clean_line(item["translation"].get("de", ""))

        # Keep only valid non-empty aligned pairs
        if en and de:
            f_en.write(en + "\n")
            f_de.write(de + "\n")
            kept += 1
        else:
            skipped += 1


print(f"Finished writing aligned data")
print(f"Kept: {kept:,} sentence pairs")
print(f"Skipped: {skipped:,} incomplete or empty entries")
print(f"Saved training files:\n  {en_path}\n  {de_path}")

val_en_path = "data/val.en"
val_de_path = "data/val.de"


with open(val_en_path, "w", encoding="utf-8") as f_en, open(val_de_path, "w", encoding="utf-8") as f_de:
    kept, skipped = 0, 0
    for item in dataset["validation"]:
        en = clean_line(item["translation"].get("en", ""))
        de = clean_line(item["translation"].get("de", ""))

        # Keep only valid non-empty aligned pairs
        if en and de:
            f_en.write(en + "\n")
            f_de.write(de + "\n")
            kept += 1
        else:
            skipped += 1

print(f"Finished writing aligned data")
print(f"Kept: {kept:,} sentence pairs")
print(f"Skipped: {skipped:,} incomplete or empty entries")
print(f"Saved validation files:\n  {val_en_path}\n  {val_de_path}")


# Concatenate for joint BPE training

print("Concatenating train.en and train.de for joint BPE training")
merged_path = "data/train.all"
os.system(f"cat {en_path} {de_path} > {merged_path}")


# Train SentencePiece BPE model tokenizer

print("Training SentencePiece tokenizer (32k vocab, BPE)")

spm.SentencePieceTrainer.train(
    input=merged_path,
    model_prefix="data/wmt_bpe",
    vocab_size=32000,
    model_type="bpe",
    character_coverage=1.0,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
)


# Test the trained tokenizer
sp = spm.SentencePieceProcessor(model_file="data/wmt_bpe.model")

# English sentence
en_text = "The agreement on the European Economic Area was signed in August 1992."
en_ids = sp.encode(en_text, out_type=int)
en_decoded = sp.decode(en_ids)

# German sentence
de_text = "Das Abkommen über den Europäischen Wirtschaftsraum wurde im August 1992 unterzeichnet."
de_ids = sp.encode(de_text, out_type=int)
de_decoded = sp.decode(de_ids)

print("Sample English Test")
print("Original (EN):", en_text)
print("Encoded IDs:", en_ids)
print("Decoded:", en_decoded)

print("Sample German Test")
print("Original (DE):", de_text)
print("Encoded IDs:", de_ids)
print("Decoded:", de_decoded)

print("Shared BPE tokenizer works for both English and German!")