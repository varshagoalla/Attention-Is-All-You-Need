import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config import Config
from utils.tokenizer import Tokenizer
from utils.dataset import TranslationDataset




config = Config()

# Test tokenizer
print("Testing tokenizer...")
tokenizer = Tokenizer(config.model_path)
print(f"✅ Tokenizer loaded! Vocab size: {tokenizer.vocab_size()}")

# Test dataset
print("\nTesting dataset...")
train_dataset = TranslationDataset(config.train_src, config.train_tgt, tokenizer)
print(f"✅ Dataset loaded! {len(train_dataset)} samples")

# Test one sample
src, tgt = train_dataset[0]
print(f"✅ Sample shape - src: {src.shape}, tgt: {tgt.shape}")

print("\n🎉 Everything looks good!")