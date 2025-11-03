import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
import json

from Transformer import Transformer
from tokenizer import Tokenizer
from dataset import TranslationDataset, collate_fn
from config import Config
from utils import WarmupLRScheduler, LabelSmoothingLoss

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device

        print(f"Using device: {self.device}")

        self.tokenizer = Tokenizer(config.model_path)
        self.vocab_size = self.tokenizer.vocab_size()

        self.train_dataset = TranslationDataset(
            config.train_src, config.train_tgt, self.tokenizer
        )
        
        self.val_dataset = TranslationDataset(
            config.val_src, config.val_tgt, self.tokenizer  
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.num_workers,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.num_workers,
            pin_memory=True
        )

        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Train batches: {len(self.train_loader)}")


        self.model = Transformer(
            src_vocab_size=self.vocab_size,
            tgt_vocab_size=self.vocab_size,
            d_model=config.d_model,
            nhead=config.n_heads,
            num_encoder_layers=config.num_layers,
            num_decoder_layers=config.num_layers,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout
        ).to(self.device)

        self.criterion = LabelSmoothingLoss(
            vocab_size=self.vocab_size,
            pad_idx=self.tokenizer.pad_id,
            smoothing=config['label_smoothing']
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps
        )

        self.scheduler = WarmupLRScheduler(
            self.optimizer,
            d_model=config.d_model,
            warmup_steps=config.warmup_steps
        )

    def train_epoch(self):
        """Train for one epoch"""
        # Training mode
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (src, tgt) in enumerate(progress_bar):
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            # Input to decoder - All tokens except last
            tgt_input = tgt[:, :-1]

            # Target output - All tokens except first
            tgt_output = tgt[:, 1:]
            
            # Create masks - TODO: Implement create_masks function
            src_mask, tgt_mask = create_masks(src, tgt_input, self.tokenizer.pad_id)
            
            # Forward pass
            self.optimizer.zero_grad() # Zero grad because we don't want to accumulate gradients across batches
            output = self.model(src, tgt_input, src_mask, tgt_mask)
            
            # Calculate loss
            # output: (B, T, vocab_size), tgt_output: (B, T)
            output = output.reshape(-1, self.vocab_size)  # (B*T, vocab_size)
            tgt_output = tgt_output.reshape(-1)  # (B*T)
            
            loss = self.criterion(output, tgt_output)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_grad'])
            
            # Update parameters
            self.optimizer.step()

            # Update learning rate using scheduler
            self.scheduler.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.6f}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def train(self):

        pass

        
        
            



if __name__ == "__main__":
    config = Config()
    trainer = Trainer(config)
