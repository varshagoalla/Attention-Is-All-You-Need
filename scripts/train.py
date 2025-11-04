import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
import json
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.Transformer import Transformer
from utils.tokenizer import Tokenizer
from utils.dataset import TranslationDataset, collate_fn
from config.config import Config
from utils.utils import WarmupLRScheduler, LabelSmoothingLoss, create_masks

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
            num_layers=config.num_layers,
            h=config.n_heads,
            d_model=config.d_model,
            d_ff=config.ff_dim,
            max_len=config.max_len,            
            dropout=config.dropout
        ).to(self.device)

        self.criterion = LabelSmoothingLoss(
            vocab_size=self.vocab_size,
            pad_idx=self.tokenizer.pad_id,
            smoothing=config.label_smoothing
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
            
            # Create masks 
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad)
            
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
    
    def evaluate(self):
        """Evaluate on validation set"""

        # Evaluation mode - no dropout etc.
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for src, tgt in tqdm(self.val_loader, desc="Evaluating"):
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                
                # (B, T)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                src_mask, tgt_mask = create_masks(src, tgt_input, self.tokenizer.pad_id)
                
                output = self.model(src, tgt_input, src_mask, tgt_mask) # (B, T, vocab_size)
                
                output = output.reshape(-1, self.vocab_size) # (B * T, vocab_size)
                tgt_output = tgt_output.reshape(-1) # (B * T)
                
                loss = self.criterion(output, tgt_output)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Loaded checkpoint from {checkpoint_path} (epoch {self.current_epoch})")

    def train(self):
        """Main training loop"""

        self.global_step = 0
        self.best_val_loss = float('inf')

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            start_time = time.time()
            train_loss = self.train_epoch()
            elapsed = time.time() - start_time

            val_loss = self.evaluate()

            print(f"Epoch {epoch + 1} completed in {elapsed:.2f}s - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # # Generate sample translation TODO: Implement generate_translation method
            # sample_src = "A man is riding a horse."
            # sample_translation = self.generate_translation(sample_src)
            # print(f"Sample Translation:")
            # print(f"  Source: {sample_src}")
            # print(f"  Translation: {sample_translation}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pt")
                print(f"  New best model! Val loss: {val_loss:.4f}")
            
            # Save regular checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt") 
            



if __name__ == "__main__":

    
    config = Config()

    # Save config
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    with open(os.path.join(config.checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config.__dict__, f, indent=2)

    trainer = Trainer(config)
