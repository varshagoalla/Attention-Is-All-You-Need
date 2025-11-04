import torch
import torch.nn as nn

class WarmupLRScheduler:
    """
    Implements the learning rate scheduler with warmup as described in
    "Attention Is All You Need" paper.
    """
    
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        lr = self._get_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _get_lr(self):
        """Calculate learning rate based on current step"""
        step_num = self.current_step
        
        # Formula from paper
        lr = (self.d_model ** -0.5) * min(step_num ** -0.5, step_num * (self.warmup_steps ** -1.5))
        
        return lr
    
    def get_last_lr(self):
        """Get current learning rate as list - matches PyTorch scheduler interface"""
        return [self._get_lr()]
    

class LabelSmoothingLoss(nn.Module):
    """
    Implements label smoothing and then cross entropy loss
    """
    
    def __init__(self, vocab_size, pad_idx, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        
        
    def forward(self, pred, target):
        """
        Compute the label smoothing loss
        pred: (B * T, vocab_size)
        target: (B * T)
        """
        
        confidence = 1.0 - self.smoothing
        low_confidence = self.smoothing / (self.vocab_size - 2)
        
        # Create smoothed target distribution
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(low_confidence)
        true_dist.scatter_(1, target.unsqueeze(1), confidence)
        true_dist[:, self.pad_idx] = 0

        # Mask out padding positions
        mask = (target != self.pad_idx)

        if mask.sum() == 0:
            # All positions are padding
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # Compute cross entropy loss
        log_probs = torch.nn.functional.log_softmax(pred, dim=1)
        loss = -(true_dist * log_probs).sum(dim=1)

        # Mask the padding tokens and compute mean loss
        loss = loss.masked_select(mask).mean()
        
        return loss
    

def create_padding_mask(seq, pad_idx=0):
    """
    Create a padding mask for sequences
    seq: (B, T)
    Returns: (B, 1, 1, T)
    """
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask 

def create_causal_mask(size):
    """
    Creates causal (look-ahead) mask for decoder
    Prevents attending to future positions
    
    size: sequence length
    
    Returns:
        mask: (1, size, size) - lower triangular matrix
    """
    mask = torch.tril(torch.ones((size, size))).unsqueeze(0)
    return mask

def create_masks(src, tgt, pad_idx=0):
    """
    Create masks for source and target sequences
    Args:
        src: (B, T_src) - source sequences
        tgt: (B, T_tgt) - target sequences
        pad_idx: padding token index
    
    Returns:
        src_mask: (B, 1, 1, T_src) - source padding mask
        tgt_mask: (B, 1, T_tgt, T_tgt) - combined causal + padding mask for target
    """
    src_mask = create_padding_mask(src, pad_idx)
    
    tgt_padding_mask = create_padding_mask(tgt, pad_idx) # (B, 1, 1, T_tgt)
    tgt_seq_len = tgt.shape[1]
    tgt_causal_mask = create_causal_mask(tgt_seq_len).to(tgt.device) # (1, T_tgt, T_tgt)
    
    tgt_mask = tgt_padding_mask & tgt_causal_mask # Broadcasting will handle dimensions (B, 1, T_tgt, T_tgt)
    
    return src_mask, tgt_mask