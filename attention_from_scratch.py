import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional



class ScaledDotProductAttention(nn.Module):
    """
    Computes Scaled Dot-Product Attention mentioned in 'Attention is All You Need'
    - B: Batch size
    - h: Number of attention heads
    - T_q: Sequence length
    - T_kv: Sequence length
    - d_k: Dimension of key/query vectors
    - d_v: Dimension of value vectors 
    In this paper d_k, d_v are same.
    T_k and T_qv is same for self-attention, different for cross-attention.
    """

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout) # Randomly zeros some of the elements, follows bernoulli distribution

    def forward(self, q:torch.Tensor, k: torch.Tensor, v:torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            q -> (B, h, T_q, d_k) : Query tensor
            k -> (B, h, T_kv, d_k) : Key tensor
            v -> (B, h, T_kv, d_v) : Value tensor
            mask -> Padding mask (B, 1, 1, T_kv) or Causal/look-ahead mask (B, 1, T_q, T_kv) : Mask tensor to prevent attention to certain positions
        returns:
            output -> (B, h, T_q, d_v) : Attention output
            attention_weights -> (B, h, T_q, T_kv) : Attention weights
        """
        d_k = q.size()[-1]

        # Compute dot product between queries and keys. (B, h, T_q, d_k) x (B, h, d_k, T_kv) -> (B, h, T_q, T_kv)
        scores = torch.matmul(q, k.transpose(-2, -1))

        # Scale by square root of d_k to prevent large dot product values and avoid softmax saturation thus eliminating vanishing gradients
        scores = scores / math.sqrt(d_k)

        # Apply mask if provided
        if mask is not None:
            # Using -1e9 avoids potential NaN issues on some GPUs or mixed precision setups.
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, -1)

        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)

        # Multiply attention weights with values. (B, h, T_q, T_kv) x (B, h, T_kv, d_v) -> B, h, T_q, d_v)
        output = torch.matmul(attention_weights, v)

        return output, attention_weights
    

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module as described in 'Attention is All You Need'
    Supports both self-attention and cross-attention.
    - h: Number of attention heads
    - d_model: Dimension of the model
    - dropout: Dropout rate
    """

    def __init__(self, h: int, d_model:int, dropout: float = 0.1):
        super().__init__()

        self.h = h
        self.d_model = d_model
        assert d_model % h == 0

        self.d_k = d_model // h
        self.d_v = d_model // h

        self.W_q = nn.Linear(d_model, self.d_k * h)
        self.W_k = nn.Linear(d_model, self.d_k * h)
        self.W_v = nn.Linear(d_model, self.d_v * h)
        self.W_o = nn.Linear(self.d_v * h, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, X: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (h, d_k or d_v) and transpose the result to get dimensions (B, h, T, d_k or d_v)
        """
        (B, T, _) = X.shape
        X = X.view(B, T, self.h, -1) # (B, T, h, d_k or d_v)
        X = X.transpose(1, 2).contiguous()   # (B, h, T, d_k or d_v)
        return X
    
    def combine_heads(self, X: torch.Tensor) -> torch.Tensor:
        """
        Combine the heads by transposing and reshaping back to (B, T, h * d_v)
        """
        (B, _, T, _) = X.shape
        X = X.transpose(1, 2).contiguous() # (B, T, h, d_v)
        X = X.view(B, T, self.d_model)     # (B, T, d_model = h * d_v)
        return X
    
    def forward(self, query_X: torch.Tensor, key_X: torch.Tensor, value_X: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            For self-attention, query_X, key_X, value_X are all the same input tensor X of shape (B, T, d_model).
            For cross-attention, query_X is the query from the decoder and key_X/value_X is from the encoder.

        returns:
            output -> (B, T_q, d_model) : Attention output
            attention_weights -> (B, h, T_q, T_kv) : Attention weights"""

        Q = self.W_q(query_X)  # (B, T_q, h * d_k)
        K = self.W_k(key_X)  # (B, T_kv, h * d_k)
        V = self.W_v(value_X)  # (B, T_kv, h * d_v)

        q = self.split_heads(Q) # (B, h, T_q, d_k)
        k = self.split_heads(K) # (B, h, T_kv, d_k)
        v = self.split_heads(V) # (B, h, T_kv, d_v)

        if mask is not None:
            mask = mask.unsqueeze(1) # (B, 1, T_q, T_kv)

        outputs, attention_weights = self.attention(q, k, v, mask)

        outputs = self.combine_heads(outputs) # (B, T_q, h * d_v)
        outputs = self.W_o(outputs)           # (B, T_q, d_model)
        outputs = self.dropout(outputs) 

        return outputs, attention_weights