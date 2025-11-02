import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional

class PositionalEncoding(nn.Module):
    """
    Positional Encoding module to inject position information into the input embeddings
    - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        
        position = torch.arange(0, max_len).unsqueeze(1).float() # position index
        
        # 1/10000^(2i/d_model) = exp((2i/d_model) * log(1/10000)) = exp(-log(10000) * (2i/d_model))
        mul_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * mul_term)
        pe[:, 1::2] = torch.cos(position * mul_term)
    
        # (1, max_len, d_model) so that it can be added to input embeddings of shape (B, T, d_model)
        pe = pe.unsqueeze(0)  

        # buffer so that it's not considered a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -> (B, T, d_model)
        returns:
            output -> (B, T, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

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
    
class PositionWiseFeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network as described in 'Attention is All You Need'
    Consists of two linear transformations with a ReLU activation in between.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -> (B, T, d_model)
        returns:
            output -> (B, T, d_model)
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class EncoderLayer(nn.Module):
    """
    Transformer Encoder layer consisting of Multi-Head Attention and Position-Wise Feed-Forward Network
    """

    def __init__(self, h: int, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(h, d_model, dropout)
        self.feed_forward_network = PositionWiseFeedForwardNetwork(d_model, d_ff, dropout)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)  
        self.dropout2 = nn.Dropout(dropout)

    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        att_outputs, att_weights = self.multi_head_attention(x, x, x, mask)
        x = x + self.dropout1(att_outputs)
        x = self.layer_norm1(x)

        ff_outputs = self.feed_forward_network(x)
        x = x + self.dropout2(ff_outputs)
        x = self.layer_norm2(x)

        return x, att_weights
    
class DecoderLayer(nn.Module):
    """
    Transformer Decoder layer consisting of Masked Multi-Head Self-Attention, Multi-Head Cross-Attention, and Position-Wise Feed-Forward Network
    """

    def __init__(self, h: int, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(h, d_model, dropout)
        self.cross_attention = MultiHeadAttention(h, d_model, dropout)
        self.feed_forward_network = PositionWiseFeedForwardNetwork(d_model, d_ff, dropout)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_outputs: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None, src_mask: Optional[torch.Tensor] = None):
        self_att_outputs, self_att_weights = self.self_attention(x, x, x, tgt_mask)
        x = x + self.dropout1(self_att_outputs)
        x = self.layer_norm1(x)

        cross_att_outputs, cross_att_weights = self.cross_attention(x, encoder_outputs, encoder_outputs, src_mask)
        x = x + self.dropout2(cross_att_outputs)
        x = self.layer_norm2(x)

        ff_outputs = self.feed_forward_network(x)
        x = x + self.dropout3(ff_outputs)
        x = self.layer_norm3(x)

        return x, (self_att_weights, cross_att_weights)
    

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder consisting of multiple Encoder layers
    """

    def __init__(self, vocab_size: int, num_layers: int, h: int, d_model: int, d_ff: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout=dropout)
        self.layers = nn.ModuleList([EncoderLayer(h, d_model, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):

        # Embed input tokens and scale by sqrt(d_model)
        x = self.embedding(x) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Pass through each encoder layer
        for layer in self.layers:
            x, _ = layer(x, mask)
        return x

class TransformerDecoder(nn.Module):
    """
    Transformer Decoder consisting of multiple Decoder layers
    """

    def __init__(self, vocab_size: int, num_layers: int, h: int, d_model: int, d_ff: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout=dropout)
        self.layers = nn.ModuleList([DecoderLayer(h, d_model, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        # Final linear layer to project to vocabulary
        self.fc_out = nn.Linear(d_model, vocab_size)
        

    def forward(self, x: torch.Tensor, enc_outputs: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None, src_mask: Optional[torch.Tensor] = None):

        # Embed input tokens and scale by sqrt(d_model)
        x = self.embedding(x) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Pass through each decoder layer
        for layer in self.layers:
            x, _ = layer(x, enc_outputs, tgt_mask, src_mask)

        # Project to vocabulary
        output = self.fc_out(x)

        return output
    

class Transformer(nn.Module):
    """
    Complete Transformer model consisting of Encoder and Decoder
    """

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, num_layers: int, h: int, d_model: int, d_ff: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab_size, num_layers, h, d_model, d_ff, max_len, dropout)
        self.decoder = TransformerDecoder(tgt_vocab_size, num_layers, h, d_model, d_ff, max_len, dropout)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None):
        enc_outputs = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_outputs, tgt_mask, src_mask)
        return output