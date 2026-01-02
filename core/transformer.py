"""
Transformer
===========
Transformer blocks with multi-head attention and feed-forward layers.
"""

import numpy as np
from typing import List, Optional, Tuple


class AttentionHead:
    """Single attention head"""
    
    def __init__(self, d_model: int, d_head: int):
        self.d_head = d_head
        
        # Xavier initialization
        scale = np.sqrt(2.0 / (d_model + d_head))
        self.W_q = np.random.randn(d_model, d_head) * scale
        self.W_k = np.random.randn(d_model, d_head) * scale
        self.W_v = np.random.randn(d_model, d_head) * scale
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute scaled dot-product attention.
        
        Args:
            x: Input tensor of shape (seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            Attention output of shape (seq_len, d_head)
        """
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        # Scaled dot-product attention
        scores = (Q @ K.T) / np.sqrt(self.d_head)
        
        if mask is not None:
            scores = scores + mask * -1e9
        
        attention_weights = self._softmax(scores)
        return attention_weights @ V
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-8)
    
    def get_weights(self) -> dict:
        """Get weights for persistence"""
        return {
            'W_q': self.W_q,
            'W_k': self.W_k,
            'W_v': self.W_v
        }
    
    def set_weights(self, weights: dict) -> None:
        """Load weights from persistence"""
        if not weights:
            return
        if 'W_q' in weights:
            self.W_q = weights['W_q']
        if 'W_k' in weights:
            self.W_k = weights['W_k']
        if 'W_v' in weights:
            self.W_v = weights['W_v']


class MultiHeadAttention:
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int):
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_model = d_model
        
        self.heads = [
            AttentionHead(d_model, self.d_head)
            for _ in range(num_heads)
        ]
        
        # Output projection
        scale = np.sqrt(2.0 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * scale
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute multi-head attention.
        
        Args:
            x: Input tensor of shape (seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            Output of shape (seq_len, d_model)
        """
        # Compute attention for each head
        head_outputs = [
            head.forward(x, mask)
            for head in self.heads
        ]
        
        # Concatenate heads
        concat = np.concatenate(head_outputs, axis=-1)
        
        # Project output
        return concat @ self.W_o
    
    def get_weights(self) -> dict:
        """Get weights for persistence"""
        return {
            'heads': [head.get_weights() for head in self.heads],
            'W_o': self.W_o
        }
    
    def set_weights(self, weights: dict) -> None:
        """Load weights from persistence"""
        if not weights:
            return
        if 'heads' in weights:
            for i, head_weights in enumerate(weights['heads']):
                if i < len(self.heads):
                    self.heads[i].set_weights(head_weights)
        if 'W_o' in weights:
            self.W_o = weights['W_o']


class FeedForward:
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int):
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Xavier initialization
        scale1 = np.sqrt(2.0 / (d_model + d_ff))
        scale2 = np.sqrt(2.0 / (d_ff + d_model))
        
        self.W1 = np.random.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Feed-forward with ReLU activation.
        
        Args:
            x: Input tensor of shape (seq_len, d_model)
        
        Returns:
            Output of shape (seq_len, d_model)
        """
        # First linear + ReLU
        hidden = np.maximum(0, x @ self.W1 + self.b1)
        # Second linear
        return hidden @ self.W2 + self.b2
    
    def get_weights(self) -> dict:
        """Get weights for persistence"""
        return {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2
        }
    
    def set_weights(self, weights: dict) -> None:
        """Load weights from persistence"""
        if not weights:
            return
        if 'W1' in weights:
            self.W1 = weights['W1']
        if 'b1' in weights:
            self.b1 = weights['b1']
        if 'W2' in weights:
            self.W2 = weights['W2']
        if 'b2' in weights:
            self.b2 = weights['b2']


class LayerNorm:
    """Layer normalization"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    
    def get_weights(self) -> dict:
        return {'gamma': self.gamma, 'beta': self.beta}
    
    def set_weights(self, weights: dict) -> None:
        if not weights:
            return
        if 'gamma' in weights:
            self.gamma = weights['gamma']
        if 'beta' in weights:
            self.beta = weights['beta']


class TransformerBlock:
    """Single transformer block with attention, feed-forward, and residual connections"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            Output of shape (seq_len, d_model)
        """
        # Self-attention with residual connection and layer norm
        attn_out = self.attention.forward(x, mask)
        x = self.norm1.forward(x + attn_out)
        
        # Feed-forward with residual connection and layer norm
        ff_out = self.ff.forward(x)
        x = self.norm2.forward(x + ff_out)
        
        return x
    
    def get_weights(self) -> dict:
        """Get weights for persistence"""
        return {
            'attention': self.attention.get_weights(),
            'ff': self.ff.get_weights(),
            'norm1': self.norm1.get_weights(),
            'norm2': self.norm2.get_weights()
        }
    
    def set_weights(self, weights: dict) -> None:
        """Load weights from persistence"""
        if not weights:
            return
        if 'attention' in weights:
            self.attention.set_weights(weights['attention'])
        if 'ff' in weights:
            self.ff.set_weights(weights['ff'])
        if 'norm1' in weights:
            self.norm1.set_weights(weights['norm1'])
        if 'norm2' in weights:
            self.norm2.set_weights(weights['norm2'])


class TransformerEncoder:
    """Stack of transformer blocks"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int):
        self.layers = [
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]
        self.d_model = d_model
        self.num_layers = num_layers
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass through all transformer blocks"""
        for layer in self.layers:
            x = layer.forward(x, mask)
        return x
    
    def get_weights(self) -> dict:
        """Get weights for persistence"""
        return {
            f'layer_{i}': layer.get_weights()
            for i, layer in enumerate(self.layers)
        }
    
    def set_weights(self, weights: dict) -> None:
        """Load weights from persistence"""
        if not weights:
            return
        for i, layer in enumerate(self.layers):
            key = f'layer_{i}'
            if key in weights:
                try:
                    layer.set_weights(weights[key])
                except Exception as e:
                    print(f"Warning: Could not load weights for {key}: {e}")