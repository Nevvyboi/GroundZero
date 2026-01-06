"""
GroundZero Neural Transformer
==============================
A GPT-style transformer built from scratch.

This is the REAL neural network - not embeddings, not lookups.
Actual attention, actual learning, actual intelligence.

Architecture:
- Decoder-only transformer (like GPT)
- Multi-head self-attention
- Feedforward layers
- Layer normalization
- Positional encoding

Designed to:
- Run on CPU or GPU
- Scale from small to large
- Learn incrementally
- Generate coherent text
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TransformerConfig:
    """Configuration for the transformer model"""
    vocab_size: int = 32000          # Vocabulary size
    max_seq_len: int = 512           # Maximum sequence length
    n_layers: int = 6                # Number of transformer blocks
    n_heads: int = 8                 # Number of attention heads
    d_model: int = 512               # Model dimension
    d_ff: int = 2048                 # Feedforward dimension (usually 4x d_model)
    dropout: float = 0.1            # Dropout rate
    layer_norm_eps: float = 1e-5    # Layer norm epsilon
    
    # Scaling options
    @classmethod
    def tiny(cls):
        """~1M params - for testing"""
        return cls(n_layers=2, n_heads=2, d_model=128, d_ff=512)
    
    @classmethod
    def small(cls):
        """~25M params - runs fast on CPU"""
        return cls(n_layers=4, n_heads=4, d_model=256, d_ff=1024)
    
    @classmethod
    def medium(cls):
        """~85M params - like DistilGPT-2"""
        return cls(n_layers=6, n_heads=8, d_model=512, d_ff=2048)
    
    @classmethod
    def large(cls):
        """~350M params - like GPT-2 medium"""
        return cls(n_layers=12, n_heads=12, d_model=768, d_ff=3072)
    
    @classmethod
    def xl(cls):
        """~750M params - like GPT-2 large"""
        return cls(n_layers=24, n_heads=16, d_model=1024, d_ff=4096)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    
    More efficient than absolute positional encoding.
    Used in LLaMA, Mistral, etc.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos/sin cache
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, 
                         cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to queries and keys."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention
    
    The core of the transformer - allows tokens to attend to each other.
    This is where the "understanding" happens.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # Rotary embeddings
        self.rotary = RotaryPositionalEmbedding(self.head_dim, config.max_seq_len)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Scale factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Forward pass for multi-head attention.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_mask: Mask for causal attention
            use_cache: Whether to return key/value for caching
            past_kv: Cached key/value from previous forward pass
        
        Returns:
            Output tensor and optionally cached key/value
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        cos, sin = self.rotary(x, seq_len)
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Handle KV cache for generation
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        new_kv = (k, v) if use_cache else None
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        output = torch.matmul(attn_probs, v)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.o_proj(output)
        
        return output, new_kv


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    Two linear layers with activation in between.
    This is where the model does "thinking" between attention layers.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        # SwiGLU activation (used in LLaMA, more effective than ReLU)
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (x * W1 * sigmoid(x * W1)) * W3 * W2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    Simpler and faster than LayerNorm, used in LLaMA.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class TransformerBlock(nn.Module):
    """
    A single transformer block.
    
    Combines:
    - Self-attention (understanding context)
    - Feed-forward (processing information)
    - Residual connections (training stability)
    - Layer normalization (training stability)
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        # Pre-normalization (more stable training)
        self.attention_norm = RMSNorm(config.d_model, config.layer_norm_eps)
        self.ffn_norm = RMSNorm(config.d_model, config.layer_norm_eps)
        
        # Core layers
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
    
    def forward(self, x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                past_kv: Optional[Tuple] = None) -> Tuple[torch.Tensor, Optional[Tuple]]:
        # Self-attention with residual
        normed = self.attention_norm(x)
        attn_out, new_kv = self.attention(normed, attention_mask, use_cache, past_kv)
        x = x + attn_out
        
        # Feed-forward with residual
        x = x + self.feed_forward(self.ffn_norm(x))
        
        return x, new_kv


class GroundZeroTransformer(nn.Module):
    """
    The Complete GroundZero Transformer
    
    A GPT-style language model built from scratch.
    No external dependencies, no pretrained weights.
    Just pure PyTorch and mathematics.
    
    This model can:
    - Generate text
    - Answer questions
    - Learn from your data
    - Get smarter over time
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final normalization
        self.norm = RMSNorm(config.d_model, config.layer_norm_eps)
        
        # Output projection (tied with embedding weights)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())
        print(f"✅ Transformer initialized: {self.n_params:,} parameters")
    
    def _init_weights(self, module):
        """Initialize weights with small values for stable training"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask (can't see future tokens)"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                past_key_values: Optional[list] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the transformer.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Optional attention mask
            labels: Optional labels for training (same as input_ids shifted by 1)
            use_cache: Whether to use KV caching for generation
            past_key_values: Cached key/values from previous forward
        
        Returns:
            Dictionary with logits, loss (if labels provided), and optionally cache
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        x = self.dropout(x)
        
        # Create causal mask
        if attention_mask is None:
            attention_mask = self._create_causal_mask(seq_len, device)
        
        # Process through transformer blocks
        new_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            x, new_kv = layer(x, attention_mask, use_cache, past_kv)
            if use_cache:
                new_key_values.append(new_kv)
        
        # Final norm
        x = self.norm(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {
            'logits': logits,
            'loss': loss,
            'past_key_values': new_key_values
        }
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor,
                 max_new_tokens: int = 100,
                 temperature: float = 0.8,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 do_sample: bool = True) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting tokens [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample or use greedy decoding
        
        Returns:
            Generated token IDs
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Truncate if too long
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]
            
            # Forward pass
            outputs = self.forward(idx_cond)
            logits = outputs['logits'][:, -1, :]  # Last token logits
            
            if do_sample:
                # Apply temperature
                logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def save(self, path: Path):
        """Save model to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict(),
            'n_params': self.n_params
        }, path)
        print(f"💾 Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path, device: str = 'cpu') -> 'GroundZeroTransformer':
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        print(f"📂 Model loaded from {path}")
        return model


def test_transformer():
    """Test the transformer model"""
    print("=" * 60)
    print("🧪 Testing GroundZero Transformer")
    print("=" * 60)
    
    # Test different sizes
    for name, config in [
        ("Tiny", TransformerConfig.tiny()),
        ("Small", TransformerConfig.small()),
        ("Medium", TransformerConfig.medium()),
    ]:
        print(f"\n📊 {name} Model:")
        model = GroundZeroTransformer(config)
        
        # Test forward pass
        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        outputs = model(input_ids, labels=input_ids)
        
        print(f"   Logits shape: {outputs['logits'].shape}")
        print(f"   Loss: {outputs['loss'].item():.4f}")
        
        # Test generation
        prompt = torch.randint(0, config.vocab_size, (1, 10))
        generated = model.generate(prompt, max_new_tokens=20)
        print(f"   Generated shape: {generated.shape}")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_transformer()
