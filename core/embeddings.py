"""
Embeddings
==========
Token and positional embedding management.
"""

import numpy as np
import threading
from typing import List, Optional


class EmbeddingLayer:
    """
    Manages token and positional embeddings.
    Supports dynamic vocabulary expansion.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_sequence_length: int,
        token_embeddings: Optional[np.ndarray] = None,
        position_embeddings: Optional[np.ndarray] = None
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self._lock = threading.Lock()
        
        # Initialize or load token embeddings
        if token_embeddings is not None:
            self.token_embeddings = token_embeddings
            self.vocab_size = token_embeddings.shape[0]
        else:
            self.token_embeddings = self._init_token_embeddings()
        
        # Initialize or load positional embeddings
        if position_embeddings is not None:
            self.position_embeddings = position_embeddings
        else:
            self.position_embeddings = self._init_positional_embeddings()
    
    def _init_token_embeddings(self) -> np.ndarray:
        """Initialize token embeddings with Xavier/Glorot initialization"""
        scale = np.sqrt(2.0 / (self.vocab_size + self.embedding_dim))
        return np.random.randn(self.vocab_size, self.embedding_dim) * scale
    
    def _init_positional_embeddings(self) -> np.ndarray:
        """Initialize sinusoidal positional embeddings"""
        pe = np.zeros((self.max_sequence_length, self.embedding_dim))
        position = np.arange(self.max_sequence_length)[:, np.newaxis]
        
        div_term = np.exp(
            np.arange(0, self.embedding_dim, 2) * 
            (-np.log(10000.0) / self.embedding_dim)
        )
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def embed(self, token_ids: List[int]) -> np.ndarray:
        """
        Get embeddings for token IDs.
        Combines token and positional embeddings.
        """
        seq_length = min(len(token_ids), self.max_sequence_length)
        token_ids = token_ids[:seq_length]
        
        with self._lock:
            # Clamp token IDs to valid range
            safe_ids = [
                min(max(tid, 0), self.token_embeddings.shape[0] - 1)
                for tid in token_ids
            ]
            
            # Get token embeddings
            token_emb = self.token_embeddings[safe_ids]
            
            # Add positional embeddings
            pos_emb = self.position_embeddings[:seq_length]
            
            return token_emb + pos_emb
    
    def get_token_embedding(self, token_id: int) -> np.ndarray:
        """Get embedding for a single token"""
        with self._lock:
            safe_id = min(max(token_id, 0), self.token_embeddings.shape[0] - 1)
            return self.token_embeddings[safe_id].copy()
    
    def expand_vocabulary(self, new_vocab_size: int) -> None:
        """Expand embedding matrix for larger vocabulary"""
        with self._lock:
            if new_vocab_size <= self.token_embeddings.shape[0]:
                return
            
            # Create new larger embedding matrix
            scale = np.sqrt(2.0 / (new_vocab_size + self.embedding_dim))
            new_embeddings = np.random.randn(new_vocab_size, self.embedding_dim) * scale
            
            # Copy existing embeddings
            new_embeddings[:self.token_embeddings.shape[0]] = self.token_embeddings
            
            self.token_embeddings = new_embeddings
            self.vocab_size = new_vocab_size
    
    def update_embedding(self, token_id: int, delta: np.ndarray, learning_rate: float) -> None:
        """Update a single token embedding"""
        with self._lock:
            if 0 <= token_id < self.token_embeddings.shape[0]:
                self.token_embeddings[token_id] += learning_rate * delta
    
    def get_similar_tokens(
        self, 
        token_id: int, 
        top_k: int = 10,
        exclude_special: bool = True
    ) -> List[tuple]:
        """Find tokens with similar embeddings"""
        with self._lock:
            query_emb = self.token_embeddings[token_id]
            
            # Compute cosine similarities
            norms = np.linalg.norm(self.token_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-8, norms)
            normalized = self.token_embeddings / norms
            
            query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
            similarities = normalized @ query_norm
            
            # Get top-k indices
            start_idx = 4 if exclude_special else 0  # Skip special tokens
            top_indices = np.argsort(similarities[start_idx:])[::-1][:top_k] + start_idx
            
            return [
                (int(idx), float(similarities[idx]))
                for idx in top_indices
                if idx != token_id
            ]
    
    def compute_sentence_embedding(self, token_ids: List[int]) -> np.ndarray:
        """Compute mean-pooled sentence embedding"""
        if not token_ids:
            return np.zeros(self.embedding_dim)
        
        embeddings = self.embed(token_ids)
        return np.mean(embeddings, axis=0)
    
    def save_state(self) -> dict:
        """Get embeddings for persistence"""
        with self._lock:
            return {
                'token_embeddings': self.token_embeddings.copy(),
                'position_embeddings': self.position_embeddings.copy()
            }
