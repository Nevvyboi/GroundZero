"""
Neural Model
============
Main model orchestrator that coordinates tokenizer, embeddings, and transformer.
"""

import numpy as np
import threading
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

from .tokenizer import Tokenizer
from .embeddings import EmbeddingLayer
from .transformer import TransformerEncoder
from storage import MemoryStore, ModelStore
from config import ModelConfig


class NeuralModel:
    """
    Main neural network model.
    Coordinates tokenization, embeddings, and transformer layers.
    Handles learning and inference.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        memory_store: MemoryStore,
        model_store: ModelStore
    ):
        self.config = config
        self.memory = memory_store
        self.model_store = model_store
        self._lock = threading.Lock()
        
        # Training state
        self.total_tokens_learned = 0
        self.training_steps = 0
        self.learning_rate = 0.001
        
        # Initialize or load components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize or load model components"""
        # Initialize tokenizer (loads from database)
        self.tokenizer = Tokenizer(
            self.memory,
            max_vocab_size=self.config.vocab_size
        )
        
        # Try to load existing model state
        saved_state = self.model_store.load_state()
        saved_embeddings = self.model_store.load_embeddings()
        saved_weights = self.model_store.load_weights()
        
        if saved_state:
            self.total_tokens_learned = saved_state.get('total_tokens', 0)
            self.training_steps = saved_state.get('training_steps', 0)
            self.learning_rate = saved_state.get('learning_rate', 0.001)
        
        # Initialize embeddings
        if saved_embeddings:
            self.embeddings = EmbeddingLayer(
                vocab_size=self.config.vocab_size,
                embedding_dim=self.config.embedding_dim,
                max_sequence_length=self.config.max_sequence_length,
                token_embeddings=saved_embeddings['token_embeddings'],
                position_embeddings=saved_embeddings['position_embeddings']
            )
        else:
            self.embeddings = EmbeddingLayer(
                vocab_size=self.config.vocab_size,
                embedding_dim=self.config.embedding_dim,
                max_sequence_length=self.config.max_sequence_length
            )
        
        # Initialize transformer
        self.transformer = TransformerEncoder(
            d_model=self.config.embedding_dim,
            num_heads=self.config.num_heads,
            d_ff=self.config.feedforward_dim,
            num_layers=self.config.num_layers
        )
        
        # Load transformer weights if available
        if saved_weights:
            try:
                self.transformer.set_weights(saved_weights)
            except Exception as e:
                print(f"Warning: Could not load transformer weights: {e}")
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to a dense vector representation.
        
        Args:
            text: Input text
            
        Returns:
            Vector representation of shape (embedding_dim,)
        """
        token_ids = self.tokenizer.encode(text)
        
        if not token_ids:
            return np.zeros(self.config.embedding_dim)
        
        # Get embeddings
        embedded = self.embeddings.embed(token_ids)
        
        # Pass through transformer
        transformed = self.transformer.forward(embedded)
        
        # Mean pooling for sentence representation
        return np.mean(transformed, axis=0)
    
    def learn_from_text(
        self,
        text: str,
        source: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Learn from a piece of text.
        
        Args:
            text: Text to learn from
            source: Source identifier (URL, user, etc.)
            
        Returns:
            Learning statistics
        """
        with self._lock:
            # Learn vocabulary
            new_word_count, new_words = self.tokenizer.learn(text)
            
            # Expand embeddings if needed
            current_vocab = self.tokenizer.vocab_size
            if current_vocab > self.embeddings.token_embeddings.shape[0]:
                self.embeddings.expand_vocabulary(current_vocab)
            
            # Encode and store knowledge
            token_ids = self.tokenizer.encode(text)
            
            # Lightweight embedding update
            self._update_embeddings(token_ids)
            
            # Update training stats
            self.total_tokens_learned += len(token_ids)
            self.training_steps += 1
            
            return {
                'tokens_processed': len(token_ids),
                'new_words': new_word_count,
                'total_tokens': self.total_tokens_learned,
                'training_steps': self.training_steps,
                'vocab_size': self.tokenizer.vocab_size
            }
    
    def _update_embeddings(self, token_ids: List[int]) -> None:
        """Lightweight contextual embedding update"""
        window_size = 3
        
        for i, token_id in enumerate(token_ids):
            if token_id >= self.embeddings.token_embeddings.shape[0]:
                continue
            
            # Get context window
            start = max(0, i - window_size)
            end = min(len(token_ids), i + window_size + 1)
            context_ids = [
                tid for tid in token_ids[start:end]
                if tid != token_id and tid < self.embeddings.token_embeddings.shape[0]
            ]
            
            if not context_ids:
                continue
            
            # Compute context mean
            context_embeddings = [
                self.embeddings.get_token_embedding(tid)
                for tid in context_ids
            ]
            context_mean = np.mean(context_embeddings, axis=0)
            
            # Update embedding towards context
            current = self.embeddings.get_token_embedding(token_id)
            delta = context_mean - current
            self.embeddings.update_embedding(
                token_id,
                delta,
                self.learning_rate * 0.01
            )
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts"""
        emb1 = self.encode_text(text1)
        emb2 = self.encode_text(text2)
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    
    def save(self) -> Dict[str, Any]:
        """Save model state to persistent storage"""
        with self._lock:
            # Save transformer weights
            self.model_store.save_weights(
                self.transformer.get_weights()
            )
            
            # Save embeddings
            emb_state = self.embeddings.save_state()
            self.model_store.save_embeddings(
                emb_state['token_embeddings'],
                emb_state['position_embeddings']
            )
            
            # Save state
            self.model_store.save_state(
                total_tokens=self.total_tokens_learned,
                training_steps=self.training_steps,
                learning_rate=self.learning_rate,
                vocab_mapping=self.tokenizer.get_mapping(),
                metadata={
                    'saved_at': datetime.now().isoformat(),
                    'vocab_size': self.tokenizer.vocab_size
                }
            )
            
            return {
                'status': 'saved',
                'vocab_size': self.tokenizer.vocab_size,
                'total_tokens': self.total_tokens_learned,
                'training_steps': self.training_steps
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        return {
            'vocab_size': self.tokenizer.vocab_size,
            'total_tokens_learned': self.total_tokens_learned,
            'training_steps': self.training_steps,
            'embedding_dim': self.config.embedding_dim,
            'num_heads': self.config.num_heads,
            'num_layers': self.config.num_layers,
            'learning_rate': self.learning_rate
        }
