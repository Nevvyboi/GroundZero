"""
Model Store
===========
Binary storage for neural network weights and embeddings.
Uses NumPy's compressed format for efficient storage.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import threading


class ModelStore:
    """Binary storage for model weights and state"""
    
    def __init__(
        self,
        weights_path: Path,
        embeddings_path: Path,
        state_path: Path
    ):
        self.weights_path = weights_path
        self.embeddings_path = embeddings_path
        self.state_path = state_path
        self._lock = threading.Lock()
    
    # === Weight Operations ===
    
    def save_weights(self, weights: Dict[str, Any]) -> None:
        """Save model weights using pickle (handles nested dicts)"""
        import pickle
        with self._lock:
            with open(self.weights_path, 'wb') as f:
                pickle.dump(weights, f)
    
    def load_weights(self) -> Optional[Dict[str, Any]]:
        """Load model weights from pickle file"""
        if not self.weights_path.exists():
            return None
        
        with self._lock:
            try:
                import pickle
                with open(self.weights_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading weights: {e}")
                # Delete corrupted file and try old format
                try:
                    # Try numpy format (old files)
                    data = np.load(self.weights_path, allow_pickle=True)
                    return {key: data[key] for key in data.files}
                except:
                    pass
                try:
                    self.weights_path.unlink()
                    print("Deleted corrupted weights file, starting fresh.")
                except:
                    pass
                return None
    
    def weights_exist(self) -> bool:
        """Check if weights file exists"""
        return self.weights_path.exists()
    
    # === Embedding Operations ===
    
    def save_embeddings(
        self,
        token_embeddings: np.ndarray,
        position_embeddings: np.ndarray,
        context_embeddings: Optional[Dict[str, np.ndarray]] = None
    ) -> None:
        """Save all embeddings to compressed binary"""
        with self._lock:
            data = {
                'token_embeddings': token_embeddings,
                'position_embeddings': position_embeddings,
            }
            
            if context_embeddings:
                # Store context embeddings with string keys
                for key, emb in context_embeddings.items():
                    data[f'ctx_{key}'] = emb
            
            np.savez_compressed(self.embeddings_path, **data)
    
    def load_embeddings(self) -> Optional[Dict[str, Any]]:
        """Load embeddings from binary"""
        if not self.embeddings_path.exists():
            return None
        
        with self._lock:
            try:
                # Try loading with allow_pickle=True for compatibility
                data = np.load(self.embeddings_path, allow_pickle=True)
                
                result = {
                    'token_embeddings': data['token_embeddings'],
                    'position_embeddings': data['position_embeddings'],
                    'context_embeddings': {}
                }
                
                # Extract context embeddings
                for key in data.files:
                    if key.startswith('ctx_'):
                        ctx_key = key[4:]  # Remove 'ctx_' prefix
                        result['context_embeddings'][ctx_key] = data[key]
                
                return result
            except Exception as e:
                print(f"Error loading embeddings: {e}")
                # Delete corrupted file
                try:
                    self.embeddings_path.unlink()
                    print("Deleted corrupted embeddings file, starting fresh.")
                except:
                    pass
                return None
    
    def embeddings_exist(self) -> bool:
        """Check if embeddings file exists"""
        return self.embeddings_path.exists()
    
    # === State Operations ===
    
    def save_state(
        self,
        total_tokens: int,
        training_steps: int,
        learning_rate: float,
        vocab_mapping: Dict[str, int],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save model state to compressed binary"""
        with self._lock:
            # Convert vocab mapping to arrays for efficient storage
            words = np.array(list(vocab_mapping.keys()), dtype=object)
            ids = np.array(list(vocab_mapping.values()), dtype=np.int32)
            
            data = {
                'total_tokens': np.array([total_tokens]),
                'training_steps': np.array([training_steps]),
                'learning_rate': np.array([learning_rate]),
                'vocab_words': words,
                'vocab_ids': ids,
                'saved_at': np.array([datetime.now().isoformat()], dtype=object)
            }
            
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (int, float)):
                        data[f'meta_{key}'] = np.array([value])
                    elif isinstance(value, str):
                        data[f'meta_{key}'] = np.array([value], dtype=object)
            
            np.savez_compressed(self.state_path, **data)
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """Load model state from binary"""
        if not self.state_path.exists():
            return None
        
        with self._lock:
            try:
                data = np.load(self.state_path, allow_pickle=True)
                
                # Reconstruct vocab mapping
                words = data['vocab_words']
                ids = data['vocab_ids']
                vocab_mapping = dict(zip(words, ids))
                
                result = {
                    'total_tokens': int(data['total_tokens'][0]),
                    'training_steps': int(data['training_steps'][0]),
                    'learning_rate': float(data['learning_rate'][0]),
                    'vocab_mapping': vocab_mapping,
                    'saved_at': str(data['saved_at'][0]) if 'saved_at' in data.files else None,
                    'metadata': {}
                }
                
                # Extract metadata
                for key in data.files:
                    if key.startswith('meta_'):
                        meta_key = key[5:]  # Remove 'meta_' prefix
                        result['metadata'][meta_key] = data[key][0]
                
                return result
            except Exception as e:
                print(f"Error loading state: {e}")
                # Delete corrupted file
                try:
                    self.state_path.unlink()
                    print("Deleted corrupted state file, starting fresh.")
                except:
                    pass
                return None
    
    def state_exists(self) -> bool:
        """Check if state file exists"""
        return self.state_path.exists()
    
    # === Utilities ===
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about stored data"""
        info = {
            'weights_exists': self.weights_exist(),
            'embeddings_exists': self.embeddings_exist(),
            'state_exists': self.state_exists(),
            'weights_size_mb': 0,
            'embeddings_size_mb': 0,
            'state_size_mb': 0
        }
        
        if self.weights_path.exists():
            info['weights_size_mb'] = self.weights_path.stat().st_size / (1024 * 1024)
        if self.embeddings_path.exists():
            info['embeddings_size_mb'] = self.embeddings_path.stat().st_size / (1024 * 1024)
        if self.state_path.exists():
            info['state_size_mb'] = self.state_path.stat().st_size / (1024 * 1024)
        
        info['total_size_mb'] = (
            info['weights_size_mb'] + 
            info['embeddings_size_mb'] + 
            info['state_size_mb']
        )
        
        return info
    
    def clear_all(self) -> None:
        """Delete all stored data"""
        with self._lock:
            for path in [self.weights_path, self.embeddings_path, self.state_path]:
                if path.exists():
                    path.unlink()