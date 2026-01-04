import numpy as np
import re
import math
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import pickle
from pathlib import Path


class EmbeddingEngine:
    """
    Creates vector embeddings from text.
    
    Embeddings are numerical representations that capture semantic meaning.
    Similar texts will have similar vectors (close in vector space).
    
    Architecture:
    1. Tokenize text into words
    2. Build vocabulary from corpus
    3. Create TF-IDF vectors
    4. Reduce dimensions with SVD (optional)
    
    Dimension: 256 (configurable)
    - Higher dimensions = more precision but more memory
    - 256-512 is good for most use cases (per the documents)
    """
    
    def __init__(self, dimension: int = 256, vocab_size: int = 10000):
        self.dimension = dimension
        self.vocab_size = vocab_size
        
        # Vocabulary
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.word_frequencies: Counter = Counter()
        self.document_frequencies: Counter = Counter()
        self.total_documents = 0
        
        # IDF cache
        self.idf_cache: Dict[str, float] = {}
        
        # Projection matrix for dimension reduction
        self.projection_matrix: Optional[np.ndarray] = None
        
        # Common stopwords to ignore
        self.stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but',
            'if', 'or', 'because', 'until', 'while', 'this', 'that', 'these',
            'those', 'what', 'which', 'who', 'whom', 'its', 'it', 'he', 'she',
            'they', 'them', 'his', 'her', 'their', 'my', 'your', 'our', 'i', 'you',
            'we', 'me', 'him', 'us', 'also', 'however', 'therefore', 'thus'
        }
        
        self._initialized = False
    
    def tokenize(self, text: str) -> List[str]:
        """Convert text to list of tokens"""
        # Lowercase and extract words
        text = text.lower()
        words = re.findall(r'\b[a-z]{2,}\b', text)
        
        # Remove stopwords
        words = [w for w in words if w not in self.stopwords]
        
        return words
    
    def add_document(self, text: str) -> None:
        """Add document to build vocabulary"""
        tokens = self.tokenize(text)
        
        # Update word frequencies
        self.word_frequencies.update(tokens)
        
        # Update document frequencies (unique words per doc)
        unique_tokens = set(tokens)
        self.document_frequencies.update(unique_tokens)
        
        self.total_documents += 1
    
    def build_vocabulary(self) -> None:
        """Build vocabulary from collected documents"""
        # Get most common words
        most_common = self.word_frequencies.most_common(self.vocab_size)
        
        self.word_to_idx = {word: idx for idx, (word, _) in enumerate(most_common)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        # Calculate IDF for each word
        for word in self.word_to_idx:
            df = self.document_frequencies.get(word, 1)
            self.idf_cache[word] = math.log((self.total_documents + 1) / (df + 1)) + 1
        
        # Create random projection matrix for dimension reduction
        # This is a simple but effective method (similar to LSH concept)
        vocab_len = len(self.word_to_idx)
        if vocab_len > self.dimension:
            np.random.seed(42)  # Reproducible
            self.projection_matrix = np.random.randn(vocab_len, self.dimension)
            self.projection_matrix /= np.linalg.norm(self.projection_matrix, axis=1, keepdims=True)
        
        self._initialized = True
        print(f"✅ Embedding engine initialized: {len(self.word_to_idx)} words, {self.dimension}D vectors")
    
    def embed(self, text: str) -> np.ndarray:
        """
        Convert text to vector embedding.
        
        Process:
        1. Tokenize text
        2. Calculate TF-IDF weights
        3. Create sparse vector
        4. Project to lower dimension
        5. Normalize (for cosine similarity)
        """
        if not self._initialized:
            # Return zero vector if not initialized
            return np.zeros(self.dimension, dtype=np.float32)
        
        tokens = self.tokenize(text)
        
        if not tokens:
            return np.zeros(self.dimension, dtype=np.float32)
        
        # Calculate TF (term frequency)
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        
        # Create TF-IDF vector
        vocab_size = len(self.word_to_idx)
        tfidf_vector = np.zeros(vocab_size, dtype=np.float32)
        
        for token, count in token_counts.items():
            if token in self.word_to_idx:
                idx = self.word_to_idx[token]
                tf = count / total_tokens
                idf = self.idf_cache.get(token, 1.0)
                tfidf_vector[idx] = tf * idf
        
        # Project to lower dimension if needed
        if self.projection_matrix is not None:
            vector = tfidf_vector @ self.projection_matrix
        else:
            # Pad or truncate to target dimension
            if len(tfidf_vector) >= self.dimension:
                vector = tfidf_vector[:self.dimension]
            else:
                vector = np.pad(tfidf_vector, (0, self.dimension - len(tfidf_vector)))
        
        # Normalize for cosine similarity
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector.astype(np.float32)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts at once"""
        embeddings = [self.embed(text) for text in texts]
        return np.array(embeddings, dtype=np.float32)
    
    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Cosine similarity measures the angle between vectors:
        - 1.0 = identical direction (same meaning)
        - 0.0 = orthogonal (unrelated)
        - -1.0 = opposite (opposite meaning)
        """
        # Vectors should already be normalized, but ensure
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def save(self, path: Path) -> None:
        """Save embedding engine state"""
        state = {
            'dimension': self.dimension,
            'vocab_size': self.vocab_size,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'word_frequencies': dict(self.word_frequencies),
            'document_frequencies': dict(self.document_frequencies),
            'total_documents': self.total_documents,
            'idf_cache': self.idf_cache,
            'projection_matrix': self.projection_matrix,
            'initialized': self._initialized
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: Path) -> bool:
        """Load embedding engine state"""
        if not path.exists():
            return False
        
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            
            self.dimension = state['dimension']
            self.vocab_size = state['vocab_size']
            self.word_to_idx = state['word_to_idx']
            self.idx_to_word = state['idx_to_word']
            self.word_frequencies = Counter(state['word_frequencies'])
            self.document_frequencies = Counter(state['document_frequencies'])
            self.total_documents = state['total_documents']
            self.idf_cache = state['idf_cache']
            self.projection_matrix = state['projection_matrix']
            self._initialized = state['initialized']
            
            return True
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return False
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding engine statistics"""
        return {
            'vocabulary_size': len(self.word_to_idx),
            'dimension': self.dimension,
            'total_documents': self.total_documents,
            'initialized': self._initialized
        }
