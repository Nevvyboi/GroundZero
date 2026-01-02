"""
Tokenizer
=========
Vocabulary management and text tokenization.
"""

import re
import threading
from typing import List, Dict, Optional, Tuple

from storage import MemoryStore


class Tokenizer:
    """
    Dynamic vocabulary tokenizer with persistence.
    Thread-safe for concurrent access.
    """
    
    # Special tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    START_TOKEN = "<START>"
    END_TOKEN = "<END>"
    
    SPECIAL_TOKENS = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
        START_TOKEN: 2,
        END_TOKEN: 3
    }
    
    def __init__(self, memory_store: MemoryStore, max_vocab_size: int = 50000):
        self.memory = memory_store
        self.max_vocab_size = max_vocab_size
        self._lock = threading.RLock()
        
        # In-memory cache for fast encoding
        self._word_to_id: Dict[str, int] = dict(self.SPECIAL_TOKENS)
        self._id_to_word: Dict[int, str] = {v: k for k, v in self.SPECIAL_TOKENS.items()}
        self._next_id = len(self.SPECIAL_TOKENS)
        
        # Load existing vocabulary from database
        self._load_vocabulary()
    
    def _load_vocabulary(self) -> None:
        """Load vocabulary from persistent storage"""
        # Get all words from database
        rows = self.memory.db.fetch_all(
            "SELECT id, word FROM vocabulary ORDER BY id"
        )
        
        with self._lock:
            for row in rows:
                word = row['word']
                word_id = row['id'] + len(self.SPECIAL_TOKENS)  # Offset for special tokens
                self._word_to_id[word] = word_id
                self._id_to_word[word_id] = word
                self._next_id = max(self._next_id, word_id + 1)
    
    def _preprocess(self, text: str) -> List[str]:
        """
        Clean and tokenize text into words.
        Handles punctuation as separate tokens.
        """
        text = text.lower().strip()
        
        # Split on whitespace and handle punctuation
        tokens = []
        current = ""
        
        for char in text:
            if char.isalnum() or char == "'":
                current += char
            else:
                if current:
                    tokens.append(current)
                    current = ""
                if char in ".,!?;:()-\"":
                    tokens.append(char)
        
        if current:
            tokens.append(current)
        
        return tokens
    
    def learn(self, text: str) -> Tuple[int, List[str]]:
        """
        Learn vocabulary from text.
        Returns (new_words_count, new_words_list).
        """
        tokens = self._preprocess(text)
        new_words = []
        
        with self._lock:
            for token in tokens:
                if not token or len(token) > 50:  # Skip empty or very long tokens
                    continue
                
                if token not in self._word_to_id:
                    if self._next_id < self.max_vocab_size:
                        # Add to memory cache
                        self._word_to_id[token] = self._next_id
                        self._id_to_word[self._next_id] = token
                        self._next_id += 1
                        new_words.append(token)
                
                # Always update frequency in database
                self.memory.add_word(token)
        
        return len(new_words), new_words
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        tokens = self._preprocess(text)
        
        with self._lock:
            return [
                self._word_to_id.get(token, self.SPECIAL_TOKENS[self.UNK_TOKEN])
                for token in tokens
            ]
    
    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text"""
        with self._lock:
            words = [
                self._id_to_word.get(i, self.UNK_TOKEN)
                for i in ids
            ]
        return " ".join(words)
    
    def get_id(self, word: str) -> int:
        """Get ID for a word"""
        with self._lock:
            return self._word_to_id.get(
                word.lower(),
                self.SPECIAL_TOKENS[self.UNK_TOKEN]
            )
    
    def get_word(self, word_id: int) -> str:
        """Get word for an ID"""
        with self._lock:
            return self._id_to_word.get(word_id, self.UNK_TOKEN)
    
    @property
    def vocab_size(self) -> int:
        """Current vocabulary size"""
        with self._lock:
            return len(self._word_to_id)
    
    def get_mapping(self) -> Dict[str, int]:
        """Get word to ID mapping (for persistence)"""
        with self._lock:
            return dict(self._word_to_id)
    
    def contains(self, word: str) -> bool:
        """Check if word is in vocabulary"""
        with self._lock:
            return word.lower() in self._word_to_id
