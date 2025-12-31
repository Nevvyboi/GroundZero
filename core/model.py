"""
NeuralMind - Custom AI Model from Scratch
A lightweight transformer-inspired architecture that learns from the internet
WITH: Reasoning Engine for logic, math, and code analysis
"""

import numpy as np
import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import threading
import queue

# Import reasoning engine
from .reasoning import ReasoningEngine, ReasoningType, ReasoningResult


class Tokenizer:
    """Custom tokenizer that learns vocabulary dynamically"""
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.word_to_id: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        self.id_to_word: Dict[int, str] = {0: "<PAD>", 1: "<UNK>", 2: "<START>", 3: "<END>"}
        self.word_freq: Dict[str, int] = {}
        self.next_id = 4
        
    def learn_vocabulary(self, text: str):
        """Learn new words from text"""
        words = self._preprocess(text)
        for word in words:
            if word not in self.word_freq:
                self.word_freq[word] = 0
            self.word_freq[word] += 1
            
            if word not in self.word_to_id and self.next_id < self.vocab_size:
                self.word_to_id[word] = self.next_id
                self.id_to_word[self.next_id] = word
                self.next_id += 1
                
    def _preprocess(self, text: str) -> List[str]:
        """Clean and tokenize text"""
        text = text.lower()
        # Simple word tokenization
        words = []
        current_word = ""
        for char in text:
            if char.isalnum():
                current_word += char
            else:
                if current_word:
                    words.append(current_word)
                    current_word = ""
                if char in ".,!?;:":
                    words.append(char)
        if current_word:
            words.append(current_word)
        return words
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        words = self._preprocess(text)
        return [self.word_to_id.get(w, 1) for w in words]  # 1 = <UNK>
    
    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text"""
        words = [self.id_to_word.get(i, "<UNK>") for i in ids]
        return " ".join(words)
    
    def vocab_count(self) -> int:
        return len(self.word_to_id)


class AttentionHead:
    """Single attention head implementation"""
    
    def __init__(self, d_model: int, d_head: int):
        self.d_head = d_head
        scale = np.sqrt(2.0 / (d_model + d_head))
        self.W_q = np.random.randn(d_model, d_head) * scale
        self.W_k = np.random.randn(d_model, d_head) * scale
        self.W_v = np.random.randn(d_model, d_head) * scale
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute attention: softmax(QK^T / sqrt(d_k)) * V"""
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        scores = Q @ K.T / np.sqrt(self.d_head)
        attention = self._softmax(scores)
        return attention @ V
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class MultiHeadAttention:
    """Multi-head attention layer"""
    
    def __init__(self, d_model: int, n_heads: int):
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.heads = [AttentionHead(d_model, self.d_head) for _ in range(n_heads)]
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        head_outputs = [head.forward(x) for head in self.heads]
        concat = np.concatenate(head_outputs, axis=-1)
        return concat @ self.W_o


class FeedForward:
    """Feed-forward network with ReLU activation"""
    
    def __init__(self, d_model: int, d_ff: int):
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros(d_model)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return hidden @ self.W2 + self.b2


class TransformerBlock:
    """Single transformer block with attention and feed-forward"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)
        
    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True) + 1e-6
        return gamma * (x - mean) / std + beta
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Self-attention with residual
        attn_out = self.attention.forward(x)
        x = self._layer_norm(x + attn_out, self.gamma1, self.beta1)
        
        # Feed-forward with residual
        ff_out = self.ff.forward(x)
        x = self._layer_norm(x + ff_out, self.gamma2, self.beta2)
        
        return x


class KnowledgeMemory:
    """Long-term memory storage for learned knowledge"""
    
    def __init__(self, memory_size: int = 100000):
        self.memory_size = memory_size
        self.memories: Dict[str, Dict] = {}
        self.context_embeddings: Dict[str, np.ndarray] = {}
        self.learning_history: List[Dict] = []
        
    def store(self, key: str, content: str, embedding: np.ndarray, source: str = "unknown"):
        """Store a piece of knowledge"""
        memory_id = hashlib.md5(key.encode()).hexdigest()[:16]
        self.memories[memory_id] = {
            "key": key,
            "content": content,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "access_count": 0
        }
        self.context_embeddings[memory_id] = embedding
        
        if len(self.memories) > self.memory_size:
            # Remove least accessed memories
            sorted_mems = sorted(self.memories.items(), 
                                 key=lambda x: x[1]["access_count"])
            to_remove = sorted_mems[0][0]
            del self.memories[to_remove]
            del self.context_embeddings[to_remove]
            
    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Retrieve most relevant memories using cosine similarity"""
        if not self.memories:
            return []
            
        similarities = []
        for mem_id, emb in self.context_embeddings.items():
            sim = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-8
            )
            similarities.append((mem_id, sim))
            
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for mem_id, sim in similarities[:top_k]:
            self.memories[mem_id]["access_count"] += 1
            results.append({**self.memories[mem_id], "similarity": float(sim)})
            
        return results
    
    def memory_count(self) -> int:
        return len(self.memories)


class NeuralMind:
    """
    The main AI model - a custom neural network that learns from the internet
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 512
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        
        # Initialize components
        self.tokenizer = Tokenizer(vocab_size)
        self.memory = KnowledgeMemory()
        self.reasoning = ReasoningEngine()  # Add reasoning capabilities
        
        # Embeddings
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.position_embedding = self._create_positional_encoding()
        
        # Transformer layers
        self.layers = [
            TransformerBlock(d_model, n_heads, d_ff) 
            for _ in range(n_layers)
        ]
        
        # Output projection
        self.output_proj = np.random.randn(d_model, vocab_size) * 0.02
        
        # Training state
        self.learning_rate = 0.001
        self.total_tokens_learned = 0
        self.training_steps = 0
        self.is_learning = False
        self.learning_queue = queue.Queue()
        
        # Stats
        self.stats = {
            "vocab_size": 4,
            "memory_size": 0,
            "tokens_learned": 0,
            "training_steps": 0,
            "sites_learned": 0,
            "last_update": None
        }
        
    def _create_positional_encoding(self) -> np.ndarray:
        """Create sinusoidal positional encodings"""
        pos = np.arange(self.max_seq_len)[:, np.newaxis]
        dim = np.arange(0, self.d_model, 2)
        
        pe = np.zeros((self.max_seq_len, self.d_model))
        pe[:, 0::2] = np.sin(pos / 10000 ** (dim / self.d_model))
        pe[:, 1::2] = np.cos(pos / 10000 ** (dim / self.d_model))
        
        return pe
    
    def embed(self, token_ids: List[int]) -> np.ndarray:
        """Convert token IDs to embeddings with positional encoding"""
        seq_len = min(len(token_ids), self.max_seq_len)
        token_ids = token_ids[:seq_len]
        
        # Get token embeddings (handle out-of-vocabulary tokens)
        safe_ids = [min(tid, self.vocab_size - 1) for tid in token_ids]
        tok_emb = self.token_embedding[safe_ids]
        pos_emb = self.position_embedding[:seq_len]
        
        return tok_emb + pos_emb
    
    def forward(self, text: str) -> np.ndarray:
        """Forward pass through the model"""
        token_ids = self.tokenizer.encode(text)
        if not token_ids:
            return np.zeros(self.d_model)
            
        x = self.embed(token_ids)
        
        for layer in self.layers:
            x = layer.forward(x)
            
        # Mean pooling for sentence embedding
        return np.mean(x, axis=0)
    
    def generate_response(self, prompt: str, max_length: int = 100) -> str:
        """Generate a response based on learned knowledge and reasoning"""
        # First, try reasoning engine for logic/math/code
        reasoning_result = self.reasoning.reason(prompt)
        
        # If reasoning produced a meaningful result, use it
        if reasoning_result.reasoning_type != ReasoningType.GENERAL and reasoning_result.success:
            if reasoning_result.final_answer:
                return reasoning_result.final_answer
        
        # Get prompt embedding for memory retrieval
        prompt_emb = self.forward(prompt)
        
        # Retrieve relevant memories
        relevant = self.memory.retrieve(prompt_emb, top_k=5)
        
        if relevant:
            # Combine context from memories
            context_parts = []
            for mem in relevant:
                if mem["similarity"] > 0.3:
                    context_parts.append(mem["content"])
            
            if context_parts:
                # Simple response generation based on context
                response = self._generate_from_context(prompt, context_parts)
                return response
        
        return self._generate_fallback(prompt)
    
    def generate_response_with_reasoning(self, prompt: str) -> Dict:
        """Generate response with full reasoning chain visible"""
        # Try reasoning engine first
        reasoning_result = self.reasoning.reason(prompt)
        
        response_data = {
            "response": "",
            "reasoning_type": reasoning_result.reasoning_type.value,
            "steps": [
                {
                    "step": s.step_num,
                    "description": s.description,
                    "operation": s.operation,
                    "result": str(s.result)
                } for s in reasoning_result.steps
            ],
            "used_reasoning": reasoning_result.reasoning_type != ReasoningType.GENERAL,
            "confidence": reasoning_result.confidence,
            "used_memory": False,
            "memory_sources": []
        }
        
        # If reasoning produced a result, use it
        if reasoning_result.reasoning_type != ReasoningType.GENERAL and reasoning_result.success and reasoning_result.final_answer:
            response_data["response"] = reasoning_result.final_answer
            return response_data
        
        # Fall back to memory retrieval
        prompt_emb = self.forward(prompt)
        relevant = self.memory.retrieve(prompt_emb, top_k=5)
        
        if relevant:
            context_parts = []
            for mem in relevant:
                if mem["similarity"] > 0.3:
                    context_parts.append(mem["content"])
                    response_data["memory_sources"].append({
                        "source": mem.get("source", "unknown"),
                        "similarity": mem["similarity"]
                    })
            
            if context_parts:
                response_data["response"] = self._generate_from_context(prompt, context_parts)
                response_data["used_memory"] = True
                return response_data
        
        response_data["response"] = self._generate_fallback(prompt)
        return response_data
    
    def _generate_from_context(self, prompt: str, contexts: List[str]) -> str:
        """Generate response using retrieved context"""
        # Find most relevant sentences from context
        prompt_lower = prompt.lower()
        prompt_words = set(self.tokenizer._preprocess(prompt_lower))
        
        best_sentences = []
        for ctx in contexts:
            sentences = ctx.replace("\n", " ").split(".")
            for sent in sentences:
                if len(sent.strip()) > 20:
                    sent_words = set(self.tokenizer._preprocess(sent.lower()))
                    overlap = len(prompt_words & sent_words)
                    if overlap > 0:
                        best_sentences.append((overlap, sent.strip()))
        
        best_sentences.sort(reverse=True)
        
        if best_sentences:
            # Construct response from top sentences
            response_parts = []
            for _, sent in best_sentences[:3]:
                if sent and sent not in response_parts:
                    response_parts.append(sent)
            
            if response_parts:
                return ". ".join(response_parts) + "."
        
        return self._generate_fallback(prompt)
    
    def _generate_fallback(self, prompt: str) -> str:
        """Fallback response when no relevant memory found"""
        responses = [
            "I'm still learning about that topic. Could you teach me more?",
            "I haven't learned enough about this yet. Want to help me learn?",
            "This is new to me! I'd love to learn more about it.",
            "My knowledge on this is limited. Shall I search for more information?",
            "I need more training on this topic. Can you provide some information?"
        ]
        return responses[hash(prompt) % len(responses)]
    
    def learn_from_text(self, text: str, source: str = "user") -> Dict:
        """Learn from a piece of text"""
        # Learn vocabulary
        self.tokenizer.learn_vocabulary(text)
        
        # Create embedding
        embedding = self.forward(text)
        
        # Store in memory
        key = text[:100]  # Use first 100 chars as key
        self.memory.store(key, text, embedding, source)
        
        # Update token embeddings for new vocabulary
        if self.tokenizer.vocab_count() > self.token_embedding.shape[0]:
            new_size = self.tokenizer.vocab_count()
            new_embeddings = np.random.randn(new_size, self.d_model) * 0.02
            new_embeddings[:self.token_embedding.shape[0]] = self.token_embedding
            self.token_embedding = new_embeddings
        
        # Simulated gradient update (lightweight training)
        self._lightweight_update(text)
        
        self.training_steps += 1
        self.total_tokens_learned += len(self.tokenizer.encode(text))
        
        # Update stats
        self.stats.update({
            "vocab_size": self.tokenizer.vocab_count(),
            "memory_size": self.memory.memory_count(),
            "tokens_learned": self.total_tokens_learned,
            "training_steps": self.training_steps,
            "last_update": datetime.now().isoformat()
        })
        
        return self.stats.copy()
    
    def _lightweight_update(self, text: str):
        """Lightweight gradient-like update to embeddings"""
        tokens = self.tokenizer.encode(text)
        
        for i, token_id in enumerate(tokens[:self.max_seq_len]):
            if token_id < self.token_embedding.shape[0]:
                # Context-aware update
                context_ids = tokens[max(0, i-3):i+4]
                context_mean = np.mean([
                    self.token_embedding[min(tid, self.vocab_size-1)] 
                    for tid in context_ids
                ], axis=0)
                
                # Gentle update towards context
                update = self.learning_rate * (context_mean - self.token_embedding[token_id])
                self.token_embedding[token_id] += update * 0.01
    
    def correct_and_learn(self, wrong_response: str, correct_info: str, source: str = "correction"):
        """Learn from corrections"""
        # Store the correction with high importance
        embedding = self.forward(correct_info)
        self.memory.store(wrong_response, correct_info, embedding, source)
        
        # Additional learning from the correction
        self.learn_from_text(correct_info, source)
        
        return {"status": "learned", "message": "Thank you for the correction!"}
    
    def save(self, path: str):
        """Save model to disk"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_state = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "stats": self.stats,
            "total_tokens_learned": self.total_tokens_learned,
            "training_steps": self.training_steps
        }
        
        with open(save_path / "model_config.json", "w") as f:
            json.dump(model_state, f, indent=2)
            
        # Save tokenizer
        tokenizer_state = {
            "word_to_id": self.tokenizer.word_to_id,
            "id_to_word": {str(k): v for k, v in self.tokenizer.id_to_word.items()},
            "word_freq": self.tokenizer.word_freq,
            "next_id": self.tokenizer.next_id
        }
        with open(save_path / "tokenizer.json", "w") as f:
            json.dump(tokenizer_state, f)
            
        # Save embeddings and weights
        np.save(save_path / "token_embedding.npy", self.token_embedding)
        
        # Save memories
        memory_state = {
            "memories": self.memory.memories,
            "learning_history": self.memory.learning_history
        }
        with open(save_path / "memory.json", "w") as f:
            json.dump(memory_state, f)
            
        # Save context embeddings
        np.save(save_path / "context_embeddings.npy", 
                dict(self.memory.context_embeddings))
                
        return {"status": "saved", "path": str(save_path)}
    
    @classmethod
    def load(cls, path: str) -> "NeuralMind":
        """Load model from disk"""
        load_path = Path(path)
        
        if not load_path.exists():
            return cls()  # Return new model if no save exists
            
        try:
            # Load config
            with open(load_path / "model_config.json", "r") as f:
                config = json.load(f)
                
            model = cls(
                vocab_size=config["vocab_size"],
                d_model=config["d_model"],
                n_heads=config["n_heads"],
                n_layers=config["n_layers"],
                d_ff=config["d_ff"],
                max_seq_len=config["max_seq_len"]
            )
            
            model.stats = config.get("stats", model.stats)
            model.total_tokens_learned = config.get("total_tokens_learned", 0)
            model.training_steps = config.get("training_steps", 0)
            
            # Load tokenizer
            if (load_path / "tokenizer.json").exists():
                with open(load_path / "tokenizer.json", "r") as f:
                    tok_state = json.load(f)
                model.tokenizer.word_to_id = tok_state["word_to_id"]
                model.tokenizer.id_to_word = {int(k): v for k, v in tok_state["id_to_word"].items()}
                model.tokenizer.word_freq = tok_state["word_freq"]
                model.tokenizer.next_id = tok_state["next_id"]
                
            # Load embeddings
            if (load_path / "token_embedding.npy").exists():
                model.token_embedding = np.load(load_path / "token_embedding.npy")
                
            # Load memories
            if (load_path / "memory.json").exists():
                with open(load_path / "memory.json", "r") as f:
                    mem_state = json.load(f)
                model.memory.memories = mem_state["memories"]
                model.memory.learning_history = mem_state.get("learning_history", [])
                
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return cls()
    
    def get_stats(self) -> Dict:
        """Get current model statistics"""
        return {
            **self.stats,
            "vocab_size": self.tokenizer.vocab_count(),
            "memory_size": self.memory.memory_count(),
            "is_learning": self.is_learning,
            "model_params": {
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers,
                "embedding_shape": self.token_embedding.shape
            }
        }