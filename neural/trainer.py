"""
GroundZero Neural Trainer
=========================
Continual learning trainer for the transformer model.

Key features:
- Incremental training (learns from new data)
- Replay buffer (prevents catastrophic forgetting)
- Elastic Weight Consolidation (protects important weights)
- Automatic checkpointing
- Background training thread

This is what makes the model actually LEARN.
"""

import os
import time
import random
import threading
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from pathlib import Path
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from .transformer import GroundZeroTransformer, TransformerConfig
from .tokenizer import BPETokenizer


@dataclass
class TrainerConfig:
    """Training configuration"""
    # Model
    model_size: str = "small"  # tiny, small, medium, large, xl
    
    # Training
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    
    # Continual learning
    replay_buffer_size: int = 10000  # Number of samples to remember
    replay_ratio: float = 0.3  # Fraction of batch from replay buffer
    ewc_lambda: float = 100.0  # Elastic weight consolidation strength
    
    # Checkpointing
    save_every_steps: int = 1000
    checkpoint_dir: Path = Path("checkpoints")
    
    # Hardware
    device: str = "auto"  # auto, cpu, cuda, mps


class TextDataset(Dataset):
    """Dataset for text sequences with proper padding"""
    
    def __init__(self, token_ids: List[List[int]], max_seq_len: int = 512, pad_id: int = 0):
        self.sequences = []
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        
        # Split into chunks of max_seq_len
        for ids in token_ids:
            for i in range(0, len(ids) - 1, max_seq_len):
                chunk = ids[i:i + max_seq_len + 1]  # +1 for labels
                if len(chunk) > 10:  # Skip very short sequences
                    self.sequences.append(chunk)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Pad sequence to max_seq_len + 1
        if len(seq) < self.max_seq_len + 1:
            seq = seq + [self.pad_id] * (self.max_seq_len + 1 - len(seq))
        
        # Input and target are same sequence (autoregressive)
        return torch.tensor(seq[:-1]), torch.tensor(seq[1:])


class ReplayBuffer:
    """
    Experience replay buffer for continual learning.
    
    Stores past training examples to mix with new data,
    preventing catastrophic forgetting.
    """
    
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add(self, sequences: List[List[int]]):
        """Add sequences to buffer"""
        for seq in sequences:
            self.buffer.append(seq)
    
    def sample(self, n: int) -> List[List[int]]:
        """Sample n sequences from buffer"""
        if len(self.buffer) == 0:
            return []
        n = min(n, len(self.buffer))
        return random.sample(list(self.buffer), n)
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, path: Path):
        """Save buffer to disk"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)
    
    def load(self, path: Path):
        """Load buffer from disk"""
        import pickle
        if path.exists():
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.buffer = deque(data, maxlen=self.max_size)


class ElasticWeightConsolidation:
    """
    Elastic Weight Consolidation (EWC)
    
    Protects important weights from being overwritten when learning new tasks.
    Based on: "Overcoming catastrophic forgetting in neural networks" (Kirkpatrick et al.)
    
    The idea: Some weights are more important than others for previous tasks.
    EWC adds a penalty to the loss when those weights change too much.
    """
    
    def __init__(self, model: nn.Module, lambda_: float = 100.0):
        self.model = model
        self.lambda_ = lambda_
        
        # Store optimal parameters and Fisher information
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self.fisher_info: Dict[str, torch.Tensor] = {}
        
        self.is_initialized = False
    
    def compute_fisher(self, dataloader: DataLoader, num_samples: int = 500):
        """
        Compute Fisher information matrix (diagonal approximation).
        
        Fisher information tells us which weights are important for the current task.
        """
        self.model.eval()
        fisher_info = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        
        samples_seen = 0
        
        for inputs, targets in dataloader:
            if samples_seen >= num_samples:
                break
            
            self.model.zero_grad()
            outputs = self.model(inputs, labels=targets)
            loss = outputs['loss']
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher_info[name] += param.grad.pow(2)
            
            samples_seen += inputs.size(0)
        
        # Normalize
        for name in fisher_info:
            fisher_info[name] /= samples_seen
        
        # Store current parameters and Fisher info
        if self.is_initialized:
            # Combine with previous Fisher info (online EWC)
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.fisher_info[name] = 0.5 * (self.fisher_info[name] + fisher_info[name])
                    self.optimal_params[name] = param.data.clone()
        else:
            self.fisher_info = fisher_info
            self.optimal_params = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}
            self.is_initialized = True
        
        print(f"📊 Fisher information computed over {samples_seen} samples")
    
    def penalty(self) -> torch.Tensor:
        """Compute EWC penalty"""
        if not self.is_initialized:
            return torch.tensor(0.0)
        
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_info:
                loss += (self.fisher_info[name] * (param - self.optimal_params[name]).pow(2)).sum()
        
        return self.lambda_ * loss


class NeuralTrainer:
    """
    The main trainer for the GroundZero neural network.
    
    This handles:
    - Training the transformer model
    - Continual learning (learning without forgetting)
    - Saving/loading checkpoints
    - Background training
    """
    
    def __init__(self, config: TrainerConfig, data_dir: Path):
        self.config = config
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine device
        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)
        
        print(f"🖥️  Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer_path = self.data_dir / "tokenizer.json"
        if self.tokenizer_path.exists():
            self.tokenizer = BPETokenizer.load(self.tokenizer_path)
        else:
            self.tokenizer = BPETokenizer(vocab_size=32000)
        
        # Initialize model
        self.model_path = self.data_dir / "model.pt"
        if self.model_path.exists():
            self.model = GroundZeroTransformer.load(self.model_path, self.device)  # weights_only=False in load method
        else:
            model_config = self._get_model_config()
            self.model = GroundZeroTransformer(model_config)
        
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Replay buffer for continual learning
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
        replay_path = self.data_dir / "replay_buffer.pkl"
        if replay_path.exists():
            self.replay_buffer.load(replay_path)
        
        # EWC for preventing catastrophic forgetting
        self.ewc = ElasticWeightConsolidation(self.model, config.ewc_lambda)
        
        # Training state
        self.global_step = 0
        self.total_tokens_trained = 0
        self.training_history = []
        
        # Background training
        self._training_thread: Optional[threading.Thread] = None
        self._stop_training = threading.Event()
        self._training_queue = deque()
        self._is_training = False
        
        # Load state if exists
        self._load_state()
        
        print(f"✅ Neural Trainer initialized")
        print(f"   Model: {self.model.n_params:,} parameters")
        print(f"   Tokenizer: {len(self.tokenizer)} tokens")
        print(f"   Replay buffer: {len(self.replay_buffer)} samples")
    
    def _get_model_config(self) -> TransformerConfig:
        """Get model config based on size"""
        configs = {
            'tiny': TransformerConfig.tiny(),
            'small': TransformerConfig.small(),
            'medium': TransformerConfig.medium(),
            'large': TransformerConfig.large(),
            'xl': TransformerConfig.xl()
        }
        config = configs.get(self.config.model_size, TransformerConfig.small())
        config.vocab_size = self.tokenizer.vocab_size
        return config
    
    def _load_state(self):
        """Load training state"""
        state_path = self.data_dir / "trainer_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location='cpu', weights_only=False)
            self.global_step = state.get('global_step', 0)
            self.total_tokens_trained = state.get('total_tokens_trained', 0)
            self.training_history = state.get('training_history', [])
            if 'optimizer' in state:
                self.optimizer.load_state_dict(state['optimizer'])
            print(f"📂 Loaded training state: step {self.global_step}, {self.total_tokens_trained:,} tokens")
    
    def _save_state(self):
        """Save training state"""
        state = {
            'global_step': self.global_step,
            'total_tokens_trained': self.total_tokens_trained,
            'training_history': self.training_history[-1000:],  # Keep last 1000
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, self.data_dir / "trainer_state.pt")
    
    def train_tokenizer(self, texts: List[str]):
        """Train or expand the tokenizer"""
        if not self.tokenizer.is_trained:
            self.tokenizer.train(texts)
            # Resize model embedding
            self._resize_model_embedding()
        else:
            old_vocab_size = len(self.tokenizer)
            self.tokenizer.expand_vocabulary(texts)
            if len(self.tokenizer) > old_vocab_size:
                self._resize_model_embedding()
        
        self.tokenizer.save(self.tokenizer_path)
    
    def _resize_model_embedding(self):
        """Resize model embedding layer if vocabulary changed"""
        new_vocab_size = len(self.tokenizer)
        old_vocab_size = self.model.config.vocab_size
        
        if new_vocab_size != old_vocab_size:
            print(f"📐 Resizing embeddings: {old_vocab_size} -> {new_vocab_size}")
            
            # Create new embeddings
            old_embedding = self.model.token_embedding
            new_embedding = nn.Embedding(new_vocab_size, self.model.config.d_model)
            
            # Copy old weights
            min_vocab = min(old_vocab_size, new_vocab_size)
            new_embedding.weight.data[:min_vocab] = old_embedding.weight.data[:min_vocab]
            
            # Initialize new tokens
            if new_vocab_size > old_vocab_size:
                nn.init.normal_(new_embedding.weight.data[old_vocab_size:], mean=0.0, std=0.02)
            
            # Replace
            self.model.token_embedding = new_embedding.to(self.device)
            self.model.lm_head = nn.Linear(self.model.config.d_model, new_vocab_size, bias=False).to(self.device)
            self.model.lm_head.weight = self.model.token_embedding.weight
            self.model.config.vocab_size = new_vocab_size
            
            # Update optimizer
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
    
    def train_on_texts(self, texts: List[str], epochs: int = 1, verbose: bool = True) -> Dict[str, Any]:
        """
        Train the model on a list of texts.
        
        This is the main training method. It:
        1. Tokenizes the texts
        2. Mixes with replay buffer
        3. Trains with EWC regularization
        4. Updates replay buffer
        
        Args:
            texts: List of text strings
            epochs: Number of epochs
            verbose: Print progress
        
        Returns:
            Training statistics
        """
        if not texts:
            return {'error': 'No texts provided'}
        
        # Ensure tokenizer is trained
        if not self.tokenizer.is_trained:
            self.train_tokenizer(texts)
        
        # Tokenize
        token_sequences = [self.tokenizer.encode(text) for text in texts]
        
        # Add to replay buffer
        self.replay_buffer.add(token_sequences)
        
        # Create dataset with replay
        all_sequences = token_sequences.copy()
        
        # Add replay samples
        replay_count = int(len(texts) * self.config.replay_ratio)
        if replay_count > 0 and len(self.replay_buffer) > 0:
            replay_samples = self.replay_buffer.sample(replay_count)
            all_sequences.extend(replay_samples)
        
        # Create dataset
        dataset = TextDataset(all_sequences, self.model.config.max_seq_len)
        
        if len(dataset) == 0:
            return {'error': 'No valid sequences after tokenization'}
        
        # Custom collate to handle any remaining size differences
        def collate_fn(batch):
            inputs, targets = zip(*batch)
            # Stack - should work now with padding
            return torch.stack(inputs), torch.stack(targets)
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn
        )
        
        # Training loop
        self.model.train()
        total_loss = 0
        total_steps = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_steps = 0
            
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs, labels=targets)
                loss = outputs['loss']
                
                # Add EWC penalty
                if self.ewc.is_initialized:
                    loss = loss + self.ewc.penalty()
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Update
                self.optimizer.step()
                
                # Stats
                epoch_loss += loss.item()
                epoch_steps += 1
                self.global_step += 1
                self.total_tokens_trained += inputs.numel()
                
                total_loss += loss.item()
                total_steps += 1
                
                # Save checkpoint
                if self.global_step % self.config.save_every_steps == 0:
                    self.save_checkpoint()
            
            if verbose:
                avg_loss = epoch_loss / max(epoch_steps, 1)
                print(f"   Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
        
        # Update EWC after training
        self.ewc.compute_fisher(dataloader)
        
        # Save
        self.save_checkpoint()
        
        # Stats
        elapsed = time.time() - start_time
        avg_loss = total_loss / max(total_steps, 1)
        
        stats = {
            'loss': avg_loss,
            'steps': total_steps,
            'tokens_trained': self.total_tokens_trained,
            'elapsed_seconds': elapsed,
            'tokens_per_second': (total_steps * self.config.batch_size * self.model.config.max_seq_len) / elapsed
        }
        
        self.training_history.append({
            'step': self.global_step,
            'loss': avg_loss,
            'timestamp': time.time()
        })
        
        if verbose:
            print(f"✅ Training complete: {stats['tokens_per_second']:.0f} tokens/sec")
        
        return stats
    
    def save_checkpoint(self):
        """Save model and training state"""
        self.model.save(self.model_path)
        self.tokenizer.save(self.tokenizer_path)
        self.replay_buffer.save(self.data_dir / "replay_buffer.pkl")
        self._save_state()
    
    def generate(self, prompt: str, max_tokens: int = 100, 
                 temperature: float = 0.8, top_k: int = 50) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Starting text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
        
        Returns:
            Generated text
        """
        if not self.tokenizer.is_trained:
            return "Error: Tokenizer not trained. Please train on some data first."
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        # Generate
        self.model.eval()
        with torch.no_grad():
            output_ids = self.model.generate(
                input_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k
            )
        
        # Decode
        generated = self.tokenizer.decode(output_ids[0].tolist())
        
        return generated
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'global_step': self.global_step,
            'total_tokens_trained': self.total_tokens_trained,
            'model_params': self.model.n_params,
            'vocab_size': len(self.tokenizer),
            'replay_buffer_size': len(self.replay_buffer),
            'device': str(self.device),
            'is_training': self._is_training,
            'recent_losses': [h['loss'] for h in self.training_history[-10:]]
        }
    
    # ============ Background Training ============
    
    def queue_training(self, texts: List[str]):
        """Add texts to training queue (for background training)"""
        self._training_queue.extend(texts)
    
    def start_background_training(self):
        """Start background training thread"""
        if self._training_thread is not None and self._training_thread.is_alive():
            print("⚠️ Background training already running")
            return
        
        self._stop_training.clear()
        self._is_training = True
        self._training_thread = threading.Thread(target=self._background_training_loop, daemon=True)
        self._training_thread.start()
        print("🚀 Background training started")
    
    def stop_background_training(self):
        """Stop background training"""
        self._stop_training.set()
        self._is_training = False
        if self._training_thread is not None:
            self._training_thread.join(timeout=5)
        print("⏹️ Background training stopped")
    
    def _background_training_loop(self):
        """Background training loop"""
        batch_texts = []
        
        while not self._stop_training.is_set():
            # Collect batch
            while len(batch_texts) < 10 and self._training_queue:
                batch_texts.append(self._training_queue.popleft())
            
            # Train on batch
            if batch_texts:
                try:
                    self.train_on_texts(batch_texts, epochs=1, verbose=False)
                except Exception as e:
                    print(f"⚠️ Training error: {e}")
                batch_texts = []
            else:
                # Wait for more data
                time.sleep(1)


def test_trainer():
    """Test the trainer"""
    import tempfile
    
    print("=" * 60)
    print("🧪 Testing Neural Trainer")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainerConfig(model_size="tiny", batch_size=2)
        trainer = NeuralTrainer(config, Path(tmpdir))
        
        # Sample texts
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming the world of technology.",
            "Python is a popular programming language for data science.",
            "Neural networks can learn complex patterns from data.",
            "The transformer architecture revolutionized natural language processing.",
        ] * 5
        
        # Train
        print("\n📚 Training on sample texts...")
        stats = trainer.train_on_texts(texts, epochs=2)
        print(f"   Stats: {stats}")
        
        # Generate
        print("\n🔮 Generating text...")
        generated = trainer.generate("The quick", max_tokens=30)
        print(f"   Generated: {generated}")
        
        # Stats
        print(f"\n📊 Trainer stats: {trainer.get_stats()}")
    
    print("\n✅ Trainer test passed!")


if __name__ == "__main__":
    test_trainer()
