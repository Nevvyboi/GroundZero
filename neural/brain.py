"""
GroundZero Neural Brain v2.0 - Intelligent Neural Integration
=============================================================
The central nervous system connecting neural networks to the AI.

UPGRADES FROM v1.0:
- Model size backfilling (migrate data when changing sizes)
- Timeline/history tracking (model evolution)
- Enhanced statistics and monitoring
- Better generation with chain-of-thought
- Async training support
- Memory efficient inference
- Model versioning

This makes the neural network:
- Automatically learn from content
- Generate coherent responses
- Track its own evolution over time
- Support model upgrades without data loss
"""

import threading
import json
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import deque
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class ModelEvent:
    """Represents an event in the model's timeline"""
    timestamp: str
    event_type: str  # 'created', 'trained', 'size_changed', 'checkpoint', 'milestone'
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ModelTimeline:
    """
    Tracks the evolution history of the model.
    
    Records:
    - Model creation
    - Training sessions
    - Size changes
    - Milestones (1M tokens, etc.)
    - Performance improvements
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.timeline_path = self.data_dir / "model_timeline.json"
        self.events: List[ModelEvent] = []
        self._load()
    
    def _load(self):
        """Load timeline from disk"""
        if self.timeline_path.exists():
            try:
                with open(self.timeline_path, 'r') as f:
                    data = json.load(f)
                self.events = [
                    ModelEvent(**e) for e in data.get('events', [])
                ]
            except Exception as e:
                print(f"Timeline load error: {e}")
    
    def _save(self):
        """Save timeline to disk"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        data = {
            'events': [e.to_dict() for e in self.events],
            'last_updated': datetime.now().isoformat()
        }
        with open(self.timeline_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_event(self, event_type: str, details: Dict[str, Any]):
        """Add an event to the timeline"""
        event = ModelEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            details=details
        )
        self.events.append(event)
        self._save()
    
    def get_timeline(self, limit: int = 50) -> List[Dict]:
        """Get timeline events"""
        return [e.to_dict() for e in self.events[-limit:]]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get timeline summary"""
        if not self.events:
            return {'total_events': 0}
        
        event_counts = {}
        for e in self.events:
            event_counts[e.event_type] = event_counts.get(e.event_type, 0) + 1
        
        return {
            'total_events': len(self.events),
            'first_event': self.events[0].timestamp if self.events else None,
            'last_event': self.events[-1].timestamp if self.events else None,
            'event_counts': event_counts
        }


class NeuralBrain:
    """
    The Neural Brain of GroundZero v2.0
    
    Provides unified interface for:
    - Learning from text
    - Generating responses
    - Answering questions
    - Model management (size changes, backfilling)
    - Timeline tracking
    """
    
    # Available model sizes
    MODEL_SIZES = ['nano', 'tiny', 'small', 'medium', 'large', 'xl', 'xxl']
    
    def __init__(self, data_dir: Path, model_size: str = "small"):
        self.data_dir = Path(data_dir)
        self.neural_dir = self.data_dir / "neural"
        self.neural_dir.mkdir(parents=True, exist_ok=True)
        
        # Lazy load trainer
        self.trainer = None
        self.model_size = model_size
        
        # Timeline tracking
        self.timeline = ModelTimeline(self.neural_dir)
        
        # Training buffer
        self.text_buffer: deque = deque(maxlen=200)
        self.buffer_lock = threading.Lock()
        
        # Stats
        self.texts_learned = 0
        self.tokens_generated = 0
        self.is_training = False
        
        # Milestones tracking
        self._last_milestone = 0
        self._milestones = [1000, 10000, 100000, 1000000, 10000000]
        
        # Initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize neural components"""
        try:
            from .trainer import NeuralTrainer, TrainerConfig
            
            config = TrainerConfig(
                model_size=self.model_size,
                batch_size=4,
                gradient_accumulation_steps=2,
                learning_rate=3e-4,
                replay_buffer_size=50000,
                save_every_steps=500,
                curriculum_learning=True,
                prioritized_replay=True
            )
            
            self.trainer = NeuralTrainer(config, self.neural_dir)
            
            # Record creation if first time
            if len(self.timeline.events) == 0:
                self.timeline.add_event('created', {
                    'model_size': self.model_size,
                    'parameters': self.trainer.model.n_params
                })
            
            print(f"      ├─  ✓ Neural Brain v2.0 initialized ({self.model_size})")
            
        except Exception as e:
            print(f"⚠️ Neural Brain initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.trainer = None
    
    @property
    def is_available(self) -> bool:
        """Check if neural components are available"""
        return self.trainer is not None
    
    def change_model_size(self, new_size: str, backfill: bool = True) -> Dict[str, Any]:
        """
        Change model size with optional data backfilling.
        
        Args:
            new_size: New model size (nano, tiny, small, medium, large, xl, xxl)
            backfill: If True, retrain new model on existing data
        
        Returns:
            Migration statistics
        """
        if not self.is_available:
            return {'error': 'Neural brain not available'}
        
        if new_size not in self.MODEL_SIZES:
            return {'error': f'Invalid size. Choose from: {self.MODEL_SIZES}'}
        
        old_size = self.model_size
        
        # Perform model change
        stats = self.trainer.change_model_size(new_size, backfill=backfill)
        
        self.model_size = new_size
        
        # Record in timeline
        self.timeline.add_event('size_changed', {
            'from_size': old_size,
            'to_size': new_size,
            'backfilled': stats.get('backfilled', False),
            'old_params': stats.get('old_params', 0),
            'new_params': stats.get('new_params', 0)
        })
        
        return stats
    
    def learn(self, text: str, source: str = "") -> Dict[str, Any]:
        """
        Learn from a piece of text.
        
        Adds text to training buffer. Training happens in batches.
        """
        if not self.is_available:
            return {'status': 'unavailable', 'error': 'Neural brain not initialized'}
        
        with self.buffer_lock:
            self.text_buffer.append(text)
            self.texts_learned += 1
        
        # Train if buffer is large enough
        if len(self.text_buffer) >= 20:
            return self.train_batch()
        
        return {
            'status': 'buffered',
            'buffer_size': len(self.text_buffer),
            'texts_learned': self.texts_learned
        }
    
    def train_batch(self, force: bool = False) -> Dict[str, Any]:
        """Train on buffered texts"""
        if not self.is_available:
            return {'status': 'unavailable'}
        
        with self.buffer_lock:
            if not self.text_buffer and not force:
                return {'status': 'empty', 'message': 'No texts in buffer'}
            
            texts = list(self.text_buffer)
            self.text_buffer.clear()
        
        if not texts:
            return {'status': 'empty'}
        
        # Train
        try:
            self.is_training = True
            stats = self.trainer.train_on_texts(texts, epochs=1, verbose=False)
            self.is_training = False
            
            # Check milestones
            self._check_milestones()
            
            # Record training event
            if stats.get('steps', 0) > 0:
                self.timeline.add_event('trained', {
                    'texts': len(texts),
                    'loss': stats.get('loss', 0),
                    'steps': stats.get('steps', 0),
                    'total_tokens': stats.get('tokens_trained', 0)
                })
            
            return {
                'status': 'trained',
                'texts': len(texts),
                **stats
            }
        except Exception as e:
            self.is_training = False
            return {'status': 'error', 'error': str(e)}
    
    def _check_milestones(self):
        """Check and record training milestones"""
        if not self.is_available:
            return
        
        tokens = self.trainer.total_tokens_trained
        
        for milestone in self._milestones:
            if tokens >= milestone > self._last_milestone:
                self._last_milestone = milestone
                self.timeline.add_event('milestone', {
                    'type': 'tokens_trained',
                    'value': milestone,
                    'formatted': f"{milestone:,}"
                })
    
    def generate(self, prompt: str, max_tokens: int = 150,
                 temperature: float = 0.8,
                 use_chain_of_thought: bool = False) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Starting text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_chain_of_thought: If True, add CoT prefix
        
        Returns:
            Generated text
        """
        if not self.is_available:
            return "Neural brain not available. Install PyTorch: pip install torch"
        
        if not self.trainer.tokenizer.is_trained:
            return "Model not trained yet. Please learn some content first."
        
        try:
            # Add chain-of-thought prefix if requested
            if use_chain_of_thought:
                prompt = f"Let me think step by step about this: {prompt}\n\nThinking:"
            
            generated = self.trainer.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            self.tokens_generated += max_tokens
            return generated
            
        except Exception as e:
            return f"Generation error: {e}"
    
    def answer(self, question: str, use_reasoning: bool = True) -> Dict[str, Any]:
        """
        Answer a question using the neural model.
        
        More sophisticated than simple generation:
        - Formats question as Q&A prompt
        - Optionally uses reasoning prefix
        - Extracts and cleans answer
        """
        if not self.is_available:
            return {
                'answer': None,
                'confidence': 0,
                'error': 'Neural brain not available'
            }
        
        # Format prompt
        if use_reasoning:
            prompt = f"Question: {question}\n\nLet me think about this carefully.\n\nReasoning:"
        else:
            prompt = f"Question: {question}\nAnswer:"
        
        # Generate
        response = self.generate(prompt, max_tokens=200, temperature=0.7)
        
        # Extract answer
        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
        elif "Therefore" in response:
            answer = response.split("Therefore")[-1].strip()
        else:
            # Take last meaningful sentence
            sentences = response.split('.')
            answer = sentences[-2] + '.' if len(sentences) > 1 else response
        
        # Clean up
        answer = answer.split('\n')[0].strip()
        if not answer.endswith('.'):
            answer += '.'
        
        return {
            'answer': answer,
            'full_response': response,
            'confidence': 0.5,
            'method': 'neural_generation',
            'used_reasoning': use_reasoning
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive neural brain statistics"""
        stats = {
            'available': self.is_available,
            'texts_learned': self.texts_learned,
            'tokens_generated': self.tokens_generated,
            'buffer_size': len(self.text_buffer),
            'is_training': self.is_training,
            'model_size': self.model_size,
            'available_sizes': self.MODEL_SIZES
        }
        
        if self.is_available:
            trainer_stats = self.trainer.get_stats()
            
            stats.update({
                'model_params': trainer_stats['model_params'],
                'vocab_size': trainer_stats['vocab_size'],
                'total_tokens_trained': trainer_stats['total_tokens_trained'],
                'global_step': trainer_stats['global_step'],
                'replay_buffer_size': trainer_stats['replay_buffer_size'],
                'device': trainer_stats['device'],
                'recent_losses': trainer_stats['recent_losses'],
                'n_layers': trainer_stats['n_layers'],
                'n_heads': trainer_stats['n_heads'],
                'd_model': trainer_stats['d_model'],
                'max_seq_len': trainer_stats['max_seq_len'],
                'learning_rate': trainer_stats['learning_rate'],
                'ewc_initialized': trainer_stats['ewc_initialized']
            })
            
            # Timeline summary
            stats['timeline'] = self.timeline.get_summary()
        
        return stats
    
    def get_timeline(self, limit: int = 50) -> List[Dict]:
        """Get model evolution timeline"""
        return self.timeline.get_timeline(limit)
    
    def start_background_training(self):
        """Start background training thread"""
        if self.is_available:
            self.trainer.start_background_training()
    
    def stop_background_training(self):
        """Stop background training"""
        if self.is_available:
            self.trainer.stop_background_training()
    
    def save(self):
        """Save neural state"""
        if self.is_available:
            self.trainer.save_checkpoint()
            
            # Record checkpoint
            self.timeline.add_event('checkpoint', {
                'global_step': self.trainer.global_step,
                'tokens_trained': self.trainer.total_tokens_trained,
                'model_params': self.trainer.model.n_params
            })


# Global instance management
_neural_brain: Optional[NeuralBrain] = None


def get_neural_brain(data_dir: Path = None, model_size: str = "small") -> NeuralBrain:
    """Get or create the neural brain instance"""
    global _neural_brain
    
    if _neural_brain is None:
        if data_dir is None:
            data_dir = Path("data")
        _neural_brain = NeuralBrain(data_dir, model_size)
    
    return _neural_brain


def reset_neural_brain():
    """Reset the global neural brain instance"""
    global _neural_brain
    _neural_brain = None
