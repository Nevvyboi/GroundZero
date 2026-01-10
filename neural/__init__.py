"""
GroundZero Neural Module v2.0
=============================
Advanced neural network components for AI learning and generation.
"""

from .transformer import GroundZeroTransformer, TransformerConfig
from .tokenizer import BPETokenizer
from .trainer import NeuralTrainer, TrainerConfig
from .brain import NeuralBrain, get_neural_brain, ModelTimeline

# Speech recognition (optional - requires PyTorch)
try:
    from .speech import (
        AudioProcessor,
        ASRVocabulary,
        ASRTrainer,
        HybridTranscriber,
        ASRTrainingData
    )
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

__all__ = [
    'GroundZeroTransformer',
    'TransformerConfig', 
    'BPETokenizer',
    'NeuralTrainer',
    'TrainerConfig',
    'NeuralBrain',
    'get_neural_brain',
    'ModelTimeline',
    # Speech (optional)
    'AudioProcessor',
    'ASRVocabulary',
    'ASRTrainer',
    'HybridTranscriber',
    'ASRTrainingData',
    'SPEECH_AVAILABLE'
]