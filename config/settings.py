"""
Configuration Settings
======================
Centralized configuration for all NeuralMind components.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class ModelConfig:
    """Neural model architecture settings"""
    vocab_size: int = 50000
    embedding_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    feedforward_dim: int = 1024
    max_sequence_length: int = 512
    dropout_rate: float = 0.1


@dataclass
class LearningConfig:
    """Learning and training settings"""
    learning_rate: float = 0.001
    batch_size: int = 32
    target_sites: int = 50  # 100% progress
    chunk_size: int = 500   # Text chunk size for learning
    min_chunk_length: int = 50
    request_delay: float = 1.0  # Seconds between web requests
    max_queue_size: int = 100


@dataclass
class StorageConfig:
    """Storage and persistence settings"""
    database_name: str = "neuralmind.db"
    weights_file: str = "model_weights.pkl"  # Using pickle for nested structures
    embeddings_file: str = "embeddings.npz"
    state_file: str = "model_state.npz"


@dataclass
class Settings:
    """Main application settings"""
    # Paths
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    DATA_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    DEBUG: bool = False
    
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    
    # Seed URLs for learning
    SEED_URLS: List[str] = field(default_factory=lambda: [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Computer_science",
        "https://en.wikipedia.org/wiki/Physics",
        "https://en.wikipedia.org/wiki/Mathematics",
        "https://en.wikipedia.org/wiki/Biology",
        "https://en.wikipedia.org/wiki/Chemistry",
        "https://en.wikipedia.org/wiki/History",
        "https://en.wikipedia.org/wiki/Philosophy",
        "https://en.wikipedia.org/wiki/Psychology",
        "https://en.wikipedia.org/wiki/Economics",
        "https://en.wikipedia.org/wiki/Literature",
        "https://en.wikipedia.org/wiki/Technology",
        "https://en.wikipedia.org/wiki/Medicine",
        "https://en.wikipedia.org/wiki/Geography",
    ])
    
    def __post_init__(self):
        """Ensure directories exist"""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    @property
    def database_path(self) -> Path:
        return self.DATA_DIR / self.storage.database_name
    
    @property
    def weights_path(self) -> Path:
        return self.DATA_DIR / self.storage.weights_file
    
    @property
    def embeddings_path(self) -> Path:
        return self.DATA_DIR / self.storage.embeddings_file
    
    @property
    def state_path(self) -> Path:
        return self.DATA_DIR / self.storage.state_file