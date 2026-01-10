from pathlib import Path
from dataclasses import dataclass


@dataclass
class Settings:
    """Application settings"""
    
    # Server
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = True
    
    # Paths
    data_dir: Path = Path("data")
    
    # Embeddings
    embedding_dimension: int = 256
    
    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
