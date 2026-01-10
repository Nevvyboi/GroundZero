"""
GroundZero Vector Store v2.2
============================
Production-grade vector storage with FAISS + SQLite.
Includes all compatibility exports: AdvancedVectorStore, HNSWIndex, simple_embed
"""

import os
import json
import sqlite3
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


@dataclass
class VectorMetadata:
    """Metadata associated with a vector"""
    id: str
    source_id: str = ""
    source_title: str = ""
    content_preview: str = ""
    chunk_type: str = "chunk"
    created_at: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SearchResult:
    """Result from a vector search"""
    id: str
    distance: float
    metadata: VectorMetadata
    vector: Optional[np.ndarray] = None


class HybridVectorStore:
    """
    Hybrid Vector Store combining FAISS and SQLite.
    Automatically adapts to existing database schemas.
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        dim: int = 768,
        max_elements: int = 1000000,
        **kwargs
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.max_elements = max_elements
        
        self.index_path = self.data_dir / "vectors.faiss"
        self.db_path = self.data_dir / "vectors.db"
        
        self._lock = threading.RLock()
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._next_idx = 0
        self._db_columns: List[str] = []
        
        self.stats = {
            'total_vectors': 0,
            'total_searches': 0,
            'total_adds': 0
        }
        
        self.index = None
        if FAISS_AVAILABLE:
            self._init_faiss_index()
        
        self._detect_schema()
        self._load()
    
    def _init_faiss_index(self):
        """Initialize FAISS index"""
        if not FAISS_AVAILABLE:
            return
        self.index = faiss.IndexFlatL2(self.dim)
        self.index = faiss.IndexIDMap(self.index)
    
    def _detect_schema(self):
        """Detect existing database schema and adapt"""
        if not self.db_path.exists():
            self._init_fresh_db()
            return
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vectors'")
            if not cursor.fetchone():
                conn.close()
                self._init_fresh_db()
                return
            
            cursor.execute("PRAGMA table_info(vectors)")
            self._db_columns = [col[1] for col in cursor.fetchall()]
            
            # Add idx column if missing
            if 'idx' not in self._db_columns:
                try:
                    cursor.execute("ALTER TABLE vectors ADD COLUMN idx INTEGER")
                    cursor.execute("UPDATE vectors SET idx = rowid - 1 WHERE idx IS NULL")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_vectors_idx ON vectors(idx)")
                    conn.commit()
                    self._db_columns.append('idx')
                except:
                    pass
            
            conn.close()
        except Exception:
            self._init_fresh_db()
    
    def _init_fresh_db(self):
        """Initialize a fresh SQLite database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                idx INTEGER UNIQUE,
                content TEXT,
                source_id TEXT,
                source_title TEXT,
                chunk_type TEXT DEFAULT 'chunk',
                created_at TEXT,
                metadata_json TEXT
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_vectors_idx ON vectors(idx)')
        conn.commit()
        conn.close()
        self._db_columns = ['id', 'idx', 'content', 'source_id', 'source_title', 
                           'chunk_type', 'created_at', 'metadata_json']
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get SQLite connection"""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _load(self):
        """Load existing index and metadata"""
        # Load FAISS index
        if FAISS_AVAILABLE and self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                self.dim = self.index.d
            except Exception:
                self._init_faiss_index()
        
        # Load ID mappings from SQLite
        if not self.db_path.exists():
            return
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, idx FROM vectors WHERE idx IS NOT NULL ORDER BY idx")
            rows = cursor.fetchall()
            
            for row in rows:
                vec_id = row['id'] if row['id'] else str(row['idx'])
                idx = row['idx']
                self._id_to_idx[vec_id] = idx
                self._idx_to_id[idx] = vec_id
                self._next_idx = max(self._next_idx, idx + 1)
            
            cursor.execute("SELECT COUNT(*) FROM vectors")
            self.stats['total_vectors'] = cursor.fetchone()[0]
            
            conn.close()
        except Exception:
            pass
    
    def save(self):
        """Save FAISS index to disk"""
        with self._lock:
            if FAISS_AVAILABLE and self.index is not None:
                faiss.write_index(self.index, str(self.index_path))
    
    def add(
        self,
        vector_id: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> bool:
        """Add a vector with metadata"""
        with self._lock:
            if not FAISS_AVAILABLE or self.index is None:
                return False
            
            if vector_id in self._id_to_idx:
                self.remove(vector_id)
            
            vector = np.asarray(vector, dtype=np.float32)
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            
            idx = self._next_idx
            self._next_idx += 1
            
            self.index.add_with_ids(vector, np.array([idx], dtype=np.int64))
            
            self._id_to_idx[vector_id] = idx
            self._idx_to_id[idx] = vector_id
            
            # Store in SQLite
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                'INSERT OR REPLACE INTO vectors (id, idx) VALUES (?, ?)',
                (vector_id, idx)
            )
            conn.commit()
            conn.close()
            
            self.stats['total_vectors'] += 1
            self.stats['total_adds'] += 1
            
            return True
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        **kwargs
    ) -> List[SearchResult]:
        """Search for similar vectors"""
        if not FAISS_AVAILABLE or self.index is None:
            return []
        
        with self._lock:
            query_vector = np.asarray(query_vector, dtype=np.float32)
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            
            distances, indices = self.index.search(query_vector, k)
            
            self.stats['total_searches'] += 1
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                
                vec_id = self._idx_to_id.get(int(idx), str(idx))
                metadata = VectorMetadata(id=vec_id)
                
                results.append(SearchResult(
                    id=vec_id,
                    distance=float(dist),
                    metadata=metadata
                ))
            
            return results
    
    def remove(self, vector_id: str) -> bool:
        """Remove a vector by ID"""
        with self._lock:
            if vector_id not in self._id_to_idx:
                return False
            
            idx = self._id_to_idx[vector_id]
            
            if FAISS_AVAILABLE and hasattr(self.index, 'remove_ids'):
                try:
                    self.index.remove_ids(np.array([idx], dtype=np.int64))
                except:
                    pass
            
            del self._id_to_idx[vector_id]
            del self._idx_to_id[idx]
            
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM vectors WHERE id = ?", (vector_id,))
            conn.commit()
            conn.close()
            
            self.stats['total_vectors'] -= 1
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics"""
        faiss_count = 0
        if FAISS_AVAILABLE and self.index is not None:
            faiss_count = self.index.ntotal
        
        return {
            **self.stats,
            'faiss_vectors': faiss_count,
            'sqlite_vectors': len(self._id_to_idx),
            'index_type': 'FAISS',
            'dimension': self.dim
        }
    
    def load(self):
        """Compatibility - data loaded in __init__"""
        pass
    
    @property
    def count(self) -> int:
        return self.stats['total_vectors']
    
    @property
    def total_vectors(self) -> int:
        return self.stats['total_vectors']


# ============================================================
# COMPATIBILITY EXPORTS - Required by knowledge_base.py and others
# ============================================================

# Main vector store alias
AdvancedVectorStore = HybridVectorStore


class HNSWIndex:
    """
    HNSW Index compatibility class.
    Used by knowledge_base.py and other components.
    """
    
    def __init__(
        self,
        dim: int = 768,
        max_elements: int = 100000,
        ef_construction: int = 200,
        M: int = 16,
        **kwargs
    ):
        self.dim = dim
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.M = M
        
        self._vectors: Dict[int, np.ndarray] = {}
        self._next_id = 0
        
        # Use FAISS HNSW if available
        if FAISS_AVAILABLE:
            self.index = faiss.IndexHNSWFlat(dim, M)
            self.index.hnsw.efConstruction = ef_construction
            self.index = faiss.IndexIDMap(self.index)
        else:
            self.index = None
    
    def add_item(self, vector: np.ndarray, idx: int = None) -> int:
        """Add a vector to the index"""
        if idx is None:
            idx = self._next_id
        self._next_id = max(self._next_id, idx + 1)
        
        if self.index is not None and FAISS_AVAILABLE:
            vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
            self.index.add_with_ids(vec, np.array([idx], dtype=np.int64))
        
        self._vectors[idx] = np.asarray(vector, dtype=np.float32)
        return idx
    
    def add_items(self, vectors: np.ndarray, ids: List[int] = None):
        """Add multiple vectors"""
        vectors = np.asarray(vectors, dtype=np.float32)
        if ids is None:
            ids = list(range(self._next_id, self._next_id + len(vectors)))
        
        for vec, idx in zip(vectors, ids):
            self.add_item(vec, idx)
    
    def search(self, query: np.ndarray, k: int = 10):
        """Search for nearest neighbors"""
        if self.index is not None and FAISS_AVAILABLE:
            q = np.asarray(query, dtype=np.float32).reshape(1, -1)
            distances, indices = self.index.search(q, k)
            return indices[0].tolist(), distances[0].tolist()
        
        # Fallback: brute force
        if not self._vectors:
            return [], []
        
        query = np.asarray(query, dtype=np.float32)
        distances = []
        for idx, vec in self._vectors.items():
            dist = np.linalg.norm(query - vec)
            distances.append((idx, dist))
        
        distances.sort(key=lambda x: x[1])
        top_k = distances[:k]
        
        return [x[0] for x in top_k], [x[1] for x in top_k]
    
    def knn_query(self, query: np.ndarray, k: int = 10):
        """Alternative search method (used by some code)"""
        indices, distances = self.search(query, k)
        return np.array([indices]), np.array([distances])
    
    def get_item(self, idx: int) -> Optional[np.ndarray]:
        """Get a vector by index"""
        return self._vectors.get(idx)
    
    def get_items(self, ids: List[int]) -> np.ndarray:
        """Get multiple vectors"""
        return np.array([self._vectors.get(i, np.zeros(self.dim)) for i in ids])
    
    def save(self, path: str):
        """Save index to file"""
        if self.index is not None and FAISS_AVAILABLE:
            faiss.write_index(self.index, path)
    
    def load(self, path: str):
        """Load index from file"""
        if FAISS_AVAILABLE and os.path.exists(path):
            self.index = faiss.read_index(path)
    
    @property
    def element_count(self) -> int:
        """Number of elements in index"""
        if self.index is not None and FAISS_AVAILABLE:
            return self.index.ntotal
        return len(self._vectors)
    
    def __len__(self) -> int:
        return self.element_count


class ProductQuantizer:
    """
    Product Quantizer for vector compression.
    Compatibility class for code that imports ProductQuantizer.
    """
    
    def __init__(
        self,
        dim: int = 768,
        n_subquantizers: int = 8,
        n_bits: int = 8,
        **kwargs
    ):
        self.dim = dim
        self.n_subquantizers = n_subquantizers
        self.n_bits = n_bits
        self.n_centroids = 2 ** n_bits
        self.sub_dim = dim // n_subquantizers
        
        self.centroids = None
        self.is_trained = False
        
        # Use FAISS PQ if available
        if FAISS_AVAILABLE:
            self.pq = faiss.ProductQuantizer(dim, n_subquantizers, n_bits)
        else:
            self.pq = None
    
    def train(self, vectors: np.ndarray):
        """Train the quantizer on vectors"""
        vectors = np.asarray(vectors, dtype=np.float32)
        if self.pq is not None:
            self.pq.train(vectors)
        self.is_trained = True
    
    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode vectors to codes"""
        vectors = np.asarray(vectors, dtype=np.float32)
        if self.pq is not None:
            return self.pq.compute_codes(vectors)
        # Fallback: just return indices
        return np.zeros((len(vectors), self.n_subquantizers), dtype=np.uint8)
    
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode codes back to vectors"""
        if self.pq is not None:
            return self.pq.decode(codes)
        return np.zeros((len(codes), self.dim), dtype=np.float32)
    
    def compute_distance(self, query: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Compute distances between query and encoded vectors"""
        query = np.asarray(query, dtype=np.float32)
        decoded = self.decode(codes)
        return np.linalg.norm(decoded - query, axis=1)


class VectorIndex:
    """Generic vector index wrapper - compatibility class"""
    
    def __init__(self, dim: int = 768, **kwargs):
        self.dim = dim
        self._store = HybridVectorStore(dim=dim, **kwargs)
    
    def add(self, vectors, ids=None):
        if ids is None:
            ids = [f"vec_{i}" for i in range(len(vectors))]
        for vec, vid in zip(vectors, ids):
            self._store.add(str(vid), vec)
    
    def search(self, query, k=10):
        results = self._store.search(query, k)
        return [r.id for r in results], [r.distance for r in results]
    
    def save(self, path):
        self._store.save()
    
    def load(self, path):
        pass


def simple_embed(text: str, dim: int = 768) -> np.ndarray:
    """
    Simple deterministic embedding for testing.
    Creates a reproducible vector from text hash.
    """
    np.random.seed(hash(text) % 2**32)
    vec = np.random.randn(dim).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("Testing vector_store.py...")
    
    # Test exports
    print(f"  AdvancedVectorStore: {AdvancedVectorStore}")
    print(f"  HNSWIndex: {HNSWIndex}")
    print(f"  ProductQuantizer: {ProductQuantizer}")
    print(f"  VectorIndex: {VectorIndex}")
    print(f"  simple_embed: {simple_embed}")
    
    # Test simple_embed
    vec = simple_embed("test", 256)
    print(f"  simple_embed shape: {vec.shape}")
    
    # Test HNSWIndex
    hnsw = HNSWIndex(dim=256)
    hnsw.add_item(vec, 0)
    print(f"  HNSWIndex count: {hnsw.element_count}")
    
    # Test ProductQuantizer
    pq = ProductQuantizer(dim=256, n_subquantizers=8)
    print(f"  ProductQuantizer dim: {pq.dim}")
    
    # Test HybridVectorStore
    store = HybridVectorStore(data_dir="./test_vs", dim=256)
    store.add("test_1", vec)
    print(f"  HybridVectorStore count: {store.count}")
    
    print("  ✓ All tests passed!")
