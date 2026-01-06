"""
Vector Store
============
Embedded vector database with full persistence.

Uses FAISS when available, falls back to brute force with SQLite storage.
All data is persisted to disk and survives restarts.

Based on Vector Database best practices:
- Vectors stored in optimized index for similarity search
- SQLite stores metadata AND vectors (for persistence without FAISS)
- Supports semantic "search by meaning"
"""

import numpy as np
import sqlite3
import threading
import json
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not available, using brute force search (slower but persistent)")


class VectorStore:
    """
    Embedded vector database with full persistence.
    
    Architecture:
    - SQLite stores: metadata + vectors (as BLOBs)
    - FAISS (if available): fast similarity search index
    - Without FAISS: brute force search using stored vectors
    
    Persistence:
    - Vectors stored in SQLite (always persisted)
    - FAISS index saved separately (optional speed boost)
    - All data survives restarts
    """
    
    def __init__(self, data_dir: Path, dimension: int = 256):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.dimension = dimension
        self.db_path = self.data_dir / "vectors.db"
        self.index_path = self.data_dir / "vectors.faiss"
        
        # Thread safety
        self._lock = threading.RLock()  # Use RLock to allow reentrant locking
        
        # In-memory vector cache (for brute force search)
        self._vector_cache: Dict[int, np.ndarray] = {}
        
        # Initialize database
        self._init_database()
        
        # Load vectors into memory/FAISS
        self._load_vectors()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings for concurrent access"""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,  # Wait up to 30 seconds for lock
            check_same_thread=False,  # Allow multi-threaded access
            isolation_level=None  # Autocommit mode
        )
        conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for better concurrency
        conn.execute("PRAGMA busy_timeout=30000")  # 30 second busy timeout
        return conn
    
    def _init_database(self) -> None:
        """Initialize SQLite database with vector storage"""
        conn = self._get_connection()
        
        # Main table with vector blob storage
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                source_url TEXT DEFAULT '',
                source_title TEXT DEFAULT '',
                confidence REAL DEFAULT 0.5,
                vector_data BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT DEFAULT '{}'
            )
        """)
        
        # Indexes for faster lookups
        conn.execute("CREATE INDEX IF NOT EXISTS idx_source_url ON vectors(source_url)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_source_title ON vectors(source_title)")
        
        conn.close()
    
    def _load_vectors(self) -> None:
        """Load all vectors from database into memory/FAISS"""
        conn = self._get_connection()
        rows = conn.execute("SELECT id, vector_data FROM vectors WHERE vector_data IS NOT NULL").fetchall()
        conn.close()
        
        if not rows:
            # No vectors to load
            if FAISS_AVAILABLE:
                self.index = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIDMap(self.index)
            return
        
        # Load vectors
        ids = []
        vectors = []
        
        for row_id, vector_blob in rows:
            if vector_blob:
                try:
                    vector = np.frombuffer(vector_blob, dtype=np.float32)
                    if len(vector) == self.dimension:
                        ids.append(row_id)
                        vectors.append(vector)
                        self._vector_cache[row_id] = vector
                except Exception as e:
                    print(f"Warning: Could not load vector {row_id}: {e}")
        
        if FAISS_AVAILABLE and vectors:
            # Build FAISS index
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIDMap(self.index)
            
            vectors_np = np.array(vectors, dtype=np.float32)
            ids_np = np.array(ids, dtype=np.int64)
            self.index.add_with_ids(vectors_np, ids_np)
            
            print(f"✅ Loaded {len(vectors)} vectors into FAISS index")
        elif vectors:
            print(f"✅ Loaded {len(vectors)} vectors into memory")
        
        if not FAISS_AVAILABLE:
            self.index = None
    
    def _save_faiss_index(self) -> None:
        """Save FAISS index to disk (optional optimization)"""
        if FAISS_AVAILABLE and self.index is not None:
            try:
                faiss.write_index(self.index, str(self.index_path))
            except Exception as e:
                print(f"Warning: Could not save FAISS index: {e}")
    
    def add(self, vector: np.ndarray, content: str, 
            source_url: str = '', source_title: str = '',
            confidence: float = 0.5, metadata: Dict = None) -> int:
        """
        Add a vector to the store.
        
        Process:
        1. Store vector as BLOB + metadata in SQLite (persistence)
        2. Add to FAISS index (fast search) or memory cache
        
        Returns: ID of the inserted vector
        """
        if metadata is None:
            metadata = {}
        
        # Ensure vector is correct shape and type
        vector = np.array(vector, dtype=np.float32).flatten()
        vector_blob = vector.tobytes()
        
        with self._lock:
            # Insert into SQLite with vector blob
            conn = self._get_connection()
            cursor = conn.execute(
                """INSERT INTO vectors (content, source_url, source_title, confidence, vector_data, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (content, source_url, source_title, confidence, vector_blob, json.dumps(metadata))
            )
            vector_id = cursor.lastrowid
            conn.close()
            
            # Add to memory cache
            self._vector_cache[vector_id] = vector
            
            # Add to FAISS index if available
            if FAISS_AVAILABLE and self.index is not None:
                vector_2d = vector.reshape(1, -1)
                ids = np.array([vector_id], dtype=np.int64)
                self.index.add_with_ids(vector_2d, ids)
            
            # Save FAISS index periodically
            if FAISS_AVAILABLE and vector_id % 100 == 0:
                self._save_faiss_index()
            
            return vector_id
    
    def search(self, query_vector: np.ndarray, top_k: int = 10, 
               min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        This is the core of semantic search:
        1. Query vector represents the "meaning" of the question
        2. FAISS finds vectors with similar "meaning"
        3. We return the associated content
        
        Uses Approximate Nearest Neighbor (ANN) - doesn't check every vector,
        but uses smart indexing to find likely matches quickly.
        
        Args:
            query_vector: The query embedding
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)
        
        Returns:
            List of matching documents with scores
        """
        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        
        results = []
        
        with self._lock:
            if FAISS_AVAILABLE:
                # FAISS search
                if self.index.ntotal == 0:
                    return []
                
                # Search returns distances and IDs
                scores, ids = self.index.search(query_vector, min(top_k * 2, self.index.ntotal))
                
                # Get metadata for each result
                conn = self._get_connection()
                conn.row_factory = sqlite3.Row
                
                for score, vec_id in zip(scores[0], ids[0]):
                    if vec_id < 0:  # FAISS returns -1 for empty slots
                        continue
                    
                    # Convert FAISS score to 0-1 range
                    # For normalized vectors, inner product is cosine similarity
                    similarity = float(score)
                    
                    if similarity < min_score:
                        continue
                    
                    row = conn.execute(
                        "SELECT * FROM vectors WHERE id = ?", (int(vec_id),)
                    ).fetchone()
                    
                    if row:
                        results.append({
                            'id': row['id'],
                            'content': row['content'],
                            'source_url': row['source_url'],
                            'source_title': row['source_title'],
                            'confidence': row['confidence'],
                            'relevance': similarity,
                            'metadata': json.loads(row['metadata'] or '{}')
                        })
                
                conn.close()
            
            else:
                # Brute force search using cached vectors
                if not self._vector_cache:
                    return []
                
                similarities = []
                for vec_id, vec in self._vector_cache.items():
                    sim = float(np.dot(query_vector.flatten(), vec))
                    if sim >= min_score:
                        similarities.append((vec_id, sim))
                
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                conn = self._get_connection()
                conn.row_factory = sqlite3.Row
                
                for vec_id, sim in similarities[:top_k]:
                    row = conn.execute(
                        "SELECT * FROM vectors WHERE id = ?", (vec_id,)
                    ).fetchone()
                    
                    if row:
                        results.append({
                            'id': row['id'],
                            'content': row['content'],
                            'source_url': row['source_url'],
                            'source_title': row['source_title'],
                            'confidence': row['confidence'],
                            'relevance': sim,
                            'metadata': json.loads(row['metadata'] or '{}')
                        })
                
                conn.close()
        
        return results[:top_k]
    
    def get_by_id(self, vector_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific entry by ID"""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM vectors WHERE id = ?", (vector_id,)).fetchone()
        conn.close()
        
        if row:
            return {
                'id': row['id'],
                'content': row['content'],
                'source_url': row['source_url'],
                'source_title': row['source_title'],
                'confidence': row['confidence'],
                'metadata': json.loads(row['metadata'] or '{}')
            }
        return None
    
    def exists(self, source_url: str) -> bool:
        """Check if a source URL already exists"""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT id FROM vectors WHERE source_url = ? LIMIT 1", (source_url,)
        ).fetchone()
        conn.close()
        return row is not None
    
    def count(self) -> int:
        """Get total number of vectors"""
        if FAISS_AVAILABLE and self.index is not None:
            return self.index.ntotal
        return len(self._vector_cache)
    
    def save(self) -> None:
        """Save FAISS index to disk (vectors already in SQLite)"""
        self._save_faiss_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            conn = self._get_connection()
            total = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
            sources = conn.execute("SELECT COUNT(DISTINCT source_url) FROM vectors WHERE source_url != ''").fetchone()[0]
            conn.close()
        except Exception as e:
            print(f"Warning: Could not get stats: {e}")
            total = len(self._vector_cache)
            sources = 0
        
        return {
            'total_vectors': self.count(),
            'total_entries': total,
            'unique_sources': sources,
            'dimension': self.dimension,
            'index_type': 'FAISS' if FAISS_AVAILABLE else 'BruteForce'
        }
    
    def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recently added entries"""
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM vectors ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            conn.close()
            
            return [{
                'id': row['id'],
                'source_title': row['source_title'],
                'source_url': row['source_url'],
                'created_at': row['created_at']
            } for row in rows]
        except Exception as e:
            print(f"Warning: Could not get recent: {e}")
            return []
    
    def get_all_knowledge(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all knowledge entries with basic info"""
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, source_title, source_url, confidence, created_at FROM vectors ORDER BY created_at DESC LIMIT ?", 
                (limit,)
            ).fetchall()
            conn.close()
            
            return [{
                'id': row['id'],
                'title': row['source_title'],
                'url': row['source_url'],
                'confidence': row['confidence'],
                'created_at': row['created_at']
            } for row in rows]
        except Exception as e:
            print(f"Warning: Could not get all knowledge: {e}")
            return []
    
    def get_all_with_content(self) -> List[Dict[str, Any]]:
        """Get all entries with their content for re-embedding"""
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT id, content FROM vectors").fetchall()
            conn.close()
            
            return [{'id': row['id'], 'content': row['content']} for row in rows]
        except Exception as e:
            print(f"Warning: Could not get all with content: {e}")
            return []
    
    def update_vector(self, entry_id: int, new_vector: np.ndarray) -> None:
        """Update the vector for an existing entry"""
        vector_blob = new_vector.astype(np.float32).tobytes()
        
        with self._lock:
            try:
                conn = self._get_connection()
                conn.execute(
                    "UPDATE vectors SET vector_data = ? WHERE id = ?",
                    (vector_blob, entry_id)
                )
                conn.close()
                
                # Update cache
                self._vector_cache[entry_id] = new_vector.astype(np.float32)
                
                # Update FAISS if available
                if FAISS_AVAILABLE and self.index is not None:
                    # FAISS doesn't support update, so we need to rebuild
                    # For now just update the cache - index will be rebuilt on save/load
                    pass
            except Exception as e:
                print(f"Warning: Could not update vector: {e}")
    
    def get_related(self, entry_id: int, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get entries related to a specific entry by vector similarity"""
        with self._lock:
            # Get the vector for this entry
            if entry_id not in self._vector_cache:
                return []
            
            query_vector = self._vector_cache[entry_id]
            query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
            
            results = []
            
            if FAISS_AVAILABLE and self.index is not None and self.index.ntotal > 1:
                # Search for similar vectors (get extra to exclude self)
                scores, ids = self.index.search(query_vector, min(top_k + 1, self.index.ntotal))
                
                try:
                    conn = self._get_connection()
                    conn.row_factory = sqlite3.Row
                    
                    for score, vec_id in zip(scores[0], ids[0]):
                        if vec_id < 0 or int(vec_id) == entry_id:
                            continue
                        
                        similarity = float(score)
                        if similarity < 0.05:  # Min threshold (5%)
                            continue
                        
                        row = conn.execute(
                            "SELECT id, source_title, confidence FROM vectors WHERE id = ?", 
                            (int(vec_id),)
                        ).fetchone()
                        
                        if row:
                            results.append({
                                'id': row['id'],
                                'title': row['source_title'],
                                'confidence': row['confidence'],
                                'similarity': round(similarity * 100, 1)
                            })
                        
                        if len(results) >= top_k:
                            break
                    
                    conn.close()
                except Exception as e:
                    print(f"Warning: Error getting related: {e}")
            
            else:
                # Brute force search
                if len(self._vector_cache) <= 1:
                    return []
                
                try:
                    conn = self._get_connection()
                    conn.row_factory = sqlite3.Row
                    
                    similarities = []
                    for vid, vec in self._vector_cache.items():
                        if vid == entry_id:
                            continue
                        vec_arr = np.array(vec, dtype=np.float32)
                        # Cosine similarity
                        sim = np.dot(query_vector.flatten(), vec_arr) / (
                            np.linalg.norm(query_vector) * np.linalg.norm(vec_arr) + 1e-9
                        )
                        similarities.append((vid, float(sim)))
                    
                    # Sort by similarity
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    
                    for vid, sim in similarities[:top_k]:
                        if sim < 0.05:  # Min threshold (5%)
                            continue
                        row = conn.execute(
                            "SELECT id, source_title, confidence FROM vectors WHERE id = ?",
                            (vid,)
                        ).fetchone()
                        if row:
                            results.append({
                                'id': row['id'],
                                'title': row['source_title'],
                                'confidence': row['confidence'],
                                'similarity': round(sim * 100, 1)
                            })
                    
                    conn.close()
                except Exception as e:
                    print(f"Warning: Error getting related: {e}")
            
            return results