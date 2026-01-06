"""
Knowledge Base
==============
High-level knowledge management combining embeddings + vector store.

This is the main interface for:
- Adding knowledge (text → embedding → store)
- Searching knowledge (query → embedding → similarity search)
- Managing vocabulary and statistics
"""

import sqlite3
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import Counter
import time
import threading

from .vector_store import VectorStore
from core.embeddings import EmbeddingEngine


class KnowledgeBase:
    """
    Manages all knowledge storage and retrieval.
    
    Combines:
    - EmbeddingEngine: Converts text to vectors
    - VectorStore: Stores and searches vectors
    - SQLite: Tracks vocabulary, sources, statistics
    
    This implements the "search by meaning" concept from the documents:
    - Query "car" finds "automobile", "vehicle", "sedan"
    - Query "king" finds related concepts like "monarch", "ruler", "queen"
    """
    
    def __init__(self, data_dir: Path, dimension: int = 256):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.dimension = dimension
        self._lock = threading.RLock()
        
        # Initialize components
        self.embeddings = EmbeddingEngine(dimension=dimension)
        self.vectors = VectorStore(data_dir, dimension=dimension)
        
        # Metadata database
        self.db_path = self.data_dir / "knowledge.db"
        self._init_database()
        
        # Load embedding model if exists
        self._load_embeddings()
        
        # Track if we need to rebuild embeddings
        self._docs_since_rebuild = 0
        self._rebuild_threshold = 100  # Rebuild vocab every 100 docs
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings for concurrent access"""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn
    
    def _init_database(self) -> None:
        """Initialize metadata database"""
        conn = self._get_connection()
        
        # Sources table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                title TEXT,
                word_count INTEGER DEFAULT 0,
                session_id INTEGER,
                learned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Vocabulary table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vocabulary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT UNIQUE NOT NULL,
                frequency INTEGER DEFAULT 1
            )
        """)
        
        # Statistics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stats (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                total_words INTEGER DEFAULT 0,
                total_sources INTEGER DEFAULT 0,
                total_knowledge INTEGER DEFAULT 0,
                total_learning_time INTEGER DEFAULT 0,
                last_learn_at TIMESTAMP
            )
        """)
        
        # Learning sessions table - tracks each learning session
        conn.execute("""
            CREATE TABLE IF NOT EXISTS learning_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                duration_seconds INTEGER DEFAULT 0,
                articles_learned INTEGER DEFAULT 0,
                words_learned INTEGER DEFAULT 0,
                knowledge_added INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active'
            )
        """)
        
        conn.execute("INSERT OR IGNORE INTO stats (id) VALUES (1)")
        conn.close()
    
    def _load_embeddings(self) -> None:
        """Load embedding model from disk"""
        embed_path = self.data_dir / "embeddings.pkl"
        if embed_path.exists():
            if self.embeddings.load(embed_path):
                print(f"✅ Loaded embeddings: {self.embeddings.get_stats()['vocabulary_size']} words")
    
    def _save_embeddings(self) -> None:
        """Save embedding model to disk"""
        embed_path = self.data_dir / "embeddings.pkl"
        self.embeddings.save(embed_path)
    
    def add_knowledge(self, content: str, source_url: str = '', 
                      source_title: str = '', confidence: float = 0.5) -> Tuple[int, bool]:
        """
        Add knowledge to the database.
        
        Process:
        1. Check for duplicates
        2. Update vocabulary (for embedding model)
        3. Create embedding vector
        4. Store in vector database
        5. Update statistics
        
        Returns: (id, is_new)
        """
        if not content or len(content) < 50:
            return 0, False
        
        # Check if already exists
        if source_url and self.vectors.exists(source_url):
            return 0, False
        
        # Add to embedding vocabulary
        self.embeddings.add_document(content)
        self._docs_since_rebuild += 1
        
        # Rebuild vocabulary periodically
        if self._docs_since_rebuild >= self._rebuild_threshold:
            self._rebuild_vocabulary()
        
        # Create embedding (may be zero if not initialized)
        vector = self.embeddings.embed(content)
        
        # Store in vector database
        knowledge_id = self.vectors.add(
            vector=vector,
            content=content,
            source_url=source_url,
            source_title=source_title,
            confidence=confidence
        )
        
        # Update vocabulary table
        self._update_vocabulary(content)
        
        # Update statistics
        word_count = len(content.split())
        self._update_stats(word_count)
        
        # Add source
        if source_url:
            self._add_source(source_url, source_title, word_count)
        
        return knowledge_id, True
    
    def _rebuild_vocabulary(self) -> None:
        """Rebuild embedding vocabulary"""
        print("🔄 Rebuilding embedding vocabulary...")
        self.embeddings.build_vocabulary()
        self._save_embeddings()
        self._docs_since_rebuild = 0
        
        # TODO: Re-embed existing vectors with new vocabulary
        # For production, you'd want to re-index everything
    
    def _update_vocabulary(self, text: str) -> None:
        """Update vocabulary table"""
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        word_counts = Counter(words)
        
        try:
            conn = self._get_connection()
            for word, count in word_counts.items():
                conn.execute("""
                    INSERT INTO vocabulary (word, frequency) VALUES (?, ?)
                    ON CONFLICT(word) DO UPDATE SET frequency = frequency + ?
                """, (word, count, count))
            conn.close()
        except Exception as e:
            print(f"Warning: Could not update vocabulary: {e}")
    
    def _update_stats(self, word_count: int) -> None:
        """Update statistics"""
        try:
            conn = self._get_connection()
            conn.execute("""
                UPDATE stats SET 
                    total_words = total_words + ?,
                    total_knowledge = total_knowledge + 1,
                    last_learn_at = CURRENT_TIMESTAMP
                WHERE id = 1
            """, (word_count,))
            conn.close()
        except Exception as e:
            print(f"Warning: Could not update stats: {e}")
    
    def _add_source(self, url: str, title: str, word_count: int) -> None:
        """Add or update source"""
        try:
            conn = self._get_connection()
            conn.execute("""
                INSERT OR IGNORE INTO sources (url, title, word_count) VALUES (?, ?, ?)
            """, (url, title, word_count))
            conn.execute("UPDATE stats SET total_sources = total_sources + 1 WHERE id = 1")
            conn.close()
        except Exception as e:
            print(f"Warning: Could not add source: {e}")
    
    def search(self, query: str, limit: int = 10, min_score: float = 0.1) -> List[Dict[str, Any]]:
        """
        Search for relevant knowledge using semantic similarity.
        
        This is "search by meaning":
        1. Convert query to vector (captures meaning)
        2. Find vectors with similar meaning
        3. Return associated content
        
        The magic: "car" will match "automobile" because their
        vectors are close in the embedding space.
        """
        if not query:
            return []
        
        # If embeddings not initialized, fall back to keyword search
        if not self.embeddings.is_initialized:
            return self._keyword_search(query, limit)
        
        # Create query embedding
        query_vector = self.embeddings.embed(query)
        
        # Search vector store
        results = self.vectors.search(query_vector, top_k=limit * 2, min_score=min_score)
        
        # Also do keyword search and merge results
        keyword_results = self._keyword_search(query, limit)
        
        # Merge results (vector results first, then keyword)
        seen_ids = {r['id'] for r in results}
        for kr in keyword_results:
            if kr['id'] not in seen_ids:
                results.append(kr)
                seen_ids.add(kr['id'])
        
        # Sort by relevance
        results.sort(key=lambda x: x.get('relevance', 0), reverse=True)
        
        return results[:limit]
    
    def _keyword_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fallback keyword-based search"""
        query_lower = query.lower()
        words = [w for w in re.findall(r'\w+', query_lower) if len(w) > 2]
        
        if not words:
            return []
        
        # Search in vector store's SQLite
        conn = sqlite3.connect(str(self.vectors.db_path))
        conn.row_factory = sqlite3.Row
        
        results = []
        seen_ids = set()
        
        # Search for full phrase
        rows = conn.execute("""
            SELECT * FROM vectors 
            WHERE LOWER(content) LIKE ? OR LOWER(source_title) LIKE ?
            ORDER BY confidence DESC
            LIMIT ?
        """, (f'%{query_lower}%', f'%{query_lower}%', limit)).fetchall()
        
        for row in rows:
            if row['id'] not in seen_ids:
                score = self._calculate_keyword_relevance(query_lower, words, dict(row))
                if score > 0.2:
                    results.append({
                        'id': row['id'],
                        'content': row['content'],
                        'source_url': row['source_url'],
                        'source_title': row['source_title'],
                        'confidence': row['confidence'],
                        'relevance': score
                    })
                    seen_ids.add(row['id'])
        
        # Search individual words
        for word in words[:5]:
            if len(results) >= limit:
                break
            
            rows = conn.execute("""
                SELECT * FROM vectors 
                WHERE LOWER(content) LIKE ? OR LOWER(source_title) LIKE ?
                ORDER BY confidence DESC
                LIMIT ?
            """, (f'%{word}%', f'%{word}%', limit)).fetchall()
            
            for row in rows:
                if row['id'] not in seen_ids:
                    score = self._calculate_keyword_relevance(query_lower, words, dict(row))
                    if score > 0.15:
                        results.append({
                            'id': row['id'],
                            'content': row['content'],
                            'source_url': row['source_url'],
                            'source_title': row['source_title'],
                            'confidence': row['confidence'],
                            'relevance': score
                        })
                        seen_ids.add(row['id'])
        
        conn.close()
        
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results[:limit]
    
    def _calculate_keyword_relevance(self, query: str, words: List[str], 
                                      result: Dict) -> float:
        """Calculate keyword-based relevance score"""
        content = result.get('content', '').lower()
        title = result.get('source_title', '').lower()
        
        score = 0.0
        
        # Full query match
        if query in content:
            score += 0.4
        if query in title:
            score += 0.4
        
        # Word matches
        if words:
            content_matches = sum(1 for w in words if w in content)
            title_matches = sum(1 for w in words if w in title)
            
            score += (content_matches / len(words)) * 0.3
            score += (title_matches / len(words)) * 0.3
        
        # Confidence factor
        score = score * 0.85 + result.get('confidence', 0.5) * 0.15
        
        return min(1.0, score)
    
    def source_exists(self, url: str) -> bool:
        """Check if source already processed"""
        return self.vectors.exists(url)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            
            stats = conn.execute("SELECT * FROM stats WHERE id = 1").fetchone()
            vocab_count = conn.execute("SELECT COUNT(*) FROM vocabulary").fetchone()[0]
            session_count = conn.execute("SELECT COUNT(*) FROM learning_sessions").fetchone()[0]
            
            conn.close()
            
            vector_stats = self.vectors.get_stats()
            embed_stats = self.embeddings.get_stats()
            
            return {
                'total_knowledge': stats['total_knowledge'] if stats else 0,
                'total_sources': stats['total_sources'] if stats else 0,
                'total_words': stats['total_words'] if stats else 0,
                'total_learning_time': stats['total_learning_time'] if stats else 0,
                'vocabulary_size': vocab_count,
                'total_sessions': session_count,
                'last_learn_at': stats['last_learn_at'] if stats else None,
                'vectors': vector_stats,
                'embeddings': embed_stats
            }
        except Exception as e:
            print(f"Warning: Could not get statistics: {e}")
            return {
                'total_knowledge': 0,
                'total_sources': 0,
                'total_words': 0,
                'total_learning_time': 0,
                'vocabulary_size': 0,
                'total_sessions': 0,
                'last_learn_at': None,
                'vectors': self.vectors.get_stats(),
                'embeddings': self.embeddings.get_stats()
            }
    
    # ==================== SESSION MANAGEMENT ====================
    
    def start_session(self) -> int:
        """Start a new learning session"""
        try:
            conn = self._get_connection()
            cursor = conn.execute(
                "INSERT INTO learning_sessions (status) VALUES ('active')"
            )
            session_id = cursor.lastrowid
            conn.close()
            return session_id
        except Exception as e:
            print(f"Warning: Could not start session: {e}")
            return 0
    
    def update_session(self, session_id: int, articles: int = 0, 
                       words: int = 0, knowledge: int = 0) -> None:
        """Update session statistics"""
        try:
            conn = self._get_connection()
            conn.execute("""
                UPDATE learning_sessions SET 
                    articles_learned = articles_learned + ?,
                    words_learned = words_learned + ?,
                    knowledge_added = knowledge_added + ?
                WHERE id = ?
            """, (articles, words, knowledge, session_id))
            conn.close()
        except Exception as e:
            print(f"Warning: Could not update session: {e}")
    
    def end_session(self, session_id: int, duration_seconds: int) -> None:
        """End a learning session"""
        try:
            conn = self._get_connection()
            conn.execute("""
                UPDATE learning_sessions SET 
                    ended_at = CURRENT_TIMESTAMP,
                    duration_seconds = ?,
                    status = 'completed'
                WHERE id = ?
            """, (duration_seconds, session_id))
            
            # Update total learning time
            conn.execute("""
                UPDATE stats SET total_learning_time = total_learning_time + ?
                WHERE id = 1
            """, (duration_seconds,))
            
            conn.close()
        except Exception as e:
            print(f"Warning: Could not end session: {e}")
    
    def get_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get learning session history"""
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            
            rows = conn.execute("""
                SELECT * FROM learning_sessions 
                ORDER BY started_at DESC 
                LIMIT ?
            """, (limit,)).fetchall()
            
            conn.close()
            
            return [{
                'id': row['id'],
                'started_at': row['started_at'],
                'ended_at': row['ended_at'],
                'duration_seconds': row['duration_seconds'],
                'articles_learned': row['articles_learned'],
                'words_learned': row['words_learned'],
                'knowledge_added': row['knowledge_added'],
                'status': row['status']
            } for row in rows]
        except Exception as e:
            print(f"Warning: Could not get sessions: {e}")
            return []
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of all learning sessions"""
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            
            summary = conn.execute("""
                SELECT 
                    COUNT(*) as total_sessions,
                    SUM(duration_seconds) as total_time,
                    SUM(articles_learned) as total_articles,
                    SUM(words_learned) as total_words,
                    SUM(knowledge_added) as total_knowledge
                FROM learning_sessions
                WHERE status = 'completed'
            """).fetchone()
            
            conn.close()
            
            return {
                'total_sessions': summary['total_sessions'] or 0,
                'total_time_seconds': summary['total_time'] or 0,
                'total_articles': summary['total_articles'] or 0,
                'total_words': summary['total_words'] or 0,
                'total_knowledge': summary['total_knowledge'] or 0
            }
        except Exception as e:
            print(f"Warning: Could not get session summary: {e}")
            return {
                'total_sessions': 0,
                'total_time_seconds': 0,
                'total_articles': 0,
                'total_words': 0,
                'total_knowledge': 0
            }
    
    def get_recent_knowledge(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recently added knowledge"""
        return self.vectors.get_recent(limit)
    
    def initialize_embeddings(self) -> None:
        """Force initialize/rebuild embeddings and re-embed all vectors"""
        if self.embeddings.total_documents > 0:
            self.embeddings.build_vocabulary()
            self._save_embeddings()
            
            # Re-embed all stored content with the new vocabulary
            all_entries = self.vectors.get_all_with_content()
            for entry in all_entries:
                new_vector = self.embeddings.embed(entry['content'])
                self.vectors.update_vector(entry['id'], new_vector)
            
            print(f"✅ Embeddings initialized: {self.embeddings.get_stats()['vocabulary_size']} words")
    
    def save(self) -> None:
        """Save all data to disk"""
        self.vectors.save()
        self._save_embeddings()