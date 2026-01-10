"""
Learning Data Manager
=====================
SQLite-based storage for learning progress data.

Replaces JSON files with proper database storage:
- vital_articles_cache.json → vital_articles table
- learned_articles.json → learned_articles table

Benefits:
- ACID transactions (no data corruption)
- Efficient queries (indexed lookups)
- Scalable (millions of articles)
- Concurrent access safe
"""

import sqlite3
import json
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from contextlib import contextmanager
import hashlib


@dataclass
class VitalArticle:
    """A vital article from Wikipedia's Vital Articles list"""
    title: str
    category: str
    level: int = 1  # Vital level (1-5)
    priority: float = 1.0
    url: str = ""
    last_fetched: Optional[str] = None
    content_hash: Optional[str] = None
    word_count: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class LearnedArticle:
    """Record of a learned article"""
    article_id: str
    title: str
    source_url: str
    category: str = ""
    learned_at: str = field(default_factory=lambda: datetime.now().isoformat())
    word_count: int = 0
    facts_extracted: int = 0
    tokens_trained: int = 0
    confidence: float = 1.0
    session_id: str = ""
    content_hash: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


class LearningDataManager:
    """
    SQLite-based manager for learning data.
    
    Provides thread-safe access to:
    - Vital articles cache
    - Learned articles history
    - Learning sessions
    - Category progress
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.data_dir / "learning.db"
        
        # Thread-local connections
        self._local = threading.local()
        self._lock = threading.RLock()
        
        # Initialize database
        self._init_db()
        
        # Migrate from JSON if exists
        self._migrate_from_json()
    
    @property
    def conn(self) -> sqlite3.Connection:
        """Get thread-local connection"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn
    
    @contextmanager
    def transaction(self):
        """Context manager for transactions"""
        with self._lock:
            try:
                yield self.conn
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                raise e
    
    def _init_db(self):
        """Initialize database schema"""
        with self.transaction() as conn:
            # Vital articles cache
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vital_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT UNIQUE NOT NULL,
                    category TEXT NOT NULL,
                    level INTEGER DEFAULT 1,
                    priority REAL DEFAULT 1.0,
                    url TEXT,
                    last_fetched TEXT,
                    content_hash TEXT,
                    word_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Indexes for vital articles
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vital_category ON vital_articles(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vital_level ON vital_articles(level)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vital_priority ON vital_articles(priority DESC)")
            
            # Learned articles history
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learned_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_id TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    source_url TEXT,
                    category TEXT,
                    learned_at TEXT NOT NULL,
                    word_count INTEGER DEFAULT 0,
                    facts_extracted INTEGER DEFAULT 0,
                    tokens_trained INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 1.0,
                    session_id TEXT,
                    content_hash TEXT
                )
            """)
            
            # Indexes for learned articles
            conn.execute("CREATE INDEX IF NOT EXISTS idx_learned_category ON learned_articles(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_learned_session ON learned_articles(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_learned_date ON learned_articles(learned_at)")
            
            # Learning sessions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_sessions (
                    id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    articles_learned INTEGER DEFAULT 0,
                    words_learned INTEGER DEFAULT 0,
                    facts_extracted INTEGER DEFAULT 0,
                    tokens_trained INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active',
                    config TEXT
                )
            """)
            
            # Category progress
            conn.execute("""
                CREATE TABLE IF NOT EXISTS category_progress (
                    category TEXT PRIMARY KEY,
                    total_articles INTEGER DEFAULT 0,
                    articles_learned INTEGER DEFAULT 0,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def _migrate_from_json(self):
        """Migrate data from JSON files if they exist"""
        # Migrate vital_articles_cache.json
        vital_json = self.data_dir / "vital_articles_cache.json"
        if vital_json.exists():
            try:
                with open(vital_json) as f:
                    data = json.load(f)
                
                # Import articles
                count = 0
                for category, articles in data.get('categories', {}).items():
                    for article in articles:
                        if isinstance(article, dict):
                            title = article.get('title', article.get('name', ''))
                        else:
                            title = str(article)
                        
                        if title:
                            self.add_vital_article(VitalArticle(
                                title=title,
                                category=category,
                                level=1
                            ))
                            count += 1
                
                # Backup and remove old file
                vital_json.rename(vital_json.with_suffix('.json.migrated'))
                print(f"Migrated {count} vital articles from JSON to SQLite")
                
            except Exception as e:
                print(f"Error migrating vital_articles_cache.json: {e}")
        
        # Migrate learned_articles.json
        learned_json = self.data_dir / "learned_articles.json"
        if learned_json.exists():
            try:
                with open(learned_json) as f:
                    data = json.load(f)
                
                count = 0
                for article_id, info in data.items():
                    if isinstance(info, dict):
                        self.mark_as_learned(LearnedArticle(
                            article_id=article_id,
                            title=info.get('title', article_id),
                            source_url=info.get('url', ''),
                            category=info.get('category', ''),
                            learned_at=info.get('learned_at', datetime.now().isoformat()),
                            word_count=info.get('word_count', 0),
                            facts_extracted=info.get('facts_extracted', 0)
                        ))
                        count += 1
                
                # Backup and remove old file
                learned_json.rename(learned_json.with_suffix('.json.migrated'))
                print(f"Migrated {count} learned articles from JSON to SQLite")
                
            except Exception as e:
                print(f"Error migrating learned_articles.json: {e}")
    
    # ========== Vital Articles ==========
    
    def add_vital_article(self, article: VitalArticle) -> bool:
        """Add or update a vital article"""
        with self.transaction() as conn:
            try:
                conn.execute("""
                    INSERT INTO vital_articles (title, category, level, priority, url, word_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(title) DO UPDATE SET
                        category = excluded.category,
                        level = excluded.level,
                        priority = excluded.priority,
                        url = excluded.url,
                        updated_at = CURRENT_TIMESTAMP
                """, (article.title, article.category, article.level, 
                      article.priority, article.url, article.word_count))
                return True
            except Exception:
                return False
    
    def add_vital_articles_batch(self, articles: List[VitalArticle]) -> int:
        """Add multiple vital articles efficiently"""
        count = 0
        with self.transaction() as conn:
            for article in articles:
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO vital_articles (title, category, level, priority, url)
                        VALUES (?, ?, ?, ?, ?)
                    """, (article.title, article.category, article.level, 
                          article.priority, article.url))
                    count += 1
                except Exception:
                    pass
        return count
    
    def get_vital_articles(self, category: Optional[str] = None, 
                           level: Optional[int] = None,
                           limit: int = 100) -> List[VitalArticle]:
        """Get vital articles with optional filters"""
        query = "SELECT * FROM vital_articles WHERE 1=1"
        params = []
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if level:
            query += " AND level = ?"
            params.append(level)
        
        query += " ORDER BY priority DESC, level ASC LIMIT ?"
        params.append(limit)
        
        cursor = self.conn.execute(query, params)
        
        return [VitalArticle(
            title=row['title'],
            category=row['category'],
            level=row['level'],
            priority=row['priority'],
            url=row['url'] or '',
            last_fetched=row['last_fetched'],
            content_hash=row['content_hash'],
            word_count=row['word_count']
        ) for row in cursor.fetchall()]
    
    def get_unlearned_vital_articles(self, category: Optional[str] = None,
                                     limit: int = 100) -> List[VitalArticle]:
        """Get vital articles that haven't been learned yet"""
        query = """
            SELECT v.* FROM vital_articles v
            LEFT JOIN learned_articles l ON v.title = l.title
            WHERE l.id IS NULL
        """
        params = []
        
        if category:
            query += " AND v.category = ?"
            params.append(category)
        
        query += " ORDER BY v.priority DESC, v.level ASC LIMIT ?"
        params.append(limit)
        
        cursor = self.conn.execute(query, params)
        
        return [VitalArticle(
            title=row['title'],
            category=row['category'],
            level=row['level'],
            priority=row['priority'],
            url=row['url'] or ''
        ) for row in cursor.fetchall()]
    
    def get_vital_categories(self) -> Dict[str, int]:
        """Get all categories with article counts"""
        cursor = self.conn.execute("""
            SELECT category, COUNT(*) as count 
            FROM vital_articles 
            GROUP BY category 
            ORDER BY count DESC
        """)
        return {row['category']: row['count'] for row in cursor.fetchall()}
    
    def update_vital_article_content(self, title: str, content_hash: str, 
                                     word_count: int) -> None:
        """Update vital article after fetching content"""
        with self.transaction() as conn:
            conn.execute("""
                UPDATE vital_articles 
                SET last_fetched = ?, content_hash = ?, word_count = ?, updated_at = ?
                WHERE title = ?
            """, (datetime.now().isoformat(), content_hash, word_count, 
                  datetime.now().isoformat(), title))
    
    # ========== Learned Articles ==========
    
    def mark_as_learned(self, article: LearnedArticle) -> bool:
        """Mark an article as learned"""
        with self.transaction() as conn:
            try:
                conn.execute("""
                    INSERT INTO learned_articles 
                    (article_id, title, source_url, category, learned_at, 
                     word_count, facts_extracted, tokens_trained, confidence, 
                     session_id, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(article_id) DO UPDATE SET
                        word_count = word_count + excluded.word_count,
                        facts_extracted = facts_extracted + excluded.facts_extracted,
                        tokens_trained = tokens_trained + excluded.tokens_trained
                """, (article.article_id, article.title, article.source_url,
                      article.category, article.learned_at, article.word_count,
                      article.facts_extracted, article.tokens_trained,
                      article.confidence, article.session_id, article.content_hash))
                
                # Update category progress
                conn.execute("""
                    INSERT INTO category_progress (category, articles_learned)
                    VALUES (?, 1)
                    ON CONFLICT(category) DO UPDATE SET
                        articles_learned = articles_learned + 1,
                        last_updated = CURRENT_TIMESTAMP
                """, (article.category,))
                
                return True
            except Exception:
                return False
    
    def is_learned(self, article_id: str) -> bool:
        """Check if an article has been learned"""
        cursor = self.conn.execute(
            "SELECT 1 FROM learned_articles WHERE article_id = ?",
            (article_id,)
        )
        return cursor.fetchone() is not None
    
    def get_learned_articles(self, category: Optional[str] = None,
                            session_id: Optional[str] = None,
                            limit: int = 100,
                            offset: int = 0) -> List[LearnedArticle]:
        """Get learned articles with optional filters"""
        query = "SELECT * FROM learned_articles WHERE 1=1"
        params = []
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        
        query += " ORDER BY learned_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor = self.conn.execute(query, params)
        
        return [LearnedArticle(
            article_id=row['article_id'],
            title=row['title'],
            source_url=row['source_url'] or '',
            category=row['category'] or '',
            learned_at=row['learned_at'],
            word_count=row['word_count'],
            facts_extracted=row['facts_extracted'],
            tokens_trained=row['tokens_trained'],
            confidence=row['confidence'],
            session_id=row['session_id'] or '',
            content_hash=row['content_hash'] or ''
        ) for row in cursor.fetchall()]
    
    def get_learned_count(self, category: Optional[str] = None) -> int:
        """Get count of learned articles"""
        if category:
            cursor = self.conn.execute(
                "SELECT COUNT(*) FROM learned_articles WHERE category = ?",
                (category,)
            )
        else:
            cursor = self.conn.execute("SELECT COUNT(*) FROM learned_articles")
        return cursor.fetchone()[0]
    
    def get_learned_article_ids(self) -> Set[str]:
        """Get set of all learned article IDs (for fast lookup)"""
        cursor = self.conn.execute("SELECT article_id FROM learned_articles")
        return {row['article_id'] for row in cursor.fetchall()}
    
    # ========== Sessions ==========
    
    def create_session(self, session_id: str, config: Optional[Dict] = None) -> None:
        """Create a new learning session"""
        with self.transaction() as conn:
            conn.execute("""
                INSERT INTO learning_sessions (id, started_at, config)
                VALUES (?, ?, ?)
            """, (session_id, datetime.now().isoformat(), 
                  json.dumps(config) if config else None))
    
    def update_session(self, session_id: str, 
                      articles: int = 0, words: int = 0, 
                      facts: int = 0, tokens: int = 0) -> None:
        """Update session statistics"""
        with self.transaction() as conn:
            conn.execute("""
                UPDATE learning_sessions SET
                    articles_learned = articles_learned + ?,
                    words_learned = words_learned + ?,
                    facts_extracted = facts_extracted + ?,
                    tokens_trained = tokens_trained + ?
                WHERE id = ?
            """, (articles, words, facts, tokens, session_id))
    
    def end_session(self, session_id: str) -> None:
        """Mark session as ended"""
        with self.transaction() as conn:
            conn.execute("""
                UPDATE learning_sessions SET
                    ended_at = ?,
                    status = 'completed'
                WHERE id = ?
            """, (datetime.now().isoformat(), session_id))
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session info"""
        cursor = self.conn.execute(
            "SELECT * FROM learning_sessions WHERE id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict]:
        """Get recent sessions"""
        cursor = self.conn.execute("""
            SELECT * FROM learning_sessions 
            ORDER BY started_at DESC LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    # ========== Statistics ==========
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        stats = {}
        
        # Total counts
        cursor = self.conn.execute("SELECT COUNT(*) FROM vital_articles")
        stats['total_vital_articles'] = cursor.fetchone()[0]
        
        cursor = self.conn.execute("SELECT COUNT(*) FROM learned_articles")
        stats['total_learned'] = cursor.fetchone()[0]
        
        # Category breakdown
        cursor = self.conn.execute("""
            SELECT 
                v.category,
                COUNT(v.id) as total,
                COUNT(l.id) as learned
            FROM vital_articles v
            LEFT JOIN learned_articles l ON v.title = l.title
            GROUP BY v.category
        """)
        stats['by_category'] = {
            row['category']: {
                'total': row['total'],
                'learned': row['learned'],
                'progress': row['learned'] / row['total'] if row['total'] > 0 else 0
            }
            for row in cursor.fetchall()
        }
        
        # Session stats
        cursor = self.conn.execute("""
            SELECT 
                COUNT(*) as total_sessions,
                SUM(articles_learned) as total_articles,
                SUM(words_learned) as total_words,
                SUM(facts_extracted) as total_facts,
                SUM(tokens_trained) as total_tokens
            FROM learning_sessions
        """)
        row = cursor.fetchone()
        stats['sessions'] = {
            'total': row['total_sessions'],
            'articles': row['total_articles'] or 0,
            'words': row['total_words'] or 0,
            'facts': row['total_facts'] or 0,
            'tokens': row['total_tokens'] or 0
        }
        
        # Recent activity (last 7 days)
        cursor = self.conn.execute("""
            SELECT DATE(learned_at) as date, COUNT(*) as count
            FROM learned_articles
            WHERE learned_at >= datetime('now', '-7 days')
            GROUP BY DATE(learned_at)
            ORDER BY date
        """)
        stats['recent_activity'] = {row['date']: row['count'] for row in cursor.fetchall()}
        
        return stats
    
    def get_category_progress(self) -> Dict[str, Dict[str, Any]]:
        """Get progress for each category"""
        cursor = self.conn.execute("""
            SELECT 
                v.category,
                COUNT(v.id) as total,
                COUNT(l.id) as learned,
                SUM(COALESCE(l.word_count, 0)) as words_learned,
                SUM(COALESCE(l.facts_extracted, 0)) as facts_extracted
            FROM vital_articles v
            LEFT JOIN learned_articles l ON v.title = l.title
            GROUP BY v.category
            ORDER BY COUNT(l.id) DESC
        """)
        
        return {
            row['category']: {
                'total': row['total'],
                'learned': row['learned'],
                'remaining': row['total'] - row['learned'],
                'progress': row['learned'] / row['total'] if row['total'] > 0 else 0,
                'words': row['words_learned'],
                'facts': row['facts_extracted']
            }
            for row in cursor.fetchall()
        }
    
    # ========== Utilities ==========
    
    @staticmethod
    def generate_article_id(title: str, url: str = "") -> str:
        """Generate unique article ID"""
        content = f"{title}:{url}".lower()
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def close(self):
        """Close database connection"""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# ============================================================
# Singleton accessor
# ============================================================

_data_manager: Optional[LearningDataManager] = None

def get_data_manager(data_dir: str = "data") -> LearningDataManager:
    """Get or create the data manager singleton"""
    global _data_manager
    if _data_manager is None:
        _data_manager = LearningDataManager(data_dir)
    return _data_manager


__all__ = [
    'LearningDataManager',
    'VitalArticle',
    'LearnedArticle',
    'get_data_manager'
]
