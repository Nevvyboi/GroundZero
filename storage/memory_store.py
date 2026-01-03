"""
Memory Store
============
High-level interface for knowledge and memory persistence.
"""

import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from .database import Database


@dataclass
class KnowledgeItem:
    """Represents a piece of stored knowledge"""
    id: int
    content: str
    summary: Optional[str]
    source_url: Optional[str]
    source_title: Optional[str]
    confidence: float
    access_count: int
    created_at: str
    last_accessed: str
    content_hash: str


@dataclass
class VocabularyItem:
    """Represents a vocabulary entry"""
    id: int
    word: str
    frequency: int
    confidence: float
    first_seen: str
    last_seen: str


@dataclass
class LearnedSource:
    """Represents a learned web source"""
    id: int
    url: str
    title: Optional[str]
    content_length: int
    chunks_learned: int
    words_learned: int
    learned_at: str


class MemoryStore:
    """Interface for knowledge and memory persistence"""
    
    def __init__(self, db: Database):
        self.db = db
    
    # === Vocabulary Operations ===
    
    def add_word(self, word: str) -> int:
        """Add or update a word in vocabulary"""
        existing = self.db.fetch_one(
            "SELECT id, frequency FROM vocabulary WHERE word = ?",
            (word,)
        )
        
        if existing:
            self.db.execute(
                """UPDATE vocabulary 
                   SET frequency = frequency + 1, 
                       last_seen = CURRENT_TIMESTAMP,
                       confidence = MIN(1.0, confidence + 0.01)
                   WHERE id = ?""",
                (existing['id'],)
            )
            return existing['id']
        else:
            return self.db.insert('vocabulary', {'word': word})
    
    def add_words_batch(self, words: List[str]) -> int:
        """Add multiple words efficiently"""
        added = 0
        for word in words:
            if word and len(word) > 0:
                self.add_word(word)
                added += 1
        return added
    
    def get_vocabulary_size(self) -> int:
        """Get total vocabulary size"""
        return self.db.count('vocabulary')
    
    def get_word_id(self, word: str) -> Optional[int]:
        """Get ID for a word"""
        return self.db.fetch_value(
            "SELECT id FROM vocabulary WHERE word = ?",
            (word,)
        )
    
    def get_top_words(self, limit: int = 50) -> List[Dict]:
        """Get most frequent words"""
        rows = self.db.fetch_all(
            """SELECT word, frequency, confidence 
               FROM vocabulary 
               ORDER BY frequency DESC 
               LIMIT ?""",
            (limit,)
        )
        return [dict(row) for row in rows]
    
    def get_recent_words(self, limit: int = 20) -> List[Dict]:
        """Get recently learned words"""
        rows = self.db.fetch_all(
            """SELECT word, first_seen, frequency 
               FROM vocabulary 
               ORDER BY first_seen DESC 
               LIMIT ?""",
            (limit,)
        )
        return [dict(row) for row in rows]
    
    # === Knowledge Operations ===
    
    def _hash_content(self, content: str) -> str:
        """Generate hash for content deduplication"""
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def store_knowledge(
        self,
        content: str,
        summary: Optional[str] = None,
        source_url: Optional[str] = None,
        source_title: Optional[str] = None,
        confidence: float = 0.5,
        topic: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Tuple[int, bool]:
        """
        Store a piece of knowledge.
        Returns (id, is_new) tuple.
        Also updates the knowledge index.
        """
        content_hash = self._hash_content(content)
        
        existing = self.db.fetch_one(
            "SELECT id FROM knowledge WHERE content_hash = ?",
            (content_hash,)
        )
        
        if existing:
            # Update access count
            self.db.execute(
                """UPDATE knowledge 
                   SET access_count = access_count + 1,
                       last_accessed = CURRENT_TIMESTAMP,
                       confidence = MIN(1.0, confidence + 0.02)
                   WHERE id = ?""",
                (existing['id'],)
            )
            return existing['id'], False
        
        knowledge_id = self.db.insert('knowledge', {
            'content_hash': content_hash,
            'content': content,
            'summary': summary,
            'source_url': source_url,
            'source_title': source_title,
            'confidence': confidence
        })
        
        # Add to knowledge index
        try:
            from core.knowledge_index import get_knowledge_index
            index = get_knowledge_index(self)
            if index:
                index.add_knowledge(
                    doc_id=knowledge_id,
                    content=content,
                    summary=summary or '',
                    source_url=source_url or '',
                    source_title=source_title or '',
                    confidence=confidence
                )
        except Exception as e:
            print(f"Failed to add to index: {e}")
        
        # Add topic/concept if provided
        if topic:
            self.add_concept(topic)
        
        return knowledge_id, True
    
    def get_knowledge_count(self) -> int:
        """Get total knowledge entries"""
        return self.db.count('knowledge')
    def search_knowledge(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search knowledge base using knowledge index first, then SQL fallback.
        """
        if not query:
            return []
        
        # Try knowledge index first (TF-IDF + topic-based search)
        try:
            from core.knowledge_index import get_knowledge_index
            index = get_knowledge_index(self)
            
            if index and index.doc_count > 0:
                results = index.search(query, limit=limit)
                if results:
                    return results
        except Exception as e:
            print(f"Knowledge index search failed, falling back to SQL: {e}")
        
        # Fallback to SQL-based search
        return self._sql_search_knowledge(query, limit)

    def _sql_search_knowledge(self, query: str, limit: int = 10) -> List[Dict]:
        """
        SQL-based fallback search with multiple strategies.
        """
        results = []
        seen_ids = set()
        
        # Strategy 1: Exact phrase match in content or title
        rows = self.db.fetch_all(
            """SELECT id, content, summary, source_url, source_title, confidence
               FROM knowledge 
               WHERE content LIKE ? OR source_title LIKE ?
               ORDER BY confidence DESC, access_count DESC
               LIMIT ?""",
            (f'%{query}%', f'%{query}%', limit)
        )
        
        for row in rows:
            if row['id'] not in seen_ids:
                seen_ids.add(row['id'])
                results.append({
                    **dict(row),
                    'relevance': 0.9
                })
        
        # Strategy 2: Individual word search
        if len(results) < limit:
            words = [w.strip() for w in query.split() if len(w.strip()) > 2]
            for word in words[:5]:
                rows = self.db.fetch_all(
                    """SELECT id, content, summary, source_url, source_title, confidence
                       FROM knowledge 
                       WHERE content LIKE ? OR source_title LIKE ?
                       ORDER BY confidence DESC
                       LIMIT ?""",
                    (f'%{word}%', f'%{word}%', limit - len(results))
                )
                
                for row in rows:
                    if row['id'] not in seen_ids:
                        seen_ids.add(row['id'])
                        results.append({
                            **dict(row),
                            'relevance': 0.6
                        })
        
        # Sort by relevance then confidence
        results.sort(key=lambda x: (x.get('relevance', 0), x.get('confidence', 0)), reverse=True)
        
        return results[:limit]


    def get_top_knowledge(self, limit: int = 20) -> List[Dict]:
        """Get most confident/accessed knowledge"""
        rows = self.db.fetch_all(
            """SELECT content, summary, source_url, source_title, confidence, access_count
               FROM knowledge 
               ORDER BY (confidence * 0.5 + (access_count / 100.0) * 0.5) DESC
               LIMIT ?""",
            (limit,)
        )
        return [dict(row) for row in rows]
    
    def get_knowledge_by_source(self, source_url: str) -> List[Dict]:
        """Get all knowledge from a specific source"""
        rows = self.db.fetch_all(
            "SELECT * FROM knowledge WHERE source_url = ?",
            (source_url,)
        )
        return [dict(row) for row in rows]
    
    # === Learned Sources Operations ===
    
    def is_source_learned(self, url: str) -> bool:
        """Check if a URL has been learned"""
        return self.db.exists('learned_sources', 'url = ?', (url,))
    
    def mark_source_learned(
        self,
        url: str,
        title: Optional[str] = None,
        content_length: int = 0,
        chunks_learned: int = 0,
        words_learned: int = 0,
        success: bool = True
    ) -> int:
        """Mark a source as learned"""
        return self.db.insert_or_ignore('learned_sources', {
            'url': url,
            'title': title,
            'content_length': content_length,
            'chunks_learned': chunks_learned,
            'words_learned': words_learned,
            'success': success
        })
    
    def get_learned_sources_count(self) -> int:
        """Get count of learned sources"""
        return self.db.count('learned_sources', 'success = 1')
    
    def get_recent_sources(self, limit: int = 20) -> List[Dict]:
        """Get recently learned sources"""
        rows = self.db.fetch_all(
            """SELECT url, title, content_length, chunks_learned, words_learned, learned_at
               FROM learned_sources 
               WHERE success = 1
               ORDER BY learned_at DESC 
               LIMIT ?""",
            (limit,)
        )
        return [dict(row) for row in rows]
    
    # === Concepts Operations ===
    
    def add_concept(self, name: str, description: Optional[str] = None) -> int:
        """Add or update a concept"""
        existing = self.db.fetch_one(
            "SELECT id FROM concepts WHERE name = ?",
            (name.lower(),)
        )
        
        if existing:
            self.db.execute(
                """UPDATE concepts 
                   SET mention_count = mention_count + 1,
                       updated_at = CURRENT_TIMESTAMP,
                       confidence = MIN(1.0, confidence + 0.01)
                   WHERE id = ?""",
                (existing['id'],)
            )
            return existing['id']
        
        return self.db.insert('concepts', {
            'name': name.lower(),
            'description': description
        })
    
    def get_top_concepts(self, limit: int = 20) -> List[Dict]:
        """Get most mentioned concepts"""
        rows = self.db.fetch_all(
            """SELECT name, description, confidence, mention_count
               FROM concepts 
               ORDER BY mention_count DESC 
               LIMIT ?""",
            (limit,)
        )
        return [dict(row) for row in rows]
    
    # === Model State Operations ===
    
    def save_state(self, key: str, value: Any) -> None:
        """Save a state value"""
        value_type = type(value).__name__
        data = {
            'key': key,
            'value_type': value_type,
            'value_int': value if isinstance(value, int) else None,
            'value_real': value if isinstance(value, float) else None,
            'value_text': str(value) if isinstance(value, str) else None,
            'updated_at': datetime.now().isoformat()
        }
        
        existing = self.db.exists('model_state', 'key = ?', (key,))
        if existing:
            self.db.update('model_state', data, 'key = ?', (key,))
        else:
            self.db.insert('model_state', data)
    
    def load_state(self, key: str, default: Any = None) -> Any:
        """Load a state value"""
        row = self.db.fetch_one(
            "SELECT * FROM model_state WHERE key = ?",
            (key,)
        )
        
        if not row:
            return default
        
        if row['value_type'] == 'int':
            return row['value_int']
        elif row['value_type'] == 'float':
            return row['value_real']
        else:
            return row['value_text']
    
    # === Statistics ===
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        return {
            'vocabulary_size': self.get_vocabulary_size(),
            'knowledge_count': self.get_knowledge_count(),
            'sources_learned': self.get_learned_sources_count(),
            'concepts_count': self.db.count('concepts'),
            'total_word_occurrences': self.db.fetch_value(
                "SELECT COALESCE(SUM(frequency), 0) FROM vocabulary"
            ),
            'avg_knowledge_confidence': self.db.fetch_value(
                "SELECT COALESCE(AVG(confidence), 0) FROM knowledge"
            )
        }