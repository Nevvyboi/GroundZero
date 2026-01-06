"""
Learning Engine
===============
Continuous learning from Wikipedia and web sources.

Features:
- Background learning thread
- Session tracking with persistent stats
- Progress tracking with callbacks
- URL ingestion
- Search and learn
- **Persistent Knowledge Graph** (facts stored in SQLite)
"""

import threading
import time
from typing import Dict, Any, List, Optional, Callable
from queue import Queue

from storage import KnowledgeBase
from web import WikipediaSearch, ContentExtractor


class LearningEngine:
    """
    Manages continuous learning from Wikipedia and the web.
    
    Architecture:
    - Background thread for continuous Wikipedia learning
    - Session tracking persisted to database
    - Manual URL ingestion
    - Progress callbacks for UI updates
    - **Persistent Knowledge Graph** for symbolic reasoning (SQLite-backed)
    """
    
    def __init__(self, knowledge_base: KnowledgeBase, graph_reasoner=None):
        self.kb = knowledge_base
        self.wikipedia = WikipediaSearch()
        self.extractor = ContentExtractor()
        
        # Persistent Knowledge Graph Reasoner (SQLite-backed)
        self.graph_reasoner = graph_reasoner
        if self.graph_reasoner:
            print("✅ Learning Engine connected to Persistent Knowledge Graph")
        
        # Learning state
        self.is_running = False
        self.is_paused = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Current session
        self.current_session_id: Optional[int] = None
        self.session_start_time: Optional[float] = None
        
        # Session stats (in-memory for current session)
        self.session_stats = {
            'articles_read': 0,
            'words_learned': 0,
            'knowledge_added': 0,
            'facts_extracted': 0,  # Track knowledge graph facts
            'start_time': None,
            'current_article': None,
            'current_url': None,
            'current_content': None
        }
        
        # Callbacks
        self.on_article_start: Optional[Callable] = None
        self.on_article_complete: Optional[Callable] = None
        self.on_progress: Optional[Callable] = None
        
        print("✅ Learning Engine initialized")
    
    def start(self) -> Dict[str, Any]:
        """Start continuous learning - creates new session"""
        if self.is_running:
            return {'status': 'already_running', 'session_id': self.current_session_id}
        
        # Create new session in database
        self.current_session_id = self.kb.start_session()
        self.session_start_time = time.time()
        
        # Reset session stats
        self.session_stats = {
            'articles_read': 0,
            'words_learned': 0,
            'knowledge_added': 0,
            'facts_extracted': 0,  # Track knowledge graph facts
            'start_time': self.session_start_time,
            'current_article': None,
            'current_url': None,
            'current_content': None
        }
        
        self._stop_event.clear()
        self.is_running = True
        self.is_paused = False
        
        self._thread = threading.Thread(target=self._learning_loop, daemon=True)
        self._thread.start()
        
        return {
            'status': 'started',
            'session_id': self.current_session_id
        }
    
    def stop(self) -> Dict[str, Any]:
        """Stop continuous learning - ends current session"""
        self._stop_event.set()
        self.is_running = False
        
        if self._thread:
            self._thread.join(timeout=2)
        
        # End session in database
        duration = 0
        if self.session_start_time:
            duration = int(time.time() - self.session_start_time)
        
        if self.current_session_id:
            self.kb.end_session(self.current_session_id, duration)
        
        # Save embeddings
        self.kb.save()
        
        session_summary = {
            'session_id': self.current_session_id,
            'duration_seconds': duration,
            'articles_read': self.session_stats['articles_read'],
            'words_learned': self.session_stats['words_learned'],
            'knowledge_added': self.session_stats['knowledge_added'],
            'facts_extracted': self.session_stats.get('facts_extracted', 0)
        }
        
        # Reset session
        self.current_session_id = None
        self.session_start_time = None
        
        return {
            'status': 'stopped',
            'session': session_summary,
            'stats': self.get_stats()
        }
    
    def pause(self) -> Dict[str, Any]:
        """Pause learning"""
        self.is_paused = True
        return {'status': 'paused'}
    
    def resume(self) -> Dict[str, Any]:
        """Resume learning"""
        self.is_paused = False
        return {'status': 'resumed'}
    
    def _learning_loop(self):
        """Main learning loop"""
        batch_count = 0
        
        while not self._stop_event.is_set():
            if self.is_paused:
                time.sleep(0.5)
                continue
            
            try:
                # Get random articles
                articles = self.wikipedia.get_random_articles(5)
                
                for article in articles:
                    if self._stop_event.is_set() or self.is_paused:
                        break
                    
                    try:
                        self._learn_article(article)
                    except Exception as e:
                        print(f"Error learning article: {e}")
                    
                    time.sleep(0.3)  # Be nice to Wikipedia
                
                batch_count += 1
                
                # Initialize embeddings after first batch (in background)
                if batch_count == 1:
                    threading.Thread(target=self._safe_init_embeddings, daemon=True).start()
                
                # Rebuild embeddings every 50 batches (less frequently, in background)
                if batch_count % 50 == 0:
                    threading.Thread(target=self._safe_init_embeddings, daemon=True).start()
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Learning loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(2)
    
    def _safe_init_embeddings(self):
        """Safely initialize embeddings without blocking"""
        try:
            self.kb.initialize_embeddings()
        except Exception as e:
            print(f"Embeddings init error: {e}")
    
    def _learn_article(self, article: Dict[str, str]) -> bool:
        """Learn from a single article"""
        title = article.get('title', '')
        url = article.get('url', '')
        
        if not title:
            return False
        
        # Check if already learned (prevents duplicate learning)
        if self.kb.source_exists(url):
            return False
        
        # Notify start and update stats
        self.session_stats['current_article'] = title
        self.session_stats['current_url'] = url
        if self.on_article_start:
            self.on_article_start(title, url)
        
        # Fetch content
        content_data = self.wikipedia.get_article_content(title)
        
        if not content_data or len(content_data.get('content', '')) < 100:
            return False
        
        # Store knowledge
        content = content_data['content']
        word_count = len(content.split())
        
        # Store current content for preview
        self.session_stats['current_content'] = content[:2000]  # First 2000 chars for preview
        
        knowledge_id, is_new = self.kb.add_knowledge(
            content=content,
            source_url=url,
            source_title=title,
            confidence=0.7
        )
        
        # =====================================================
        # PERSISTENT KNOWLEDGE GRAPH: Extract facts (SQLite-backed)
        # =====================================================
        facts_added = 0
        if self.graph_reasoner and is_new:
            try:
                graph_result = self.graph_reasoner.learn(content, title)
                facts_added = graph_result.get('facts_added', 0)
                # Safely update facts_extracted
                if 'facts_extracted' not in self.session_stats:
                    self.session_stats['facts_extracted'] = 0
                self.session_stats['facts_extracted'] += facts_added
            except Exception as e:
                print(f"Knowledge graph extraction error: {e}")
        
        if is_new:
            # Update in-memory session stats
            self.session_stats['articles_read'] += 1
            self.session_stats['words_learned'] += word_count
            self.session_stats['knowledge_added'] += 1
            
            # Update session in database
            if self.current_session_id:
                self.kb.update_session(
                    self.current_session_id,
                    articles=1,
                    words=word_count,
                    knowledge=1
                )
        
        # Notify complete
        if self.on_article_complete:
            self.on_article_complete(title, word_count)
        
        return True
    
    def learn_from_url(self, url: str) -> Dict[str, Any]:
        """Learn from a specific URL"""
        # Check if exists (prevents duplicate learning)
        if self.kb.source_exists(url):
            return {
                'success': False,
                'reason': 'already_learned',
                'message': 'This URL has already been learned',
                'url': url
            }
        
        # Extract content
        if 'wikipedia.org' in url:
            content_data = self.wikipedia.get_article_by_url(url)
        else:
            content_data = self.extractor.extract(url)
        
        if not content_data or len(content_data.get('content', '')) < 100:
            return {
                'success': False,
                'reason': 'extraction_failed',
                'message': 'Could not extract content from URL',
                'url': url
            }
        
        # Store
        content = content_data['content']
        title = content_data.get('title', url)
        word_count = len(content.split())
        
        self.kb.add_knowledge(
            content=content,
            source_url=url,
            source_title=title,
            confidence=0.7
        )
        
        # Update embeddings
        self.kb.initialize_embeddings()
        
        return {
            'success': True,
            'url': url,
            'title': title,
            'word_count': word_count
        }
    
    def search_and_learn(self, query: str, max_articles: int = 3) -> Dict[str, Any]:
        """Search Wikipedia and learn from results"""
        learned_from = []
        
        articles = self.wikipedia.search(query, limit=max_articles)
        
        for article in articles:
            result = self._learn_article(article)
            if result:
                learned_from.append({
                    'title': article.get('title'),
                    'url': article.get('url')
                })
        
        # Update embeddings after learning
        if learned_from:
            self.kb.initialize_embeddings()
        
        return {
            'query': query,
            'count': len(learned_from),
            'learned_from': learned_from
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics including session history"""
        kb_stats = self.kb.get_statistics()
        session_summary = self.kb.get_session_summary()
        
        # Current session time
        session_time = 0
        if self.session_start_time:
            session_time = time.time() - self.session_start_time
        
        return {
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'current_article': self.session_stats.get('current_article'),
            'current_url': self.session_stats.get('current_url'),
            'current_content': self.session_stats.get('current_content'),
            'current_session': {
                'id': self.current_session_id,
                'articles_read': self.session_stats.get('articles_read', 0),
                'words_learned': self.session_stats.get('words_learned', 0),
                'knowledge_added': self.session_stats.get('knowledge_added', 0),
                'facts_extracted': self.session_stats.get('facts_extracted', 0),
                'duration_seconds': int(session_time)
            },
            'all_sessions': session_summary,
            'total': kb_stats
        }
    
    def get_session_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get detailed session history"""
        return self.kb.get_sessions(limit)