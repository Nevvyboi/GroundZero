"""
Continuous Learner
==================
Orchestrates background learning from web sources.
"""

import threading
import time
import random
import re
from queue import Queue, Empty
from typing import Optional, Callable, List, Dict, Any
from datetime import datetime

from web import WikipediaCrawler, ContentExtractor, WikipediaSearch
from storage import MemoryStore


class ContinuousLearner:
    """
    Background learning orchestrator.
    Continuously fetches and processes web content using Wikipedia API.
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        neural_model,
        seed_urls: List[str],
        target_sites: int = 50,
        chunk_size: int = 500,
        request_delay: float = 1.0
    ):
        self.memory = memory_store
        self.model = neural_model
        self.seed_urls = seed_urls
        self.target_sites = target_sites
        self.chunk_size = chunk_size
        
        # Components - WikipediaSearch uses API (more reliable than HTML crawling)
        self.wiki_search = WikipediaSearch()
        self.extractor = ContentExtractor()
        
        # State
        self.topic_queue: Queue = Queue()
        self.is_running = False
        self.learning_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.sites_learned = 0
        self.sites_skipped = 0
        self.total_chunks = 0
        self.total_words = 0
        self.errors: List[Dict] = []
        
        # Current state
        self.current_url: Optional[str] = None
        self.current_title: Optional[str] = None
        self.current_preview: Optional[str] = None
        
        # Callbacks
        self.on_progress: Optional[Callable] = None
        self.on_content: Optional[Callable] = None
        self.on_complete: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Initialize queue
        self._seed_queue()
    
    def _seed_queue(self) -> None:
        """Populate queue with seed topics from URLs"""
        topics = []
        for url in self.seed_urls:
            if '/wiki/' in url:
                topic = url.split('/wiki/')[-1].replace('_', ' ')
                topics.append(topic)
        
        random.shuffle(topics)
        for topic in topics:
            self.topic_queue.put(topic)
    
    def start(self) -> Dict[str, Any]:
        """Start background learning"""
        if self.is_running:
            return {'status': 'already_running'}
        
        if self.sites_learned >= self.target_sites:
            return {'status': 'complete', 'message': 'Target reached. Reset to continue.'}
        
        self.is_running = True
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        
        return {'status': 'started'}
    
    def stop(self) -> Dict[str, Any]:
        """Stop background learning"""
        self.is_running = False
        if self.learning_thread:
            self.learning_thread.join(timeout=2)
        
        # Save model state
        self.model.save()
        
        return {
            'status': 'stopped',
            'sites_learned': self.sites_learned,
            'total_chunks': self.total_chunks
        }
    
    def reset(self) -> Dict[str, Any]:
        """Reset learning progress"""
        self.stop()
        
        self.sites_learned = 0
        self.sites_skipped = 0
        self.total_chunks = 0
        self.total_words = 0
        self.errors = []
        
        # Clear and reseed queue
        while not self.topic_queue.empty():
            try:
                self.topic_queue.get_nowait()
            except Empty:
                break
        
        self._seed_queue()
        
        return {'status': 'reset'}
    
    def _learning_loop(self) -> None:
        """Main learning loop"""
        while self.is_running:
            # Check if target reached
            if self.sites_learned >= self.target_sites:
                self._complete()
                break
            
            try:
                # Get next topic
                topic = None
                try:
                    topic = self.topic_queue.get(timeout=1)
                except Empty:
                    # Get random articles if queue empty
                    self._add_random_topics()
                    continue
                
                if topic:
                    result = self._learn_from_topic(topic)
                    
                    if result.get('success'):
                        self.sites_learned += 1
                    
                    # Emit progress
                    if self.on_progress:
                        self.on_progress(self.get_stats())
                
                # Rate limiting - be nice to Wikipedia
                time.sleep(1.5)
                
            except Exception as e:
                self._handle_error(str(e))
                time.sleep(2.0)
    
    def _learn_from_topic(self, topic: str) -> Dict[str, Any]:
        """Learn from a topic using Wikipedia API"""
        url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        
        # Skip if already learned
        if self.memory.is_source_learned(url):
            self.sites_skipped += 1
            return {'success': False, 'reason': 'already_learned'}
        
        self.current_url = url
        self.current_title = topic
        
        # Get content via API (more reliable than HTML scraping)
        content = self.wiki_search.get_content(topic)
        
        if not content or not content.get('content'):
            # Try searching for the topic
            search_results = self.wiki_search.search(topic, limit=1)
            if search_results:
                actual_title = search_results[0]['title']
                content = self.wiki_search.get_content(actual_title)
        
        if not content or not content.get('content'):
            return {'success': False, 'reason': 'no_content'}
        
        text = content['content']
        title = content.get('title', topic)
        
        if len(text) < 100:
            return {'success': False, 'reason': 'insufficient_content'}
        
        self.current_title = title
        self.current_preview = text[:300]
        
        # Emit content update
        if self.on_content:
            self.on_content({
                'url': url,
                'title': title,
                'preview': self.current_preview,
                'length': len(text)
            })
        
        # Chunk and learn
        chunks = self.extractor.chunk_text(text, self.chunk_size)
        chunks_learned = 0
        words_learned = 0
        
        for chunk in chunks:
            if len(chunk.strip()) < 50:
                continue
            
            # Learn from chunk
            learn_result = self.model.learn_from_text(chunk, source=url)
            chunks_learned += 1
            words_learned += learn_result.get('new_words', 0)
            
            # Store knowledge
            summary = chunk[:200] if len(chunk) > 200 else chunk
            self.memory.store_knowledge(
                content=chunk,
                summary=summary,
                source_url=url,
                source_title=title
            )
        
        # Mark source as learned
        self.memory.mark_source_learned(
            url=url,
            title=title,
            content_length=len(text),
            chunks_learned=chunks_learned,
            words_learned=words_learned
        )
        
        self.total_chunks += chunks_learned
        self.total_words += words_learned
        
        # Extract related topics and add to queue
        self._add_related_topics(text)
        
        return {
            'success': True,
            'chunks': chunks_learned,
            'words': words_learned
        }
    
    def _add_related_topics(self, text: str) -> None:
        """Extract and add related topics from text"""
        # Find capitalized terms (potential Wikipedia topics)
        potential_topics = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        
        # Filter and add unique topics
        seen = set()
        count = 0
        for topic in potential_topics:
            if topic not in seen and len(topic) > 3 and self.topic_queue.qsize() < 100:
                seen.add(topic)
                self.topic_queue.put(topic)
                count += 1
                if count >= 5:  # Add up to 5 related topics
                    break
    
    def _add_random_topics(self) -> None:
        """Add random Wikipedia articles to queue"""
        try:
            random_articles = self.wiki_search.get_random_articles(5)
            for article in random_articles:
                self.topic_queue.put(article['title'])
        except Exception as e:
            print(f"Error getting random articles: {e}")
    
    def _complete(self) -> None:
        """Handle learning completion"""
        self.is_running = False
        self.model.save()
        
        if self.on_complete:
            self.on_complete(self.get_stats())
    
    def _handle_error(self, error: str) -> None:
        """Handle and log errors"""
        self.errors.append({
            'url': self.current_url,
            'error': error,
            'time': datetime.now().isoformat()
        })
        
        if self.on_error:
            self.on_error(error)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        progress = min(100, (self.sites_learned / self.target_sites) * 100)
        
        return {
            'is_running': self.is_running,
            'progress': progress,
            'is_complete': progress >= 100,
            'sites_learned': self.sites_learned,
            'sites_skipped': self.sites_skipped,
            'total_chunks': self.total_chunks,
            'total_words': self.total_words,
            'queue_size': self.topic_queue.qsize(),
            'current_url': self.current_url,
            'current_title': self.current_title,
            'current_preview': self.current_preview,
            'error_count': len(self.errors)
        }
    
    def learn_from_url(self, url: str) -> Dict[str, Any]:
        """Learn from a specific URL (manual trigger)"""
        if self.memory.is_source_learned(url):
            return {'success': False, 'reason': 'already_learned'}
        
        # Extract topic from URL
        if '/wiki/' in url:
            topic = url.split('/wiki/')[-1].replace('_', ' ')
            result = self._learn_from_topic(topic)
        else:
            return {'success': False, 'reason': 'not_wikipedia_url'}
        
        if result.get('success'):
            self.sites_learned += 1
            self.model.save()
        
        return result
    
    def search_and_learn(self, query: str, max_articles: int = 3) -> Dict[str, Any]:
        """Search for a topic and learn from results"""
        results = self.wiki_search.search(query, limit=max_articles)
        learned_from = []
        
        if not results:
            return {
                'success': False,
                'query': query,
                'learned_from': [],
                'count': 0,
                'error': 'No search results found'
            }
        
        for result in results:
            title = result['title']
            url = result['url']
            
            if self.memory.is_source_learned(url):
                continue
            
            learn_result = self._learn_from_topic(title)
            
            if learn_result.get('success'):
                self.sites_learned += 1
                learned_from.append({
                    'title': title,
                    'url': url
                })
        
        if learned_from:
            self.model.save()
        
        return {
            'success': len(learned_from) > 0,
            'query': query,
            'learned_from': learned_from,
            'count': len(learned_from)
        }
    
    def learn_from_any_url(self, url: str) -> Dict[str, Any]:
        """
        Learn from ANY URL - not just Wikipedia.
        Uses universal scraper to extract content.
        """
        from web import UniversalScraper
        
        if self.memory.is_source_learned(url):
            return {'success': False, 'reason': 'already_learned'}
        
        scraper = UniversalScraper()
        result = scraper.fetch(url)
        
        if not result.get('success'):
            return {'success': False, 'reason': result.get('error', 'Failed to fetch')}
        
        content = result.get('content', '')
        title = result.get('title', 'Unknown')
        
        if len(content) < 100:
            return {'success': False, 'reason': 'insufficient_content'}
        
        # Chunk and learn
        chunks = self.extractor.chunk_text(content, self.chunk_size)
        chunks_learned = 0
        words_learned = 0
        
        for chunk in chunks:
            if len(chunk.strip()) < 50:
                continue
            
            learn_result = self.model.learn_from_text(chunk, source=url)
            chunks_learned += 1
            words_learned += learn_result.get('new_words', 0)
            
            summary = chunk[:200] if len(chunk) > 200 else chunk
            self.memory.store_knowledge(
                content=chunk,
                summary=summary,
                source_url=url,
                source_title=title
            )
        
        # Mark as learned
        self.memory.mark_source_learned(
            url=url,
            title=title,
            content_length=len(content),
            chunks_learned=chunks_learned,
            words_learned=words_learned
        )
        
        self.total_chunks += chunks_learned
        self.total_words += words_learned
        self.sites_learned += 1
        
        # Save
        self.model.save()
        
        return {
            'success': True,
            'title': title,
            'chunks': chunks_learned,
            'words': words_learned,
            'content_length': len(content)
        }
    
    def web_search_and_learn(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search the web and learn from top results.
        Uses DuckDuckGo for search (no API key needed).
        """
        from web import WebSearcher, UniversalScraper
        
        searcher = WebSearcher()
        scraper = UniversalScraper()
        
        # Search
        results = searcher.search(query, max_results=max_results)
        
        if not results:
            return {
                'success': False,
                'query': query,
                'error': 'No search results found',
                'learned_from': [],
                'count': 0
            }
        
        learned_from = []
        
        for result in results:
            url = result['url']
            
            # Skip if already learned
            if self.memory.is_source_learned(url):
                continue
            
            # Skip some domains that are problematic
            skip_domains = ['youtube.com', 'facebook.com', 'twitter.com', 'instagram.com', 'tiktok.com']
            if any(d in url.lower() for d in skip_domains):
                continue
            
            # Try to learn
            try:
                learn_result = self.learn_from_any_url(url)
                
                if learn_result.get('success'):
                    learned_from.append({
                        'title': learn_result.get('title', result.get('title', 'Unknown')),
                        'url': url,
                        'chunks': learn_result.get('chunks', 0)
                    })
            except Exception as e:
                print(f"Error learning from {url}: {e}")
                continue
            
            # Limit to avoid too many requests
            if len(learned_from) >= 3:
                break
        
        return {
            'success': len(learned_from) > 0,
            'query': query,
            'learned_from': learned_from,
            'count': len(learned_from),
            'results_found': len(results)
        }