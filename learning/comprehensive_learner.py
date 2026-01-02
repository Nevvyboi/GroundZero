"""
Comprehensive Web Learning System
=================================
Uses the whole internet for learning, not just Wikipedia.
Implements robust web scraping and search integration.
"""

import re
import time
import random
from typing import List, Dict, Any, Optional, Callable
from urllib.parse import urlparse, urljoin
import threading
from queue import Queue, Empty
from datetime import datetime

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False


class ComprehensiveWebSearcher:
    """
    Searches the web using multiple search engines.
    Primary: DuckDuckGo (no API key needed)
    Fallback: Wikipedia API
    """
    
    def __init__(self):
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        self.headers = {'User-Agent': self.user_agent}
        self.last_search_time = 0
        self.min_delay = 1.5  # Respect rate limits
    
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search the web for a query.
        Returns list of results with title, url, snippet.
        """
        results = []
        
        # Rate limiting
        elapsed = time.time() - self.last_search_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        
        # Try DuckDuckGo first
        if HAS_DDGS:
            try:
                results = self._search_duckduckgo(query, max_results)
                if results:
                    self.last_search_time = time.time()
                    return results
            except Exception as e:
                print(f"DuckDuckGo search error: {e}")
        
        # Fallback to Wikipedia
        results = self._search_wikipedia(query, max_results)
        self.last_search_time = time.time()
        
        return results
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo"""
        results = []
        
        try:
            with DDGS() as ddgs:
                ddg_results = list(ddgs.text(query, max_results=max_results))
                
                for r in ddg_results:
                    results.append({
                        'title': r.get('title', ''),
                        'url': r.get('href', r.get('link', '')),
                        'snippet': r.get('body', r.get('snippet', '')),
                        'source': 'duckduckgo'
                    })
        except Exception as e:
            print(f"DuckDuckGo error: {e}")
        
        return results
    
    def _search_wikipedia(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Fallback: Search Wikipedia API"""
        results = []
        
        try:
            url = 'https://en.wikipedia.org/w/api.php'
            params = {
                'action': 'query',
                'list': 'search',
                'srsearch': query,
                'srlimit': max_results,
                'format': 'json'
            }
            
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            data = response.json()
            
            for item in data.get('query', {}).get('search', []):
                title = item.get('title', '')
                results.append({
                    'title': title,
                    'url': f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    'snippet': re.sub(r'<[^>]+>', '', item.get('snippet', '')),
                    'source': 'wikipedia'
                })
        except Exception as e:
            print(f"Wikipedia search error: {e}")
        
        return results


class RobustWebScraper:
    """
    Robust web scraper that extracts clean text from any URL.
    Handles various HTML structures and edge cases.
    """
    
    def __init__(self):
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        self.headers = {'User-Agent': self.user_agent}
        self.timeout = 15
        
        # Content removal patterns
        self.remove_patterns = [
            r'<script[^>]*>.*?</script>',
            r'<style[^>]*>.*?</style>',
            r'<nav[^>]*>.*?</nav>',
            r'<footer[^>]*>.*?</footer>',
            r'<header[^>]*>.*?</header>',
            r'<aside[^>]*>.*?</aside>',
            r'<!--.*?-->',
        ]
        
        # Skip domains
        self.skip_domains = [
            'youtube.com', 'facebook.com', 'twitter.com', 'instagram.com',
            'tiktok.com', 'reddit.com', 'pinterest.com', 'linkedin.com'
        ]
    
    def fetch(self, url: str) -> Dict[str, Any]:
        """
        Fetch and extract content from URL.
        Returns: title, content, word_count, success
        """
        # Check domain
        domain = urlparse(url).netloc
        if any(skip in domain for skip in self.skip_domains):
            return {'success': False, 'error': f'Skipped domain: {domain}'}
        
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Detect encoding
            if response.encoding and response.encoding.lower() != 'utf-8':
                html = response.content.decode(response.encoding, errors='ignore')
            else:
                html = response.text
            
            # Parse HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract title
            title = self._extract_title(soup, url)
            
            # Extract main content
            content = self._extract_content(soup)
            
            if not content or len(content) < 100:
                return {'success': False, 'error': 'Insufficient content'}
            
            return {
                'success': True,
                'url': url,
                'title': title,
                'content': content,
                'word_count': len(content.split())
            }
            
        except requests.Timeout:
            return {'success': False, 'error': 'Request timeout'}
        except requests.RequestException as e:
            return {'success': False, 'error': str(e)}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_title(self, soup: BeautifulSoup, url: str) -> str:
        """Extract page title"""
        # Try various title sources
        if soup.title:
            return soup.title.string.strip() if soup.title.string else ''
        
        # Try h1
        h1 = soup.find('h1')
        if h1:
            return h1.get_text().strip()
        
        # Try og:title
        og_title = soup.find('meta', property='og:title')
        if og_title:
            return og_title.get('content', '')
        
        return urlparse(url).netloc
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main text content"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 
                           'aside', 'form', 'button', 'input', 'select']):
            element.decompose()
        
        # Try article/main content first
        main_content = None
        for selector in ['article', 'main', '[role="main"]', '.content', 
                        '#content', '.post-content', '.article-content']:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            # Fallback to body
            body = soup.body
            if body:
                text = body.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)
        
        # Clean text
        text = self._clean_text(text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common junk
        junk_patterns = [
            r'Cookie Policy.*?Accept',
            r'Subscribe to our newsletter.*?Email',
            r'Sign up.*?account',
            r'Read more articles',
            r'Share this article',
            r'Advertisement',
            r'Loading\.\.\.',
        ]
        
        for pattern in junk_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove excess whitespace again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


class LiveLearningSystem:
    """
    Real-time learning system with live progress updates.
    Provides callbacks for UI updates during learning.
    """
    
    def __init__(self, memory_store, neural_model):
        self.memory = memory_store
        self.model = neural_model
        self.searcher = ComprehensiveWebSearcher()
        self.scraper = RobustWebScraper()
        
        # Learning state
        self.is_learning = False
        self.current_url = None
        self.current_title = None
        self.current_text = None
        self.current_word_index = 0
        
        # Statistics
        self.sources_learned = 0
        self.total_words = 0
        self.total_chunks = 0
        
        # Callbacks
        self.on_start: Optional[Callable] = None
        self.on_reading: Optional[Callable] = None  # Called as text is being read
        self.on_word: Optional[Callable] = None     # Called for each word being processed
        self.on_source_complete: Optional[Callable] = None
        self.on_complete: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
    
    def search_and_learn(self, query: str, max_sources: int = 3) -> Dict[str, Any]:
        """
        Search for a topic and learn from top results.
        Provides real-time updates via callbacks.
        """
        self.is_learning = True
        learned_from = []
        
        # Notify start
        if self.on_start:
            self.on_start({
                'query': query,
                'status': 'Searching...',
                'phase': 'search'
            })
        
        # Search
        results = self.searcher.search(query, max_results=max_sources * 2)
        
        if not results:
            self.is_learning = False
            if self.on_error:
                self.on_error({'error': 'No search results found', 'query': query})
            return {'success': False, 'error': 'No results found'}
        
        # Learn from each result
        for result in results:
            if len(learned_from) >= max_sources:
                break
            
            url = result['url']
            
            # Skip if already learned
            if self.memory.is_source_learned(url):
                continue
            
            # Learn from this URL
            learn_result = self._learn_from_url_with_updates(url, result.get('title', ''))
            
            if learn_result.get('success'):
                learned_from.append({
                    'title': learn_result['title'],
                    'url': url,
                    'words': learn_result.get('words', 0),
                    'chunks': learn_result.get('chunks', 0)
                })
        
        self.is_learning = False
        
        # Notify complete
        if self.on_complete:
            self.on_complete({
                'query': query,
                'learned_from': learned_from,
                'total_sources': len(learned_from)
            })
        
        # Save model
        self.model.save()
        
        return {
            'success': len(learned_from) > 0,
            'query': query,
            'learned_from': learned_from,
            'count': len(learned_from)
        }
    
    def _learn_from_url_with_updates(self, url: str, hint_title: str = '') -> Dict[str, Any]:
        """Learn from URL with real-time updates"""
        self.current_url = url
        
        # Fetch content
        fetch_result = self.scraper.fetch(url)
        
        if not fetch_result.get('success'):
            if self.on_error:
                self.on_error({'url': url, 'error': fetch_result.get('error', 'Failed to fetch')})
            return {'success': False}
        
        content = fetch_result['content']
        title = fetch_result['title'] or hint_title
        self.current_title = title
        self.current_text = content
        
        # Notify reading started
        if self.on_reading:
            self.on_reading({
                'url': url,
                'title': title,
                'total_words': len(content.split()),
                'preview': content[:500]
            })
        
        # Chunk content
        chunks = self._chunk_text(content, chunk_size=400)
        chunks_learned = 0
        words_learned = 0
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue
            
            # Simulate word-by-word reading for UI
            words = chunk.split()
            for j, word in enumerate(words):
                self.current_word_index = j
                if self.on_word and j % 10 == 0:  # Update every 10 words
                    self.on_word({
                        'word': word,
                        'word_index': j,
                        'total_words': len(words),
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'progress': (i * len(words) + j) / (len(chunks) * len(words)) * 100
                    })
            
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
            content_length=len(content),
            chunks_learned=chunks_learned,
            words_learned=words_learned
        )
        
        # Update stats
        self.sources_learned += 1
        self.total_words += words_learned
        self.total_chunks += chunks_learned
        
        # Notify source complete
        if self.on_source_complete:
            self.on_source_complete({
                'url': url,
                'title': title,
                'words': words_learned,
                'chunks': chunks_learned,
                'total_sources': self.sources_learned
            })
        
        return {
            'success': True,
            'title': title,
            'words': words_learned,
            'chunks': chunks_learned
        }
    
    def _chunk_text(self, text: str, chunk_size: int = 400) -> List[str]:
        """Split text into chunks"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            words = len(sentence.split())
            
            if current_size + words > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = words
            else:
                current_chunk.append(sentence)
                current_size += words
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            'sources_learned': self.sources_learned,
            'total_words': self.total_words,
            'total_chunks': self.total_chunks,
            'is_learning': self.is_learning,
            'current_url': self.current_url,
            'current_title': self.current_title
        }


class AutoLearner:
    """
    Autonomous learning system that decides what to learn.
    Uses knowledge graph to identify gaps.
    """
    
    def __init__(self, memory_store, neural_model, knowledge_graph=None):
        self.memory = memory_store
        self.model = neural_model
        self.knowledge_graph = knowledge_graph
        self.live_learner = LiveLearningSystem(memory_store, neural_model)
        
        # Learning queue
        self.topic_queue: Queue = Queue()
        self.is_running = False
        self.learning_thread: Optional[threading.Thread] = None
        
        # Seed topics for initial learning
        self.seed_topics = [
            'artificial intelligence', 'machine learning', 'computer science',
            'physics', 'chemistry', 'biology', 'mathematics', 'history',
            'geography', 'economics', 'philosophy', 'literature', 'art',
            'music', 'technology', 'engineering', 'medicine', 'psychology'
        ]
    
    def start_auto_learning(self, target_sources: int = 50) -> None:
        """Start autonomous learning"""
        if self.is_running:
            return
        
        self.is_running = True
        self.target_sources = target_sources
        
        # Seed the queue
        for topic in self.seed_topics:
            self.topic_queue.put(topic)
        
        # Start learning thread
        self.learning_thread = threading.Thread(target=self._auto_learn_loop, daemon=True)
        self.learning_thread.start()
    
    def stop_auto_learning(self) -> None:
        """Stop autonomous learning"""
        self.is_running = False
        if self.learning_thread:
            self.learning_thread.join(timeout=2)
    
    def _auto_learn_loop(self) -> None:
        """Main autonomous learning loop"""
        while self.is_running:
            if self.live_learner.sources_learned >= self.target_sources:
                self.is_running = False
                break
            
            try:
                # Get next topic
                topic = self.topic_queue.get(timeout=2)
                
                # Learn about this topic
                self.live_learner.search_and_learn(topic, max_sources=2)
                
                # Add related topics to queue
                if self.knowledge_graph:
                    suggestions = self.knowledge_graph.suggest_topics_to_learn(3)
                    for s in suggestions:
                        self.topic_queue.put(s)
                
                # Rate limiting
                time.sleep(2)
                
            except Empty:
                # Add random topic
                topic = random.choice(self.seed_topics)
                self.topic_queue.put(topic)
            except Exception as e:
                print(f"Auto-learn error: {e}")
                time.sleep(2)
    
    def learn_about_question(self, question: str) -> Dict[str, Any]:
        """Learn specifically to answer a question"""
        return self.live_learner.search_and_learn(question, max_sources=3)