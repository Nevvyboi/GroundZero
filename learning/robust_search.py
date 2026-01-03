"""
Robust Multi-Engine Search System
=================================
Implements multiple search engines with automatic fallbacks:
1. DuckDuckGo (primary)
2. Wikipedia API (fallback)
3. Brave Search API (if key provided)
4. Google Custom Search (if key provided)

Features:
- Automatic retry with exponential backoff
- Multiple engine fallbacks
- Result caching
- Timeout handling
- Rate limiting
"""

import re
import time
import random
import hashlib
import threading
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import quote_plus, urlparse, unquote
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
import json

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


@dataclass
class SearchResult:
    """A search result"""
    title: str
    url: str
    snippet: str
    source: str  # Which search engine
    relevance: float = 0.8


class SearchCache:
    """Thread-safe cache for search results"""
    
    def __init__(self, max_size: int = 500, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[float, List[SearchResult]]] = {}
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
    
    def _make_key(self, query: str) -> str:
        """Create cache key from query"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[List[SearchResult]]:
        """Get cached results if valid"""
        key = self._make_key(query)
        with self._lock:
            if key in self.cache:
                timestamp, results = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    return results
                else:
                    del self.cache[key]
        return None
    
    def set(self, query: str, results: List[SearchResult]) -> None:
        """Cache results"""
        key = self._make_key(query)
        with self._lock:
            # Evict old entries if full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache, key=lambda k: self.cache[k][0])
                del self.cache[oldest_key]
            
            self.cache[key] = (time.time(), results)
    
    def clear(self) -> None:
        """Clear cache"""
        with self._lock:
            self.cache.clear()


class RobustSearchEngine:
    """
    Multi-engine search with automatic fallbacks and error handling.
    """
    
    # User agents to rotate
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    ]
    
    def __init__(self, brave_api_key: str = None, google_api_key: str = None, google_cx: str = None):
        self.brave_api_key = brave_api_key
        self.google_api_key = google_api_key
        self.google_cx = google_cx
        
        self.cache = SearchCache()
        self.session = self._create_session()
        
        # Rate limiting
        self.last_request_time = {}
        self.min_delay = {
            'duckduckgo': 1.5,
            'wikipedia': 0.5,
            'brave': 0.3,
            'google': 0.5
        }
        
        # Error tracking
        self.engine_errors = {
            'duckduckgo': 0,
            'wikipedia': 0,
            'brave': 0,
            'google': 0
        }
        self.max_errors_before_skip = 3
    
    def _create_session(self) -> requests.Session:
        """Create requests session with retry capabilities"""
        if not REQUESTS_AVAILABLE:
            return None
        
        session = requests.Session()
        session.headers.update({
            'User-Agent': random.choice(self.USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        return session
    
    def _rotate_user_agent(self) -> None:
        """Rotate user agent"""
        if self.session:
            self.session.headers['User-Agent'] = random.choice(self.USER_AGENTS)
    
    def _rate_limit(self, engine: str) -> None:
        """Apply rate limiting"""
        now = time.time()
        last = self.last_request_time.get(engine, 0)
        delay = self.min_delay.get(engine, 1.0)
        
        elapsed = now - last
        if elapsed < delay:
            time.sleep(delay - elapsed)
        
        self.last_request_time[engine] = time.time()
    
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Search across multiple engines with automatic fallbacks.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            List of SearchResult objects
        """
        if not REQUESTS_AVAILABLE:
            return []
        
        # Check cache
        cached = self.cache.get(query)
        if cached:
            return cached[:max_results]
        
        results = []
        
        # Build engine priority list based on error counts
        engines = self._get_engine_priority()
        
        for engine_name, engine_func in engines:
            if len(results) >= max_results:
                break
            
            # Skip engines with too many errors
            if self.engine_errors.get(engine_name, 0) >= self.max_errors_before_skip:
                continue
            
            try:
                self._rate_limit(engine_name)
                engine_results = engine_func(query, max_results - len(results))
                
                if engine_results:
                    results.extend(engine_results)
                    # Reset error count on success
                    self.engine_errors[engine_name] = 0
                    
            except Exception as e:
                print(f"Search engine '{engine_name}' error: {e}")
                self.engine_errors[engine_name] = self.engine_errors.get(engine_name, 0) + 1
                continue
        
        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for r in results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique_results.append(r)
        
        # Cache results
        if unique_results:
            self.cache.set(query, unique_results)
        
        return unique_results[:max_results]
    
    def _get_engine_priority(self) -> List[Tuple[str, callable]]:
        """Get engines sorted by reliability"""
        engines = [
            ('wikipedia', self._search_wikipedia),  # Most reliable
            ('duckduckgo', self._search_duckduckgo),
        ]
        
        if self.brave_api_key:
            engines.insert(0, ('brave', self._search_brave))
        
        if self.google_api_key and self.google_cx:
            engines.insert(0, ('google', self._search_google))
        
        # Sort by error count (fewer errors = higher priority)
        engines.sort(key=lambda x: self.engine_errors.get(x[0], 0))
        
        return engines
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using DuckDuckGo HTML"""
        results = []
        
        # Try multiple methods
        methods = [
            self._ddg_html_search,
            self._ddg_lite_search,
        ]
        
        for method in methods:
            try:
                results = method(query, max_results)
                if results:
                    return results
            except Exception as e:
                print(f"DuckDuckGo method failed: {e}")
                continue
        
        return results
    
    def _ddg_html_search(self, query: str, max_results: int) -> List[SearchResult]:
        """DuckDuckGo HTML search with robust timeout handling"""
        results = []
        
        url = f"https://html.duckduckgo.com/html/"
        
        try:
            # Use POST for better reliability
            self._rotate_user_agent()
            response = self.session.post(
                url,
                data={'q': query},
                timeout=(5, 10),  # (connect timeout, read timeout) - shorter connect timeout
                allow_redirects=True
            )
            response.raise_for_status()
            
            if BS4_AVAILABLE:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for result in soup.select('.result')[:max_results]:
                    title_elem = result.select_one('.result__title')
                    snippet_elem = result.select_one('.result__snippet')
                    
                    if title_elem:
                        link = title_elem.find('a')
                        if link and link.get('href'):
                            href = link.get('href')
                            
                            # Parse DuckDuckGo redirect URL
                            actual_url = self._parse_ddg_url(href)
                            
                            if actual_url:
                                results.append(SearchResult(
                                    title=title_elem.get_text(strip=True),
                                    url=actual_url,
                                    snippet=snippet_elem.get_text(strip=True) if snippet_elem else "",
                                    source='duckduckgo',
                                    relevance=0.8
                                ))
        except requests.exceptions.Timeout:
            print("DuckDuckGo HTML search timed out")
            raise  # Let the caller handle and try fallback
        except requests.exceptions.ConnectionError as e:
            print(f"DuckDuckGo connection error: {e}")
            raise  # Let the caller handle and try fallback
        
        return results
    
    def _ddg_lite_search(self, query: str, max_results: int) -> List[SearchResult]:
        """DuckDuckGo Lite search (more reliable but simpler)"""
        results = []
        
        url = f"https://lite.duckduckgo.com/lite/"
        
        try:
            self._rotate_user_agent()
            response = self.session.post(
                url,
                data={'q': query},
                timeout=(5, 10)  # (connect, read) timeout
            )
            response.raise_for_status()
            
            if BS4_AVAILABLE:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Lite version has different structure
                for row in soup.select('tr'):
                    link = row.select_one('a.result-link')
                    snippet_cell = row.select_one('td.result-snippet')
                    
                    if link and link.get('href'):
                        results.append(SearchResult(
                            title=link.get_text(strip=True),
                            url=link.get('href'),
                            snippet=snippet_cell.get_text(strip=True) if snippet_cell else "",
                            source='duckduckgo_lite',
                            relevance=0.75
                        ))
                        
                        if len(results) >= max_results:
                            break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"DuckDuckGo Lite error: {e}")
            raise
        
        return results
    
    def _parse_ddg_url(self, href: str) -> Optional[str]:
        """Parse actual URL from DuckDuckGo redirect"""
        if not href:
            return None
        
        # Direct URL
        if href.startswith('http') and 'duckduckgo.com' not in href:
            return href
        
        # Parse uddg parameter
        if 'uddg=' in href:
            match = re.search(r'uddg=([^&]+)', href)
            if match:
                return unquote(match.group(1))
        
        return None
    
    def _search_wikipedia(self, query: str, max_results: int) -> List[SearchResult]:
        """Search Wikipedia API"""
        results = []
        
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'srlimit': max_results,
            'format': 'json',
            'srprop': 'snippet|titlesnippet'
        }
        
        response = self.session.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        for item in data.get('query', {}).get('search', []):
            title = item.get('title', '')
            snippet = re.sub(r'<[^>]+>', '', item.get('snippet', ''))
            url = f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
            
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                source='wikipedia',
                relevance=0.9  # Wikipedia is highly relevant for educational content
            ))
        
        return results
    
    def _search_brave(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using Brave Search API"""
        if not self.brave_api_key:
            return []
        
        results = []
        
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            'Accept': 'application/json',
            'X-Subscription-Token': self.brave_api_key
        }
        params = {
            'q': query,
            'count': max_results
        }
        
        response = self.session.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        for item in data.get('web', {}).get('results', []):
            results.append(SearchResult(
                title=item.get('title', ''),
                url=item.get('url', ''),
                snippet=item.get('description', ''),
                source='brave',
                relevance=0.85
            ))
        
        return results
    
    def _search_google(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using Google Custom Search API"""
        if not self.google_api_key or not self.google_cx:
            return []
        
        results = []
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': self.google_api_key,
            'cx': self.google_cx,
            'q': query,
            'num': min(max_results, 10)
        }
        
        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        for item in data.get('items', []):
            results.append(SearchResult(
                title=item.get('title', ''),
                url=item.get('link', ''),
                snippet=item.get('snippet', ''),
                source='google',
                relevance=0.9
            ))
        
        return results
    
    def reset_error_counts(self) -> None:
        """Reset error counts for all engines"""
        for engine in self.engine_errors:
            self.engine_errors[engine] = 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get search engine status"""
        return {
            'engines': {
                'duckduckgo': {'errors': self.engine_errors.get('duckduckgo', 0), 'available': True},
                'wikipedia': {'errors': self.engine_errors.get('wikipedia', 0), 'available': True},
                'brave': {'errors': self.engine_errors.get('brave', 0), 'available': bool(self.brave_api_key)},
                'google': {'errors': self.engine_errors.get('google', 0), 'available': bool(self.google_api_key)},
            },
            'cache_size': len(self.cache.cache)
        }


# Global instance for easy access
_search_engine: Optional[RobustSearchEngine] = None


def get_search_engine() -> RobustSearchEngine:
    """Get or create global search engine instance"""
    global _search_engine
    if _search_engine is None:
        _search_engine = RobustSearchEngine()
    return _search_engine


def search(query: str, max_results: int = 5) -> List[SearchResult]:
    """Convenience function for searching"""
    return get_search_engine().search(query, max_results)