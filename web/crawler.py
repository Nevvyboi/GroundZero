"""
Web Crawler
===========
Fetches and parses web pages for knowledge extraction.
"""

import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict, Any, List
from urllib.parse import urljoin, urlparse
import time
import threading


class WebCrawler:
    """Handles web page fetching and parsing"""
    
    # Wikipedia requires a proper User-Agent with contact info
    DEFAULT_HEADERS = {
        'User-Agent': 'NeuralMind/1.0 (Educational AI Project; https://github.com/neuralmind) Python/3.x',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive'
    }
    
    def __init__(self, timeout: int = 15, delay: float = 1.0):
        self.timeout = timeout
        self.delay = delay
        self.last_request_time = 0
        self._lock = threading.Lock()
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests"""
        with self._lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            self.last_request_time = time.time()
    
    def fetch(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a web page.
        
        Returns dict with:
            - url: The fetched URL
            - title: Page title
            - html: Raw HTML content
            - status_code: HTTP status code
        """
        self._rate_limit()
        
        try:
            response = requests.get(
                url,
                headers=self.DEFAULT_HEADERS,
                timeout=self.timeout,
                allow_redirects=True
            )
            response.raise_for_status()
            
            return {
                'url': response.url,
                'title': self._extract_title(response.text),
                'html': response.text,
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type', ''),
                'content_length': len(response.text)
            }
        except requests.RequestException as e:
            return {
                'url': url,
                'error': str(e),
                'status_code': getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            }
    
    def _extract_title(self, html: str) -> str:
        """Extract page title from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            if soup.title and soup.title.string:
                return soup.title.string.strip()
        except:
            pass
        return "Unknown"
    
    def extract_links(
        self,
        html: str,
        base_url: str,
        filter_pattern: Optional[str] = None
    ) -> List[str]:
        """
        Extract links from HTML.
        
        Args:
            html: HTML content
            base_url: Base URL for resolving relative links
            filter_pattern: Optional pattern to filter links (e.g., '/wiki/')
        """
        links = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            for a in soup.find_all('a', href=True):
                href = a['href']
                
                # Skip anchors, javascript, mailto
                if href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
                    continue
                
                # Resolve relative URLs
                full_url = urljoin(base_url, href)
                
                # Apply filter if specified
                if filter_pattern and filter_pattern not in full_url:
                    continue
                
                # Skip non-HTTP URLs
                if not full_url.startswith('http'):
                    continue
                
                links.append(full_url)
        except:
            pass
        
        return list(set(links))  # Remove duplicates
    
    def get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        parsed = urlparse(url)
        return parsed.netloc


class WikipediaCrawler(WebCrawler):
    """Specialized crawler for Wikipedia"""
    
    WIKI_API = "https://en.wikipedia.org/w/api.php"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create session with proper headers for Wikipedia
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NeuralMind/1.0 (Educational AI Project; https://github.com/neuralmind) Python/3.x',
            'Accept': 'application/json'
        })
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        """
        Search Wikipedia.
        
        Returns list of dicts with: title, snippet, url
        """
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'format': 'json',
            'srlimit': limit
        }
        
        try:
            response = self.session.get(self.WIKI_API, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get('query', {}).get('search', []):
                # Clean snippet (remove HTML tags)
                snippet = BeautifulSoup(item.get('snippet', ''), 'html.parser').get_text()
                
                results.append({
                    'title': item['title'],
                    'snippet': snippet,
                    'url': f"https://en.wikipedia.org/wiki/{item['title'].replace(' ', '_')}"
                })
            
            return results
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return []
    
    def get_article_content(self, title: str) -> Optional[str]:
        """Get plain text content of a Wikipedia article"""
        params = {
            'action': 'query',
            'titles': title,
            'prop': 'extracts',
            'exintro': False,
            'explaintext': True,
            'format': 'json'
        }
        
        try:
            response = self.session.get(self.WIKI_API, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            pages = data.get('query', {}).get('pages', {})
            for page_data in pages.values():
                if 'extract' in page_data:
                    return page_data['extract']
        except Exception as e:
            print(f"Wikipedia content error: {e}")
        
        return None
    
    def extract_wiki_links(self, html: str, base_url: str) -> List[str]:
        """Extract Wikipedia article links from HTML"""
        links = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Focus on content area
            content = soup.find('div', {'id': 'mw-content-text'}) or soup
            
            for a in content.find_all('a', href=True):
                href = a['href']
                
                # Only internal wiki links
                if href.startswith('/wiki/') and ':' not in href:
                    full_url = urljoin(base_url, href)
                    links.append(full_url)
        except:
            pass
        
        return list(set(links))
