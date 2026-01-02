"""
Universal Scraper
=================
Scrapes and extracts content from any website.
"""

import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict, Any, List
from urllib.parse import urljoin, urlparse
import re


class UniversalScraper:
    """
    Scrapes content from any website.
    Handles various page structures and content types.
    """
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    # Tags to remove
    REMOVE_TAGS = [
        'script', 'style', 'nav', 'footer', 'header', 'aside',
        'noscript', 'iframe', 'form', 'button', 'input', 'select',
        'svg', 'canvas', 'video', 'audio', 'advertisement', 'ads'
    ]
    
    # Classes/IDs to remove
    REMOVE_PATTERNS = [
        'sidebar', 'navigation', 'nav-', 'menu', 'footer', 'header',
        'advertisement', 'ad-', 'ads-', 'banner', 'popup', 'modal',
        'social', 'share', 'comment', 'related', 'recommended',
        'cookie', 'newsletter', 'subscribe', 'promo'
    ]
    
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
    
    def fetch(self, url: str) -> Dict[str, Any]:
        """
        Fetch a web page and extract its content.
        
        Returns dict with: url, title, content, success, error
        """
        try:
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'text/html' not in content_type and 'application/xhtml' not in content_type:
                return {
                    'url': url,
                    'success': False,
                    'error': f'Unsupported content type: {content_type}'
                }
            
            html = response.text
            title = self._extract_title(html)
            content = self._extract_content(html, url)
            
            return {
                'url': response.url,  # May differ if redirected
                'title': title,
                'content': content,
                'content_length': len(content),
                'success': True
            }
            
        except requests.Timeout:
            return {'url': url, 'success': False, 'error': 'Request timed out'}
        except requests.RequestException as e:
            return {'url': url, 'success': False, 'error': str(e)}
    
    def _extract_title(self, html: str) -> str:
        """Extract page title"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Try og:title first
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                return og_title['content'].strip()
            
            # Try regular title
            if soup.title and soup.title.string:
                return soup.title.string.strip()
            
            # Try h1
            h1 = soup.find('h1')
            if h1:
                return h1.get_text(strip=True)
                
        except:
            pass
        return "Unknown"
    
    def _extract_content(self, html: str, url: str) -> str:
        """Extract main text content from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        self._clean_soup(soup)
        
        # Try to find main content area
        content_area = self._find_main_content(soup, url)
        
        # Extract text
        text = self._extract_text(content_area)
        
        # Clean up
        text = self._clean_text(text)
        
        return text
    
    def _clean_soup(self, soup: BeautifulSoup) -> None:
        """Remove unwanted elements"""
        # Remove specific tags
        for tag in self.REMOVE_TAGS:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Remove elements with certain classes/IDs
        for pattern in self.REMOVE_PATTERNS:
            for element in soup.find_all(class_=re.compile(pattern, re.I)):
                element.decompose()
            for element in soup.find_all(id=re.compile(pattern, re.I)):
                element.decompose()
    
    def _find_main_content(self, soup: BeautifulSoup, url: str) -> BeautifulSoup:
        """Find the main content area"""
        # Site-specific selectors
        domain = urlparse(url).netloc.lower()
        
        # Wikipedia
        if 'wikipedia.org' in domain:
            content = soup.find('div', {'id': 'mw-content-text'})
            if content:
                return content
        
        # Medium
        if 'medium.com' in domain:
            content = soup.find('article')
            if content:
                return content
        
        # Generic article detection
        selectors = [
            ('article', {}),
            ('main', {}),
            ('div', {'role': 'main'}),
            ('div', {'id': 'content'}),
            ('div', {'id': 'main-content'}),
            ('div', {'class': 'content'}),
            ('div', {'class': 'article'}),
            ('div', {'class': 'post-content'}),
            ('div', {'class': 'entry-content'}),
            ('div', {'class': 'article-content'}),
            ('div', {'class': 'story-body'}),
        ]
        
        for tag, attrs in selectors:
            element = soup.find(tag, attrs)
            if element and len(element.get_text(strip=True)) > 200:
                return element
        
        # Fallback: find the div with most text
        return self._find_largest_text_block(soup)
    
    def _find_largest_text_block(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Find the div with the most text content"""
        best = soup.body or soup
        best_length = 0
        
        for div in soup.find_all(['div', 'article', 'section']):
            text_length = len(div.get_text(strip=True))
            # Prefer elements with reasonable depth
            if text_length > best_length and text_length < 100000:
                # Check if it has actual paragraphs
                paragraphs = div.find_all('p')
                if len(paragraphs) >= 2:
                    best = div
                    best_length = text_length
        
        return best
    
    def _extract_text(self, content: BeautifulSoup) -> str:
        """Extract text from content area"""
        parts = []
        
        for element in content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'td', 'th']):
            text = element.get_text(separator=' ', strip=True)
            
            if not text or len(text) < 10:
                continue
            
            # Add headers with formatting
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                parts.append(f"\n## {text}\n")
            elif element.name == 'li':
                parts.append(f"• {text}")
            else:
                parts.append(text)
        
        return '\n'.join(parts)
    
    def _clean_text(self, text: str) -> str:
        """Clean up extracted text"""
        # Remove citation markers
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\[citation needed\]', '', text, flags=re.I)
        text = re.sub(r'\[edit\]', '', text, flags=re.I)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove very short lines
        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 20 or line.startswith('#')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def extract_links(self, html: str, base_url: str, same_domain: bool = True) -> List[str]:
        """Extract links from page"""
        soup = BeautifulSoup(html, 'html.parser')
        base_domain = urlparse(base_url).netloc
        links = []
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            
            # Skip anchors and special links
            if href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
                continue
            
            # Resolve relative URLs
            full_url = urljoin(base_url, href)
            
            # Skip non-HTTP
            if not full_url.startswith('http'):
                continue
            
            # Filter by domain if requested
            if same_domain:
                if urlparse(full_url).netloc != base_domain:
                    continue
            
            links.append(full_url)
        
        return list(set(links))


class WebSearcher:
    """
    Searches the web using DuckDuckGo (no API key required).
    """
    
    SEARCH_URL = "https://html.duckduckgo.com/html/"
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
    
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """
        Search the web and return results.
        
        Returns list of dicts with: title, url, snippet
        """
        try:
            response = self.session.post(
                self.SEARCH_URL,
                data={'q': query, 'b': ''},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Find result elements
            for result in soup.find_all('div', class_='result'):
                if len(results) >= max_results:
                    break
                
                # Get title and URL
                title_elem = result.find('a', class_='result__a')
                if not title_elem:
                    continue
                
                title = title_elem.get_text(strip=True)
                url = title_elem.get('href', '')
                
                # DuckDuckGo uses redirect URLs, extract the actual URL
                if 'uddg=' in url:
                    import urllib.parse
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                    url = parsed.get('uddg', [url])[0]
                
                # Get snippet
                snippet_elem = result.find('a', class_='result__snippet')
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                
                if url and url.startswith('http'):
                    results.append({
                        'title': title,
                        'url': url,
                        'snippet': snippet
                    })
            
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []