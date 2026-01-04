"""
Web Module
==========
Wikipedia search and web content extraction.
"""

import re
import requests
from typing import Dict, Any, List, Optional
from urllib.parse import quote
import time


class WikipediaSearch:
    """
    Search and fetch Wikipedia articles.
    Uses the Wikipedia API for content extraction.
    """
    
    API_URL = "https://en.wikipedia.org/w/api.php"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GroundZero/2.0 (Learning AI; educational project)'
        })
    
    def get_random_articles(self, count: int = 5) -> List[Dict[str, str]]:
        """Get random Wikipedia articles"""
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'random',
                'rnnamespace': 0,
                'rnlimit': count
            }
            
            response = self.session.get(self.API_URL, params=params, timeout=10)
            data = response.json()
            
            articles = []
            for item in data.get('query', {}).get('random', []):
                title = item.get('title', '')
                if title:
                    articles.append({
                        'title': title,
                        'url': f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                    })
            
            return articles
        except Exception as e:
            print(f"Error getting random articles: {e}")
            return []
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        """Search Wikipedia for articles"""
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'srlimit': limit
            }
            
            response = self.session.get(self.API_URL, params=params, timeout=10)
            data = response.json()
            
            articles = []
            for item in data.get('query', {}).get('search', []):
                title = item.get('title', '')
                if title:
                    articles.append({
                        'title': title,
                        'url': f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                    })
            
            return articles
        except Exception as e:
            print(f"Error searching Wikipedia: {e}")
            return []
    
    def get_article_content(self, title: str) -> Optional[Dict[str, Any]]:
        """Get full article content"""
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts|info',
                'explaintext': True,
                'exsectionformat': 'plain',
                'inprop': 'url'
            }
            
            response = self.session.get(self.API_URL, params=params, timeout=15)
            data = response.json()
            
            pages = data.get('query', {}).get('pages', {})
            
            for page_id, page in pages.items():
                if page_id == '-1':
                    continue
                
                content = page.get('extract', '')
                
                if content and len(content) > 100:
                    return {
                        'title': page.get('title', title),
                        'content': content,
                        'url': page.get('fullurl', f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"),
                        'word_count': len(content.split())
                    }
            
            return None
        except Exception as e:
            print(f"Error fetching '{title}': {e}")
            return None
    
    def get_article_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Get article by URL"""
        match = re.search(r'wikipedia\.org/wiki/(.+?)(?:\?|#|$)', url)
        if match:
            title = match.group(1).replace('_', ' ')
            return self.get_article_content(title)
        return None


class ContentExtractor:
    """
    Extract clean text from web pages.
    Used for non-Wikipedia URLs.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def extract(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract content from URL"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            html = response.text
            
            # Extract title
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else url
            
            # Clean HTML
            html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
            
            # Extract paragraphs
            paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html, re.DOTALL | re.IGNORECASE)
            
            text_parts = []
            for p in paragraphs:
                text = re.sub(r'<[^>]+>', '', p)
                text = ' '.join(text.split())
                if len(text) > 50:
                    text_parts.append(text)
            
            content = '\n\n'.join(text_parts)
            
            if len(content) < 100:
                text = re.sub(r'<[^>]+>', ' ', html)
                content = ' '.join(text.split())[:10000]
            
            return {
                'title': title[:200],
                'content': content,
                'url': url,
                'word_count': len(content.split())
            }
        except Exception as e:
            print(f"Error extracting {url}: {e}")
            return None
