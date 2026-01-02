"""
Web Search
==========
Search functionality for knowledge acquisition.
"""

import requests
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup


class WikipediaSearch:
    """Search and retrieve content from Wikipedia"""
    
    API_URL = "https://en.wikipedia.org/w/api.php"
    
    # Wikipedia requires proper User-Agent
    HEADERS = {
        'User-Agent': 'NeuralMind/1.0 (Educational AI Project; https://github.com/neuralmind) Python/3.x',
        'Accept': 'application/json'
    }
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search Wikipedia for articles.
        
        Returns list of dicts with: title, snippet, url, pageid
        """
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'format': 'json',
            'srlimit': limit,
            'srprop': 'snippet|titlesnippet'
        }
        
        try:
            response = self.session.get(self.API_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get('query', {}).get('search', []):
                snippet = BeautifulSoup(item.get('snippet', ''), 'html.parser').get_text()
                title = item['title']
                
                results.append({
                    'title': title,
                    'snippet': snippet,
                    'url': f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    'pageid': item.get('pageid')
                })
            
            return results
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return []
    
    def get_content(self, title: str) -> Optional[Dict[str, Any]]:
        """
        Get full content of a Wikipedia article.
        
        Returns dict with: title, content, url, sections
        """
        params = {
            'action': 'query',
            'titles': title,
            'prop': 'extracts|info',
            'exintro': False,
            'explaintext': True,
            'format': 'json',
            'inprop': 'url'
        }
        
        try:
            response = self.session.get(self.API_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                if page_id == '-1':  # Page not found
                    return None
                
                return {
                    'title': page_data.get('title', title),
                    'content': page_data.get('extract', ''),
                    'url': page_data.get('fullurl', f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"),
                    'pageid': page_id
                }
        except Exception as e:
            print(f"Wikipedia content error: {e}")
        
        return None
    
    def get_summary(self, title: str) -> Optional[str]:
        """Get just the introduction/summary of an article"""
        params = {
            'action': 'query',
            'titles': title,
            'prop': 'extracts',
            'exintro': True,
            'explaintext': True,
            'format': 'json'
        }
        
        try:
            response = self.session.get(self.API_URL, params=params, timeout=self.timeout)
            data = response.json()
            
            pages = data.get('query', {}).get('pages', {})
            for page_data in pages.values():
                if 'extract' in page_data:
                    return page_data['extract']
        except:
            pass
        
        return None
    
    def get_random_articles(self, count: int = 5) -> List[Dict[str, str]]:
        """Get random Wikipedia articles"""
        params = {
            'action': 'query',
            'list': 'random',
            'rnlimit': count,
            'rnnamespace': 0,  # Main namespace only
            'format': 'json'
        }
        
        try:
            response = self.session.get(self.API_URL, params=params, timeout=self.timeout)
            data = response.json()
            
            results = []
            for item in data.get('query', {}).get('random', []):
                title = item['title']
                results.append({
                    'title': title,
                    'url': f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                })
            
            return results
        except:
            return []


class SearchAggregator:
    """Aggregates results from multiple search sources"""
    
    def __init__(self):
        self.wikipedia = WikipediaSearch()
    
    def search(
        self,
        query: str,
        sources: List[str] = ['wikipedia'],
        limit: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Search across multiple sources.
        
        Returns dict mapping source name to results.
        """
        results = {}
        
        if 'wikipedia' in sources:
            results['wikipedia'] = self.wikipedia.search(query, limit)
        
        return results
    
    def search_and_get_content(
        self,
        query: str,
        max_articles: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search and retrieve full content for top results.
        
        Returns list of articles with full content.
        """
        # Search
        search_results = self.wikipedia.search(query, limit=max_articles)
        
        # Get content for each result
        articles = []
        for result in search_results:
            content = self.wikipedia.get_content(result['title'])
            if content:
                articles.append(content)
        
        return articles
