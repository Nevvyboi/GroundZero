"""
Web Learning Engine - Fetches and learns from websites across the internet
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import random
import time
import threading
from queue import Queue, Empty
from typing import List, Dict, Optional, Callable
from datetime import datetime
import json
import re


class WebLearner:
    """
    Autonomous web learning system that crawls and learns from websites
    """
    
    # Seed URLs to start learning from (diverse knowledge sources)
    SEED_URLS = [
        # Science & Technology
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Computer_science",
        "https://en.wikipedia.org/wiki/Physics",
        "https://en.wikipedia.org/wiki/Mathematics",
        "https://en.wikipedia.org/wiki/Biology",
        "https://en.wikipedia.org/wiki/Chemistry",
        
        # History & Culture
        "https://en.wikipedia.org/wiki/World_history",
        "https://en.wikipedia.org/wiki/Philosophy",
        "https://en.wikipedia.org/wiki/Art",
        "https://en.wikipedia.org/wiki/Literature",
        
        # General Knowledge
        "https://en.wikipedia.org/wiki/Geography",
        "https://en.wikipedia.org/wiki/Economics",
        "https://en.wikipedia.org/wiki/Psychology",
        "https://en.wikipedia.org/wiki/Medicine",
        
        # Programming
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "https://en.wikipedia.org/wiki/JavaScript",
        "https://en.wikipedia.org/wiki/Data_science",
        
        # Current Events
        "https://en.wikipedia.org/wiki/2024",
        "https://en.wikipedia.org/wiki/Technology",
    ]
    
    def __init__(self, model=None):
        self.model = model
        self.url_queue: Queue = Queue()
        self.visited_urls: set = set()
        self.learning_active = False
        self.learning_thread: Optional[threading.Thread] = None
        self.current_url: Optional[str] = None
        self.current_content: Optional[str] = None
        
        # Stats
        self.sites_learned = 0
        self.total_text_learned = 0
        self.learning_history: List[Dict] = []
        self.errors: List[Dict] = []
        
        # Callbacks for UI updates
        self.on_progress: Optional[Callable] = None
        self.on_content: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Rate limiting
        self.request_delay = 1.0  # seconds between requests
        self.last_request_time = 0
        
        # Initialize with seed URLs
        self._seed_queue()
        
    def _seed_queue(self):
        """Add seed URLs to the queue"""
        random.shuffle(self.SEED_URLS)
        for url in self.SEED_URLS:
            self.url_queue.put(url)
            
    def set_model(self, model):
        """Set the model to learn with"""
        self.model = model
        
    def start_learning(self):
        """Start the autonomous learning process"""
        if self.learning_active:
            return {"status": "already_running"}
            
        self.learning_active = True
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        
        return {"status": "started", "message": "Learning started!"}
        
    def stop_learning(self):
        """Stop the learning process"""
        self.learning_active = False
        if self.learning_thread:
            self.learning_thread.join(timeout=2)
            
        return {"status": "stopped", "message": "Learning stopped"}
        
    def _learning_loop(self):
        """Main learning loop"""
        while self.learning_active:
            try:
                # Rate limiting
                elapsed = time.time() - self.last_request_time
                if elapsed < self.request_delay:
                    time.sleep(self.request_delay - elapsed)
                    
                # Get next URL
                try:
                    url = self.url_queue.get(timeout=1)
                except Empty:
                    # Refill queue if empty
                    self._seed_queue()
                    continue
                    
                if url in self.visited_urls:
                    continue
                    
                self.visited_urls.add(url)
                self.current_url = url
                self.last_request_time = time.time()
                
                # Fetch and process
                result = self._fetch_and_learn(url)
                
                if result["success"]:
                    self.sites_learned += 1
                    if self.on_progress:
                        self.on_progress(self.get_stats())
                        
            except Exception as e:
                self.errors.append({
                    "url": self.current_url,
                    "error": str(e),
                    "time": datetime.now().isoformat()
                })
                if self.on_error:
                    self.on_error(str(e))
                    
    def _fetch_and_learn(self, url: str) -> Dict:
        """Fetch a URL and learn from its content"""
        try:
            headers = {
                "User-Agent": "NeuralMind-Learner/1.0 (Educational AI Research Bot)"
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
                
            # Extract text content
            text_content = self._extract_text(soup)
            
            if len(text_content) < 100:
                return {"success": False, "reason": "Not enough content"}
                
            self.current_content = text_content[:500]  # For display
            
            if self.on_content:
                self.on_content({
                    "url": url,
                    "title": soup.title.string if soup.title else "Unknown",
                    "preview": text_content[:300],
                    "length": len(text_content)
                })
            
            # Learn from content
            if self.model:
                # Split into chunks for learning
                chunks = self._chunk_text(text_content, chunk_size=500)
                for chunk in chunks:
                    self.model.learn_from_text(chunk, source=url)
                    self.total_text_learned += len(chunk)
                    
            # Extract and queue new links (Wikipedia only for safety)
            new_links = self._extract_wiki_links(soup, url)
            for link in new_links[:10]:  # Limit new links per page
                if link not in self.visited_urls:
                    self.url_queue.put(link)
                    
            # Record in history
            self.learning_history.append({
                "url": url,
                "title": soup.title.string if soup.title else "Unknown",
                "chars_learned": len(text_content),
                "chunks": len(chunks) if self.model else 0,
                "time": datetime.now().isoformat()
            })
            
            return {"success": True, "chars": len(text_content)}
            
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text from HTML"""
        # Get main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find('div', {'id': 'content'})
        
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)
            
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
        
    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks for learning"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
        
    def _extract_wiki_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract Wikipedia article links"""
        links = []
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            
            # Only follow Wikipedia article links
            if href.startswith('/wiki/') and ':' not in href:
                full_url = urljoin(base_url, href)
                if full_url not in self.visited_urls:
                    links.append(full_url)
                    
        random.shuffle(links)
        return links
        
    def learn_from_url(self, url: str) -> Dict:
        """Manually learn from a specific URL"""
        self.current_url = url
        result = self._fetch_and_learn(url)
        
        if result["success"]:
            self.sites_learned += 1
            
        return {
            **result,
            "stats": self.get_stats()
        }
        
    def search_and_learn(self, query: str) -> Dict:
        """Search for information and learn from results"""
        # Use Wikipedia API for search
        try:
            search_url = f"https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 5
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("query", {}).get("search", [])
            
            learned_from = []
            for result in results[:3]:  # Learn from top 3 results
                title = result["title"]
                article_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                
                learn_result = self.learn_from_url(article_url)
                if learn_result.get("success"):
                    learned_from.append({
                        "title": title,
                        "url": article_url
                    })
                    
            return {
                "success": True,
                "query": query,
                "learned_from": learned_from,
                "count": len(learned_from)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    def get_stats(self) -> Dict:
        """Get current learning statistics"""
        return {
            "is_learning": self.learning_active,
            "current_url": self.current_url,
            "current_preview": self.current_content[:200] if self.current_content else None,
            "sites_learned": self.sites_learned,
            "total_chars_learned": self.total_text_learned,
            "urls_in_queue": self.url_queue.qsize(),
            "urls_visited": len(self.visited_urls),
            "recent_history": self.learning_history[-10:] if self.learning_history else [],
            "error_count": len(self.errors)
        }
        
    def get_recent_sites(self, n: int = 10) -> List[Dict]:
        """Get most recently learned sites"""
        return self.learning_history[-n:][::-1]


class SearchEngine:
    """Simple search functionality for corrections and fact-checking"""
    
    @staticmethod
    def search_wikipedia(query: str) -> List[Dict]:
        """Search Wikipedia for information"""
        try:
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 5
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            data = response.json()
            
            results = []
            for item in data.get("query", {}).get("search", []):
                results.append({
                    "title": item["title"],
                    "snippet": BeautifulSoup(item["snippet"], "html.parser").get_text(),
                    "url": f"https://en.wikipedia.org/wiki/{item['title'].replace(' ', '_')}"
                })
                
            return results
            
        except Exception as e:
            return [{"error": str(e)}]
            
    @staticmethod
    def get_wikipedia_content(title: str) -> Optional[str]:
        """Get full content from a Wikipedia article"""
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "titles": title,
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "format": "json"
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if "extract" in page_data:
                    return page_data["extract"]
                    
            return None
            
        except Exception:
            return None