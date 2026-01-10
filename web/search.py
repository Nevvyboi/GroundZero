"""
Web Module - Enhanced Version with Selenium Support
===================================================
Wikipedia search and web content extraction.
Uses async/threading to prevent blocking the event loop.
Enhanced content extraction for various website types.
Includes Selenium for JavaScript-rendered sites.
Supports Chrome, Edge, and Firefox browsers.
"""

import re
import asyncio
import requests
from typing import Dict, Any, List, Optional
from urllib.parse import quote, unquote, urlparse
from concurrent.futures import ThreadPoolExecutor
import threading
import json
import time
import os
import subprocess
import shutil

# Thread pool for running blocking requests - prevents event loop blocking
_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="web_fetch")

# Selenium availability flag
SELENIUM_AVAILABLE = False
AVAILABLE_BROWSERS = []

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    pass

# Check for Chrome
try:
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
    AVAILABLE_BROWSERS.append('chrome')
except ImportError:
    pass

# Check for Edge
try:
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.webdriver.edge.service import Service as EdgeService
    AVAILABLE_BROWSERS.append('edge')
except ImportError:
    pass

# Check for Firefox
try:
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.firefox.service import Service as FirefoxService
    AVAILABLE_BROWSERS.append('firefox')
except ImportError:
    pass

# WebDriver manager for auto-installing drivers
WEBDRIVER_MANAGER_AVAILABLE = False
try:
    from webdriver_manager.chrome import ChromeDriverManager
    from webdriver_manager.microsoft import EdgeChromiumDriverManager
    from webdriver_manager.firefox import GeckoDriverManager
    WEBDRIVER_MANAGER_AVAILABLE = True
except ImportError:
    pass

def detect_installed_browsers() -> List[str]:
    """Detect which browsers are actually installed on the system"""
    installed = []
    
    # Windows paths
    if os.name == 'nt':
        brave_paths = [
            os.path.expandvars(r'%LocalAppData%\BraveSoftware\Brave-Browser\Application\brave.exe'),
            os.path.expandvars(r'%ProgramFiles%\BraveSoftware\Brave-Browser\Application\brave.exe'),
            os.path.expandvars(r'%ProgramFiles(x86)%\BraveSoftware\Brave-Browser\Application\brave.exe'),
        ]
        chrome_paths = [
            os.path.expandvars(r'%ProgramFiles%\Google\Chrome\Application\chrome.exe'),
            os.path.expandvars(r'%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe'),
            os.path.expandvars(r'%LocalAppData%\Google\Chrome\Application\chrome.exe'),
        ]
        edge_paths = [
            os.path.expandvars(r'%ProgramFiles%\Microsoft\Edge\Application\msedge.exe'),
            os.path.expandvars(r'%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe'),
        ]
        firefox_paths = [
            os.path.expandvars(r'%ProgramFiles%\Mozilla Firefox\firefox.exe'),
            os.path.expandvars(r'%ProgramFiles(x86)%\Mozilla Firefox\firefox.exe'),
        ]
        
        # Check Brave first (preferred if available)
        for p in brave_paths:
            if os.path.exists(p):
                installed.append('brave')
                # Store the path for later use
                os.environ['BRAVE_PATH'] = p
                break
        
        if any(os.path.exists(p) for p in chrome_paths):
            installed.append('chrome')
        if any(os.path.exists(p) for p in edge_paths):
            installed.append('edge')
        if any(os.path.exists(p) for p in firefox_paths):
            installed.append('firefox')
    else:
        # Linux/Mac - use which command
        if shutil.which('brave') or shutil.which('brave-browser'):
            installed.append('brave')
            # Try to find the path
            brave_path = shutil.which('brave') or shutil.which('brave-browser')
            if brave_path:
                os.environ['BRAVE_PATH'] = brave_path
        if shutil.which('google-chrome') or shutil.which('chrome') or shutil.which('chromium'):
            installed.append('chrome')
        if shutil.which('microsoft-edge') or shutil.which('msedge'):
            installed.append('edge')
        if shutil.which('firefox'):
            installed.append('firefox')
    
    return installed


def get_browser_version(browser_path: str) -> Optional[str]:
    """Get the version of a Chromium-based browser"""
    try:
        if os.name == 'nt':
            # Windows: Use PowerShell to get version
            cmd = f'(Get-Item "{browser_path}").VersionInfo.ProductVersion'
            result = subprocess.run(
                ['powershell', '-Command', cmd],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                # Return just the major version number (e.g., "131" from "131.0.6778.86")
                if version:
                    return version.split('.')[0]
        else:
            # Linux/Mac: Run browser with --version
            result = subprocess.run(
                [browser_path, '--version'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                # Extract version number from output like "Brave 131.1.73.97 Chromium: 131.0.6778.86"
                output = result.stdout
                # Try to find Chromium version first
                chromium_match = re.search(r'Chromium[:\s]+(\d+)', output)
                if chromium_match:
                    return chromium_match.group(1)
                # Otherwise find any version number
                version_match = re.search(r'(\d+)\.\d+\.\d+', output)
                if version_match:
                    return version_match.group(1)
    except Exception:
        pass  # Browser version detection failed
    return None

# Detect browsers at module load
INSTALLED_BROWSERS = detect_installed_browsers()

if SELENIUM_AVAILABLE:
    if not INSTALLED_BROWSERS:
        SELENIUM_AVAILABLE = False


class WikipediaSearch:
    """
    Search and fetch Wikipedia articles.
    Uses the Wikipedia API for content extraction.
    Thread-safe with connection pooling.
    """
    
    API_URL = "https://en.wikipedia.org/w/api.php"
    
    def __init__(self):
        self._local = threading.local()
    
    @property
    def session(self):
        """Thread-local session for thread safety"""
        if not hasattr(self._local, 'session'):
            self._local.session = requests.Session()
            self._local.session.headers.update({
                'User-Agent': 'GroundZero/2.0 (Learning AI; educational project)'
            })
        return self._local.session
    
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
            title = unquote(title)
            return self.get_article_content(title)
        return None
    
    # Async versions that don't block the event loop
    async def get_random_articles_async(self, count: int = 5) -> List[Dict[str, str]]:
        """Non-blocking version of get_random_articles"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.get_random_articles, count)
    
    async def search_async(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        """Non-blocking version of search"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.search, query, limit)
    
    async def get_article_content_async(self, title: str) -> Optional[Dict[str, Any]]:
        """Non-blocking version of get_article_content"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.get_article_content, title)
    
    async def get_article_by_url_async(self, url: str) -> Optional[Dict[str, Any]]:
        """Non-blocking version of get_article_by_url"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.get_article_by_url, url)


class ContentExtractor:
    """
    Enhanced content extractor for various website types.
    Handles static HTML, JavaScript-rendered pages (via meta/JSON),
    and special site-specific extractors.
    Thread-safe implementation.
    """
    
    # Common user agents for different scenarios
    USER_AGENTS = {
        'browser': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'mobile': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
        'bot': 'GroundZero/2.0 (Learning AI; educational project)'
    }
    
    def __init__(self):
        self._local = threading.local()
    
    @property
    def session(self):
        """Thread-local session for thread safety"""
        if not hasattr(self._local, 'session'):
            self._local.session = requests.Session()
            self._local.session.headers.update({
                'User-Agent': self.USER_AGENTS['browser'],
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            })
        return self._local.session
    
    def extract(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Extract content from URL with multiple strategies.
        Tries different methods to get the best content.
        """
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace('www.', '')
        
        
        # Try site-specific extractors first
        result = self._try_site_specific(url, domain)
        if result and result.get('content') and len(result['content']) > 100:
            return result
        
        # Try standard HTML extraction
        result = self._extract_html(url)
        if result and result.get('content') and len(result['content']) > 100:
            return result
        
        # Try to find JSON-LD structured data
        result = self._extract_json_ld(url)
        if result and result.get('content') and len(result['content']) > 100:
            return result
        
        # Try meta tags extraction (for JS-rendered sites)
        result = self._extract_meta(url)
        if result and result.get('content') and len(result['content']) > 50:
            return result
        
        # Last resort: Try Selenium for JS-rendered sites
        if SELENIUM_AVAILABLE:
            pass
            result = self._extract_with_selenium(url)
            if result and result.get('content') and len(result['content']) > 100:
                pass
                return result
        
        pass
        return None
    
    def extract_with_browser(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Force extraction using Selenium browser.
        Use this for known JS-heavy sites.
        """
        if not SELENIUM_AVAILABLE:
            pass
            return None
        return self._extract_with_selenium(url)
    
    def _extract_with_selenium(self, url: str, wait_time: int = 5) -> Optional[Dict[str, Any]]:
        """Extract content using Selenium headless browser (tries Brave, Edge, Chrome, Firefox)"""
        
        # Try browsers in order of preference (Brave first if available)
        browsers_to_try = []
        if 'brave' in INSTALLED_BROWSERS:
            browsers_to_try.append('brave')
        if 'edge' in INSTALLED_BROWSERS:
            browsers_to_try.append('edge')
        if 'chrome' in INSTALLED_BROWSERS:
            browsers_to_try.append('chrome')
        if 'firefox' in INSTALLED_BROWSERS:
            browsers_to_try.append('firefox')
        
        if not browsers_to_try:
            pass
            return None
        
        for browser in browsers_to_try:
            result = self._try_browser(browser, url, wait_time)
            if result:
                return result
        
        return None
    
    def _try_browser(self, browser: str, url: str, wait_time: int) -> Optional[Dict[str, Any]]:
        """Try to extract content with a specific browser"""
        driver = None
        try:
            pass
            
            if browser == 'brave':
                # Brave uses Chrome's WebDriver since it's Chromium-based
                # Key: Set binary_location to Brave executable and match driver version
                options = ChromeOptions()
                options.add_argument('--headless=new')  # New headless mode
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--disable-gpu')
                options.add_argument('--window-size=1920,1080')
                options.add_argument(f'--user-agent={self.USER_AGENTS["browser"]}')
                
                # IMPORTANT: Set Brave binary location
                brave_path = os.environ.get('BRAVE_PATH')
                if not brave_path:
                    pass
                    return None
                options.binary_location = brave_path
                
                # Get Brave's Chromium version for matching ChromeDriver
                brave_version = get_browser_version(brave_path)
                pass
                
                if WEBDRIVER_MANAGER_AVAILABLE and brave_version:
                    # Use version-specific ChromeDriver
                    service = ChromeService(
                        ChromeDriverManager(driver_version=f"{brave_version}").install()
                    )
                    driver = webdriver.Chrome(service=service, options=options)
                elif WEBDRIVER_MANAGER_AVAILABLE:
                    # Fallback: let it try to auto-detect
                    service = ChromeService(ChromeDriverManager().install())
                    driver = webdriver.Chrome(service=service, options=options)
                else:
                    driver = webdriver.Chrome(options=options)
            
            elif browser == 'edge':
                options = EdgeOptions()
                options.add_argument('--headless=new')  # New headless mode
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--disable-gpu')
                options.add_argument('--window-size=1920,1080')
                options.add_argument(f'--user-agent={self.USER_AGENTS["browser"]}')
                
                if WEBDRIVER_MANAGER_AVAILABLE:
                    service = EdgeService(EdgeChromiumDriverManager().install())
                    driver = webdriver.Edge(service=service, options=options)
                else:
                    driver = webdriver.Edge(options=options)
                    
            elif browser == 'chrome':
                options = ChromeOptions()
                options.add_argument('--headless=new')  # New headless mode
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--disable-gpu')
                options.add_argument('--window-size=1920,1080')
                options.add_argument(f'--user-agent={self.USER_AGENTS["browser"]}')
                
                if WEBDRIVER_MANAGER_AVAILABLE:
                    service = ChromeService(ChromeDriverManager().install())
                    driver = webdriver.Chrome(service=service, options=options)
                else:
                    driver = webdriver.Chrome(options=options)
                    
            elif browser == 'firefox':
                options = FirefoxOptions()
                options.add_argument('--headless')
                options.set_preference('general.useragent.override', self.USER_AGENTS["browser"])
                
                if WEBDRIVER_MANAGER_AVAILABLE:
                    service = FirefoxService(GeckoDriverManager().install())
                    driver = webdriver.Firefox(service=service, options=options)
                else:
                    driver = webdriver.Firefox(options=options)
            
            if not driver:
                return None
            
            # Set timeouts
            driver.set_page_load_timeout(30)
            driver.implicitly_wait(wait_time)
            
            # Load the page
            driver.get(url)
            
            # Wait for body to be present
            WebDriverWait(driver, wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Additional wait for JS to render
            time.sleep(2)
            
            # Try to wait for common content indicators
            content_selectors = [
                "main", "article", "[role='main']", 
                ".content", "#content", ".main-content",
                ".post-content", ".article-content", ".entry-content"
            ]
            
            for selector in content_selectors:
                try:
                    WebDriverWait(driver, 2).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    break
                except TimeoutException:
                    continue
            
            # Get page source after JS execution
            html = driver.page_source
            
            # Get title
            title = driver.title or self._extract_title(html, url)
            
            # Extract content from rendered HTML
            content = self._clean_and_extract(html)
            
            # Also try to get text directly from body
            if len(content) < 200:
                try:
                    body = driver.find_element(By.TAG_NAME, "body")
                    body_text = body.text
                    if len(body_text) > len(content):
                        content = body_text
                except:
                    pass
            
            if content and len(content) > 50:
                return {
                    'title': title[:300],
                    'content': content,
                    'url': url,
                    'word_count': len(content.split()),
                    'extraction_method': f'selenium-{browser}'
                }
            
            return None
            
        except WebDriverException as e:
            error_msg = str(e).split('\n')[0][:100]  # First line, truncated
            pass
            return None
        except Exception as e:
            pass
            return None
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
    
    def _try_site_specific(self, url: str, domain: str) -> Optional[Dict[str, Any]]:
        """Try site-specific extraction methods"""
        
        # Known JS-heavy sites that need Selenium
        js_heavy_sites = [
            'tfcertification.com',
            'tensorflow.org',
            'reactjs.org',
            'vuejs.org',
            'angular.io',
            'vercel.app',
            'netlify.app',
            'streamlit.app',
            'gradio.app',
        ]
        
        if SELENIUM_AVAILABLE and any(site in domain for site in js_heavy_sites):
            pass
            result = self._extract_with_selenium(url, wait_time=8)
            if result:
                return result
        
        # Medium articles
        if 'medium.com' in domain or domain.endswith('.medium.com'):
            return self._extract_medium(url)
        
        # GitHub README
        if 'github.com' in domain:
            return self._extract_github(url)
        
        # News sites often have article schema
        news_domains = ['bbc.com', 'cnn.com', 'reuters.com', 'nytimes.com', 'theguardian.com']
        if any(d in domain for d in news_domains):
            return self._extract_news_article(url)
        
        return None
    
    def _extract_html(self, url: str) -> Optional[Dict[str, Any]]:
        """Standard HTML content extraction"""
        try:
            response = self.session.get(url, timeout=20, allow_redirects=True)
            response.raise_for_status()
            
            html = response.text
            
            # Extract title
            title = self._extract_title(html, url)
            
            # Clean and extract content
            content = self._clean_and_extract(html)
            
            if content and len(content) > 50:
                return {
                    'title': title[:300],
                    'content': content,
                    'url': response.url,  # Final URL after redirects
                    'word_count': len(content.split()),
                    'extraction_method': 'html'
                }
            
            return None
        except Exception as e:
            pass
            return None
    
    def _extract_title(self, html: str, fallback_url: str) -> str:
        """Extract the best title from HTML"""
        # Try Open Graph title first
        og_match = re.search(r'<meta\s+property=["\']og:title["\']\s+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
        if not og_match:
            og_match = re.search(r'<meta\s+content=["\']([^"\']+)["\']\s+property=["\']og:title["\']', html, re.IGNORECASE)
        if og_match:
            return og_match.group(1).strip()
        
        # Try Twitter card title
        tw_match = re.search(r'<meta\s+name=["\']twitter:title["\']\s+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
        if not tw_match:
            tw_match = re.search(r'<meta\s+content=["\']([^"\']+)["\']\s+name=["\']twitter:title["\']', html, re.IGNORECASE)
        if tw_match:
            return tw_match.group(1).strip()
        
        # Try standard title tag
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
        if title_match:
            title = title_match.group(1).strip()
            # Clean up common suffixes
            title = re.sub(r'\s*[\|–-]\s*[^|–-]+$', '', title)
            return title
        
        # Try h1 tag
        h1_match = re.search(r'<h1[^>]*>([^<]+)</h1>', html, re.IGNORECASE)
        if h1_match:
            return self._clean_text(h1_match.group(1))
        
        # Fallback to domain
        parsed = urlparse(fallback_url)
        return parsed.netloc
    
    def _clean_and_extract(self, html: str) -> str:
        """Clean HTML and extract text content"""
        # Remove unwanted elements
        patterns_to_remove = [
            r'<script[^>]*>.*?</script>',
            r'<style[^>]*>.*?</style>',
            r'<nav[^>]*>.*?</nav>',
            r'<footer[^>]*>.*?</footer>',
            r'<header[^>]*>.*?</header>',
            r'<aside[^>]*>.*?</aside>',
            r'<form[^>]*>.*?</form>',
            r'<iframe[^>]*>.*?</iframe>',
            r'<noscript[^>]*>.*?</noscript>',
            r'<!--.*?-->',
            r'<svg[^>]*>.*?</svg>',
            # Common ad/tracking divs
            r'<div[^>]*class=["\'][^"\']*(?:ad-|ads-|advertisement|sidebar|cookie|popup|modal|banner)[^"\']*["\'][^>]*>.*?</div>',
        ]
        
        cleaned = html
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # Try to find main content area
        main_content = self._find_main_content(cleaned)
        if main_content:
            cleaned = main_content
        
        # Extract text from paragraphs, headings, and list items
        text_parts = []
        
        # Extract headings
        for tag in ['h1', 'h2', 'h3']:
            headings = re.findall(f'<{tag}[^>]*>(.*?)</{tag}>', cleaned, re.DOTALL | re.IGNORECASE)
            for h in headings:
                text = self._clean_text(h)
                if text and len(text) > 5:
                    text_parts.append(f"\n{text}\n")
        
        # Extract paragraphs
        paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', cleaned, re.DOTALL | re.IGNORECASE)
        for p in paragraphs:
            text = self._clean_text(p)
            if text and len(text) > 30:
                text_parts.append(text)
        
        # Extract list items
        list_items = re.findall(r'<li[^>]*>(.*?)</li>', cleaned, re.DOTALL | re.IGNORECASE)
        for li in list_items:
            text = self._clean_text(li)
            if text and len(text) > 20:
                text_parts.append(f"• {text}")
        
        # Extract divs with substantial text (for JS-light sites)
        if len(text_parts) < 3:
            divs = re.findall(r'<div[^>]*>(.*?)</div>', cleaned, re.DOTALL | re.IGNORECASE)
            for div in divs:
                text = self._clean_text(div)
                if text and len(text) > 100:
                    text_parts.append(text)
        
        # Extract spans with text (some sites use spans for content)
        if len(text_parts) < 3:
            spans = re.findall(r'<span[^>]*>(.*?)</span>', cleaned, re.DOTALL | re.IGNORECASE)
            for span in spans:
                text = self._clean_text(span)
                if text and len(text) > 50:
                    text_parts.append(text)
        
        content = '\n\n'.join(text_parts)
        
        # If still not enough content, try brute force
        if len(content) < 200:
            text = re.sub(r'<[^>]+>', ' ', cleaned)
            content = ' '.join(text.split())
        
        # Remove duplicate lines
        lines = content.split('\n')
        seen = set()
        unique_lines = []
        for line in lines:
            line_clean = line.strip()
            if line_clean and line_clean not in seen:
                seen.add(line_clean)
                unique_lines.append(line)
        content = '\n'.join(unique_lines)
        
        return content[:50000]  # Limit to ~50k chars
    
    def _find_main_content(self, html: str) -> Optional[str]:
        """Try to find the main content area"""
        # Common main content selectors
        main_patterns = [
            r'<main[^>]*>(.*?)</main>',
            r'<article[^>]*>(.*?)</article>',
            r'<div[^>]*class=["\'][^"\']*(?:content|main|article|post|entry|body)[^"\']*["\'][^>]*>(.*?)</div>',
            r'<div[^>]*id=["\'](?:content|main|article|post|body)["\'][^>]*>(.*?)</div>',
            r'<section[^>]*class=["\'][^"\']*(?:content|main)[^"\']*["\'][^>]*>(.*?)</section>',
        ]
        
        for pattern in main_patterns:
            match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
            if match and len(match.group(1)) > 500:
                return match.group(1)
        
        return None
    
    def _clean_text(self, html_text: str) -> str:
        """Clean HTML text to plain text"""
        # Remove tags but keep text
        text = re.sub(r'<br\s*/?>', '\n', html_text)
        text = re.sub(r'<[^>]+>', ' ', text)
        # Decode HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        text = text.replace('&apos;', "'")
        text = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))) if int(m.group(1)) < 65536 else '', text)
        text = re.sub(r'&\w+;', '', text)  # Remove remaining entities
        # Clean whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def _extract_json_ld(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract content from JSON-LD structured data"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            html = response.text
            
            # Find JSON-LD scripts
            json_ld_matches = re.findall(
                r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
                html, re.DOTALL | re.IGNORECASE
            )
            
            for json_str in json_ld_matches:
                try:
                    data = json.loads(json_str.strip())
                    
                    # Handle array of schemas
                    if isinstance(data, list):
                        data = data[0] if data else {}
                    
                    # Handle @graph structure
                    if '@graph' in data:
                        for item in data['@graph']:
                            if item.get('@type') in ['Article', 'NewsArticle', 'BlogPosting', 'WebPage']:
                                data = item
                                break
                    
                    # Extract article content
                    if data.get('@type') in ['Article', 'NewsArticle', 'BlogPosting', 'WebPage', 'Organization', 'WebSite']:
                        title = data.get('headline') or data.get('name', '')
                        content = data.get('articleBody') or data.get('description', '')
                        
                        if content and len(content) > 50:
                            return {
                                'title': title[:300],
                                'content': content,
                                'url': url,
                                'word_count': len(content.split()),
                                'extraction_method': 'json-ld'
                            }
                except json.JSONDecodeError:
                    continue
            
            return None
        except Exception as e:
            pass
            return None
    
    def _extract_meta(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract content from meta tags (for JS-rendered sites)"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            html = response.text
            
            # Get title
            title = self._extract_title(html, url)
            
            # Try to get description from various meta tags
            description = ''
            
            # Open Graph description
            og_desc = re.search(r'<meta\s+property=["\']og:description["\']\s+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
            if not og_desc:
                og_desc = re.search(r'<meta\s+content=["\']([^"\']+)["\']\s+property=["\']og:description["\']', html, re.IGNORECASE)
            if og_desc:
                description = og_desc.group(1)
            
            # Twitter description
            if not description:
                tw_desc = re.search(r'<meta\s+name=["\']twitter:description["\']\s+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
                if not tw_desc:
                    tw_desc = re.search(r'<meta\s+content=["\']([^"\']+)["\']\s+name=["\']twitter:description["\']', html, re.IGNORECASE)
                if tw_desc:
                    description = tw_desc.group(1)
            
            # Standard meta description
            if not description:
                meta_desc = re.search(r'<meta\s+name=["\']description["\']\s+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
                if not meta_desc:
                    meta_desc = re.search(r'<meta\s+content=["\']([^"\']+)["\']\s+name=["\']description["\']', html, re.IGNORECASE)
                if meta_desc:
                    description = meta_desc.group(1)
            
            if description:
                return {
                    'title': title[:300],
                    'content': self._clean_text(description),
                    'url': url,
                    'word_count': len(description.split()),
                    'extraction_method': 'meta',
                    'note': 'Limited content extracted (JavaScript-rendered site)'
                }
            
            return None
        except Exception as e:
            pass
            return None
    
    def _extract_medium(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract content from Medium articles"""
        try:
            # Medium has a nice JSON endpoint
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            # Try JSON-LD first (Medium usually has it)
            result = self._extract_json_ld(url)
            if result:
                return result
            
            # Fall back to HTML extraction
            return self._extract_html(url)
        except Exception as e:
            pass
            return None
    
    def _extract_github(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract README from GitHub repos"""
        try:
            parsed = urlparse(url)
            path_parts = parsed.path.strip('/').split('/')
            
            if len(path_parts) >= 2:
                owner, repo = path_parts[0], path_parts[1]
                
                # Try to get README via API
                api_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
                response = self.session.get(api_url, timeout=15, headers={
                    'Accept': 'application/vnd.github.v3.raw'
                })
                
                if response.status_code == 200:
                    content = response.text
                    # Convert markdown to plain text (basic)
                    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)  # Remove images
                    content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)  # Links to text
                    content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)  # Remove code blocks
                    content = re.sub(r'`[^`]+`', '', content)  # Remove inline code
                    content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)  # Remove heading markers
                    content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)  # Bold to plain
                    content = re.sub(r'\*([^*]+)\*', r'\1', content)  # Italic to plain
                    
                    return {
                        'title': f"{owner}/{repo} - GitHub Repository",
                        'content': content.strip(),
                        'url': url,
                        'word_count': len(content.split()),
                        'extraction_method': 'github-api'
                    }
            
            # Fall back to HTML
            return self._extract_html(url)
        except Exception as e:
            pass
            return self._extract_html(url)
    
    def _extract_news_article(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract content from news sites"""
        # Try JSON-LD first (news sites usually have article schema)
        result = self._extract_json_ld(url)
        if result:
            return result
        
        # Fall back to HTML
        return self._extract_html(url)
    
    async def extract_async(self, url: str) -> Optional[Dict[str, Any]]:
        """Non-blocking version of extract"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.extract, url)


# Helper function for routes to use
async def fetch_url_content(url: str) -> Optional[Dict[str, Any]]:
    """
    Fetch content from any URL without blocking.
    Automatically detects Wikipedia vs other sites.
    """
    if 'wikipedia.org' in url:
        wiki = WikipediaSearch()
        return await wiki.get_article_by_url_async(url)
    else:
        extractor = ContentExtractor()
        return await extractor.extract_async(url)

