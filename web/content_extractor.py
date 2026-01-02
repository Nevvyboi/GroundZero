"""
Content Extractor
=================
Clean text extraction from HTML content.
"""

import re
from bs4 import BeautifulSoup
from typing import List, Optional, Dict, Any


class ContentExtractor:
    """Extracts clean, readable text from HTML"""
    
    # Tags to remove entirely
    REMOVE_TAGS = [
        'script', 'style', 'nav', 'footer', 'header', 'aside',
        'noscript', 'iframe', 'form', 'button', 'input', 'select',
        'svg', 'canvas', 'video', 'audio', 'figure', 'figcaption'
    ]
    
    # Classes/IDs that typically contain non-content
    REMOVE_PATTERNS = [
        'sidebar', 'navigation', 'nav-', 'menu', 'footer', 'header',
        'advertisement', 'ad-', 'banner', 'popup', 'modal',
        'social', 'share', 'comment', 'related', 'recommended'
    ]
    
    def extract(
        self,
        html: str,
        min_paragraph_length: int = 50,
        include_headers: bool = True
    ) -> str:
        """
        Extract clean text from HTML.
        
        Args:
            html: Raw HTML content
            min_paragraph_length: Minimum length for paragraphs to include
            include_headers: Whether to include header text
        
        Returns:
            Clean text content
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        self._remove_unwanted(soup)
        
        # Try to find main content area
        content = self._find_main_content(soup)
        
        # Extract text
        if include_headers:
            text = self._extract_with_structure(content, min_paragraph_length)
        else:
            text = self._extract_paragraphs(content, min_paragraph_length)
        
        # Clean up
        text = self._clean_text(text)
        
        return text
    
    def _remove_unwanted(self, soup: BeautifulSoup) -> None:
        """Remove unwanted elements from soup"""
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
        
        # Remove Wikipedia-specific elements
        for element in soup.find_all(['sup', 'span'], class_=['reference', 'mw-editsection']):
            element.decompose()
    
    def _find_main_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Find the main content area of the page"""
        # Try common content selectors
        selectors = [
            ('div', {'id': 'mw-content-text'}),  # Wikipedia
            ('article', {}),
            ('main', {}),
            ('div', {'class': 'content'}),
            ('div', {'class': 'article'}),
            ('div', {'id': 'content'}),
            ('div', {'role': 'main'}),
        ]
        
        for tag, attrs in selectors:
            element = soup.find(tag, attrs)
            if element:
                return element
        
        return soup.body or soup
    
    def _extract_paragraphs(
        self,
        content: BeautifulSoup,
        min_length: int
    ) -> str:
        """Extract text from paragraphs"""
        paragraphs = []
        
        for p in content.find_all('p'):
            text = p.get_text(separator=' ', strip=True)
            if len(text) >= min_length:
                paragraphs.append(text)
        
        return '\n\n'.join(paragraphs)
    
    def _extract_with_structure(
        self,
        content: BeautifulSoup,
        min_length: int
    ) -> str:
        """Extract text while preserving some structure"""
        parts = []
        
        for element in content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li']):
            text = element.get_text(separator=' ', strip=True)
            
            if element.name in ['h1', 'h2', 'h3', 'h4']:
                if len(text) > 3:  # Skip very short headers
                    parts.append(f"\n## {text}\n")
            elif element.name == 'li':
                if len(text) >= 20:
                    parts.append(f"• {text}")
            else:  # paragraph
                if len(text) >= min_length:
                    parts.append(text)
        
        return '\n'.join(parts)
    
    def _clean_text(self, text: str) -> str:
        """Clean up extracted text"""
        # Remove citation markers like [1], [2]
        text = re.sub(r'\[\d+\]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[str]:
        """
        Split text into chunks for processing.
        
        Args:
            text: Text to chunk
            chunk_size: Target size for each chunk
            overlap: Number of characters to overlap between chunks
        """
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if overlap > 0 and current_chunk:
                    current_chunk = current_chunk[-overlap:] + sentence + " "
                else:
                    current_chunk = sentence + " "
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def extract_metadata(self, html: str) -> Dict[str, Any]:
        """Extract metadata from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        
        metadata = {
            'title': None,
            'description': None,
            'keywords': [],
            'author': None
        }
        
        # Title
        if soup.title:
            metadata['title'] = soup.title.string
        
        # Meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', '').lower()
            content = meta.get('content', '')
            
            if name == 'description':
                metadata['description'] = content
            elif name == 'keywords':
                metadata['keywords'] = [k.strip() for k in content.split(',')]
            elif name == 'author':
                metadata['author'] = content
        
        return metadata
