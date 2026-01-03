"""
Knowledge Indexing & Retrieval System
======================================
Implements proper knowledge storage and retrieval using:
1. Inverted index for fast keyword lookup
2. TF-IDF scoring for relevance
3. Topic tagging for semantic grouping
4. Multiple search strategies

This fixes the "No response" issue by properly indexing and retrieving knowledge.
"""

import re
import math
import hashlib
import threading
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json


@dataclass
class IndexedKnowledge:
    """A piece of indexed knowledge"""
    id: int
    content: str
    summary: str
    source_url: str
    source_title: str
    confidence: float
    topics: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    word_count: int = 0


class TextProcessor:
    """
    Text processing utilities for indexing and search.
    """
    
    # Comprehensive stopwords list
    STOPWORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but',
        'if', 'or', 'because', 'until', 'while', 'what', 'which', 'who',
        'whom', 'this', 'that', 'these', 'those', 'am', 'it', 'its', 'i',
        'me', 'my', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they',
        'them', 'their', 'we', 'us', 'our', 'also', 'any', 'both', 'even',
        'every', 'get', 'got', 'go', 'gone', 'came', 'come', 'make', 'made',
        'much', 'many', 'new', 'now', 'old', 'see', 'way', 'well', 'back',
        'being', 'been', 'thing', 'things', 'time', 'year', 'years', 'first',
        'last', 'long', 'great', 'little', 'own', 'still', 'take', 'say',
        'says', 'said', 'put', 'know', 'known', 'use', 'used', 'work', 'part'
    }
    
    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """
        Tokenize text into normalized words.
        Handles hyphenated words, numbers, and special characters.
        """
        if not text:
            return []
        
        text = text.lower()
        
        # Handle hyphenated words (keep as both combined and separate)
        # e.g., "six-day" -> ["sixday", "six", "day"]
        hyphenated = re.findall(r'\b(\w+)-(\w+)\b', text)
        extra_tokens = []
        for w1, w2 in hyphenated:
            extra_tokens.append(w1 + w2)  # Combined
            extra_tokens.append(w1)        # First part
            extra_tokens.append(w2)        # Second part
        
        # Standard tokenization
        tokens = re.findall(r'\b\w+\b', text)
        
        # Add extra tokens from hyphenated words
        tokens.extend(extra_tokens)
        
        # Filter out very short tokens and numbers-only
        tokens = [t for t in tokens if len(t) > 1 and not t.isdigit()]
        
        return tokens
    
    @classmethod
    def normalize(cls, text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        return re.sub(r'\s+', ' ', text.lower().strip())
    
    @classmethod
    def extract_keywords(cls, text: str, max_keywords: int = 20) -> List[str]:
        """Extract important keywords from text"""
        tokens = cls.tokenize(text)
        
        # Filter stopwords
        keywords = [t for t in tokens if t not in cls.STOPWORDS and len(t) > 2]
        
        # Count frequencies
        freq = Counter(keywords)
        
        # Return most common
        return [word for word, _ in freq.most_common(max_keywords)]
    
    @classmethod
    def extract_topics(cls, text: str, title: str = "") -> List[str]:
        """
        Extract topic-like phrases from text.
        Topics are typically capitalized phrases or key concepts.
        """
        topics = []
        
        # Extract from title first (highest priority)
        if title:
            # Title words are likely topics
            title_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', title)
            topics.extend(title_words)
            
            # Also add normalized title words
            for word in title.split():
                clean = re.sub(r'[^\w]', '', word)
                if len(clean) > 2:
                    topics.append(clean.lower())
        
        # Extract capitalized phrases from content
        caps_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        topics.extend(caps_phrases[:10])  # Limit
        
        # Extract multi-word terms (2-3 word combinations that appear multiple times)
        words = text.lower().split()
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        bigram_freq = Counter(bigrams)
        for bigram, count in bigram_freq.most_common(5):
            if count >= 2 and all(w not in cls.STOPWORDS for w in bigram.split()):
                topics.append(bigram)
        
        # Normalize and deduplicate
        seen = set()
        unique_topics = []
        for topic in topics:
            norm = topic.lower().strip()
            if norm not in seen and len(norm) > 2:
                seen.add(norm)
                unique_topics.append(norm)
        
        return unique_topics[:15]


class InvertedIndex:
    """
    Inverted index for fast keyword-based retrieval.
    Maps terms to document IDs with term frequencies.
    """
    
    def __init__(self):
        # term -> {doc_id: term_frequency}
        self.index: Dict[str, Dict[int, float]] = defaultdict(dict)
        # doc_id -> document length (for normalization)
        self.doc_lengths: Dict[int, int] = {}
        # Total documents
        self.total_docs = 0
        # Lock for thread safety
        self._lock = threading.Lock()
    
    def add_document(self, doc_id: int, tokens: List[str]) -> None:
        """Add a document to the index"""
        if not tokens:
            return
        
        with self._lock:
            # Count term frequencies
            term_freq = Counter(tokens)
            doc_length = len(tokens)
            
            # Add to index
            for term, freq in term_freq.items():
                # Normalized TF
                tf = freq / doc_length
                self.index[term][doc_id] = tf
            
            self.doc_lengths[doc_id] = doc_length
            self.total_docs = len(self.doc_lengths)
    
    def remove_document(self, doc_id: int) -> None:
        """Remove a document from the index"""
        with self._lock:
            # Remove from all term entries
            terms_to_remove = []
            for term, docs in self.index.items():
                if doc_id in docs:
                    del docs[doc_id]
                if not docs:
                    terms_to_remove.append(term)
            
            # Clean up empty terms
            for term in terms_to_remove:
                del self.index[term]
            
            # Remove doc length
            if doc_id in self.doc_lengths:
                del self.doc_lengths[doc_id]
                self.total_docs = len(self.doc_lengths)
    
    def search(self, query_tokens: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for documents matching query tokens.
        Returns list of (doc_id, score) tuples sorted by relevance.
        """
        if not query_tokens:
            return []
        
        with self._lock:
            # Calculate TF-IDF scores for each document
            doc_scores: Dict[int, float] = defaultdict(float)
            
            for term in query_tokens:
                if term not in self.index:
                    continue
                
                # IDF: log((N + 1) / (df + 1)) + 1
                df = len(self.index[term])
                idf = math.log((self.total_docs + 1) / (df + 1)) + 1
                
                # Add TF-IDF score for each doc containing this term
                for doc_id, tf in self.index[term].items():
                    doc_scores[doc_id] += tf * idf
            
            # Sort by score
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            
            return sorted_docs[:top_k]
    
    def get_term_docs(self, term: str) -> Set[int]:
        """Get all document IDs containing a term"""
        with self._lock:
            return set(self.index.get(term, {}).keys())


class TopicIndex:
    """
    Topic-based index for semantic grouping.
    Maps topics to documents and vice versa.
    """
    
    def __init__(self):
        # topic -> set of doc_ids
        self.topic_to_docs: Dict[str, Set[int]] = defaultdict(set)
        # doc_id -> list of topics
        self.doc_to_topics: Dict[int, List[str]] = {}
        self._lock = threading.Lock()
    
    def add_document(self, doc_id: int, topics: List[str]) -> None:
        """Add document topics to index"""
        with self._lock:
            normalized_topics = [t.lower().strip() for t in topics if t]
            
            for topic in normalized_topics:
                self.topic_to_docs[topic].add(doc_id)
            
            self.doc_to_topics[doc_id] = normalized_topics
    
    def remove_document(self, doc_id: int) -> None:
        """Remove document from topic index"""
        with self._lock:
            if doc_id in self.doc_to_topics:
                for topic in self.doc_to_topics[doc_id]:
                    if topic in self.topic_to_docs:
                        self.topic_to_docs[topic].discard(doc_id)
                del self.doc_to_topics[doc_id]
    
    def search_by_topic(self, topic: str) -> Set[int]:
        """Find documents related to a topic"""
        topic_norm = topic.lower().strip()
        
        with self._lock:
            results = set()
            
            # Exact match
            if topic_norm in self.topic_to_docs:
                results.update(self.topic_to_docs[topic_norm])
            
            # Partial match (topic contains query or query contains topic)
            for t, docs in self.topic_to_docs.items():
                if topic_norm in t or t in topic_norm:
                    results.update(docs)
            
            return results
    
    def search_by_topics(self, topics: List[str]) -> Dict[int, int]:
        """
        Search by multiple topics.
        Returns doc_id -> number of matching topics.
        """
        with self._lock:
            doc_scores = defaultdict(int)
            
            for topic in topics:
                matching_docs = self.search_by_topic(topic)
                for doc_id in matching_docs:
                    doc_scores[doc_id] += 1
            
            return dict(doc_scores)


class KnowledgeIndex:
    """
    Main knowledge indexing and retrieval system.
    Combines inverted index and topic index for comprehensive search.
    """
    
    def __init__(self, memory_store=None):
        self.memory = memory_store
        self.processor = TextProcessor()
        self.inverted_index = InvertedIndex()
        self.topic_index = TopicIndex()
        
        # Document cache
        self.documents: Dict[int, IndexedKnowledge] = {}
        self._lock = threading.Lock()
        
        # Load existing knowledge if memory store provided
        if memory_store:
            self._load_from_database()
    
    @property
    def doc_count(self) -> int:
        """Number of documents in the index"""
        return len(self.documents)
    
    def _load_from_database(self) -> None:
        """Load existing knowledge from database into index"""
        if not self.memory:
            return
        
        try:
            # Get all knowledge entries
            rows = self.memory.db.fetch_all(
                """SELECT id, content, summary, source_url, source_title, confidence 
                   FROM knowledge ORDER BY id"""
            )
            
            for row in rows:
                doc = IndexedKnowledge(
                    id=row['id'],
                    content=row['content'] or '',
                    summary=row['summary'] or '',
                    source_url=row['source_url'] or '',
                    source_title=row['source_title'] or '',
                    confidence=row['confidence'] or 0.5
                )
                self._index_document(doc)
            
            print(f"📚 Loaded {len(self.documents)} knowledge entries into index")
        
        except Exception as e:
            print(f"Error loading knowledge index: {e}")
    
    def _index_document(self, doc: IndexedKnowledge) -> None:
        """Index a single document"""
        with self._lock:
            # Tokenize content
            content_tokens = self.processor.tokenize(doc.content)
            title_tokens = self.processor.tokenize(doc.source_title)
            
            # Combine tokens (title tokens weighted higher by repetition)
            all_tokens = content_tokens + title_tokens * 3
            
            # Extract keywords and topics
            doc.keywords = self.processor.extract_keywords(doc.content)
            doc.topics = self.processor.extract_topics(doc.content, doc.source_title)
            doc.word_count = len(content_tokens)
            
            # Add to inverted index
            self.inverted_index.add_document(doc.id, all_tokens)
            
            # Add to topic index
            self.topic_index.add_document(doc.id, doc.topics + doc.keywords[:5])
            
            # Store document
            self.documents[doc.id] = doc
    
    def add_knowledge(self, doc_id: int, content: str, summary: str = "",
                     source_url: str = "", source_title: str = "",
                     confidence: float = 0.5) -> None:
        """Add new knowledge to the index"""
        doc = IndexedKnowledge(
            id=doc_id,
            content=content,
            summary=summary,
            source_url=source_url,
            source_title=source_title,
            confidence=confidence
        )
        self._index_document(doc)
    
    def remove_knowledge(self, doc_id: int) -> None:
        """Remove knowledge from index"""
        with self._lock:
            if doc_id in self.documents:
                self.inverted_index.remove_document(doc_id)
                self.topic_index.remove_document(doc_id)
                del self.documents[doc_id]
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Main search function with multiple strategies.
        
        Strategies:
        1. Topic-based search (highest priority)
        2. Keyword/TF-IDF search
        3. Combined scoring
        """
        if not query or not self.documents:
            return []
        
        # Normalize query
        query_norm = self.processor.normalize(query)
        query_tokens = self.processor.tokenize(query)
        query_keywords = [t for t in query_tokens if t not in self.processor.STOPWORDS]
        
        if not query_keywords:
            # If all stopwords, use full query
            query_keywords = query_tokens
        
        # Extract topics from query
        query_topics = self.processor.extract_topics(query, "")
        query_topics.extend(query_keywords)  # Also use keywords as topics
        
        # Strategy 1: Topic-based search
        topic_scores = self.topic_index.search_by_topics(query_topics)
        
        # Strategy 2: Inverted index search
        tfidf_results = self.inverted_index.search(query_keywords, top_k=limit * 2)
        tfidf_scores = dict(tfidf_results)
        
        # Combine scores
        all_doc_ids = set(topic_scores.keys()) | set(tfidf_scores.keys())
        
        combined_scores = []
        for doc_id in all_doc_ids:
            doc = self.documents.get(doc_id)
            if not doc:
                continue
            
            # Topic score (normalized)
            topic_score = topic_scores.get(doc_id, 0) / max(len(query_topics), 1)
            
            # TF-IDF score (normalized)
            max_tfidf = max(tfidf_scores.values()) if tfidf_scores else 1
            tfidf_score = tfidf_scores.get(doc_id, 0) / max(max_tfidf, 0.001)
            
            # Title match bonus
            title_bonus = 0
            for keyword in query_keywords:
                if keyword in doc.source_title.lower():
                    title_bonus += 0.2
            title_bonus = min(title_bonus, 0.5)
            
            # Exact phrase match bonus
            phrase_bonus = 0.3 if query_norm in self.processor.normalize(doc.content) else 0
            
            # Confidence factor
            conf_factor = doc.confidence
            
            # Combined score
            final_score = (
                topic_score * 0.35 +
                tfidf_score * 0.25 +
                title_bonus * 0.2 +
                phrase_bonus * 0.1 +
                conf_factor * 0.1
            )
            
            combined_scores.append((doc_id, final_score))
        
        # Sort by score
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for doc_id, score in combined_scores[:limit]:
            doc = self.documents.get(doc_id)
            if doc:
                results.append({
                    'id': doc.id,
                    'content': doc.content,
                    'summary': doc.summary,
                    'source_url': doc.source_url,
                    'source_title': doc.source_title,
                    'confidence': doc.confidence,
                    'relevance': min(1.0, score * 1.5),  # Scale up for display
                    'topics': doc.topics[:5],
                    'keywords': doc.keywords[:10]
                })
        
        return results
    
    def search_by_topic(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search specifically by topic"""
        doc_ids = self.topic_index.search_by_topic(topic)
        
        results = []
        for doc_id in list(doc_ids)[:limit]:
            doc = self.documents.get(doc_id)
            if doc:
                results.append({
                    'id': doc.id,
                    'content': doc.content,
                    'summary': doc.summary,
                    'source_url': doc.source_url,
                    'source_title': doc.source_title,
                    'confidence': doc.confidence,
                    'relevance': 0.8,
                    'topics': doc.topics[:5]
                })
        
        return results
    
    def has_knowledge_about(self, topic: str) -> Tuple[bool, float, int]:
        """
        Check if we have knowledge about a topic.
        Returns: (has_knowledge, confidence, count)
        """
        results = self.search(topic, limit=5)
        
        if not results:
            return False, 0.0, 0
        
        # Calculate overall confidence
        avg_relevance = sum(r['relevance'] for r in results) / len(results)
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        
        overall_confidence = (avg_relevance + avg_confidence) / 2
        
        return True, overall_confidence, len(results)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            'total_documents': len(self.documents),
            'total_terms': len(self.inverted_index.index),
            'total_topics': len(self.topic_index.topic_to_docs),
            'avg_doc_length': sum(self.inverted_index.doc_lengths.values()) / max(len(self.documents), 1)
        }


# Singleton instance
_knowledge_index = None
_index_lock = threading.Lock()

def get_knowledge_index(memory_store=None) -> KnowledgeIndex:
    """Get or create the knowledge index singleton"""
    global _knowledge_index
    
    with _index_lock:
        if _knowledge_index is None and memory_store is not None:
            _knowledge_index = KnowledgeIndex(memory_store)
        return _knowledge_index

def reset_knowledge_index() -> None:
    """Reset the knowledge index (for testing)"""
    global _knowledge_index
    with _index_lock:
        _knowledge_index = None

def rebuild_knowledge_index(memory_store) -> Dict[str, Any]:
    """
    Rebuild the knowledge index from scratch.
    Useful when you have existing knowledge that wasn't indexed.
    """
    global _knowledge_index
    with _index_lock:
        _knowledge_index = KnowledgeIndex(memory_store)
        return _knowledge_index.get_statistics()