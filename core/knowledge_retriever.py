"""
Knowledge Retrieval System
==========================
Implements RAG-like knowledge retrieval with semantic search.
Checks knowledge first before searching the web.
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import numpy as np


class SemanticMatcher:
    """
    Semantic matching for knowledge retrieval.
    Uses TF-IDF like scoring with word embeddings approximation.
    """
    
    def __init__(self):
        # Stop words for filtering
        self.stopwords = {
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
            'them', 'their', 'we', 'us', 'our'
        }
        
        # Word frequency for IDF-like weighting
        self.word_freq = Counter()
        self.total_docs = 0
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize and normalize text"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        return [t for t in tokens if t not in self.stopwords and len(t) > 2]
    
    def update_frequencies(self, texts: List[str]) -> None:
        """Update word frequencies from corpus"""
        for text in texts:
            tokens = set(self.tokenize(text))
            self.word_freq.update(tokens)
            self.total_docs += 1
    
    def compute_tfidf(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """Compute TF-IDF-like similarity score"""
        if not query_tokens or not doc_tokens:
            return 0.0
        
        # Count term frequencies in document
        doc_freq = Counter(doc_tokens)
        doc_len = len(doc_tokens)
        
        score = 0.0
        for term in query_tokens:
            if term in doc_freq:
                # TF: term frequency in document
                tf = doc_freq[term] / doc_len
                
                # IDF: inverse document frequency
                idf = math.log(1 + self.total_docs / (1 + self.word_freq.get(term, 1)))
                
                score += tf * idf
        
        # Normalize by query length
        return score / len(query_tokens) if query_tokens else 0.0
    
    def semantic_similarity(self, query: str, document: str) -> float:
        """
        Compute semantic similarity between query and document.
        Uses word overlap + position weighting + phrase matching.
        """
        query_tokens = self.tokenize(query)
        doc_tokens = self.tokenize(document)
        
        if not query_tokens or not doc_tokens:
            return 0.0
        
        # 1. Basic TF-IDF score
        tfidf_score = self.compute_tfidf(query_tokens, doc_tokens)
        
        # 2. Word overlap (Jaccard-like)
        query_set = set(query_tokens)
        doc_set = set(doc_tokens)
        overlap = len(query_set & doc_set)
        union = len(query_set | doc_set)
        jaccard = overlap / union if union > 0 else 0
        
        # 3. Coverage: what fraction of query terms are in doc
        coverage = overlap / len(query_set) if query_set else 0
        
        # 4. Position bonus: terms appearing early in doc are more relevant
        position_bonus = 0.0
        doc_lower = document.lower()
        for i, term in enumerate(query_tokens[:3]):  # Check first 3 query terms
            pos = doc_lower.find(term)
            if pos != -1 and pos < 500:  # In first 500 chars
                position_bonus += 0.1 * (1 - pos / 500)
        
        # Combine scores
        score = (
            tfidf_score * 0.3 +
            jaccard * 0.2 +
            coverage * 0.4 +
            position_bonus * 0.1
        )
        
        return min(1.0, score)


class KnowledgeRetriever:
    """
    RAG-like knowledge retrieval system.
    Retrieves relevant knowledge before generating responses.
    """
    
    def __init__(self, memory_store):
        self.memory = memory_store
        self.matcher = SemanticMatcher()
        self.confidence_threshold = 0.4
        
        # Cache for faster retrieval
        self.knowledge_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def has_knowledge(self, query: str) -> Tuple[bool, float, List[Dict]]:
        """
        Check if we have relevant knowledge for the query.
        Returns: (has_knowledge, confidence, relevant_docs)
        """
        # Extract concepts
        concepts = self._extract_concepts(query)
        
        if not concepts:
            return False, 0.0, []
        
        # Search knowledge base
        all_results = []
        
        for concept in concepts:
            results = self.memory.search_knowledge(concept, limit=5)
            all_results.extend(results)
        
        if not all_results:
            return False, 0.0, []
        
        # Score and rank results
        scored_results = []
        for result in all_results:
            content = result.get('content', '')
            similarity = self.matcher.semantic_similarity(query, content)
            
            # Combine with stored confidence
            stored_conf = result.get('confidence', 0.5)
            final_score = similarity * 0.7 + stored_conf * 0.3
            
            scored_results.append({
                **result,
                'relevance_score': final_score
            })
        
        # Remove duplicates and sort
        seen = set()
        unique_results = []
        for r in scored_results:
            content_hash = hash(r.get('content', '')[:100])
            if content_hash not in seen:
                seen.add(content_hash)
                unique_results.append(r)
        
        unique_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Determine if we have sufficient knowledge
        if unique_results:
            top_score = unique_results[0]['relevance_score']
            has_knowledge = top_score >= self.confidence_threshold
            return has_knowledge, top_score, unique_results[:5]
        
        return False, 0.0, []
    
    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieve relevant knowledge for answering a query.
        """
        has_knowledge, confidence, results = self.has_knowledge(query)
        
        if not has_knowledge:
            return {
                'has_knowledge': False,
                'confidence': confidence,
                'documents': [],
                'answer_hint': None,
                'needs_search': True
            }
        
        # Extract relevant passages
        passages = []
        sources = []
        
        for doc in results[:top_k]:
            content = doc.get('content', '')
            
            # Extract most relevant sentences
            relevant_text = self._extract_relevant_sentences(query, content)
            if relevant_text:
                passages.append(relevant_text)
            
            # Track sources
            url = doc.get('source_url', '')
            title = doc.get('source_title', '')
            if url and url not in [s['url'] for s in sources]:
                sources.append({'url': url, 'title': title})
        
        # Generate answer hint
        answer_hint = self._generate_answer_hint(query, passages)
        
        return {
            'has_knowledge': True,
            'confidence': confidence,
            'documents': results[:top_k],
            'passages': passages,
            'sources': sources,
            'answer_hint': answer_hint,
            'needs_search': confidence < 0.6  # Might want to augment with search
        }
    
    def _extract_concepts(self, query: str) -> List[str]:
        """Extract key concepts from query"""
        tokens = self.matcher.tokenize(query)
        
        # Filter to meaningful concepts
        concepts = []
        for token in tokens:
            if len(token) > 3:  # Skip very short words
                concepts.append(token)
        
        # Also try phrase extraction (bigrams)
        words = query.lower().split()
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if len(bigram) > 5 and all(w not in self.matcher.stopwords for w in [words[i], words[i+1]]):
                concepts.append(bigram)
        
        return concepts[:10]  # Limit to top 10
    
    def _extract_relevant_sentences(self, query: str, content: str, max_sentences: int = 3) -> str:
        """Extract most relevant sentences from content"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return content[:300]
        
        # Score sentences
        query_tokens = set(self.matcher.tokenize(query))
        scored = []
        
        for sent in sentences:
            sent_tokens = set(self.matcher.tokenize(sent))
            if sent_tokens:
                overlap = len(query_tokens & sent_tokens)
                score = overlap / len(query_tokens) if query_tokens else 0
                scored.append((sent, score))
        
        # Sort by relevance
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Return top sentences
        top_sentences = [s[0] for s in scored[:max_sentences] if s[1] > 0]
        
        if top_sentences:
            return '. '.join(top_sentences) + '.'
        
        return sentences[0] if sentences else content[:300]
    
    def _generate_answer_hint(self, query: str, passages: List[str]) -> Optional[str]:
        """Generate a hint/summary for answering"""
        if not passages:
            return None
        
        # Simple: return most relevant passage
        combined = ' '.join(passages[:2])
        
        # Clean up
        combined = re.sub(r'\s+', ' ', combined).strip()
        
        if len(combined) > 500:
            combined = combined[:500] + '...'
        
        return combined


class SmartResponseSystem:
    """
    Intelligent response system that checks knowledge first,
    then searches only if needed.
    """
    
    def __init__(self, memory_store, learner=None):
        self.memory = memory_store
        self.learner = learner
        self.retriever = KnowledgeRetriever(memory_store)
        
        # Thresholds
        self.knowledge_threshold = 0.5
        self.search_threshold = 0.35
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query intelligently:
        1. Check if we have knowledge
        2. If yes and confident, respond from knowledge
        3. If no or uncertain, search and learn
        """
        # Step 1: Check knowledge
        retrieval = self.retriever.retrieve(query)
        
        if retrieval['has_knowledge'] and retrieval['confidence'] >= self.knowledge_threshold:
            # We have good knowledge - respond from it
            return {
                'action': 'respond_from_knowledge',
                'confidence': retrieval['confidence'],
                'passages': retrieval.get('passages', []),
                'sources': retrieval.get('sources', []),
                'answer_hint': retrieval.get('answer_hint'),
                'needs_search': False
            }
        
        elif retrieval['confidence'] >= self.search_threshold:
            # We have some knowledge but might want to augment
            return {
                'action': 'augment_with_search',
                'confidence': retrieval['confidence'],
                'passages': retrieval.get('passages', []),
                'sources': retrieval.get('sources', []),
                'answer_hint': retrieval.get('answer_hint'),
                'needs_search': True,
                'search_query': query
            }
        
        else:
            # No knowledge - need to search
            return {
                'action': 'search_and_learn',
                'confidence': 0.0,
                'passages': [],
                'sources': [],
                'answer_hint': None,
                'needs_search': True,
                'search_query': query
            }
    
    async def search_and_respond(self, query: str) -> Dict[str, Any]:
        """
        Search the web, learn, and respond.
        This is called when knowledge is insufficient.
        """
        if not self.learner:
            return {
                'success': False,
                'error': 'No learner configured'
            }
        
        # Search and learn
        learn_result = self.learner.web_search_and_learn(query, max_results=3)
        
        if not learn_result.get('success'):
            return {
                'success': False,
                'error': 'Could not find relevant information'
            }
        
        # Now retrieve from new knowledge
        retrieval = self.retriever.retrieve(query)
        
        return {
            'success': True,
            'learned_from': learn_result.get('learned_from', []),
            'confidence': retrieval['confidence'],
            'passages': retrieval.get('passages', []),
            'sources': retrieval.get('sources', []),
            'answer_hint': retrieval.get('answer_hint')
        }


class KnowledgeGraph:
    """
    Simple knowledge graph for tracking what the AI knows.
    Helps AI understand its own knowledge gaps.
    """
    
    def __init__(self, memory_store):
        self.memory = memory_store
        self.concept_connections = {}  # concept -> related concepts
        self.concept_coverage = {}     # concept -> coverage score (0-1)
        self.learned_topics = set()
        self.topic_sources = {}        # topic -> list of sources
    
    def add_knowledge(self, topic: str, content: str, source: str) -> None:
        """Add knowledge and update graph"""
        self.learned_topics.add(topic.lower())
        
        # Track source
        if topic.lower() not in self.topic_sources:
            self.topic_sources[topic.lower()] = []
        self.topic_sources[topic.lower()].append(source)
        
        # Extract concepts from content
        concepts = self._extract_concepts(content)
        
        # Update coverage
        for concept in concepts:
            if concept not in self.concept_coverage:
                self.concept_coverage[concept] = 0.0
            self.concept_coverage[concept] = min(1.0, self.concept_coverage[concept] + 0.1)
        
        # Update connections
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i+1:]:
                if c1 not in self.concept_connections:
                    self.concept_connections[c1] = {}
                if c2 not in self.concept_connections[c1]:
                    self.concept_connections[c1][c2] = 0
                self.concept_connections[c1][c2] += 1
    
    def knows_about(self, topic: str) -> Tuple[bool, float]:
        """Check if AI knows about a topic"""
        topic_lower = topic.lower()
        
        if topic_lower in self.learned_topics:
            return True, self.concept_coverage.get(topic_lower, 0.5)
        
        # Check partial matches
        for known_topic in self.learned_topics:
            if topic_lower in known_topic or known_topic in topic_lower:
                return True, self.concept_coverage.get(known_topic, 0.3)
        
        return False, 0.0
    
    def get_knowledge_gaps(self, query: str) -> List[str]:
        """Identify what we don't know related to query"""
        concepts = self._extract_concepts(query)
        gaps = []
        
        for concept in concepts:
            if concept not in self.learned_topics:
                gaps.append(concept)
        
        return gaps
    
    def suggest_topics_to_learn(self, n: int = 5) -> List[str]:
        """Suggest topics to learn based on knowledge gaps"""
        # Find concepts we've seen but don't know well
        weak_concepts = [
            (c, score) for c, score in self.concept_coverage.items()
            if score < 0.5
        ]
        weak_concepts.sort(key=lambda x: x[1])
        
        return [c[0] for c in weak_concepts[:n]]
    
    def get_related_topics(self, topic: str) -> List[str]:
        """Get topics related to a given topic"""
        topic_lower = topic.lower()
        
        if topic_lower in self.concept_connections:
            related = self.concept_connections[topic_lower]
            sorted_related = sorted(related.items(), key=lambda x: x[1], reverse=True)
            return [r[0] for r in sorted_related[:5]]
        
        return []
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract concepts from text"""
        # Simple: extract nouns and noun phrases
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'to', 'of', 'in', 'for'}
        concepts = [w for w in words if w not in stopwords and len(w) > 3]
        
        return list(set(concepts))[:20]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        return {
            'total_topics': len(self.learned_topics),
            'total_concepts': len(self.concept_coverage),
            'total_connections': sum(len(c) for c in self.concept_connections.values()),
            'average_coverage': sum(self.concept_coverage.values()) / max(1, len(self.concept_coverage))
        }