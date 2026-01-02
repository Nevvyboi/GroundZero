"""
Knowledge Processor
===================
Processes and summarizes learned content.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter


class KnowledgeProcessor:
    """Processes text into structured knowledge"""
    
    def __init__(self, memory_store=None):
        self.memory = memory_store
    
    def process(
        self,
        text: str,
        source: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Process text into structured knowledge.
        
        Returns dict with:
            - summary: Brief summary
            - concepts: Extracted concepts
            - facts: Key facts
            - keywords: Important keywords
        """
        # Generate summary
        summary = self._summarize(text)
        
        # Extract concepts
        concepts = self._extract_concepts(text)
        
        # Extract keywords
        keywords = self._extract_keywords(text)
        
        # Extract facts (simple sentence extraction)
        facts = self._extract_facts(text)
        
        return {
            'summary': summary,
            'concepts': concepts,
            'keywords': keywords,
            'facts': facts,
            'source': source,
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text))
        }
    
    def _summarize(self, text: str, max_sentences: int = 3) -> str:
        """Generate extractive summary"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) <= max_sentences:
            return text
        
        # Score sentences by position and content
        scored = []
        for i, sent in enumerate(sentences):
            score = 0
            
            # Position score (first sentences are usually important)
            if i < 3:
                score += (3 - i) * 0.5
            
            # Length score (prefer medium-length sentences)
            word_count = len(sent.split())
            if 10 <= word_count <= 30:
                score += 1.0
            
            # Keyword density
            keywords = self._extract_keywords(sent)
            score += len(keywords) * 0.2
            
            scored.append((score, i, sent))
        
        # Sort by score and get top sentences
        scored.sort(reverse=True)
        top = sorted(scored[:max_sentences], key=lambda x: x[1])  # Restore order
        
        return ' '.join([s[2] for s in top])
    
    def _extract_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Extract noun phrases as concepts"""
        concepts = []
        
        # Simple pattern matching for concepts
        # Look for capitalized phrases (proper nouns)
        proper_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        
        concept_counts = Counter(proper_nouns)
        
        for concept, count in concept_counts.most_common(10):
            if len(concept) > 2:  # Skip very short matches
                concepts.append({
                    'name': concept,
                    'count': count,
                    'type': 'proper_noun'
                })
        
        # Look for defined terms (X is a/an Y)
        definitions = re.findall(
            r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s+(?:is|are)\s+(?:a|an|the)\s+([a-z]+(?:\s+[a-z]+)*)',
            text
        )
        
        for term, definition in definitions[:5]:
            concepts.append({
                'name': term,
                'definition': definition,
                'type': 'defined_term'
            })
        
        return concepts
    
    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract important keywords using TF heuristics"""
        # Tokenize
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'that', 'this', 'with', 'from', 'have', 'been', 'were', 'they',
            'their', 'what', 'when', 'where', 'which', 'there', 'these',
            'those', 'about', 'would', 'could', 'should', 'into', 'more',
            'some', 'such', 'than', 'then', 'them', 'only', 'over', 'also',
            'after', 'most', 'other', 'very', 'just', 'being', 'through'
        }
        
        words = [w for w in words if w not in stop_words]
        
        # Count and return top keywords
        word_counts = Counter(words)
        return [word for word, _ in word_counts.most_common(top_n)]
    
    def _extract_facts(self, text: str, max_facts: int = 5) -> List[str]:
        """Extract fact-like sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        facts = []
        
        for sent in sentences:
            # Look for factual patterns
            factual_patterns = [
                r'^.+\s+is\s+.+$',  # X is Y
                r'^.+\s+was\s+.+$',  # X was Y
                r'^.+\s+are\s+.+$',  # X are Y
                r'^.+\s+were\s+.+$',  # X were Y
                r'^.+\s+has\s+.+$',  # X has Y
                r'^.+\s+have\s+.+$',  # X have Y
                r'^\d+.+',  # Starts with number (likely statistical)
            ]
            
            for pattern in factual_patterns:
                if re.match(pattern, sent, re.IGNORECASE):
                    if 10 <= len(sent.split()) <= 30:  # Reasonable length
                        facts.append(sent.strip())
                        break
            
            if len(facts) >= max_facts:
                break
        
        return facts
    
    def assess_relevance(
        self,
        text: str,
        query: str
    ) -> float:
        """
        Assess how relevant text is to a query.
        Returns score between 0 and 1.
        """
        text_lower = text.lower()
        query_lower = query.lower()
        
        # Extract query terms
        query_terms = set(re.findall(r'\b[a-z]{3,}\b', query_lower))
        
        if not query_terms:
            return 0.0
        
        # Count matching terms
        text_words = set(re.findall(r'\b[a-z]{3,}\b', text_lower))
        matches = query_terms & text_words
        
        # Base score from term overlap
        score = len(matches) / len(query_terms)
        
        # Bonus for exact phrase match
        if query_lower in text_lower:
            score = min(1.0, score + 0.3)
        
        return score
    
    def deduplicate(self, texts: List[str], threshold: float = 0.8) -> List[str]:
        """Remove near-duplicate texts"""
        if not texts:
            return []
        
        unique = [texts[0]]
        
        for text in texts[1:]:
            is_duplicate = False
            
            for existing in unique:
                similarity = self._text_similarity(text, existing)
                if similarity > threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(text)
        
        return unique
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple Jaccard similarity between texts"""
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
