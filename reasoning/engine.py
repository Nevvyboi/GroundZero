"""
Reasoning Engine
================
Intelligent question answering using semantic search.

Features:
- Question classification
- Semantic knowledge retrieval
- Confidence calibration
- Answer synthesis
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from storage import KnowledgeBase


class QuestionType(Enum):
    """Types of questions"""
    DEFINITION = "definition"      # What is X?
    FACTUAL = "factual"           # Who/When/Where?
    CAUSAL = "causal"             # Why?
    PROCEDURAL = "procedural"     # How to?
    COMPARATIVE = "comparative"   # Compare X and Y
    GREETING = "greeting"         # Hi, hello
    META = "meta"                 # About the AI


@dataclass
class ReasoningResult:
    """Result of reasoning"""
    answer: Optional[str]
    confidence: float
    sources: List[Dict[str, str]]
    question_type: QuestionType
    thought_process: List[Dict[str, Any]]
    needs_search: bool = False


class ReasoningEngine:
    """
    Analyzes questions and generates reasoned answers.
    
    Uses semantic search to find relevant knowledge,
    then synthesizes an answer from the results.
    """
    
    # Question patterns for classification
    PATTERNS = {
        QuestionType.DEFINITION: [
            r'^what is\b', r'^what are\b', r'^define\b', r'^explain what\b',
            r'^who is\b', r'^who are\b', r'^tell me about\b', r'^describe\b'
        ],
        QuestionType.FACTUAL: [
            r'^when\b', r'^where\b', r'^which\b', r'^how many\b', r'^how much\b'
        ],
        QuestionType.CAUSAL: [
            r'^why\b', r'^how come\b', r'^what caused\b', r'^reason for\b'
        ],
        QuestionType.PROCEDURAL: [
            r'^how to\b', r'^how do\b', r'^how can\b', r'^steps to\b'
        ],
        QuestionType.COMPARATIVE: [
            r'compare\b', r'difference\b', r'versus\b', r'\bvs\b', r'better\b'
        ],
        QuestionType.GREETING: [
            r'^hi\b', r'^hello\b', r'^hey\b', r'^good morning\b', 
            r'^good afternoon\b', r'^good evening\b', r'^thanks\b'
        ],
        QuestionType.META: [
            r'\byou\b.*\bcan\b', r'\byour\b', r'groundzero', r'what can you',
            r'who are you', r'what are you', r'about yourself'
        ]
    }
    
    # Greeting responses
    GREETINGS = {
        'hi': "Hi there! How can I help you? Ask me anything!",
        'hello': "Hello! What would you like to know?",
        'hey': "Hey! What can I do for you?",
        'good morning': "Good morning! How can I assist you?",
        'good afternoon': "Good afternoon! What would you like to explore?",
        'good evening': "Good evening! How can I help?",
        'thanks': "You're welcome! Is there anything else you'd like to know?",
        'thank you': "You're welcome! Feel free to ask me anything.",
        'bye': "Goodbye! Come back anytime!",
        'goodbye': "Goodbye! It was nice chatting with you.",
        'how are you': "I'm doing great, always learning! What can I help you with?",
        "what's up": "Not much, just ready to help! What's on your mind?",
        'ok': "Great! Let me know if you have any questions.",
        'okay': "Alright! I'm here if you need anything.",
        'help': "I can help you with:\n\n• **Ask questions** - I'll search my knowledge\n• **Teach me** - Add knowledge directly\n• **Learn from URLs** - Paste any URL and I'll read it\n\nJust ask anything!"
    }
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
    
    def classify_question(self, query: str) -> QuestionType:
        """Classify the type of question"""
        query_lower = query.lower().strip()
        
        for q_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return q_type
        
        return QuestionType.DEFINITION
    
    def reason(self, query: str) -> ReasoningResult:
        """
        Main reasoning method.
        
        Process:
        1. Check for greetings/meta
        2. Classify question type
        3. Search knowledge semantically
        4. Calculate confidence
        5. Synthesize answer
        """
        query_clean = query.strip()
        query_lower = query_clean.lower()
        
        # Check greetings first
        for greeting, response in self.GREETINGS.items():
            if query_lower == greeting or query_lower.startswith(greeting + ' '):
                return ReasoningResult(
                    answer=response,
                    confidence=1.0,
                    sources=[],
                    question_type=QuestionType.GREETING,
                    thought_process=[],
                    needs_search=False
                )
        
        # Classify question
        q_type = self.classify_question(query_lower)
        
        # Handle meta questions
        if q_type == QuestionType.META:
            return self._handle_meta_question()
        
        # Build thought process
        thoughts = []
        thoughts.append({
            'step': f"Analyzing: {query_clean[:50]}...",
            'type': 'analysis',
            'confidence': 0.9
        })
        
        # Semantic search
        results = self.kb.search(query_clean, limit=10, min_score=0.1)
        
        thoughts.append({
            'step': f"Found {len(results)} relevant entries",
            'type': 'retrieval',
            'confidence': 0.8
        })
        
        # No results
        if not results:
            return ReasoningResult(
                answer=None,
                confidence=0.1,
                sources=[],
                question_type=q_type,
                thought_process=thoughts,
                needs_search=True
            )
        
        # Check confidence threshold
        best = results[0]
        confidence = best.get('relevance', 0)
        
        thoughts.append({
            'step': f"Best match: {confidence:.0%} confidence",
            'type': 'evaluation',
            'confidence': confidence
        })
        
        # Too low confidence - need web search
        if confidence < 0.25:
            return ReasoningResult(
                answer=None,
                confidence=confidence,
                sources=[],
                question_type=q_type,
                thought_process=thoughts,
                needs_search=True
            )
        
        # Extract answer
        answer = self._extract_answer(query_clean, best['content'])
        
        # Build sources
        sources = []
        seen = set()
        for r in results[:3]:
            url = r.get('source_url', '')
            if url and url not in seen:
                sources.append({
                    'url': url,
                    'title': r.get('source_title', 'Source')
                })
                seen.add(url)
        
        return ReasoningResult(
            answer=answer,
            confidence=confidence,
            sources=sources,
            question_type=q_type,
            thought_process=thoughts,
            needs_search=False
        )
    
    def _handle_meta_question(self) -> ReasoningResult:
        """Handle questions about the AI"""
        stats = self.kb.get_statistics()
        
        response = f"""I'm **GroundZero**, an AI that learns from scratch!

📚 **My Knowledge:**
- {stats['total_knowledge']:,} facts learned
- {stats['total_sources']:,} sources read
- {stats['vocabulary_size']:,} words in vocabulary
- {stats['total_words']:,} total words processed

🧠 **Vector Database:**
- {stats['vectors']['total_vectors']:,} embeddings stored
- {stats['embeddings']['dimension']}D vectors
- Index: {stats['vectors']['index_type']}

I learn by reading Wikipedia and websites, then store knowledge as vectors for semantic search. Ask me about any topic I've learned!"""
        
        return ReasoningResult(
            answer=response,
            confidence=1.0,
            sources=[],
            question_type=QuestionType.META,
            thought_process=[{
                'step': 'Self-reflection',
                'type': 'meta',
                'confidence': 1.0
            }],
            needs_search=False
        )
    
    def _extract_answer(self, query: str, content: str) -> str:
        """Extract relevant answer from content"""
        if not content:
            return "I found a reference but couldn't extract details."
        
        if len(content) < 600:
            return content
        
        # Find sentences with query words
        query_words = set(w.lower() for w in re.findall(r'\w+', query) if len(w) > 3)
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        relevant = []
        for sent in sentences:
            if len(sent) < 20:
                continue
            sent_lower = sent.lower()
            matches = sum(1 for w in query_words if w in sent_lower)
            if matches > 0:
                relevant.append((sent, matches))
        
        if relevant:
            relevant.sort(key=lambda x: x[1], reverse=True)
            answer = ' '.join(s[0] for s in relevant[:4])
            if len(answer) > 600:
                answer = answer[:600] + '...'
            return answer
        
        return content[:500] + ('...' if len(content) > 500 else '')


class ResponseGenerator:
    """
    Generates responses using the reasoning engine.
    Provides a simple interface for the API.
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.reasoning = ReasoningEngine(knowledge_base)
        print("✅ Response Generator initialized")
    
    def generate(self, query: str) -> Dict[str, Any]:
        """Generate a response"""
        result = self.reasoning.reason(query)
        
        return {
            'response': result.answer,
            'confidence': result.confidence,
            'sources': result.sources,
            'needs_search': result.needs_search,
            'thought_process': result.thought_process,
            'question_type': result.question_type.value
        }
    
    def generate_after_learning(self, query: str, 
                                learned: List[Dict] = None) -> Dict[str, Any]:
        """Generate response after learning"""
        result = self.reasoning.reason(query)
        
        sources = list(result.sources)
        if learned:
            seen = {s.get('url') for s in sources}
            for lc in learned:
                url = lc.get('url', '')
                if url and url not in seen:
                    sources.append({
                        'url': url,
                        'title': lc.get('title', 'Learned')
                    })
                    seen.add(url)
        
        return {
            'response': result.answer,
            'confidence': result.confidence,
            'sources': sources[:5],
            'needs_search': result.needs_search,
            'thought_process': result.thought_process,
            'question_type': result.question_type.value,
            'learned_from': learned
        }
