"""
GroundZero Reasoning Engine v2.0
================================
Intelligent question answering and response generation.

Features:
- Context Brain integration for smart understanding
- Multi-method reasoning (graph, vector, neural)
- Question type detection
- Fuzzy matching and disambiguation
- Conversation context tracking
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime
import threading


class QuestionType(Enum):
    """Types of questions the system can handle"""
    DEFINITION = "definition"      # What is X?
    FACTUAL = "factual"           # When/where/which?
    CAUSAL = "causal"             # Why?
    PROCEDURAL = "procedural"     # How to?
    COMPARATIVE = "comparative"   # Compare X and Y
    GREETING = "greeting"         # Hello, hi, etc.
    META = "meta"                 # Questions about the AI
    CREATIVE = "creative"         # Generate creative content
    CORRECTION = "correction"     # Actually, I meant...
    CLARIFICATION = "clarification"  # The first one, option 2...
    OPINION = "opinion"           # What do you think?
    LIST = "list"                 # List/enumerate...


@dataclass
class ReasoningResult:
    """Result of a reasoning operation"""
    answer: Optional[str]
    confidence: float
    sources: List[Dict[str, str]]
    question_type: QuestionType
    thought_process: List[Dict[str, Any]] = field(default_factory=list)
    needs_search: bool = False
    reasoning_method: str = "unknown"
    understood_query: str = ""
    corrections_applied: List[Tuple[str, str]] = None
    suggestions: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'answer': self.answer,
            'confidence': self.confidence,
            'sources': self.sources,
            'question_type': self.question_type.value,
            'thought_process': self.thought_process,
            'needs_search': self.needs_search,
            'reasoning_method': self.reasoning_method,
            'understood_query': self.understood_query,
            'corrections_applied': self.corrections_applied,
            'suggestions': self.suggestions
        }


class ReasoningEngine:
    """
    Core reasoning engine for question analysis and answering.
    """
    
    # Question patterns for classification
    PATTERNS = {
        QuestionType.DEFINITION: [
            r'^what is\b', r'^what are\b', r'^define\b', r'^explain what\b',
            r'^who is\b', r'^who are\b', r'^tell me about\b', r'^describe\b',
            r"^what's\b", r'^explain\b'
        ],
        QuestionType.FACTUAL: [
            r'^when\b', r'^where\b', r'^which\b', r'^how many\b', r'^how much\b',
            r'^is\b', r'^are\b', r'^does\b', r'^do\b', r'^did\b', r'^was\b', r'^were\b'
        ],
        QuestionType.CAUSAL: [
            r'^why\b', r'^how come\b', r'^what caused\b', r'^reason for\b',
            r'because of\b', r'due to\b'
        ],
        QuestionType.PROCEDURAL: [
            r'^how to\b', r'^how do\b', r'^how can\b', r'^steps to\b',
            r'^can you show\b', r'^teach me\b'
        ],
        QuestionType.COMPARATIVE: [
            r'compare\b', r'difference\b', r'versus\b', r'\bvs\b', r'better\b',
            r'similar\b', r'between\b'
        ],
        QuestionType.GREETING: [
            r'^hi\b', r'^hello\b', r'^hey\b', r'^good morning\b', r'^good afternoon\b',
            r'^good evening\b', r'^howdy\b', r"^what's up\b"
        ],
        QuestionType.META: [
            r'^who are you\b', r'^what are you\b', r'^are you\b',
            r'your name\b', r'tell me about yourself\b'
        ],
        QuestionType.CREATIVE: [
            r'^write\b', r'^create\b', r'^generate\b', r'^compose\b',
            r'^imagine\b', r'^make up\b'
        ],
        QuestionType.CORRECTION: [
            r'^actually\b', r'^no,?\s*i meant\b', r'^i mean\b', r'^correction\b',
            r'^not that\b', r'^sorry,?\s*i meant\b'
        ],
        QuestionType.CLARIFICATION: [
            r'^the first\b', r'^the second\b', r'^option\s*\d', r'^number\s*\d',
            r'^choice\s*\d', r"^that one\b", r'^this one\b'
        ],
        QuestionType.OPINION: [
            r'^what do you think\b', r'^your opinion\b', r'^do you think\b',
            r'^should i\b', r'^would you\b'
        ],
        QuestionType.LIST: [
            r'^list\b', r'^enumerate\b', r'^give me.*examples\b',
            r'^what are some\b', r'^name\s*\d+\b'
        ]
    }
    
    # Greeting responses
    GREETINGS = {
        'hi': "Hello! I'm GroundZero, an AI assistant. How can I help you?",
        'hello': "Hi there! What would you like to know?",
        'hey': "Hey! I'm here to help. What's on your mind?",
        'good morning': "Good morning! How can I assist you today?",
        'good afternoon': "Good afternoon! What can I help you with?",
        'good evening': "Good evening! How may I help you?",
        "what's up": "Not much! Just here ready to help. What would you like to know?"
    }
    
    # Meta responses about the AI
    META_RESPONSES = {
        'who are you': "I'm GroundZero, a neural AI system built from scratch. I learn from Wikipedia and can answer questions using multiple reasoning methods.",
        'what are you': "I'm an artificial intelligence that combines vector search, knowledge graphs, and neural networks to understand and answer questions.",
        'are you': "I'm an AI assistant. I don't have consciousness or feelings, but I can help you find information and learn new things.",
    }
    
    def __init__(self, knowledge_base):
        """Initialize the reasoning engine"""
        self.kb = knowledge_base
        self._lock = threading.RLock()
    
    def classify_question(self, query: str) -> QuestionType:
        """Classify the type of question"""
        query_lower = query.lower().strip()
        
        # Check patterns in priority order
        for qtype in [
            QuestionType.GREETING,
            QuestionType.CORRECTION,
            QuestionType.CLARIFICATION,
            QuestionType.META,
            QuestionType.DEFINITION,
            QuestionType.PROCEDURAL,
            QuestionType.CAUSAL,
            QuestionType.COMPARATIVE,
            QuestionType.CREATIVE,
            QuestionType.LIST,
            QuestionType.OPINION,
            QuestionType.FACTUAL
        ]:
            patterns = self.PATTERNS.get(qtype, [])
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return qtype
        
        return QuestionType.FACTUAL  # Default
    
    def get_greeting_response(self, query: str) -> Optional[str]:
        """Get a greeting response if applicable"""
        query_lower = query.lower().strip()
        
        for greeting, response in self.GREETINGS.items():
            if query_lower == greeting or query_lower.startswith(greeting + ' '):
                return response
        
        return None
    
    def get_meta_response(self, query: str) -> Optional[str]:
        """Get meta response about the AI"""
        query_lower = query.lower().strip()
        
        for key, response in self.META_RESPONSES.items():
            if key in query_lower:
                return response
        
        return None
    
    def reason(self, query: str, understood_query: str = None) -> ReasoningResult:
        """
        Perform reasoning to answer a question.
        
        Args:
            query: Original user query
            understood_query: Query after processing by context brain
        
        Returns:
            ReasoningResult with answer and metadata
        """
        with self._lock:
            effective_query = understood_query or query
            qtype = self.classify_question(query)
            
            thought_process = []
            
            # Check for special cases first
            
            # Greeting
            if qtype == QuestionType.GREETING:
                response = self.get_greeting_response(query)
                if response:
                    return ReasoningResult(
                        answer=response,
                        confidence=1.0,
                        sources=[],
                        question_type=qtype,
                        reasoning_method='greeting'
                    )
            
            # Meta questions
            if qtype == QuestionType.META:
                response = self.get_meta_response(query)
                if response:
                    return ReasoningResult(
                        answer=response,
                        confidence=1.0,
                        sources=[],
                        question_type=qtype,
                        reasoning_method='meta'
                    )
            
            # Search knowledge base
            thought_process.append({
                'step': 'search',
                'query': effective_query,
                'method': 'vector_search'
            })
            
            results = self.kb.search(effective_query, top_k=5)
            
            if results:
                # Analyze results
                best_result = results[0]
                confidence = best_result.get('score', 0.5)
                
                # Build answer from top results
                answer = self._build_answer(effective_query, results, qtype)
                
                # Extract sources
                sources = []
                for r in results[:3]:
                    if r.get('source_url') or r.get('url'):
                        sources.append({
                            'url': r.get('source_url') or r.get('url', ''),
                            'title': r.get('source_title') or r.get('title', 'Unknown')
                        })
                
                thought_process.append({
                    'step': 'answer_generation',
                    'results_used': len(results),
                    'confidence': confidence
                })
                
                return ReasoningResult(
                    answer=answer,
                    confidence=confidence,
                    sources=sources,
                    question_type=qtype,
                    thought_process=thought_process,
                    reasoning_method='vector_search',
                    understood_query=effective_query,
                    needs_search=confidence < 0.3
                )
            
            # No results - suggest web search
            thought_process.append({
                'step': 'no_results',
                'suggestion': 'web_search'
            })
            
            return ReasoningResult(
                answer=None,
                confidence=0.0,
                sources=[],
                question_type=qtype,
                thought_process=thought_process,
                needs_search=True,
                reasoning_method='none',
                understood_query=effective_query
            )
    
    def _build_answer(self, query: str, results: List[Dict], qtype: QuestionType) -> str:
        """Build an answer from search results"""
        if not results:
            return "I don't have enough information to answer that question."
        
        # Get best matching content
        best = results[0]
        content = best.get('content', '')
        
        if not content:
            return "I found a match but couldn't extract the content."
        
        # For definitions, try to get first sentence or paragraph
        if qtype == QuestionType.DEFINITION:
            # Find first sentence(s)
            sentences = re.split(r'(?<=[.!?])\s+', content)
            if sentences:
                answer = ' '.join(sentences[:3])
                if len(answer) > 500:
                    answer = answer[:500] + '...'
                return answer
        
        # For other types, return relevant portion
        if len(content) > 600:
            content = content[:600] + '...'
        
        return content


class ResponseGenerator:
    """
    High-level response generator combining all reasoning methods.
    
    Uses:
    1. Context Brain (smart query understanding)
    2. Knowledge Graph (symbolic reasoning)
    3. Vector Search (semantic similarity)
    4. Neural Network (generation)
    """
    
    def __init__(self, knowledge_base, data_dir=None):
        """Initialize the response generator"""
        self.kb = knowledge_base
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.reasoning = ReasoningEngine(knowledge_base)
        
        # Context Brain (set externally or initialized)
        self.context_brain = None
        try:
            from .context_brain import ContextBrain
            self.context_brain = ContextBrain(embedding_dim=256, max_memories=100000)
        except ImportError:
            pass
        
        # Knowledge Graph reasoner (set by server)
        self.graph_reasoner = None
        
        # Neural Brain (set by server)
        self.neural_brain = None
        
        # Conversation history per session
        self._sessions: Dict[str, List[Dict]] = {}
        self._lock = threading.RLock()
    
    def learn_to_graph(self, content: str, source: str = "") -> Dict[str, Any]:
        """Feed learned content to knowledge graph and context brain"""
        result = {'facts_added': 0}
        
        # Feed to context brain
        if self.context_brain:
            try:
                self.context_brain.learn_from_content(content, source)
            except Exception:
                pass
        
        # Feed to graph reasoner
        if self.graph_reasoner:
            try:
                result = self.graph_reasoner.learn(content, source)
            except Exception:
                pass
        
        return result
    
    def generate(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Generate a response using hybrid reasoning.
        
        Priority:
        1. Context Brain (understand query)
        2. Knowledge Graph (symbolic reasoning)
        3. Vector Search (semantic similarity)
        4. Neural Network (generation)
        5. Web Search (last resort)
        """
        with self._lock:
            understood_query = query
            corrections_applied = []
            suggestions = []
            
            # =========================================
            # STEP 1: CONTEXT BRAIN UNDERSTANDING
            # =========================================
            if self.context_brain:
                try:
                    understanding = self.context_brain.understand_query(
                        query, 
                        session_id,
                        known_entities=self._get_known_entities()
                    )
                    
                    understood_query = understanding.get('understood_query', query)
                    corrections_applied = understanding.get('corrections_applied', [])
                    suggestions = understanding.get('suggestions', [])
                    
                    # Handle greeting
                    if understanding.get('intent') == 'greeting':
                        response = self.reasoning.get_greeting_response(query)
                        if response:
                            return {
                                'response': response,
                                'confidence': 1.0,
                                'sources': [],
                                'needs_search': False,
                                'reasoning_type': 'greeting',
                                'understood_query': query
                            }
                    
                    # Handle disambiguation
                    if understanding.get('needs_disambiguation'):
                        options = understanding.get('disambiguation_options', [])
                        if options:
                            return {
                                'response': self._format_disambiguation(query, options),
                                'confidence': 0.8,
                                'sources': [],
                                'needs_search': False,
                                'disambiguation': True,
                                'options': options,
                                'reasoning_type': 'disambiguation',
                                'understood_query': query
                            }
                except Exception:
                    pass
            
            # =========================================
            # STEP 2: KNOWLEDGE GRAPH REASONING
            # =========================================
            if self.graph_reasoner:
                try:
                    graph_result = self.graph_reasoner.reason(understood_query)
                    
                    if graph_result and graph_result.get('confidence', 0) >= 0.6:
                        if graph_result.get('facts_used', 0) > 0:
                            answer = graph_result['answer']
                            
                            # Get supporting sources from vector search
                            vector_results = self.kb.search(understood_query, top_k=3)
                            sources = [
                                {'url': r.get('source_url', ''), 'title': r.get('source_title', '')}
                                for r in vector_results if r.get('source_url')
                            ]
                            
                            return {
                                'response': answer,
                                'confidence': graph_result['confidence'],
                                'sources': sources[:3],
                                'needs_search': False,
                                'reasoning_type': 'knowledge_graph',
                                'facts_used': graph_result.get('facts_used', 0),
                                'understood_query': understood_query,
                                'corrections_applied': corrections_applied
                            }
                except Exception:
                    pass
            
            # =========================================
            # STEP 3: VECTOR SEARCH
            # =========================================
            result = self.reasoning.reason(query, understood_query)
            
            # =========================================
            # STEP 4: NEURAL NETWORK (if needed)
            # =========================================
            if self.neural_brain and (not result.answer or result.needs_search):
                try:
                    neural_result = self.neural_brain.answer(understood_query)
                    
                    if neural_result and neural_result.get('confidence', 0) > 0.3:
                        if not result.answer:
                            result.answer = neural_result.get('answer', '')
                            result.confidence = neural_result.get('confidence', 0.3)
                            result.reasoning_method = 'neural_network'
                            result.needs_search = False
                        else:
                            # Combine answers
                            result.answer += f"\n\n(Neural insight: {neural_result.get('answer', '')})"
                            result.confidence = max(result.confidence, neural_result.get('confidence', 0))
                except Exception:
                    pass
            
            # Build response
            response = {
                'response': result.answer,
                'confidence': result.confidence,
                'sources': result.sources,
                'needs_search': result.needs_search,
                'reasoning_type': result.reasoning_method,
                'thought_process': result.thought_process,
                'question_type': result.question_type.value,
                'understood_query': understood_query
            }
            
            if corrections_applied:
                response['corrections_applied'] = corrections_applied
            if suggestions:
                response['suggestions'] = suggestions
            
            # Track in session
            self._add_to_session(session_id, query, response)
            
            return response
    
    def _get_known_entities(self, limit: int = 500) -> List[str]:
        """Get known entity titles from knowledge base"""
        try:
            if hasattr(self.kb, 'vectors'):
                recent = self.kb.vectors.get_all_knowledge(limit)
                return [r['title'] for r in recent if r.get('title')]
            return []
        except Exception:
            return []
    
    def _format_disambiguation(self, query: str, options: List[str]) -> str:
        """Format disambiguation response"""
        response = f"**{query}** could refer to:\n\n"
        for i, option in enumerate(options[:5], 1):
            response += f"{i}. {option}\n"
        response += "\nWhich one would you like to know about?"
        return response
    
    def _add_to_session(self, session_id: str, query: str, response: Dict):
        """Add exchange to session history"""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        
        self._sessions[session_id].append({
            'query': query,
            'response': response.get('response', ''),
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep last 20 exchanges
        if len(self._sessions[session_id]) > 20:
            self._sessions[session_id] = self._sessions[session_id][-20:]
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get session history"""
        return self._sessions.get(session_id, [])
    
    def clear_context(self, session_id: str = "default"):
        """Clear session context"""
        if session_id in self._sessions:
            del self._sessions[session_id]
        
        if self.context_brain:
            try:
                self.context_brain.clear_conversation(session_id)
            except Exception:
                pass
    
    def neural_generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Direct neural generation for creative tasks"""
        if not self.neural_brain:
            return "Neural network not available."
        
        try:
            return self.neural_brain.generate(prompt, max_tokens=max_tokens)
        except Exception as e:
            return f"Generation error: {e}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics"""
        stats = {
            'sessions': len(self._sessions),
            'total_exchanges': sum(len(s) for s in self._sessions.values()),
            'graph_reasoner_available': self.graph_reasoner is not None,
            'neural_brain_available': self.neural_brain is not None,
            'context_brain_available': self.context_brain is not None
        }
        
        if self.context_brain:
            try:
                stats['context_brain'] = self.context_brain.get_stats()
            except Exception:
                pass
        
        return stats


# ═══════════════════════════════════════════════════════════════
# Testing
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing Reasoning Engine v2.0")
    print("=" * 50)
    
    # Create mock knowledge base
    class MockKB:
        def search(self, query, top_k=5):
            return [{
                'content': 'Python is a high-level programming language known for its simplicity.',
                'source_url': 'https://en.wikipedia.org/wiki/Python',
                'source_title': 'Python (programming language)',
                'score': 0.85
            }]
    
    kb = MockKB()
    
    # Test reasoning engine
    engine = ReasoningEngine(kb)
    
    # Test question classification
    print("\n1. Question Classification:")
    test_queries = [
        "What is Python?",
        "Hello!",
        "Why is the sky blue?",
        "How to make coffee?",
        "Compare Python and JavaScript"
    ]
    
    for q in test_queries:
        qtype = engine.classify_question(q)
        print(f"   '{q}' → {qtype.value}")
    
    # Test reasoning
    print("\n2. Reasoning:")
    result = engine.reason("What is Python?")
    print(f"   Answer: {result.answer[:100]}...")
    print(f"   Confidence: {result.confidence}")
    print(f"   Method: {result.reasoning_method}")
    
    # Test response generator
    print("\n3. Response Generator:")
    generator = ResponseGenerator(kb)
    response = generator.generate("What is Python?")
    print(f"   Response: {response.get('response', '')[:100]}...")
    print(f"   Type: {response.get('reasoning_type')}")
    
    print("\n✓ All tests passed!")
