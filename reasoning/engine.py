"""
Reasoning Engine
================
Intelligent question answering using semantic search AND
symbolic reasoning with persistent knowledge graph.

Features:
- Persistent Knowledge Graph reasoning (SQLite-backed, survives restarts)
- Semantic knowledge retrieval
- Question classification
- Inference engine
- Confidence calibration
- Answer synthesis
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from storage import KnowledgeBase

# Import the advanced reasoner
try:
    from .advanced_reasoner import AdvancedReasoner
    REASONER_AVAILABLE = True
except ImportError:
    try:
        from advanced_reasoner import AdvancedReasoner
        REASONER_AVAILABLE = True
    except ImportError:
        REASONER_AVAILABLE = False
        print("⚠️ AdvancedReasoner not available")


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
            r'\byou\b.*\bcan\b', r'\byour\b', r'neuralmind', r'what can you',
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
        
        response = f"""I'm **NeuralMind**, an AI that learns from scratch!

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
    Generates responses using BOTH:
    1. Vector search (semantic similarity)
    2. Advanced Knowledge Graph (symbolic AI with common sense)
    
    This combines retrieval with actual reasoning!
    """
    
    def __init__(self, knowledge_base: KnowledgeBase, data_dir=None):
        self.kb = knowledge_base
        self.reasoning = ReasoningEngine(knowledge_base)
        
        # Initialize the advanced reasoner (with common sense, analogies, multi-hop)
        if REASONER_AVAILABLE and data_dir:
            self.graph_reasoner = AdvancedReasoner(data_dir)
            print("✅ Advanced Knowledge Graph Reasoner initialized")
        else:
            self.graph_reasoner = None
        
        # Import context manager
        from .context import get_context, ConversationContext
        self.get_context = get_context
        
        print("✅ Response Generator initialized (with context + advanced reasoning)")
    
    def learn_to_graph(self, content: str, source: str = "") -> Dict[str, Any]:
        """
        Feed learned content to the persistent knowledge graph.
        Called automatically when new knowledge is added.
        """
        if not self.graph_reasoner:
            return {'facts_added': 0}
        
        result = self.graph_reasoner.learn(content, source)
        return result
    
    def generate(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """Generate a response with conversation context AND reasoning"""
        context = self.get_context(session_id)
        
        # First, try to resolve any references in the query
        resolved_entity, clarified_query = context.resolve_reference(query)
        
        if resolved_entity:
            # User is referring to a previous entity
            # Search specifically for that entity
            results = self.kb.search(resolved_entity.name, limit=5)
            
            if results:
                # Find the best match for this specific entity
                best_match = None
                for r in results:
                    if resolved_entity.name.lower() in r.get('source_title', '').lower():
                        best_match = r
                        break
                
                if not best_match:
                    best_match = results[0]
                
                answer = self._extract_focused_answer(resolved_entity.name, best_match.get('content', ''))
                
                # Update context
                context.add_user_message(query)
                context.add_assistant_response(answer, [{
                    'name': resolved_entity.name,
                    'content': answer[:200],
                    'source_url': best_match.get('source_url', ''),
                    'source_title': best_match.get('source_title', ''),
                    'confidence': best_match.get('relevance', 0.7)
                }])
                
                return {
                    'response': answer,
                    'confidence': best_match.get('relevance', 0.7),
                    'sources': [{'url': best_match.get('source_url', ''), 
                                'title': best_match.get('source_title', '')}],
                    'needs_search': False,
                    'resolved_from': resolved_entity.name,
                    'thought_process': [{'step': f'Resolved reference to: {resolved_entity.name}', 
                                        'type': 'context', 'confidence': 0.9}],
                    'question_type': 'definition'
                }
        
        # =====================================================
        # HYBRID REASONING: Try Knowledge Graph FIRST
        # =====================================================
        graph_result = None
        if self.graph_reasoner:
            graph_result = self.graph_reasoner.reason(query)
            
            # If knowledge graph has a confident answer, use it
            if graph_result and graph_result['confidence'] >= 0.6 and graph_result['facts_used'] > 0:
                # High confidence from knowledge graph!
                answer = graph_result['answer']
                
                # Still search vectors for sources/additional info
                vector_results = self.kb.search(query, limit=3)
                sources = [{'url': r.get('source_url', ''), 'title': r.get('source_title', '')} 
                          for r in vector_results if r.get('source_url')]
                
                # Update context
                context.add_user_message(query)
                context.add_assistant_response(answer, [{
                    'name': query,
                    'content': answer[:200],
                    'source_url': sources[0]['url'] if sources else '',
                    'source_title': sources[0]['title'] if sources else '',
                    'confidence': graph_result['confidence']
                }])
                
                return {
                    'response': answer,
                    'confidence': graph_result['confidence'],
                    'sources': sources[:3],
                    'needs_search': False,
                    'reasoning_type': 'knowledge_graph',
                    'facts_used': graph_result['facts_used'],
                    'thought_process': [
                        {'step': 'Knowledge Graph Reasoning', 'type': 'graph', 
                         'confidence': graph_result['confidence']},
                        {'step': graph_result.get('reasoning_trace', ''), 'type': 'trace'}
                    ],
                    'question_type': graph_result.get('question_type', 'factual')
                }
        
        # =====================================================
        # FALLBACK: Use vector search + extraction
        # =====================================================
        result = self.reasoning.reason(query)
        
        # Check if we need disambiguation
        if result.answer and not result.needs_search:
            raw_results = self.kb.search(query, limit=10)
            
            if context.needs_disambiguation(raw_results):
                disambig_response, entities = context.format_disambiguation(query, raw_results)
                
                if disambig_response and entities:
                    context.add_user_message(query)
                    context.add_assistant_response(disambig_response, entities)
                    
                    sources = [{'url': e['source_url'], 'title': e['source_title']} 
                              for e in entities if e['source_url']]
                    
                    return {
                        'response': disambig_response,
                        'confidence': 0.8,
                        'sources': sources[:5],
                        'needs_search': False,
                        'disambiguation': True,
                        'options': [e['name'] for e in entities],
                        'thought_process': [{'step': 'Multiple matches found - asking for clarification',
                                            'type': 'disambiguation', 'confidence': 0.8}],
                        'question_type': 'disambiguation'
                    }
        
        # Combine with graph reasoning if available but low confidence
        if graph_result and graph_result['confidence'] > 0.1 and result.confidence < 0.5:
            # Graph has something, vector search weak - blend them
            if graph_result['answer'] and "don't have" not in graph_result['answer']:
                result.answer = f"{graph_result['answer']}\n\n{result.answer}" if result.answer else graph_result['answer']
                result.confidence = max(result.confidence, graph_result['confidence'])
        
        # Update context with this exchange
        context.add_user_message(query)
        
        if result.answer:
            entities = context.extract_entities_from_response(result.answer, result.sources)
            context.add_assistant_response(result.answer, entities if entities else [{
                'name': query,
                'content': result.answer[:200] if result.answer else '',
                'source_url': result.sources[0]['url'] if result.sources else '',
                'source_title': result.sources[0]['title'] if result.sources else '',
                'confidence': result.confidence
            }])
        
        return {
            'response': result.answer,
            'confidence': result.confidence,
            'sources': result.sources,
            'needs_search': result.needs_search,
            'reasoning_type': 'vector_search',
            'thought_process': result.thought_process,
            'question_type': result.question_type.value
        }
    
    def _extract_focused_answer(self, entity_name: str, content: str) -> str:
        """Extract answer focused on a specific entity"""
        if not content:
            return f"I found information about {entity_name} but couldn't extract details."
        
        # Get sentences mentioning the entity
        sentences = re.split(r'(?<=[.!?])\s+', content)
        relevant = []
        
        entity_lower = entity_name.lower()
        entity_parts = set(entity_lower.split())
        
        for sent in sentences:
            sent_lower = sent.lower()
            # Check if sentence mentions entity
            if entity_lower in sent_lower or any(part in sent_lower for part in entity_parts if len(part) > 3):
                relevant.append(sent)
        
        if relevant:
            answer = ' '.join(relevant[:4])
            if len(answer) > 600:
                answer = answer[:600] + '...'
            return answer
        
        # Fallback to first part of content
        return content[:500] + ('...' if len(content) > 500 else '')
    
    def generate_after_learning(self, query: str, 
                                learned: List[Dict] = None,
                                session_id: str = "default") -> Dict[str, Any]:
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
        
        # Update context
        context = self.get_context(session_id)
        context.add_user_message(query)
        if result.answer:
            entities = [{
                'name': lc.get('title', query),
                'content': result.answer[:200] if result.answer else '',
                'source_url': lc.get('url', ''),
                'source_title': lc.get('title', ''),
                'confidence': result.confidence
            } for lc in (learned or [])]
            context.add_assistant_response(result.answer, entities)
        
        return {
            'response': result.answer,
            'confidence': result.confidence,
            'sources': sources[:5],
            'needs_search': result.needs_search,
            'thought_process': result.thought_process,
            'question_type': result.question_type.value,
            'learned_from': learned
        }
    
    def clear_context(self, session_id: str = "default") -> None:
        """Clear conversation context"""
        from .context import clear_context
        clear_context(session_id)