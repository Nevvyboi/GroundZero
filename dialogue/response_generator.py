"""
Response Generator v2
=====================
Intelligent response generation with knowledge-first approach.
- Check knowledge FIRST before searching
- Actually learn when needed, then respond
- Better reasoning with proper knowledge retrieval
- Self-aware of what it knows and doesn't know
"""

from typing import Dict, Any, Optional, List, Callable
import re
import time

from reasoning import ReasoningEngine, ReasoningType, Metacognition
from storage import MemoryStore


class ResponseGenerator:
    """
    Generates intelligent responses with knowledge-first approach.
    
    Flow:
    1. Check if it's a greeting/meta query
    2. Search INTERNAL knowledge first
    3. If confident (>50%) → respond from knowledge
    4. If not confident → trigger learning, then respond
    """
    
    # Greetings
    GREETINGS = {
        'hi': "Hi there! How can I help you? Ask me anything!",
        'hello': "Hello! What would you like to know?",
        'hey': "Hey! What can I do for you?",
        'good morning': "Good morning! How can I assist you?",
        'good afternoon': "Good afternoon! What would you like to explore?",
        'good evening': "Good evening! How can I help?",
        'thanks': "You're welcome! Is there anything else you'd like to know?",
        'thank you': "You're welcome! Feel free to ask me anything.",
        'bye': "Goodbye! Come back anytime.",
        'goodbye': "Goodbye! It was nice chatting with you.",
        'how are you': "I'm doing great, always learning! What can I help you with?",
        "what's up": "Not much, just ready to help! What's on your mind?",
        'ok': "Great! Let me know if you have any questions.",
        'okay': "Alright! I'm here if you need anything.",
        'help': "I can help you with:\n\n• **Ask questions** - I'll answer from my knowledge or search the web\n• **Teach me** - Add knowledge directly\n• **Learn from URLs** - Paste any URL and I'll read it\n\nJust ask anything!"
    }
    
    # Confidence thresholds
    CONFIDENT_THRESHOLD = 0.45  # Answer from knowledge if above this
    LEARN_THRESHOLD = 0.3       # Trigger learning if below this
    
    def __init__(
        self,
        memory_store: MemoryStore,
        neural_model,
        reasoning_engine: ReasoningEngine,
        metacognition: Metacognition,
        learner=None
    ):
        self.memory = memory_store
        self.model = neural_model
        self.reasoning = reasoning_engine
        self.metacognition = metacognition
        self.learner = learner
        
        # Learning callback for UI updates
        self.on_learning_start: Optional[Callable] = None
        self.on_learning_progress: Optional[Callable] = None
        self.on_learning_complete: Optional[Callable] = None
    
    def generate(self, query: str) -> Dict[str, Any]:
        """
        Generate a response to the query.
        
        Returns dict with:
            - response: The response text
            - confidence: Confidence level (0-1)
            - sources: List of source dicts
            - needs_search: Whether learning was triggered
            - thought_process: Reasoning steps for display
            - learning_triggered: If learning happened
        """
        query_lower = query.lower().strip()
        
        # Step 1: Check for simple greetings
        greeting = self._check_greeting(query_lower)
        if greeting:
            return greeting
        
        # Step 2: Check for self-awareness queries
        if self._is_self_query(query_lower):
            return self._handle_self_query(query)
        
        # Step 3: Try specialized reasoning (math, logic)
        specialized = self._try_specialized(query)
        if specialized:
            return specialized
        
        # Step 4: Search internal knowledge FIRST
        knowledge_result = self._search_knowledge(query)
        
        # Step 5: Decision - respond or learn
        if knowledge_result['confidence'] >= self.CONFIDENT_THRESHOLD:
            # We have enough knowledge - respond!
            return self._respond_from_knowledge(query, knowledge_result)
        else:
            # Not enough knowledge - signal need to learn
            return {
                'response': None,  # Signal that learning is needed
                'confidence': knowledge_result['confidence'],
                'sources': [],
                'needs_search': True,
                'search_query': self._extract_search_query(query),
                'thought_process': [
                    {'step': f"Analyzing question: {query[:50]}...", 'type': 'observation', 'confidence': 0.9},
                    {'step': f"Searching knowledge base...", 'type': 'analysis', 'confidence': 0.8},
                    {'step': f"Knowledge confidence: {knowledge_result['confidence']*100:.0f}% - Need to learn more", 'type': 'reflection', 'confidence': knowledge_result['confidence']}
                ],
                'partial_knowledge': knowledge_result.get('best_match', '')
            }
    
    def generate_after_learning(self, query: str, learned_content: List[Dict]) -> Dict[str, Any]:
        """
        Generate response AFTER learning has completed.
        This is called by the API after learning finishes.
        """
        # Re-search knowledge with new learned content
        knowledge_result = self._search_knowledge(query)
        
        # Now we should have better knowledge
        if knowledge_result['entries']:
            return self._respond_from_knowledge(query, knowledge_result, learned_from=learned_content)
        else:
            # Still no good knowledge - return what we learned
            if learned_content:
                summary = learned_content[0].get('summary', learned_content[0].get('content', '')[:500])
                return {
                    'response': f"I just learned about this! {summary}",
                    'confidence': 0.7,
                    'sources': [{'url': lc.get('url', ''), 'title': lc.get('title', '')} for lc in learned_content[:3]],
                    'needs_search': False,
                    'thought_process': [
                        {'step': 'Learned new information', 'type': 'synthesis', 'confidence': 0.7}
                    ]
                }
            else:
                return {
                    'response': "I couldn't find information about this topic. Could you try rephrasing or asking about something else?",
                    'confidence': 0.2,
                    'sources': [],
                    'needs_search': False,
                    'thought_process': []
                }
    
    def _check_greeting(self, query: str) -> Optional[Dict[str, Any]]:
        """Check if query is a simple greeting"""
        clean = re.sub(r'[^\w\s]', '', query).strip()
        
        for greeting, response in self.GREETINGS.items():
            if clean == greeting or query.strip() == greeting:
                return {
                    'response': response,
                    'confidence': 1.0,
                    'sources': [],
                    'needs_search': False,
                    'thought_process': []
                }
        return None
    
    def _is_self_query(self, query: str) -> bool:
        """Check if asking about the AI itself"""
        patterns = [
            'who are you', 'what are you', 'about yourself',
            'what can you do', 'your capabilities', 'what do you do',
            'what do you know', 'what have you learned', 'what topics',
            'your knowledge', 'how do you work', 'tell me about yourself',
            'how many words', 'vocabulary', 'how much do you know'
        ]
        return any(p in query for p in patterns)
    
    def _handle_self_query(self, query: str) -> Dict[str, Any]:
        """Handle questions about the AI itself"""
        stats = self.memory.get_statistics()
        
        vocab_size = stats.get('vocabulary_size', 0)
        knowledge_count = stats.get('knowledge_count', 0)
        sources = stats.get('sources_learned', 0)
        tokens = stats.get('total_tokens', 0)
        
        query_lower = query.lower()
        
        if 'how many words' in query_lower or 'vocabulary' in query_lower:
            response = f"I currently know **{vocab_size:,}** words in my vocabulary! I've processed **{tokens:,}** tokens from **{sources}** different sources."
        elif 'what do you know' in query_lower or 'what topics' in query_lower:
            # Get top concepts
            concepts = self.memory.get_top_concepts(8)
            if concepts:
                topic_list = ', '.join([c['name'] for c in concepts])
                response = f"I've learned about: **{topic_list}**\n\nI have **{knowledge_count}** pieces of knowledge from **{sources}** sources. Ask me about any of these topics!"
            else:
                response = f"I have **{knowledge_count}** pieces of knowledge from **{sources}** sources. I'm still learning - ask me questions or help me learn!"
        else:
            response = f"""I'm **NeuralMind**, an AI that learns from scratch!

**My current knowledge:**
• **{vocab_size:,}** words in vocabulary
• **{tokens:,}** tokens processed  
• **{knowledge_count}** knowledge entries
• **{sources}** sources learned

I think step-by-step, check what I know, and learn when I don't know something. Ask me anything!"""
        
        return {
            'response': response,
            'confidence': 1.0,
            'sources': [],
            'needs_search': False,
            'thought_process': [
                {'step': 'Reflecting on my own capabilities...', 'type': 'reflection', 'confidence': 1.0}
            ]
        }
    
    def _try_specialized(self, query: str) -> Optional[Dict[str, Any]]:
        """Try specialized reasoning (math, logic, code)"""
        result = self.reasoning.reason(query)
        
        if result.reasoning_type != ReasoningType.GENERAL and result.success:
            return {
                'response': result.final_answer,
                'confidence': result.confidence,
                'sources': [],
                'needs_search': False,
                'thought_process': [
                    {'step': s.step, 'type': 'specialized', 'confidence': s.confidence}
                    for s in result.steps
                ]
            }
        return None
    
    def _search_knowledge(self, query: str) -> Dict[str, Any]:
        """
        Search internal knowledge base.
        Returns confidence and matching entries.
        """
        query_terms = set(self._tokenize(query.lower()))
        
        if not query_terms:
            return {'confidence': 0, 'entries': [], 'best_match': ''}
        
        # Search knowledge
        entries = self.memory.search_knowledge(query, limit=10)
        
        if not entries:
            return {'confidence': 0, 'entries': [], 'best_match': ''}
        
        # Score entries by relevance
        scored = []
        for entry in entries:
            content = entry.get('content', '').lower()
            content_terms = set(self._tokenize(content))
            
            # Calculate term overlap
            if content_terms:
                overlap = len(query_terms & content_terms) / len(query_terms)
            else:
                overlap = 0
            
            # Factor in stored confidence
            stored_conf = entry.get('confidence', 0.5)
            score = (overlap * 0.6) + (stored_conf * 0.4)
            
            if score > 0.1:
                scored.append({**entry, 'relevance': score})
        
        # Sort by relevance
        scored.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Calculate overall confidence
        if scored:
            # Use top match score, boosted slightly
            confidence = min(1.0, scored[0]['relevance'] * 1.3)
            best_match = scored[0].get('content', '')[:300]
        else:
            confidence = 0
            best_match = ''
        
        return {
            'confidence': confidence,
            'entries': scored[:5],
            'best_match': best_match
        }
    
    def _respond_from_knowledge(self, query: str, knowledge: Dict, learned_from: List = None) -> Dict[str, Any]:
        """Generate response from retrieved knowledge"""
        entries = knowledge.get('entries', [])
        
        if not entries:
            return {
                'response': "I couldn't find relevant information.",
                'confidence': 0,
                'sources': [],
                'needs_search': True,
                'thought_process': []
            }
        
        # Build response from top entries
        response_parts = []
        sources = []
        seen = set()
        
        for entry in entries[:3]:
            content = entry.get('content', '')
            source_url = entry.get('source_url', '')
            source_title = entry.get('source_title', '')
            
            # Skip duplicates
            content_hash = hash(content[:100])
            if content_hash in seen:
                continue
            seen.add(content_hash)
            
            # Extract relevant sentences
            relevant = self._extract_relevant(query, content)
            if relevant:
                response_parts.append(relevant)
            
            # Track source
            if source_url and source_url not in [s.get('url') for s in sources]:
                sources.append({
                    'url': source_url,
                    'title': source_title or 'Source'
                })
        
        if response_parts:
            response = ' '.join(response_parts)
            response = self._clean_response(response)
        else:
            response = entries[0].get('content', '')[:500]
        
        # Add learned sources if applicable
        if learned_from:
            for lc in learned_from:
                if lc.get('url') and lc.get('url') not in [s.get('url') for s in sources]:
                    sources.append({
                        'url': lc.get('url', ''),
                        'title': lc.get('title', 'Learned Source')
                    })
        
        thought_process = [
            {'step': f"Found {len(entries)} relevant knowledge entries", 'type': 'observation', 'confidence': 0.8},
            {'step': f"Best match relevance: {entries[0].get('relevance', 0)*100:.0f}%", 'type': 'analysis', 'confidence': entries[0].get('relevance', 0.5)},
            {'step': f"Synthesizing response from {len(response_parts)} sources", 'type': 'synthesis', 'confidence': knowledge['confidence']}
        ]
        
        return {
            'response': response,
            'confidence': knowledge['confidence'],
            'sources': sources[:3],
            'needs_search': False,
            'thought_process': thought_process
        }
    
    def _extract_relevant(self, query: str, content: str, max_sentences: int = 4) -> str:
        """Extract sentences most relevant to query"""
        query_terms = set(self._tokenize(query.lower()))
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return content[:400]
        
        # Score sentences
        scored = []
        for sent in sentences:
            sent_terms = set(self._tokenize(sent.lower()))
            if sent_terms:
                overlap = len(query_terms & sent_terms) / max(len(query_terms), 1)
                scored.append((sent, overlap))
        
        # Sort by relevance
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Take top sentences
        top = [s[0] for s in scored[:max_sentences] if s[1] > 0]
        
        if top:
            return '. '.join(top) + '.'
        return '. '.join(sentences[:2]) + '.'
    
    def _clean_response(self, text: str) -> str:
        """Clean up response text"""
        # Remove citations and noise
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\[citation needed\]', '', text, flags=re.I)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    def _extract_search_query(self, query: str) -> str:
        """Extract the best search query from the user's question"""
        # Remove question words and get core topic
        cleaned = re.sub(r'\b(what|who|where|when|why|how|is|are|was|were|do|does|did|can|could|would|should|tell|me|about|the|a|an)\b', '', query.lower())
        cleaned = re.sub(r'[^\w\s]', '', cleaned)
        cleaned = ' '.join(cleaned.split())  # Normalize spaces
        
        return cleaned.strip() or query
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text, removing stopwords"""
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.lower().split()
        
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'to', 'of',
            'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between', 'under',
            'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 'just', 'and', 'but', 'if', 'or', 'because', 'until', 'while',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'am', 'it', 'its', 'i', 'me', 'my', 'you', 'your', 'he', 'she', 'they'
        }
        
        return [t for t in tokens if t not in stopwords and len(t) > 1]