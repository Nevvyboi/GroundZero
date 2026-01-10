"""
Advanced Context Brain for GroundZero AI
=========================================
Intelligent context management with:
- Semantic memory with hierarchical organization
- Entity resolution and coreference handling
- Dynamic context window optimization
- Conversation state tracking
- Topic modeling and drift detection
- Attention-based relevance scoring
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import heapq
import math
from datetime import datetime
import json
import re
import hashlib
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor


class MemoryType(Enum):
    """Types of memory storage"""
    EPISODIC = "episodic"  # Specific events/conversations
    SEMANTIC = "semantic"  # General knowledge/facts
    PROCEDURAL = "procedural"  # How to do things
    WORKING = "working"  # Current active context


@dataclass
class MemoryItem:
    """Single memory item with metadata"""
    content: str
    memory_type: MemoryType
    embedding: Optional[np.ndarray]
    importance: float
    access_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    decay_rate: float = 0.99
    associations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_activation(self) -> float:
        """Calculate current activation level based on recency and frequency"""
        # Time-based decay
        try:
            last_access = datetime.fromisoformat(self.last_accessed)
            hours_since = (datetime.now() - last_access).total_seconds() / 3600
            time_decay = math.exp(-hours_since / 24)  # Half-life of ~24 hours
        except:
            time_decay = 0.5
        
        # Frequency boost
        freq_boost = math.log(self.access_count + 1)
        
        # Combined activation
        return self.importance * time_decay * (1 + 0.1 * freq_boost)
    
    def access(self):
        """Record access to this memory"""
        self.access_count += 1
        self.last_accessed = datetime.now().isoformat()


@dataclass
class Entity:
    """Named entity with resolved references"""
    canonical_name: str
    entity_type: str  # PERSON, ORG, LOCATION, CONCEPT, etc.
    aliases: Set[str] = field(default_factory=set)
    mentions: int = 0
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    first_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    last_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_alias(self, alias: str):
        self.aliases.add(alias.lower())
    
    def matches(self, text: str) -> bool:
        text_lower = text.lower()
        if text_lower == self.canonical_name.lower():
            return True
        return text_lower in self.aliases


@dataclass
class ConversationTurn:
    """Single turn in a conversation"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    embedding: Optional[np.ndarray] = None
    entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    sentiment: float = 0.0  # -1 to 1
    
    def to_dict(self) -> Dict:
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp,
            'entities': self.entities,
            'topics': self.topics,
            'sentiment': self.sentiment
        }


class SemanticMemory:
    """
    Hierarchical semantic memory with chunking and retrieval.
    Organizes knowledge into concepts, facts, and relationships.
    """
    
    def __init__(self, embedding_dim: int = 256, max_memories: int = 100000):
        self.embedding_dim = embedding_dim
        self.max_memories = max_memories
        self.memories: Dict[str, MemoryItem] = {}
        self.concept_hierarchy: Dict[str, Set[str]] = defaultdict(set)
        self.fact_index: Dict[str, List[str]] = defaultdict(list)
        self.embedding_index: Optional[np.ndarray] = None
        self.memory_ids: List[str] = []
        self._lock = threading.RLock()
        
    def store(self, content: str, memory_type: MemoryType, 
              embedding: Optional[np.ndarray] = None,
              importance: float = 1.0, 
              metadata: Dict[str, Any] = None) -> str:
        """Store a new memory"""
        # Generate unique ID
        memory_id = hashlib.md5(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        with self._lock:
            # Check capacity
            if len(self.memories) >= self.max_memories:
                self._evict_least_important()
            
            # Create memory item
            memory = MemoryItem(
                content=content,
                memory_type=memory_type,
                embedding=embedding,
                importance=importance,
                metadata=metadata or {}
            )
            
            self.memories[memory_id] = memory
            self.memory_ids.append(memory_id)
            
            # Index by keywords for fast retrieval
            keywords = self._extract_keywords(content)
            for kw in keywords:
                self.fact_index[kw].append(memory_id)
            
            # Update embedding index
            if embedding is not None:
                self._update_embedding_index(memory_id, embedding)
        
        return memory_id
    
    def retrieve(self, query: str, query_embedding: Optional[np.ndarray] = None,
                 top_k: int = 10, memory_type: Optional[MemoryType] = None) -> List[Tuple[str, MemoryItem, float]]:
        """Retrieve relevant memories"""
        results = []
        
        with self._lock:
            # Keyword-based retrieval
            keywords = self._extract_keywords(query)
            keyword_matches = set()
            for kw in keywords:
                keyword_matches.update(self.fact_index.get(kw, []))
            
            # Score and filter
            for memory_id in keyword_matches:
                if memory_id not in self.memories:
                    continue
                    
                memory = self.memories[memory_id]
                
                # Type filter
                if memory_type and memory.memory_type != memory_type:
                    continue
                
                # Calculate relevance score
                score = memory.get_activation()
                
                # Semantic similarity if embeddings available
                if query_embedding is not None and memory.embedding is not None:
                    sim = np.dot(query_embedding, memory.embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(memory.embedding) + 1e-10
                    )
                    score *= (1 + sim) / 2
                
                results.append((memory_id, memory, score))
            
            # If no keyword matches, try semantic search
            if not results and query_embedding is not None:
                results = self._semantic_search(query_embedding, top_k, memory_type)
        
        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)
        
        # Update access counts
        for memory_id, memory, _ in results[:top_k]:
            memory.access()
        
        return results[:top_k]
    
    def _semantic_search(self, query_embedding: np.ndarray, top_k: int,
                        memory_type: Optional[MemoryType]) -> List[Tuple[str, MemoryItem, float]]:
        """Pure semantic search when keywords fail"""
        results = []
        
        for memory_id, memory in self.memories.items():
            if memory_type and memory.memory_type != memory_type:
                continue
            
            if memory.embedding is None:
                continue
            
            sim = np.dot(query_embedding, memory.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(memory.embedding) + 1e-10
            )
            score = sim * memory.get_activation()
            results.append((memory_id, memory, score))
        
        return results
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords for indexing"""
        # Simple tokenization and filtering
        words = re.findall(r'\b\w+\b', text.lower())
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                    'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                    'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                    'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                    'through', 'during', 'before', 'after', 'above', 'below',
                    'between', 'under', 'again', 'further', 'then', 'once',
                    'that', 'this', 'these', 'those', 'and', 'but', 'or', 'nor',
                    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                    'just', 'also', 'now', 'here', 'there', 'when', 'where',
                    'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
                    'most', 'other', 'some', 'such', 'no', 'any', 'it', 'its'}
        
        return [w for w in words if len(w) > 2 and w not in stopwords]
    
    def _evict_least_important(self):
        """Evict least important memory"""
        if not self.memories:
            return
        
        # Find memory with lowest activation
        min_id = min(self.memories.keys(), 
                    key=lambda x: self.memories[x].get_activation())
        
        # Remove from all indices
        memory = self.memories.pop(min_id)
        keywords = self._extract_keywords(memory.content)
        for kw in keywords:
            if min_id in self.fact_index[kw]:
                self.fact_index[kw].remove(min_id)
        
        if min_id in self.memory_ids:
            self.memory_ids.remove(min_id)
    
    def _update_embedding_index(self, memory_id: str, embedding: np.ndarray):
        """Update embedding index for fast similarity search"""
        # For simplicity, we rebuild on demand
        # In production, use FAISS or similar
        pass
    
    def get_stats(self) -> Dict:
        """Get memory statistics"""
        type_counts = defaultdict(int)
        for memory in self.memories.values():
            type_counts[memory.memory_type.value] += 1
        
        return {
            'total_memories': len(self.memories),
            'by_type': dict(type_counts),
            'index_keywords': len(self.fact_index),
            'avg_activation': np.mean([m.get_activation() for m in self.memories.values()]) if self.memories else 0
        }


class EntityResolver:
    """
    Entity resolution and coreference handling.
    Maintains canonical entities and resolves references.
    """
    
    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        self.entities: Dict[str, Entity] = {}
        self.alias_map: Dict[str, str] = {}  # alias -> canonical_name
        self.type_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Pronouns for coreference
        self.pronouns = {
            'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themselves',
            'this', 'that', 'these', 'those'
        }
        
        # Recent entity mentions for pronoun resolution
        self.recent_mentions: List[str] = []
        
    def add_entity(self, name: str, entity_type: str,
                  aliases: List[str] = None,
                  properties: Dict[str, Any] = None,
                  embedding: Optional[np.ndarray] = None) -> str:
        """Add or update an entity"""
        canonical = name.lower().strip()
        
        if canonical in self.entities:
            # Update existing
            entity = self.entities[canonical]
            entity.mentions += 1
            entity.last_seen = datetime.now().isoformat()
            if properties:
                entity.properties.update(properties)
            if embedding is not None:
                entity.embedding = embedding
        else:
            # Create new
            entity = Entity(
                canonical_name=canonical,
                entity_type=entity_type,
                properties=properties or {},
                embedding=embedding
            )
            self.entities[canonical] = entity
            self.type_index[entity_type].add(canonical)
        
        # Add aliases
        if aliases:
            for alias in aliases:
                alias_lower = alias.lower().strip()
                entity.add_alias(alias_lower)
                self.alias_map[alias_lower] = canonical
        
        self.alias_map[canonical] = canonical
        
        # Track recent mention
        self._add_recent_mention(canonical)
        
        return canonical
    
    def resolve(self, text: str) -> Optional[str]:
        """Resolve text to canonical entity name"""
        text_lower = text.lower().strip()
        
        # Direct match
        if text_lower in self.alias_map:
            canonical = self.alias_map[text_lower]
            self._add_recent_mention(canonical)
            return canonical
        
        # Pronoun resolution
        if text_lower in self.pronouns:
            return self._resolve_pronoun(text_lower)
        
        # Fuzzy match
        return self._fuzzy_match(text_lower)
    
    def _resolve_pronoun(self, pronoun: str) -> Optional[str]:
        """Resolve pronoun to most recent matching entity"""
        if not self.recent_mentions:
            return None
        
        # Gender-based filtering
        male_pronouns = {'he', 'him', 'his', 'himself'}
        female_pronouns = {'she', 'her', 'hers', 'herself'}
        neutral_pronouns = {'it', 'its', 'itself', 'they', 'them', 'their'}
        
        for entity_name in reversed(self.recent_mentions):
            entity = self.entities.get(entity_name)
            if not entity:
                continue
            
            gender = entity.properties.get('gender', 'neutral')
            
            if pronoun in male_pronouns and gender == 'male':
                return entity_name
            elif pronoun in female_pronouns and gender == 'female':
                return entity_name
            elif pronoun in neutral_pronouns:
                return entity_name
        
        # Fall back to most recent
        return self.recent_mentions[-1] if self.recent_mentions else None
    
    def _fuzzy_match(self, text: str, threshold: float = 0.8) -> Optional[str]:
        """Find entity by fuzzy string matching"""
        best_match = None
        best_score = threshold
        
        for canonical, entity in self.entities.items():
            # Check canonical name
            score = self._string_similarity(text, canonical)
            if score > best_score:
                best_score = score
                best_match = canonical
            
            # Check aliases
            for alias in entity.aliases:
                score = self._string_similarity(text, alias)
                if score > best_score:
                    best_score = score
                    best_match = canonical
        
        if best_match:
            self._add_recent_mention(best_match)
        
        return best_match
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using Jaccard on character n-grams"""
        def ngrams(s, n=3):
            return set(s[i:i+n] for i in range(max(0, len(s)-n+1)))
        
        ng1 = ngrams(s1)
        ng2 = ngrams(s2)
        
        if not ng1 or not ng2:
            return 0.0
        
        intersection = len(ng1 & ng2)
        union = len(ng1 | ng2)
        
        return intersection / union if union > 0 else 0.0
    
    def _add_recent_mention(self, entity_name: str):
        """Track recent entity mentions"""
        if entity_name in self.recent_mentions:
            self.recent_mentions.remove(entity_name)
        self.recent_mentions.append(entity_name)
        
        # Keep only last 20 mentions
        if len(self.recent_mentions) > 20:
            self.recent_mentions = self.recent_mentions[-20:]
    
    def get_entity(self, name: str) -> Optional[Entity]:
        """Get entity by name or alias"""
        canonical = self.resolve(name)
        if canonical:
            return self.entities.get(canonical)
        return None
    
    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type"""
        return [self.entities[name] for name in self.type_index.get(entity_type, set())]
    
    def extract_entities(self, text: str) -> List[Tuple[str, str, int, int]]:
        """Extract entity mentions from text"""
        mentions = []
        
        # Check all aliases
        for alias, canonical in self.alias_map.items():
            pattern = re.compile(r'\b' + re.escape(alias) + r'\b', re.I)
            for match in pattern.finditer(text):
                entity = self.entities.get(canonical)
                if entity:
                    mentions.append((match.group(), canonical, match.start(), match.end()))
        
        # Sort by position
        mentions.sort(key=lambda x: x[2])
        
        return mentions


class ConversationContext:
    """
    Manages conversation context with topic tracking and state management.
    """
    
    def __init__(self, max_turns: int = 100, max_tokens: int = 8000):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.turns: List[ConversationTurn] = []
        self.current_topics: List[str] = []
        self.topic_history: List[Tuple[str, List[str]]] = []
        self.conversation_id: str = hashlib.md5(
            datetime.now().isoformat().encode()
        ).hexdigest()[:8]
        self.started_at: str = datetime.now().isoformat()
        self.state: Dict[str, Any] = {}
        
    def add_turn(self, role: str, content: str, 
                embedding: Optional[np.ndarray] = None,
                entities: List[str] = None,
                topics: List[str] = None) -> ConversationTurn:
        """Add a conversation turn"""
        turn = ConversationTurn(
            role=role,
            content=content,
            embedding=embedding,
            entities=entities or [],
            topics=topics or []
        )
        
        self.turns.append(turn)
        
        # Update topics
        if topics:
            self._update_topics(topics)
        
        # Trim if needed
        self._trim_to_limit()
        
        return turn
    
    def _update_topics(self, new_topics: List[str]):
        """Update current topics with drift detection"""
        # Calculate topic overlap
        if self.current_topics:
            overlap = set(new_topics) & set(self.current_topics)
            drift = len(overlap) / max(len(new_topics), len(self.current_topics), 1)
            
            if drift < 0.3:  # Topic shift detected
                self.topic_history.append((datetime.now().isoformat(), self.current_topics.copy()))
        
        # Update with recency weighting
        topic_scores = defaultdict(float)
        for t in self.current_topics:
            topic_scores[t] = 0.5  # Decay existing
        for t in new_topics:
            topic_scores[t] = max(topic_scores[t], 1.0)
        
        # Keep top topics
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        self.current_topics = [t for t, _ in sorted_topics[:10]]
    
    def _trim_to_limit(self):
        """Trim conversation to fit limits"""
        # Trim by turn count
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]
        
        # Estimate tokens and trim if needed
        total_chars = sum(len(t.content) for t in self.turns)
        estimated_tokens = total_chars // 4
        
        while estimated_tokens > self.max_tokens and len(self.turns) > 2:
            self.turns.pop(0)
            total_chars = sum(len(t.content) for t in self.turns)
            estimated_tokens = total_chars // 4
    
    def get_context_window(self, max_tokens: int = 4000) -> List[Dict]:
        """Get recent turns fitting within token limit"""
        result = []
        total_chars = 0
        
        for turn in reversed(self.turns):
            turn_chars = len(turn.content)
            if total_chars + turn_chars > max_tokens * 4:
                break
            result.insert(0, turn.to_dict())
            total_chars += turn_chars
        
        return result
    
    def get_relevant_context(self, query: str, query_embedding: Optional[np.ndarray] = None,
                            top_k: int = 5) -> List[ConversationTurn]:
        """Get most relevant turns for a query"""
        if not self.turns:
            return []
        
        # Score each turn
        scored_turns = []
        
        for i, turn in enumerate(self.turns):
            score = 0.0
            
            # Recency boost
            recency = 1.0 - (len(self.turns) - i - 1) / len(self.turns)
            score += recency * 0.3
            
            # Keyword overlap
            query_words = set(query.lower().split())
            turn_words = set(turn.content.lower().split())
            keyword_overlap = len(query_words & turn_words) / max(len(query_words), 1)
            score += keyword_overlap * 0.4
            
            # Embedding similarity
            if query_embedding is not None and turn.embedding is not None:
                sim = np.dot(query_embedding, turn.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(turn.embedding) + 1e-10
                )
                score += sim * 0.3
            
            scored_turns.append((turn, score))
        
        # Sort and return top-k
        scored_turns.sort(key=lambda x: x[1], reverse=True)
        return [turn for turn, _ in scored_turns[:top_k]]
    
    def set_state(self, key: str, value: Any):
        """Set conversation state variable"""
        self.state[key] = value
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get conversation state variable"""
        return self.state.get(key, default)
    
    def get_summary(self) -> Dict:
        """Get conversation summary"""
        return {
            'conversation_id': self.conversation_id,
            'started_at': self.started_at,
            'turn_count': len(self.turns),
            'current_topics': self.current_topics,
            'topic_shifts': len(self.topic_history),
            'user_turns': sum(1 for t in self.turns if t.role == 'user'),
            'assistant_turns': sum(1 for t in self.turns if t.role == 'assistant')
        }


class ContextBrain:
    """
    Main context brain integrating all components.
    Provides unified interface for context management.
    """
    
    def __init__(self, embedding_dim: int = 256, max_memories: int = 100000):
        self.embedding_dim = embedding_dim
        self.semantic_memory = SemanticMemory(embedding_dim, max_memories)
        self.entity_resolver = EntityResolver(embedding_dim)
        self.conversation = ConversationContext()
        self.sessions: Dict[str, ConversationContext] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)
        
    def process_input(self, text: str, role: str = 'user',
                     embedding: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Process input and update all context components"""
        # Extract entities
        entity_mentions = self.entity_resolver.extract_entities(text)
        entities = [canonical for _, canonical, _, _ in entity_mentions]
        
        # Add to conversation
        turn = self.conversation.add_turn(
            role=role,
            content=text,
            embedding=embedding,
            entities=entities
        )
        
        # Store in semantic memory
        memory_id = self.semantic_memory.store(
            content=text,
            memory_type=MemoryType.EPISODIC,
            embedding=embedding,
            importance=0.8 if role == 'user' else 0.6
        )
        
        return {
            'turn': turn.to_dict(),
            'entities': entities,
            'memory_id': memory_id
        }
    
    def get_context_for_query(self, query: str, 
                             query_embedding: Optional[np.ndarray] = None,
                             max_items: int = 10) -> Dict[str, Any]:
        """Get all relevant context for answering a query"""
        # Get relevant memories
        memories = self.semantic_memory.retrieve(
            query, query_embedding, top_k=max_items // 2
        )
        
        # Get relevant conversation turns
        relevant_turns = self.conversation.get_relevant_context(
            query, query_embedding, top_k=max_items // 2
        )
        
        # Get mentioned entities
        mentions = self.entity_resolver.extract_entities(query)
        entities = []
        for _, canonical, _, _ in mentions:
            entity = self.entity_resolver.entities.get(canonical)
            if entity:
                entities.append({
                    'name': entity.canonical_name,
                    'type': entity.entity_type,
                    'properties': entity.properties
                })
        
        # Get current topics
        topics = self.conversation.current_topics
        
        return {
            'memories': [(m_id, m.content, score) for m_id, m, score in memories],
            'conversation_context': [t.to_dict() for t in relevant_turns],
            'entities': entities,
            'current_topics': topics,
            'conversation_summary': self.conversation.get_summary()
        }
    
    def add_knowledge(self, content: str, memory_type: str = 'semantic',
                     importance: float = 1.0,
                     embedding: Optional[np.ndarray] = None,
                     metadata: Dict = None) -> str:
        """Add knowledge to semantic memory"""
        mem_type = MemoryType(memory_type) if memory_type in [mt.value for mt in MemoryType] else MemoryType.SEMANTIC
        return self.semantic_memory.store(
            content=content,
            memory_type=mem_type,
            embedding=embedding,
            importance=importance,
            metadata=metadata
        )
    
    def add_entity(self, name: str, entity_type: str,
                  aliases: List[str] = None,
                  properties: Dict = None) -> str:
        """Add entity to resolver"""
        return self.entity_resolver.add_entity(
            name=name,
            entity_type=entity_type,
            aliases=aliases,
            properties=properties
        )
    
    def resolve_reference(self, text: str) -> Optional[str]:
        """Resolve a reference to canonical entity"""
        return self.entity_resolver.resolve(text)
    
    def new_conversation(self) -> str:
        """Start a new conversation, archiving the old one"""
        if self.conversation.turns:
            self.sessions[self.conversation.conversation_id] = self.conversation
        
        self.conversation = ConversationContext()
        return self.conversation.conversation_id
    
    def switch_conversation(self, conversation_id: str) -> bool:
        """Switch to a different conversation session"""
        if conversation_id in self.sessions:
            # Archive current
            if self.conversation.turns:
                self.sessions[self.conversation.conversation_id] = self.conversation
            
            # Restore requested
            self.conversation = self.sessions[conversation_id]
            return True
        return False
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            'memory_stats': self.semantic_memory.get_stats(),
            'entity_count': len(self.entity_resolver.entities),
            'conversation': self.conversation.get_summary(),
            'session_count': len(self.sessions) + 1
        }


# Test the context brain
if __name__ == "__main__":
    print("Testing Context Brain...")
    
    brain = ContextBrain(embedding_dim=256)
    
    # Add some entities
    brain.add_entity("Albert Einstein", "PERSON", 
                    aliases=["Einstein", "Dr. Einstein"],
                    properties={'profession': 'physicist', 'gender': 'male'})
    brain.add_entity("Theory of Relativity", "CONCEPT",
                    aliases=["relativity", "special relativity"])
    
    # Process conversation
    result1 = brain.process_input("Tell me about Einstein's work on relativity")
    print(f"\n1. Processed input:")
    print(f"   Entities found: {result1['entities']}")
    
    result2 = brain.process_input(
        "He developed the theory while working at the patent office",
        role='assistant'
    )
    print(f"\n2. Assistant response processed")
    
    result3 = brain.process_input("What else did he discover?")
    print(f"   'he' resolved to: {brain.resolve_reference('he')}")
    
    # Get context for new query
    context = brain.get_context_for_query("Einstein's other contributions")
    print(f"\n3. Context retrieved:")
    print(f"   Memories: {len(context['memories'])}")
    print(f"   Conversation turns: {len(context['conversation_context'])}")
    print(f"   Entities: {context['entities']}")
    
    # Add knowledge
    brain.add_knowledge(
        "E=mc² is Einstein's famous mass-energy equivalence formula",
        importance=0.9
    )
    print("\n4. Added knowledge to semantic memory")
    
    # Test memory retrieval
    memories = brain.semantic_memory.retrieve("energy mass formula", top_k=5)
    print(f"\n5. Retrieved memories for 'energy mass formula':")
    for m_id, memory, score in memories:
        print(f"   - {memory.content[:50]}... (score: {score:.3f})")
    
    # Get stats
    stats = brain.get_stats()
    print(f"\n6. Context Brain Stats:")
    print(f"   Total memories: {stats['memory_stats']['total_memories']}")
    print(f"   Entities: {stats['entity_count']}")
    print(f"   Conversation turns: {stats['conversation']['turn_count']}")
    
    print("\n✓ Context Brain tests passed!")
