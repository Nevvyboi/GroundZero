"""
Conversation Context Manager
============================
Tracks conversation state to handle follow-ups and references.

Features:
- Tracks entities mentioned in conversation
- Resolves references like "2nd option", "the first one", "it", "that"
- Maintains conversation history for context
- Handles disambiguation when multiple matches exist
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict


@dataclass
class Entity:
    """An entity mentioned in conversation"""
    name: str
    content: str
    source_url: str
    source_title: str
    confidence: float
    mentioned_at: datetime = field(default_factory=datetime.now)
    index: int = 0  # Position when multiple options given


@dataclass
class ConversationTurn:
    """A single turn in conversation"""
    role: str  # 'user' or 'assistant'
    message: str
    entities: List[Entity] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class ConversationContext:
    """
    Manages conversation state for contextual understanding.
    
    Tracks:
    - Recent entities mentioned (for "the 2nd option", "it", etc.)
    - Conversation history (for context)
    - Current topic focus
    """
    
    # Patterns for reference resolution
    ORDINAL_PATTERNS = {
        r'\b(first|1st)\b': 0,
        r'\b(second|2nd)\b': 1,
        r'\b(third|3rd)\b': 2,
        r'\b(fourth|4th)\b': 3,
        r'\b(fifth|5th)\b': 4,
        r'\b(last)\b': -1,
    }
    
    PRONOUN_PATTERNS = [
        r'\b(it|that|this)\b',
        r'\b(them|those|these)\b',
        r'\b(he|she|they)\b',
        r'\b(him|her)\b',
    ]
    
    REFERENCE_PATTERNS = [
        r'\bthe (first|second|third|1st|2nd|3rd|last) (one|option|result|person|thing)\b',
        r'\boption (\d+|one|two|three)\b',
        r'\b(\d+)(st|nd|rd|th) option\b',
        r'\bthe (former|latter)\b',
    ]
    
    def __init__(self, max_history: int = 20):
        self.history: List[ConversationTurn] = []
        self.max_history = max_history
        
        # Entity tracking
        self.recent_entities: List[Entity] = []  # Most recent entities (options given)
        self.all_entities: Dict[str, Entity] = {}  # name → entity
        
        # Current focus
        self.current_topic: Optional[str] = None
        self.disambiguation_pending: bool = False
    
    def add_user_message(self, message: str) -> None:
        """Add a user message to history"""
        turn = ConversationTurn(role='user', message=message)
        self.history.append(turn)
        self._trim_history()
    
    def add_assistant_response(self, message: str, entities: List[Dict] = None) -> None:
        """Add assistant response with any entities mentioned"""
        entity_objects = []
        
        if entities:
            # Clear previous options if we're giving new ones
            self.recent_entities = []
            
            for i, e in enumerate(entities):
                entity = Entity(
                    name=e.get('name', ''),
                    content=e.get('content', ''),
                    source_url=e.get('source_url', ''),
                    source_title=e.get('source_title', ''),
                    confidence=e.get('confidence', 0.5),
                    index=i
                )
                entity_objects.append(entity)
                self.recent_entities.append(entity)
                
                # Also track by name
                if entity.name:
                    self.all_entities[entity.name.lower()] = entity
        
        turn = ConversationTurn(
            role='assistant',
            message=message,
            entities=entity_objects
        )
        self.history.append(turn)
        self._trim_history()
    
    def resolve_reference(self, message: str) -> Tuple[Optional[Entity], str]:
        """
        Resolve references in user message to actual entities.
        
        Returns: (resolved_entity, clarified_message)
        
        Examples:
        - "tell me about the 2nd option" → resolves to 2nd entity
        - "what about it?" → resolves to most recent single entity
        - "I mean Sam Battle the youtuber" → resolves by matching description
        """
        message_lower = message.lower()
        
        # Check for ordinal references ("2nd option", "the first one")
        for pattern, index in self.ORDINAL_PATTERNS.items():
            if re.search(pattern, message_lower):
                # Check if this is specifically about an option
                if re.search(r'option|one|result|choice', message_lower) or \
                   re.search(r'referring|mean|talking about', message_lower):
                    if self.recent_entities:
                        actual_index = index if index >= 0 else len(self.recent_entities) + index
                        if 0 <= actual_index < len(self.recent_entities):
                            entity = self.recent_entities[actual_index]
                            return entity, f"Tell me about {entity.name}"
        
        # Check for numeric option reference ("option 2", "2nd")
        match = re.search(r'option\s*(\d+)', message_lower)
        if match:
            index = int(match.group(1)) - 1  # Convert to 0-indexed
            if 0 <= index < len(self.recent_entities):
                entity = self.recent_entities[index]
                return entity, f"Tell me about {entity.name}"
        
        # Check for "the X one" where X describes the entity
        match = re.search(r'the\s+(\w+)\s+one', message_lower)
        if match:
            descriptor = match.group(1)
            for entity in self.recent_entities:
                if descriptor in entity.content.lower() or descriptor in entity.name.lower():
                    return entity, f"Tell me about {entity.name}"
        
        # Check for pronoun references with context clues
        if re.search(r'\b(it|that|this)\b', message_lower):
            # If only one recent entity, resolve to it
            if len(self.recent_entities) == 1:
                entity = self.recent_entities[0]
                # Replace pronoun with entity name
                clarified = re.sub(r'\b(it|that|this)\b', entity.name, message, flags=re.IGNORECASE)
                return entity, clarified
        
        # Check for partial name matches
        for entity in self.recent_entities:
            name_parts = entity.name.lower().split()
            for part in name_parts:
                if len(part) > 3 and part in message_lower:
                    # Found a name match
                    return entity, message
        
        # Check for descriptive disambiguation ("the youtuber", "the police officer")
        descriptors = {
            'youtuber': ['youtuber', 'youtube', 'musician', 'electronics'],
            'police': ['police', 'officer', 'commissioner', 'american'],
        }
        
        for category, keywords in descriptors.items():
            if any(kw in message_lower for kw in keywords):
                for entity in self.recent_entities:
                    if any(kw in entity.content.lower() for kw in keywords):
                        return entity, f"Tell me about {entity.name}"
        
        return None, message
    
    def get_context_summary(self) -> str:
        """Get a summary of recent context for the AI"""
        if not self.history:
            return ""
        
        summary_parts = []
        
        # Recent conversation
        recent = self.history[-6:]  # Last 3 exchanges
        for turn in recent:
            role = "User" if turn.role == 'user' else "Assistant"
            summary_parts.append(f"{role}: {turn.message[:200]}")
        
        # Recent entities
        if self.recent_entities:
            entities_str = ", ".join(e.name for e in self.recent_entities[:5])
            summary_parts.append(f"Recent entities discussed: {entities_str}")
        
        return "\n".join(summary_parts)
    
    def extract_entities_from_response(self, response: str, sources: List[Dict]) -> List[Dict]:
        """
        Extract entity information from a response that lists multiple options.
        
        Detects patterns like:
        - "X may refer to:"
        - "1. Name - description"
        - "• Name - description"
        """
        entities = []
        
        # Check if this is a disambiguation response
        if 'may refer to' in response.lower() or 'could mean' in response.lower():
            # Extract bullet points or numbered items
            lines = response.split('\n')
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Match patterns like "Name - description" or "• Name - description"
                match = re.match(r'^[\d\.\)\-\•\*]?\s*([^-–]+)\s*[-–]\s*(.+)', line)
                if match:
                    name = match.group(1).strip()
                    description = match.group(2).strip()
                    
                    # Clean up name
                    name = re.sub(r'^(aka|also known as)\s+', '', name, flags=re.IGNORECASE)
                    
                    if len(name) > 2 and len(description) > 10:
                        # Try to find matching source
                        source_url = ''
                        source_title = ''
                        for src in sources:
                            if name.lower() in src.get('title', '').lower():
                                source_url = src.get('url', '')
                                source_title = src.get('title', '')
                                break
                        
                        entities.append({
                            'name': name,
                            'content': description,
                            'source_url': source_url,
                            'source_title': source_title,
                            'confidence': 0.7
                        })
        
        return entities
    
    def needs_disambiguation(self, results: List[Dict]) -> bool:
        """Check if results need disambiguation (multiple different entities)"""
        if len(results) <= 1:
            return False
        
        # Check if results are about different entities
        titles = [r.get('source_title', '').lower() for r in results]
        unique_titles = set(titles)
        
        return len(unique_titles) > 1
    
    def format_disambiguation(self, query: str, results: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Format a disambiguation response when multiple entities match.
        
        Returns: (response_text, entities_list)
        """
        # Group results by likely entity
        entities = []
        seen_titles = set()
        
        for r in results:
            title = r.get('source_title', '')
            if title and title not in seen_titles:
                seen_titles.add(title)
                
                # Extract a brief description from content
                content = r.get('content', '')
                first_sentence = content.split('.')[0] if content else ''
                
                entities.append({
                    'name': title,
                    'content': first_sentence[:200],
                    'source_url': r.get('source_url', ''),
                    'source_title': title,
                    'confidence': r.get('relevance', 0.5)
                })
        
        if len(entities) <= 1:
            return None, []
        
        # Format response
        response_lines = [f"**{query}** may refer to:\n"]
        
        for i, entity in enumerate(entities[:5], 1):
            response_lines.append(f"{entity['name']} - {entity['content']}")
        
        response_lines.append("\nWhich one would you like to know about?")
        
        return '\n'.join(response_lines), entities
    
    def _trim_history(self) -> None:
        """Keep history within limits"""
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def clear(self) -> None:
        """Clear conversation state"""
        self.history = []
        self.recent_entities = []
        self.all_entities = {}
        self.current_topic = None
        self.disambiguation_pending = False


# Global conversation context (per session in production, use session ID)
_contexts: Dict[str, ConversationContext] = {}


def get_context(session_id: str = "default") -> ConversationContext:
    """Get or create conversation context for a session"""
    if session_id not in _contexts:
        _contexts[session_id] = ConversationContext()
    return _contexts[session_id]


def clear_context(session_id: str = "default") -> None:
    """Clear conversation context for a session"""
    if session_id in _contexts:
        _contexts[session_id].clear()