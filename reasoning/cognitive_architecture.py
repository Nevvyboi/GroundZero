"""
Cognitive Architecture
======================
High-level cognitive system that integrates reasoning, memory, and learning.
Implements System 1 (fast) and System 2 (slow) thinking modes.

Inspired by dual-process theory and modern AI reasoning systems.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import time

from .advanced_reasoning import (
    AdvancedReasoningEngine, 
    ReasoningResult,
    ChainOfThought,
    ThoughtType
)


class ThinkingMode(Enum):
    """Dual-process thinking modes"""
    SYSTEM_1 = "fast"      # Quick, intuitive responses
    SYSTEM_2 = "slow"      # Deep, deliberate reasoning


@dataclass
class CognitiveState:
    """Current cognitive state"""
    thinking_mode: ThinkingMode
    attention_focus: str
    working_memory: List[str]
    confidence_level: float
    uncertainty_areas: List[str]
    requires_search: bool


class CognitiveArchitecture:
    """
    Main cognitive architecture that coordinates:
    - Perception (understanding input)
    - Attention (focusing on relevant info)
    - Working Memory (holding context)
    - Reasoning (generating conclusions)
    - Metacognition (monitoring own thinking)
    - Response Generation (forming output)
    """
    
    def __init__(self, memory_store=None, neural_model=None):
        self.memory = memory_store
        self.model = neural_model
        
        # Initialize reasoning engine
        self.reasoning = AdvancedReasoningEngine()
        
        # Working memory capacity (chunks)
        self.working_memory_capacity = 7  # Miller's law
        self.working_memory: List[str] = []
        
        # Confidence thresholds
        self.confident_threshold = 0.6
        self.search_threshold = 0.4
        
        # Response patterns for different situations
        self.response_patterns = {
            'confident': "Based on my analysis: {answer}",
            'uncertain': "I believe {answer}, though I'm not entirely certain.",
            'speculative': "I think {answer}, but this is speculative.",
            'need_info': "I don't have enough information. {reason}",
        }
    
    def process(self, query: str, context: List[str] = None) -> Dict[str, Any]:
        """
        Main cognitive processing pipeline.
        
        1. Perceive: Understand the input
        2. Attend: Focus on relevant information
        3. Retrieve: Get relevant knowledge
        4. Reason: Apply appropriate reasoning
        5. Monitor: Check reasoning quality
        6. Respond: Generate appropriate response
        """
        start_time = time.time()
        
        # Step 1: Perception - Understand the input
        perception = self._perceive(query)
        
        # Step 2: Attention - Determine what to focus on
        attention = self._attend(query, perception)
        
        # Step 3: Retrieval - Get relevant knowledge
        knowledge = self._retrieve_knowledge(attention['focus_concepts'], context)
        
        # Step 4: Decide thinking mode (System 1 or 2)
        thinking_mode = self._select_thinking_mode(perception, knowledge)
        
        # Step 5: Reasoning
        if thinking_mode == ThinkingMode.SYSTEM_1:
            result = self._fast_thinking(query, knowledge)
        else:
            result = self._slow_thinking(query, knowledge)
        
        # Step 6: Metacognitive monitoring
        state = self._monitor_state(result, knowledge)
        
        # Step 7: Generate response
        response = self._generate_response(result, state)
        
        processing_time = time.time() - start_time
        
        return {
            'response': response,
            'confidence': result.confidence,
            'thinking_mode': thinking_mode.value,
            'thought_process': result.thought_process,
            'needs_search': state.requires_search,
            'uncertainty_areas': state.uncertainty_areas,
            'verification': result.verification_status,
            'processing_time': processing_time,
            'reasoning_type': result.reasoning_type
        }
    
    def _perceive(self, query: str) -> Dict[str, Any]:
        """Understand the input query"""
        query_lower = query.lower().strip()
        
        # Determine query intent
        if any(w in query_lower for w in ['what is', 'what are', 'define', 'meaning of']):
            intent = 'definition'
        elif any(w in query_lower for w in ['why', 'reason', 'cause']):
            intent = 'explanation'
        elif any(w in query_lower for w in ['how to', 'how do', 'steps']):
            intent = 'procedural'
        elif any(w in query_lower for w in ['compare', 'difference', 'versus', 'vs']):
            intent = 'comparison'
        elif any(w in query_lower for w in ['opinion', 'think', 'believe', 'should']):
            intent = 'opinion'
        elif any(w in query_lower for w in ['list', 'examples', 'types']):
            intent = 'enumeration'
        elif '?' not in query and len(query.split()) < 5:
            intent = 'acknowledgment'
        else:
            intent = 'factual'
        
        # Determine complexity
        complexity = self._assess_complexity(query)
        
        # Extract entities
        entities = self._extract_entities(query)
        
        return {
            'intent': intent,
            'complexity': complexity,
            'entities': entities,
            'word_count': len(query.split()),
            'is_question': '?' in query
        }
    
    def _attend(self, query: str, perception: Dict) -> Dict[str, Any]:
        """Determine attention focus"""
        # Extract key concepts to focus on
        concepts = self._extract_concepts(query)
        
        # Prioritize by importance
        prioritized = self._prioritize_concepts(concepts, perception['intent'])
        
        # Update working memory
        self._update_working_memory(prioritized)
        
        return {
            'focus_concepts': prioritized[:3],  # Top 3 concepts
            'secondary_concepts': prioritized[3:],
            'attention_strength': 1.0 if perception['complexity'] > 0.5 else 0.7
        }
    
    def _retrieve_knowledge(self, concepts: List[str], context: List[str] = None) -> Dict[str, Any]:
        """Retrieve relevant knowledge from memory"""
        retrieved = []
        confidence_scores = []
        
        # First, check provided context
        if context:
            for concept in concepts:
                for ctx in context:
                    if concept.lower() in ctx.lower():
                        retrieved.append(ctx)
                        confidence_scores.append(0.8)
        
        # Then check memory store if available
        if self.memory:
            for concept in concepts:
                results = self.memory.search_knowledge(concept, limit=3)
                for r in results:
                    content = r.get('content', '')
                    if content and content not in retrieved:
                        retrieved.append(content)
                        confidence_scores.append(r.get('confidence', 0.5))
        
        # Calculate overall knowledge confidence
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            avg_confidence = 0.0
        
        return {
            'facts': retrieved[:5],  # Top 5 facts
            'confidence': avg_confidence,
            'coverage': len(retrieved) / max(len(concepts), 1),
            'has_knowledge': len(retrieved) > 0
        }
    
    def _select_thinking_mode(self, perception: Dict, knowledge: Dict) -> ThinkingMode:
        """Select appropriate thinking mode"""
        # Use System 2 (slow) for:
        # - Complex queries
        # - Explanatory questions
        # - Low knowledge confidence
        # - Comparison/analysis
        
        if perception['complexity'] > 0.6:
            return ThinkingMode.SYSTEM_2
        
        if perception['intent'] in ['explanation', 'comparison', 'opinion']:
            return ThinkingMode.SYSTEM_2
        
        if knowledge['confidence'] < 0.5:
            return ThinkingMode.SYSTEM_2
        
        # Default to System 1 for simple, well-known queries
        return ThinkingMode.SYSTEM_1
    
    def _fast_thinking(self, query: str, knowledge: Dict) -> ReasoningResult:
        """System 1: Fast, intuitive thinking"""
        # Quick pattern matching response
        path = self.reasoning.cot.reason(query, knowledge['facts'])
        
        return ReasoningResult(
            answer=path.final_answer,
            confidence=path.confidence * knowledge['confidence'],
            reasoning_type='intuitive',
            thought_process=[{
                'step': t.content,
                'type': t.thought_type.value,
                'confidence': t.confidence
            } for t in path.thoughts[:3]],  # Abbreviated
            alternative_answers=[],
            uncertainty_areas=[],
            verification_status='Quick',
            metacognitive_notes=[]
        )
    
    def _slow_thinking(self, query: str, knowledge: Dict) -> ReasoningResult:
        """System 2: Deliberate, analytical thinking"""
        # Full reasoning pipeline
        return self.reasoning.think(query, knowledge['facts'], deep_think=True)
    
    def _monitor_state(self, result: ReasoningResult, knowledge: Dict) -> CognitiveState:
        """Metacognitive monitoring of cognitive state"""
        # Determine if we need to search
        requires_search = (
            result.confidence < self.search_threshold or
            not knowledge['has_knowledge'] or
            'need more information' in result.answer.lower()
        )
        
        # Collect uncertainty areas
        uncertainties = list(result.uncertainty_areas)
        if not knowledge['has_knowledge']:
            uncertainties.append("No knowledge available on this topic")
        if result.confidence < 0.5:
            uncertainties.append("Low confidence in reasoning")
        
        return CognitiveState(
            thinking_mode=ThinkingMode.SYSTEM_2 if result.reasoning_type == 'deep_analysis' else ThinkingMode.SYSTEM_1,
            attention_focus=self.working_memory[0] if self.working_memory else '',
            working_memory=self.working_memory.copy(),
            confidence_level=result.confidence,
            uncertainty_areas=uncertainties,
            requires_search=requires_search
        )
    
    def _generate_response(self, result: ReasoningResult, state: CognitiveState) -> str:
        """Generate final response based on reasoning and state"""
        if state.requires_search:
            return f"I don't have enough knowledge about this yet. Let me search and learn..."
        
        # Select response pattern based on confidence
        if state.confidence_level >= 0.7:
            pattern = self.response_patterns['confident']
        elif state.confidence_level >= 0.5:
            pattern = self.response_patterns['uncertain']
        else:
            pattern = self.response_patterns['speculative']
        
        # Format response
        response = pattern.format(answer=result.answer)
        
        # Add metacognitive notes if relevant
        if result.metacognitive_notes and state.confidence_level < 0.7:
            response += f"\n\n(Note: {result.metacognitive_notes[0]})"
        
        return response
    
    def _assess_complexity(self, query: str) -> float:
        """Assess query complexity"""
        score = 0.3
        
        words = query.split()
        if len(words) > 15:
            score += 0.2
        elif len(words) > 8:
            score += 0.1
        
        # Complex question words
        if any(w in query.lower() for w in ['why', 'how', 'analyze', 'compare', 'evaluate']):
            score += 0.2
        
        # Multiple clauses
        if ',' in query or ' and ' in query.lower():
            score += 0.1
        
        return min(1.0, score)
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query"""
        # Simple entity extraction (capitalized words)
        words = query.split()
        entities = []
        
        for word in words:
            clean = re.sub(r'[^\w]', '', word)
            if clean and clean[0].isupper() and len(clean) > 1:
                entities.append(clean)
        
        return entities
    
    def _extract_concepts(self, query: str) -> List[str]:
        """Extract key concepts from query"""
        stopwords = {
            'what', 'is', 'are', 'the', 'a', 'an', 'how', 'why', 'when', 'where',
            'who', 'which', 'does', 'do', 'can', 'could', 'would', 'should',
            'will', 'about', 'tell', 'me', 'please', 'explain', 'describe',
            'this', 'that', 'these', 'those', 'it', 'they', 'them', 'i', 'you'
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        concepts = [w for w in words if w not in stopwords and len(w) > 2]
        
        return concepts
    
    def _prioritize_concepts(self, concepts: List[str], intent: str) -> List[str]:
        """Prioritize concepts based on intent"""
        if not concepts:
            return []
        
        # For definitions, first noun is most important
        if intent == 'definition':
            return concepts
        
        # For comparisons, both items are important
        if intent == 'comparison':
            return concepts
        
        # Default: longer words tend to be more specific/important
        return sorted(concepts, key=len, reverse=True)
    
    def _update_working_memory(self, concepts: List[str]) -> None:
        """Update working memory with new concepts"""
        for concept in concepts:
            if concept not in self.working_memory:
                self.working_memory.insert(0, concept)
        
        # Enforce capacity limit
        self.working_memory = self.working_memory[:self.working_memory_capacity]
    
    def get_thinking_summary(self) -> str:
        """Get summary of current cognitive state"""
        return f"""
Current Working Memory: {', '.join(self.working_memory[:5])}
Memory Capacity: {len(self.working_memory)}/{self.working_memory_capacity}
"""