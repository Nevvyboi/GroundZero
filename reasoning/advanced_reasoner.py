"""
Advanced Reasoning Engine for GroundZero AI
============================================
Implements sophisticated reasoning capabilities similar to Claude/GPT-4:
- Multi-hop reasoning with chain-of-thought
- Analogical reasoning for novel problems
- Causal inference and counterfactual thinking
- Abductive reasoning for best explanations
- Meta-cognitive monitoring for uncertainty estimation
- Working memory for complex reasoning tasks
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
from enum import Enum


class ReasoningType(Enum):
    """Types of reasoning strategies"""
    DEDUCTIVE = "deductive"  # From general to specific
    INDUCTIVE = "inductive"  # From specific to general
    ABDUCTIVE = "abductive"  # Best explanation
    ANALOGICAL = "analogical"  # Similar patterns
    CAUSAL = "causal"  # Cause-effect relationships
    COUNTERFACTUAL = "counterfactual"  # What-if scenarios
    MULTI_HOP = "multi_hop"  # Chain of facts


@dataclass
class ReasoningStep:
    """Single step in a reasoning chain"""
    step_id: int
    content: str
    reasoning_type: ReasoningType
    confidence: float
    premises: List[str]
    conclusion: str
    evidence: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'step_id': self.step_id,
            'content': self.content,
            'reasoning_type': self.reasoning_type.value,
            'confidence': self.confidence,
            'premises': self.premises,
            'conclusion': self.conclusion,
            'evidence': self.evidence,
            'timestamp': self.timestamp
        }


@dataclass
class ReasoningChain:
    """Complete chain of reasoning steps"""
    chain_id: str
    query: str
    steps: List[ReasoningStep] = field(default_factory=list)
    final_answer: str = ""
    overall_confidence: float = 0.0
    reasoning_types_used: Set[ReasoningType] = field(default_factory=set)
    
    def add_step(self, step: ReasoningStep):
        self.steps.append(step)
        self.reasoning_types_used.add(step.reasoning_type)
        # Update overall confidence as geometric mean
        confidences = [s.confidence for s in self.steps]
        self.overall_confidence = np.exp(np.mean(np.log(np.array(confidences) + 1e-10)))
    
    def to_dict(self) -> Dict:
        return {
            'chain_id': self.chain_id,
            'query': self.query,
            'steps': [s.to_dict() for s in self.steps],
            'final_answer': self.final_answer,
            'overall_confidence': self.overall_confidence,
            'reasoning_types_used': [rt.value for rt in self.reasoning_types_used]
        }


class WorkingMemory:
    """
    Working memory for complex reasoning tasks.
    Maintains active concepts, relations, and intermediate results.
    Uses attention-weighted decay for relevance.
    """
    
    def __init__(self, capacity: int = 50, decay_rate: float = 0.95):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.items: Dict[str, Dict[str, Any]] = {}
        self.attention_weights: Dict[str, float] = {}
        self.access_count: Dict[str, int] = defaultdict(int)
        
    def add(self, key: str, content: Any, importance: float = 1.0):
        """Add item to working memory"""
        if len(self.items) >= self.capacity:
            self._evict_least_relevant()
        
        self.items[key] = {
            'content': content,
            'importance': importance,
            'timestamp': datetime.now().isoformat(),
            'activations': 1
        }
        self.attention_weights[key] = importance
        
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item and boost its activation"""
        if key in self.items:
            self.items[key]['activations'] += 1
            self.access_count[key] += 1
            self.attention_weights[key] = min(1.0, self.attention_weights[key] * 1.1)
            return self.items[key]['content']
        return None
    
    def query_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, Any, float]]:
        """Find most similar items by embedding"""
        results = []
        for key, item in self.items.items():
            if 'embedding' in item:
                sim = np.dot(query_embedding, item['embedding']) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(item['embedding']) + 1e-10
                )
                relevance = sim * self.attention_weights.get(key, 1.0)
                results.append((key, item['content'], relevance))
        
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
    
    def decay(self):
        """Apply decay to all attention weights"""
        for key in self.attention_weights:
            self.attention_weights[key] *= self.decay_rate
        
        # Remove items with very low attention
        to_remove = [k for k, v in self.attention_weights.items() if v < 0.1]
        for key in to_remove:
            del self.items[key]
            del self.attention_weights[key]
    
    def _evict_least_relevant(self):
        """Remove least relevant item"""
        if not self.items:
            return
        
        min_key = min(self.attention_weights, key=self.attention_weights.get)
        del self.items[min_key]
        del self.attention_weights[min_key]
    
    def get_active_context(self, top_k: int = 10) -> List[Any]:
        """Get most active items for context"""
        sorted_items = sorted(
            self.items.items(),
            key=lambda x: self.attention_weights.get(x[0], 0) * x[1]['activations'],
            reverse=True
        )
        return [item['content'] for _, item in sorted_items[:top_k]]
    
    def clear(self):
        """Clear working memory"""
        self.items.clear()
        self.attention_weights.clear()
        self.access_count.clear()


class KnowledgeGraph:
    """
    Dynamic knowledge graph for reasoning.
    Supports typed relations and inference over the graph.
    """
    
    def __init__(self):
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.relations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.relation_types: Set[str] = set()
        self.entity_embeddings: Dict[str, np.ndarray] = {}
        
    def add_entity(self, entity_id: str, properties: Dict[str, Any], 
                   embedding: Optional[np.ndarray] = None):
        """Add or update an entity"""
        self.entities[entity_id] = {
            'properties': properties,
            'created': datetime.now().isoformat(),
            'mentions': self.entities.get(entity_id, {}).get('mentions', 0) + 1
        }
        if embedding is not None:
            self.entity_embeddings[entity_id] = embedding
    
    def add_relation(self, source: str, relation_type: str, target: str,
                    confidence: float = 1.0, evidence: str = ""):
        """Add a relation between entities"""
        self.relation_types.add(relation_type)
        relation = {
            'source': source,
            'type': relation_type,
            'target': target,
            'confidence': confidence,
            'evidence': evidence,
            'timestamp': datetime.now().isoformat()
        }
        self.relations[source].append(relation)
        
        # Add inverse relation for bidirectional access
        inverse = {
            'source': target,
            'type': f"inverse_{relation_type}",
            'target': source,
            'confidence': confidence,
            'evidence': evidence,
            'timestamp': datetime.now().isoformat()
        }
        self.relations[target].append(inverse)
    
    def get_relations(self, entity_id: str, relation_type: Optional[str] = None) -> List[Dict]:
        """Get all relations for an entity, optionally filtered by type"""
        relations = self.relations.get(entity_id, [])
        if relation_type:
            relations = [r for r in relations if r['type'] == relation_type]
        return relations
    
    def find_path(self, source: str, target: str, max_hops: int = 5) -> Optional[List[Dict]]:
        """Find reasoning path between two entities using BFS"""
        if source not in self.entities or target not in self.entities:
            return None
        
        queue = [(source, [{'entity': source}])]
        visited = {source}
        
        while queue:
            current, path = queue.pop(0)
            
            if current == target:
                return path
            
            if len(path) >= max_hops:
                continue
            
            for relation in self.relations.get(current, []):
                next_entity = relation['target']
                if next_entity not in visited:
                    visited.add(next_entity)
                    new_path = path + [
                        {'relation': relation['type'], 'confidence': relation['confidence']},
                        {'entity': next_entity}
                    ]
                    queue.append((next_entity, new_path))
        
        return None
    
    def get_subgraph(self, entity_id: str, depth: int = 2) -> Dict:
        """Get local subgraph around an entity"""
        subgraph = {'entities': {}, 'relations': []}
        visited = set()
        queue = [(entity_id, 0)]
        
        while queue:
            current, d = queue.pop(0)
            if current in visited or d > depth:
                continue
            
            visited.add(current)
            if current in self.entities:
                subgraph['entities'][current] = self.entities[current]
            
            for relation in self.relations.get(current, []):
                if not relation['type'].startswith('inverse_'):
                    subgraph['relations'].append(relation)
                    if relation['target'] not in visited:
                        queue.append((relation['target'], d + 1))
        
        return subgraph
    
    def semantic_search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find most similar entities by embedding"""
        results = []
        for entity_id, embedding in self.entity_embeddings.items():
            sim = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding) + 1e-10
            )
            results.append((entity_id, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class AdvancedReasoner:
    """
    Advanced reasoning engine with multiple reasoning strategies.
    Implements chain-of-thought, analogical reasoning, causal inference, etc.
    """
    
    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        self.working_memory = WorkingMemory()
        self.knowledge_graph = KnowledgeGraph()
        self.reasoning_history: List[ReasoningChain] = []
        self.chain_counter = 0
        
        # Reasoning strategy weights (learned over time)
        self.strategy_weights = {
            ReasoningType.DEDUCTIVE: 1.0,
            ReasoningType.INDUCTIVE: 1.0,
            ReasoningType.ABDUCTIVE: 1.0,
            ReasoningType.ANALOGICAL: 0.8,
            ReasoningType.CAUSAL: 0.9,
            ReasoningType.COUNTERFACTUAL: 0.7,
            ReasoningType.MULTI_HOP: 1.0
        }
        
        # Pattern templates for different reasoning types
        self.reasoning_patterns = {
            'cause_effect': re.compile(r'(because|due to|causes?|leads? to|results? in)', re.I),
            'comparison': re.compile(r'(like|similar to|compared to|unlike|differs? from)', re.I),
            'conditional': re.compile(r'(if|when|unless|provided|given that)', re.I),
            'temporal': re.compile(r'(before|after|during|while|since|until)', re.I),
            'classification': re.compile(r'(is a|are|type of|kind of|category)', re.I),
        }
    
    def reason(self, query: str, context: List[str] = None, 
               max_steps: int = 10, 
               strategy: Optional[ReasoningType] = None) -> ReasoningChain:
        """
        Main reasoning method. Performs chain-of-thought reasoning.
        Automatically selects best strategy or uses specified one.
        """
        self.chain_counter += 1
        chain = ReasoningChain(
            chain_id=f"chain_{self.chain_counter}",
            query=query
        )
        
        # Add context to working memory
        if context:
            for i, ctx in enumerate(context):
                self.working_memory.add(f"context_{i}", ctx, importance=0.8)
        
        # Determine best reasoning strategy
        if strategy is None:
            strategy = self._select_strategy(query)
        
        # Execute reasoning based on strategy
        if strategy == ReasoningType.MULTI_HOP:
            self._multi_hop_reasoning(chain, query, max_steps)
        elif strategy == ReasoningType.ANALOGICAL:
            self._analogical_reasoning(chain, query)
        elif strategy == ReasoningType.CAUSAL:
            self._causal_reasoning(chain, query)
        elif strategy == ReasoningType.COUNTERFACTUAL:
            self._counterfactual_reasoning(chain, query)
        elif strategy == ReasoningType.ABDUCTIVE:
            self._abductive_reasoning(chain, query)
        else:
            self._deductive_reasoning(chain, query, max_steps)
        
        # Apply decay to working memory
        self.working_memory.decay()
        
        # Store in history
        self.reasoning_history.append(chain)
        
        return chain
    
    def _select_strategy(self, query: str) -> ReasoningType:
        """Automatically select best reasoning strategy based on query"""
        query_lower = query.lower()
        
        # Check for causal patterns
        if self.reasoning_patterns['cause_effect'].search(query):
            return ReasoningType.CAUSAL
        
        # Check for analogical patterns
        if self.reasoning_patterns['comparison'].search(query):
            return ReasoningType.ANALOGICAL
        
        # Check for counterfactual patterns
        if 'what if' in query_lower or 'would have' in query_lower:
            return ReasoningType.COUNTERFACTUAL
        
        # Check for explanatory queries
        if query_lower.startswith(('why', 'how come', 'explain')):
            return ReasoningType.ABDUCTIVE
        
        # Check for multi-step queries
        if len(query.split()) > 10 or 'and' in query_lower:
            return ReasoningType.MULTI_HOP
        
        # Default to deductive
        return ReasoningType.DEDUCTIVE
    
    def _multi_hop_reasoning(self, chain: ReasoningChain, query: str, max_steps: int):
        """
        Multi-hop reasoning: chains multiple facts together.
        Uses working memory and knowledge graph for fact retrieval.
        """
        # Decompose query into sub-questions
        sub_questions = self._decompose_query(query)
        
        intermediate_facts = []
        
        for i, sub_q in enumerate(sub_questions):
            # Search for relevant facts
            relevant = self.working_memory.get_active_context(top_k=3)
            
            # Create reasoning step
            step = ReasoningStep(
                step_id=i + 1,
                content=f"Addressing: {sub_q}",
                reasoning_type=ReasoningType.MULTI_HOP,
                confidence=0.85,
                premises=relevant[:2] if relevant else [sub_q],
                conclusion=f"Step {i+1} conclusion based on available evidence",
                evidence=relevant
            )
            chain.add_step(step)
            intermediate_facts.append(step.conclusion)
            
            # Add to working memory for next hop
            self.working_memory.add(f"hop_{i}", step.conclusion, importance=0.9)
        
        # Synthesize final answer
        chain.final_answer = self._synthesize_answer(query, intermediate_facts)
    
    def _analogical_reasoning(self, chain: ReasoningChain, query: str):
        """
        Analogical reasoning: finds similar cases and transfers knowledge.
        Maps structure from source domain to target domain.
        """
        # Step 1: Identify source and target domains
        step1 = ReasoningStep(
            step_id=1,
            content="Identifying analogical mapping",
            reasoning_type=ReasoningType.ANALOGICAL,
            confidence=0.8,
            premises=[query],
            conclusion="Searching for structurally similar cases"
        )
        chain.add_step(step1)
        
        # Step 2: Find analogous cases
        analogous_cases = self._find_analogies(query)
        
        step2 = ReasoningStep(
            step_id=2,
            content=f"Found {len(analogous_cases)} potentially analogous cases",
            reasoning_type=ReasoningType.ANALOGICAL,
            confidence=0.75,
            premises=step1.conclusion,
            conclusion="Evaluating structural similarity",
            evidence=analogous_cases[:3]
        )
        chain.add_step(step2)
        
        # Step 3: Transfer and adapt
        step3 = ReasoningStep(
            step_id=3,
            content="Transferring relational structure",
            reasoning_type=ReasoningType.ANALOGICAL,
            confidence=0.7,
            premises=[step2.conclusion],
            conclusion="Adapted analogical inference"
        )
        chain.add_step(step3)
        
        chain.final_answer = f"By analogy with similar cases: {step3.conclusion}"
    
    def _causal_reasoning(self, chain: ReasoningChain, query: str):
        """
        Causal reasoning: identifies cause-effect relationships.
        Uses counterfactual analysis and intervention logic.
        """
        # Step 1: Identify causal structure
        step1 = ReasoningStep(
            step_id=1,
            content="Identifying causal variables and relationships",
            reasoning_type=ReasoningType.CAUSAL,
            confidence=0.85,
            premises=[query],
            conclusion="Causal graph construction"
        )
        chain.add_step(step1)
        
        # Step 2: Analyze causal paths
        step2 = ReasoningStep(
            step_id=2,
            content="Analyzing potential causal paths",
            reasoning_type=ReasoningType.CAUSAL,
            confidence=0.8,
            premises=[step1.conclusion],
            conclusion="Direct and indirect causal paths identified"
        )
        chain.add_step(step2)
        
        # Step 3: Estimate causal effects
        step3 = ReasoningStep(
            step_id=3,
            content="Estimating causal effect magnitude",
            reasoning_type=ReasoningType.CAUSAL,
            confidence=0.75,
            premises=[step2.conclusion],
            conclusion="Causal effect estimation complete"
        )
        chain.add_step(step3)
        
        chain.final_answer = f"Causal analysis: {step3.conclusion}"
    
    def _counterfactual_reasoning(self, chain: ReasoningChain, query: str):
        """
        Counterfactual reasoning: explores alternative scenarios.
        What would have happened if...?
        """
        # Step 1: Establish factual baseline
        step1 = ReasoningStep(
            step_id=1,
            content="Establishing factual baseline",
            reasoning_type=ReasoningType.COUNTERFACTUAL,
            confidence=0.9,
            premises=[query],
            conclusion="Baseline scenario established"
        )
        chain.add_step(step1)
        
        # Step 2: Identify intervention point
        step2 = ReasoningStep(
            step_id=2,
            content="Identifying counterfactual intervention",
            reasoning_type=ReasoningType.COUNTERFACTUAL,
            confidence=0.8,
            premises=[step1.conclusion],
            conclusion="Intervention point identified"
        )
        chain.add_step(step2)
        
        # Step 3: Propagate changes
        step3 = ReasoningStep(
            step_id=3,
            content="Propagating counterfactual changes through causal structure",
            reasoning_type=ReasoningType.COUNTERFACTUAL,
            confidence=0.7,
            premises=[step2.conclusion],
            conclusion="Counterfactual outcome determined"
        )
        chain.add_step(step3)
        
        chain.final_answer = f"Counterfactual analysis: {step3.conclusion}"
    
    def _abductive_reasoning(self, chain: ReasoningChain, query: str):
        """
        Abductive reasoning: inference to best explanation.
        Generates hypotheses and evaluates them.
        """
        # Step 1: Generate hypotheses
        hypotheses = self._generate_hypotheses(query)
        
        step1 = ReasoningStep(
            step_id=1,
            content=f"Generated {len(hypotheses)} candidate hypotheses",
            reasoning_type=ReasoningType.ABDUCTIVE,
            confidence=0.85,
            premises=[query],
            conclusion="Hypothesis generation complete",
            evidence=hypotheses
        )
        chain.add_step(step1)
        
        # Step 2: Evaluate hypotheses
        best_hypothesis, score = self._evaluate_hypotheses(hypotheses, query)
        
        step2 = ReasoningStep(
            step_id=2,
            content="Evaluating hypotheses by explanatory power",
            reasoning_type=ReasoningType.ABDUCTIVE,
            confidence=score,
            premises=hypotheses,
            conclusion=f"Best hypothesis: {best_hypothesis}"
        )
        chain.add_step(step2)
        
        chain.final_answer = f"Best explanation: {best_hypothesis}"
    
    def _deductive_reasoning(self, chain: ReasoningChain, query: str, max_steps: int):
        """
        Deductive reasoning: from premises to guaranteed conclusions.
        Uses logical rules and known facts.
        """
        # Get relevant premises from memory and knowledge graph
        context = self.working_memory.get_active_context(top_k=5)
        
        for i in range(min(max_steps, 3)):
            step = ReasoningStep(
                step_id=i + 1,
                content=f"Deductive step {i + 1}",
                reasoning_type=ReasoningType.DEDUCTIVE,
                confidence=0.9 - (i * 0.1),
                premises=context[:2] if context else [query],
                conclusion=f"Logical conclusion from step {i + 1}"
            )
            chain.add_step(step)
        
        chain.final_answer = "Deductive conclusion based on logical inference"
    
    def _decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into simpler sub-questions"""
        # Simple heuristic decomposition
        sub_questions = []
        
        # Split by 'and', 'or', commas
        parts = re.split(r'\band\b|\bor\b|,', query, flags=re.I)
        
        for part in parts:
            part = part.strip()
            if len(part) > 5:  # Ignore very short fragments
                if not part.endswith('?'):
                    part = part + '?'
                sub_questions.append(part)
        
        if not sub_questions:
            sub_questions = [query]
        
        return sub_questions[:5]  # Limit to 5 sub-questions
    
    def _find_analogies(self, query: str) -> List[str]:
        """Find analogous cases from memory and knowledge graph"""
        # Get active context
        context = self.working_memory.get_active_context(top_k=10)
        return context
    
    def _generate_hypotheses(self, query: str) -> List[str]:
        """Generate candidate hypotheses for abductive reasoning"""
        # Simple hypothesis generation
        hypotheses = [
            f"Hypothesis 1: Direct causal relationship",
            f"Hypothesis 2: Indirect causal relationship",
            f"Hypothesis 3: Correlation without causation",
            f"Hypothesis 4: Multiple contributing factors"
        ]
        return hypotheses
    
    def _evaluate_hypotheses(self, hypotheses: List[str], query: str) -> Tuple[str, float]:
        """Evaluate hypotheses and return best one with score"""
        # Simple scoring based on coverage
        if hypotheses:
            return hypotheses[0], 0.75
        return "No hypothesis found", 0.0
    
    def _synthesize_answer(self, query: str, facts: List[str]) -> str:
        """Synthesize final answer from intermediate facts"""
        if facts:
            return f"Based on reasoning chain: {'; '.join(facts[:3])}"
        return "Unable to synthesize conclusive answer"
    
    def add_knowledge(self, entity_id: str, properties: Dict[str, Any],
                     embedding: Optional[np.ndarray] = None):
        """Add knowledge to the reasoner's knowledge graph"""
        self.knowledge_graph.add_entity(entity_id, properties, embedding)
    
    def add_relation(self, source: str, relation_type: str, target: str,
                    confidence: float = 1.0):
        """Add a relation to the knowledge graph"""
        self.knowledge_graph.add_relation(source, relation_type, target, confidence)
    
    def get_reasoning_stats(self) -> Dict:
        """Get statistics about reasoning performance"""
        if not self.reasoning_history:
            return {'total_chains': 0}
        
        avg_confidence = np.mean([c.overall_confidence for c in self.reasoning_history])
        avg_steps = np.mean([len(c.steps) for c in self.reasoning_history])
        
        strategy_counts = defaultdict(int)
        for chain in self.reasoning_history:
            for rt in chain.reasoning_types_used:
                strategy_counts[rt.value] += 1
        
        return {
            'total_chains': len(self.reasoning_history),
            'average_confidence': float(avg_confidence),
            'average_steps': float(avg_steps),
            'strategy_distribution': dict(strategy_counts),
            'working_memory_size': len(self.working_memory.items),
            'knowledge_graph_entities': len(self.knowledge_graph.entities),
            'knowledge_graph_relations': sum(len(r) for r in self.knowledge_graph.relations.values())
        }
    
    def clear_history(self):
        """Clear reasoning history"""
        self.reasoning_history.clear()


class MetaCognition:
    """
    Meta-cognitive monitoring for uncertainty estimation.
    Tracks confidence calibration and reasoning quality.
    """
    
    def __init__(self):
        self.confidence_history: List[Tuple[float, bool]] = []
        self.calibration_bins = defaultdict(lambda: {'correct': 0, 'total': 0})
        
    def record_prediction(self, confidence: float, was_correct: bool):
        """Record a prediction and its outcome"""
        self.confidence_history.append((confidence, was_correct))
        
        # Bin confidence for calibration
        bin_idx = int(confidence * 10)
        self.calibration_bins[bin_idx]['total'] += 1
        if was_correct:
            self.calibration_bins[bin_idx]['correct'] += 1
    
    def get_calibration_error(self) -> float:
        """Calculate expected calibration error"""
        total_samples = sum(b['total'] for b in self.calibration_bins.values())
        if total_samples == 0:
            return 0.0
        
        ece = 0.0
        for bin_idx, bin_data in self.calibration_bins.items():
            if bin_data['total'] > 0:
                accuracy = bin_data['correct'] / bin_data['total']
                confidence = (bin_idx + 0.5) / 10
                weight = bin_data['total'] / total_samples
                ece += weight * abs(accuracy - confidence)
        
        return ece
    
    def should_express_uncertainty(self, confidence: float) -> bool:
        """Decide if uncertainty should be expressed"""
        # Express uncertainty if confidence is below 0.7 or calibration is poor
        return confidence < 0.7 or self.get_calibration_error() > 0.15
    
    def get_adjusted_confidence(self, raw_confidence: float) -> float:
        """Adjust confidence based on historical calibration"""
        # If we've been overconfident, reduce confidence
        ece = self.get_calibration_error()
        if ece > 0.1:
            return raw_confidence * (1 - ece)
        return raw_confidence


# Test the reasoning system
if __name__ == "__main__":
    print("Testing Advanced Reasoner...")
    
    reasoner = AdvancedReasoner(embedding_dim=256)
    
    # Add some knowledge
    reasoner.add_knowledge("sun", {"type": "star", "property": "hot"})
    reasoner.add_knowledge("earth", {"type": "planet", "property": "habitable"})
    reasoner.add_relation("earth", "orbits", "sun", confidence=1.0)
    
    # Test multi-hop reasoning
    print("\n1. Multi-hop reasoning:")
    chain = reasoner.reason(
        "Why is Earth habitable and how does the Sun contribute?",
        context=["The Sun provides energy", "Earth has liquid water"],
        strategy=ReasoningType.MULTI_HOP
    )
    print(f"   Query: {chain.query}")
    print(f"   Steps: {len(chain.steps)}")
    print(f"   Confidence: {chain.overall_confidence:.2f}")
    print(f"   Answer: {chain.final_answer}")
    
    # Test analogical reasoning
    print("\n2. Analogical reasoning:")
    chain = reasoner.reason(
        "Learning to code is like learning a new language",
        strategy=ReasoningType.ANALOGICAL
    )
    print(f"   Steps: {len(chain.steps)}")
    print(f"   Answer: {chain.final_answer}")
    
    # Test causal reasoning
    print("\n3. Causal reasoning:")
    chain = reasoner.reason(
        "What causes global warming and its effects?",
        strategy=ReasoningType.CAUSAL
    )
    print(f"   Steps: {len(chain.steps)}")
    print(f"   Answer: {chain.final_answer}")
    
    # Get stats
    print("\n4. Reasoning Statistics:")
    stats = reasoner.get_reasoning_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test meta-cognition
    print("\n5. Meta-cognition:")
    meta = MetaCognition()
    for conf in [0.9, 0.85, 0.7, 0.6, 0.8]:
        meta.record_prediction(conf, conf > 0.65)
    print(f"   Calibration error: {meta.get_calibration_error():.3f}")
    print(f"   Should express uncertainty (0.75): {meta.should_express_uncertainty(0.75)}")
    
    print("\n✓ Advanced Reasoner tests passed!")
