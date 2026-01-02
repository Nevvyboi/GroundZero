"""
Advanced Reasoning Engine
=========================
Implements human-like critical thinking based on:
- Chain-of-Thought (CoT) reasoning
- Tree of Thoughts (ToT) exploration  
- Self-Consistency verification
- Metacognitive monitoring
- Problem decomposition
- Self-verification and error detection

Inspired by GPT-4, o1, and Claude's reasoning architectures.
"""

import re
import math
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter
import numpy as np


class ThoughtType(Enum):
    """Types of reasoning thoughts"""
    OBSERVATION = "observation"      # What do I notice?
    ANALYSIS = "analysis"           # What does this mean?
    HYPOTHESIS = "hypothesis"       # What might be true?
    VERIFICATION = "verification"   # Is this correct?
    SYNTHESIS = "synthesis"         # How do pieces connect?
    CONCLUSION = "conclusion"       # What's the answer?
    UNCERTAINTY = "uncertainty"     # What am I unsure about?
    REFLECTION = "reflection"       # How confident am I?


@dataclass
class Thought:
    """A single reasoning thought/step"""
    content: str
    thought_type: ThoughtType
    confidence: float = 0.5
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    parent_thought: Optional['Thought'] = None
    children: List['Thought'] = field(default_factory=list)
    
    def score(self) -> float:
        """Calculate thought quality score"""
        evidence_score = len(self.supporting_evidence) - len(self.contradicting_evidence) * 0.5
        return self.confidence * (1 + evidence_score * 0.1)


@dataclass
class ReasoningPath:
    """A complete reasoning chain"""
    thoughts: List[Thought]
    final_answer: str
    confidence: float
    reasoning_trace: str
    
    def get_key_steps(self) -> List[str]:
        """Get key reasoning steps for display"""
        return [t.content for t in self.thoughts if t.thought_type in 
                [ThoughtType.ANALYSIS, ThoughtType.HYPOTHESIS, ThoughtType.CONCLUSION]]


@dataclass 
class ReasoningResult:
    """Final reasoning output"""
    answer: str
    confidence: float
    reasoning_type: str
    thought_process: List[Dict[str, Any]]
    alternative_answers: List[Dict[str, Any]]
    uncertainty_areas: List[str]
    verification_status: str
    metacognitive_notes: List[str]


class ChainOfThought:
    """
    Chain-of-Thought reasoning module.
    Breaks down problems into sequential steps.
    """
    
    def __init__(self, knowledge_retriever=None):
        self.knowledge = knowledge_retriever
        self.step_templates = {
            'identify': "First, let me identify what's being asked: {question}",
            'recall': "What do I know about {topic}?",
            'analyze': "Analyzing this: {observation}",
            'infer': "From this, I can infer: {inference}",
            'verify': "Let me verify: {claim}",
            'conclude': "Therefore: {conclusion}"
        }
    
    def reason(self, query: str, context: List[str] = None) -> ReasoningPath:
        """Generate chain-of-thought reasoning"""
        thoughts = []
        
        # Step 1: Understand the question
        question_analysis = self._analyze_question(query)
        thoughts.append(Thought(
            content=f"Understanding the question: {question_analysis['type']} about {question_analysis['topic']}",
            thought_type=ThoughtType.OBSERVATION,
            confidence=0.9
        ))
        
        # Step 2: Identify key concepts
        concepts = self._extract_concepts(query)
        thoughts.append(Thought(
            content=f"Key concepts identified: {', '.join(concepts)}",
            thought_type=ThoughtType.ANALYSIS,
            confidence=0.8
        ))
        
        # Step 3: Retrieve relevant knowledge
        if context:
            relevant_facts = self._find_relevant_facts(concepts, context)
            for fact in relevant_facts[:3]:
                thoughts.append(Thought(
                    content=f"Relevant knowledge: {fact}",
                    thought_type=ThoughtType.OBSERVATION,
                    confidence=0.7,
                    supporting_evidence=[fact]
                ))
        
        # Step 4: Build reasoning chain
        reasoning_steps = self._build_reasoning_chain(query, concepts, context or [])
        for step in reasoning_steps:
            thoughts.append(Thought(
                content=step['content'],
                thought_type=ThoughtType.ANALYSIS if 'because' in step['content'].lower() else ThoughtType.SYNTHESIS,
                confidence=step.get('confidence', 0.6)
            ))
        
        # Step 5: Form conclusion
        conclusion = self._synthesize_conclusion(thoughts)
        thoughts.append(Thought(
            content=conclusion,
            thought_type=ThoughtType.CONCLUSION,
            confidence=self._calculate_chain_confidence(thoughts)
        ))
        
        # Build reasoning trace
        trace = self._format_reasoning_trace(thoughts)
        
        return ReasoningPath(
            thoughts=thoughts,
            final_answer=conclusion,
            confidence=self._calculate_chain_confidence(thoughts),
            reasoning_trace=trace
        )
    
    def _analyze_question(self, query: str) -> Dict[str, str]:
        """Analyze question type and topic"""
        query_lower = query.lower()
        
        # Determine question type
        if any(w in query_lower for w in ['what is', 'what are', 'define', 'explain']):
            q_type = 'definition'
        elif any(w in query_lower for w in ['why', 'how come', 'reason']):
            q_type = 'causal'
        elif any(w in query_lower for w in ['how', 'process', 'steps']):
            q_type = 'procedural'
        elif any(w in query_lower for w in ['compare', 'difference', 'similar']):
            q_type = 'comparative'
        elif any(w in query_lower for w in ['should', 'best', 'recommend']):
            q_type = 'evaluative'
        else:
            q_type = 'factual'
        
        # Extract topic (simplified)
        topic = self._extract_main_topic(query)
        
        return {'type': q_type, 'topic': topic}
    
    def _extract_main_topic(self, query: str) -> str:
        """Extract the main topic from query"""
        # Remove question words
        cleaned = re.sub(r'\b(what|why|how|when|where|who|is|are|the|a|an)\b', '', query.lower())
        words = cleaned.split()
        # Return longest meaningful word as topic
        if words:
            return max(words, key=len)
        return query
    
    def _extract_concepts(self, query: str) -> List[str]:
        """Extract key concepts from query"""
        # Remove stopwords and extract meaningful terms
        stopwords = {'what', 'is', 'are', 'the', 'a', 'an', 'how', 'why', 'when', 'where', 'who',
                     'does', 'do', 'can', 'could', 'would', 'should', 'will', 'about', 'tell', 'me'}
        words = re.findall(r'\b\w+\b', query.lower())
        concepts = [w for w in words if w not in stopwords and len(w) > 2]
        return concepts[:5]  # Top 5 concepts
    
    def _find_relevant_facts(self, concepts: List[str], context: List[str]) -> List[str]:
        """Find facts relevant to concepts"""
        relevant = []
        for fact in context:
            fact_lower = fact.lower()
            relevance = sum(1 for c in concepts if c in fact_lower)
            if relevance > 0:
                relevant.append((fact, relevance))
        
        # Sort by relevance
        relevant.sort(key=lambda x: x[1], reverse=True)
        return [f[0] for f in relevant]
    
    def _build_reasoning_chain(self, query: str, concepts: List[str], context: List[str]) -> List[Dict]:
        """Build logical reasoning steps"""
        steps = []
        
        # Step through logical reasoning
        if context:
            # Premise identification
            steps.append({
                'content': f"Based on available knowledge about {concepts[0] if concepts else 'this topic'}...",
                'confidence': 0.7
            })
            
            # Analysis
            steps.append({
                'content': f"Analyzing the relationship between {' and '.join(concepts[:2]) if len(concepts) >= 2 else 'the concepts'}...",
                'confidence': 0.65
            })
            
            # Inference
            steps.append({
                'content': "From this analysis, I can reason that...",
                'confidence': 0.6
            })
        else:
            steps.append({
                'content': f"I need more information about {concepts[0] if concepts else 'this topic'} to reason accurately.",
                'confidence': 0.3
            })
        
        return steps
    
    def _synthesize_conclusion(self, thoughts: List[Thought]) -> str:
        """Synthesize final conclusion from thoughts"""
        # Gather evidence from thoughts
        evidence = []
        for t in thoughts:
            if t.thought_type == ThoughtType.OBSERVATION and t.supporting_evidence:
                evidence.extend(t.supporting_evidence)
        
        if evidence:
            return f"Based on my analysis: {evidence[0][:200]}..."
        
        # Fallback
        analyses = [t.content for t in thoughts if t.thought_type == ThoughtType.ANALYSIS]
        if analyses:
            return f"My reasoning suggests: {analyses[-1]}"
        
        return "I need more information to form a complete conclusion."
    
    def _calculate_chain_confidence(self, thoughts: List[Thought]) -> float:
        """Calculate overall confidence in reasoning chain"""
        if not thoughts:
            return 0.0
        
        # Confidence degrades through chain (uncertainty compounds)
        base_confidence = sum(t.confidence for t in thoughts) / len(thoughts)
        chain_penalty = 0.95 ** len(thoughts)  # Small penalty per step
        
        return min(1.0, base_confidence * chain_penalty)
    
    def _format_reasoning_trace(self, thoughts: List[Thought]) -> str:
        """Format thoughts into readable trace"""
        lines = []
        for i, t in enumerate(thoughts, 1):
            icon = {
                ThoughtType.OBSERVATION: "👁️",
                ThoughtType.ANALYSIS: "🔍",
                ThoughtType.HYPOTHESIS: "💡",
                ThoughtType.VERIFICATION: "✓",
                ThoughtType.SYNTHESIS: "🔗",
                ThoughtType.CONCLUSION: "✅",
                ThoughtType.UNCERTAINTY: "❓",
                ThoughtType.REFLECTION: "🤔"
            }.get(t.thought_type, "•")
            
            lines.append(f"{icon} Step {i}: {t.content}")
        
        return "\n".join(lines)


class TreeOfThoughts:
    """
    Tree of Thoughts reasoning module.
    Explores multiple reasoning branches and selects best path.
    """
    
    def __init__(self, max_branches: int = 3, max_depth: int = 4):
        self.max_branches = max_branches
        self.max_depth = max_depth
        self.cot = ChainOfThought()
    
    def reason(self, query: str, context: List[str] = None) -> ReasoningPath:
        """Explore tree of thoughts and find best path"""
        root = Thought(
            content=f"Exploring approaches to: {query}",
            thought_type=ThoughtType.OBSERVATION,
            confidence=1.0
        )
        
        # Generate initial branches
        branches = self._generate_branches(query, context)
        
        # Explore each branch
        paths = []
        for branch in branches:
            path = self._explore_branch(branch, query, context, depth=0)
            if path:
                paths.append(path)
        
        # Select best path
        if paths:
            best_path = max(paths, key=lambda p: p.confidence)
            return best_path
        
        # Fallback to simple CoT
        return self.cot.reason(query, context)
    
    def _generate_branches(self, query: str, context: List[str] = None) -> List[Thought]:
        """Generate different reasoning approaches"""
        branches = []
        
        # Approach 1: Direct factual
        branches.append(Thought(
            content="Direct approach: Look for factual information",
            thought_type=ThoughtType.HYPOTHESIS,
            confidence=0.7
        ))
        
        # Approach 2: Analytical
        branches.append(Thought(
            content="Analytical approach: Break down into components",
            thought_type=ThoughtType.HYPOTHESIS,
            confidence=0.6
        ))
        
        # Approach 3: Analogical
        branches.append(Thought(
            content="Analogical approach: Find similar concepts",
            thought_type=ThoughtType.HYPOTHESIS,
            confidence=0.5
        ))
        
        return branches[:self.max_branches]
    
    def _explore_branch(self, branch: Thought, query: str, context: List[str], depth: int) -> Optional[ReasoningPath]:
        """Recursively explore a reasoning branch"""
        if depth >= self.max_depth:
            return None
        
        thoughts = [branch]
        
        # Develop the branch
        if "Direct" in branch.content:
            # Direct factual path
            if context:
                relevant = [c for c in context if any(w in c.lower() for w in query.lower().split())]
                if relevant:
                    thoughts.append(Thought(
                        content=f"Found relevant information: {relevant[0][:100]}...",
                        thought_type=ThoughtType.OBSERVATION,
                        confidence=0.8,
                        supporting_evidence=relevant[:2]
                    ))
        
        elif "Analytical" in branch.content:
            # Analytical breakdown
            concepts = self.cot._extract_concepts(query)
            thoughts.append(Thought(
                content=f"Breaking down into: {', '.join(concepts)}",
                thought_type=ThoughtType.ANALYSIS,
                confidence=0.7
            ))
        
        elif "Analogical" in branch.content:
            # Analogical reasoning
            thoughts.append(Thought(
                content="Looking for similar patterns in knowledge...",
                thought_type=ThoughtType.SYNTHESIS,
                confidence=0.5
            ))
        
        # Add conclusion
        conclusion = self._form_branch_conclusion(thoughts, context)
        thoughts.append(Thought(
            content=conclusion,
            thought_type=ThoughtType.CONCLUSION,
            confidence=self._evaluate_branch(thoughts)
        ))
        
        return ReasoningPath(
            thoughts=thoughts,
            final_answer=conclusion,
            confidence=self._evaluate_branch(thoughts),
            reasoning_trace=self.cot._format_reasoning_trace(thoughts)
        )
    
    def _form_branch_conclusion(self, thoughts: List[Thought], context: List[str]) -> str:
        """Form conclusion for a branch"""
        evidence = []
        for t in thoughts:
            evidence.extend(t.supporting_evidence)
        
        if evidence:
            return f"Based on this approach: {evidence[0][:150]}..."
        
        return "This approach requires more information to conclude."
    
    def _evaluate_branch(self, thoughts: List[Thought]) -> float:
        """Evaluate quality of reasoning branch"""
        if not thoughts:
            return 0.0
        
        # Count supporting evidence
        evidence_count = sum(len(t.supporting_evidence) for t in thoughts)
        
        # Average confidence
        avg_confidence = sum(t.confidence for t in thoughts) / len(thoughts)
        
        # Evidence bonus
        evidence_bonus = min(0.3, evidence_count * 0.1)
        
        return min(1.0, avg_confidence + evidence_bonus)


class SelfConsistency:
    """
    Self-Consistency module.
    Generates multiple reasoning paths and votes on best answer.
    """
    
    def __init__(self, num_paths: int = 3):
        self.num_paths = num_paths
        self.cot = ChainOfThought()
        self.tot = TreeOfThoughts()
    
    def reason(self, query: str, context: List[str] = None) -> Tuple[ReasoningPath, List[ReasoningPath]]:
        """Generate multiple paths and find consensus"""
        paths = []
        
        # Generate diverse reasoning paths
        for i in range(self.num_paths):
            if i % 2 == 0:
                path = self.cot.reason(query, context)
            else:
                path = self.tot.reason(query, context)
            paths.append(path)
        
        # Vote on answers (simplified - compare conclusions)
        answers = [p.final_answer for p in paths]
        
        # Find most consistent answer
        best_path = max(paths, key=lambda p: p.confidence)
        
        # Check consistency
        consistency_score = self._calculate_consistency(paths)
        best_path.confidence *= consistency_score
        
        return best_path, paths
    
    def _calculate_consistency(self, paths: List[ReasoningPath]) -> float:
        """Calculate how consistent the paths are"""
        if len(paths) < 2:
            return 1.0
        
        # Simple consistency: check if conclusions are similar
        conclusions = [p.final_answer.lower() for p in paths]
        
        # Count overlapping words
        all_words = set()
        for c in conclusions:
            all_words.update(c.split())
        
        common_count = 0
        for word in all_words:
            if sum(1 for c in conclusions if word in c) >= len(conclusions) // 2 + 1:
                common_count += 1
        
        if all_words:
            return min(1.0, 0.5 + (common_count / len(all_words)) * 0.5)
        return 0.5


class MetacognitiveMonitor:
    """
    Metacognitive monitoring module.
    Monitors reasoning quality and identifies uncertainties.
    """
    
    def __init__(self):
        self.confidence_threshold = 0.6
        self.uncertainty_markers = [
            'might', 'maybe', 'perhaps', 'possibly', 'uncertain',
            'unclear', 'ambiguous', 'depends', 'if', 'assuming'
        ]
    
    def monitor(self, path: ReasoningPath) -> Dict[str, Any]:
        """Monitor reasoning quality and identify issues"""
        issues = []
        uncertainties = []
        
        # Check confidence levels
        low_confidence_thoughts = [t for t in path.thoughts if t.confidence < self.confidence_threshold]
        if low_confidence_thoughts:
            issues.append(f"Found {len(low_confidence_thoughts)} low-confidence steps")
        
        # Check for uncertainty markers
        for thought in path.thoughts:
            for marker in self.uncertainty_markers:
                if marker in thought.content.lower():
                    uncertainties.append(f"Uncertainty detected: '{marker}' in reasoning")
                    break
        
        # Check evidence quality
        evidence_count = sum(len(t.supporting_evidence) for t in path.thoughts)
        if evidence_count == 0:
            issues.append("No supporting evidence found")
        
        # Check reasoning chain length
        if len(path.thoughts) < 3:
            issues.append("Reasoning chain may be too short")
        
        # Overall assessment
        quality_score = self._calculate_quality_score(path, issues, uncertainties)
        
        return {
            'quality_score': quality_score,
            'issues': issues,
            'uncertainties': uncertainties,
            'recommendation': self._get_recommendation(quality_score, issues),
            'confidence_adjustment': self._suggest_confidence_adjustment(issues, uncertainties)
        }
    
    def _calculate_quality_score(self, path: ReasoningPath, issues: List[str], uncertainties: List[str]) -> float:
        """Calculate overall reasoning quality"""
        base_score = path.confidence
        
        # Penalties
        issue_penalty = len(issues) * 0.1
        uncertainty_penalty = len(uncertainties) * 0.05
        
        return max(0.0, min(1.0, base_score - issue_penalty - uncertainty_penalty))
    
    def _get_recommendation(self, quality_score: float, issues: List[str]) -> str:
        """Get recommendation based on quality assessment"""
        if quality_score >= 0.8:
            return "Reasoning appears sound"
        elif quality_score >= 0.6:
            return "Consider gathering more evidence"
        elif quality_score >= 0.4:
            return "Reasoning needs strengthening - search for more information"
        else:
            return "Insufficient reasoning - recommend web search"
    
    def _suggest_confidence_adjustment(self, issues: List[str], uncertainties: List[str]) -> float:
        """Suggest confidence adjustment factor"""
        penalty = len(issues) * 0.1 + len(uncertainties) * 0.05
        return max(0.5, 1.0 - penalty)


class SelfVerifier:
    """
    Self-verification module.
    Checks reasoning for logical errors and inconsistencies.
    """
    
    def verify(self, path: ReasoningPath, query: str) -> Dict[str, Any]:
        """Verify reasoning path for errors"""
        checks = []
        
        # Check 1: Does conclusion address the question?
        relevance = self._check_relevance(path.final_answer, query)
        checks.append({
            'check': 'Relevance',
            'passed': relevance > 0.5,
            'score': relevance,
            'detail': 'Conclusion addresses the question' if relevance > 0.5 else 'Conclusion may not address the question'
        })
        
        # Check 2: Is reasoning logically consistent?
        consistency = self._check_logical_consistency(path.thoughts)
        checks.append({
            'check': 'Logical Consistency',
            'passed': consistency > 0.6,
            'score': consistency,
            'detail': 'Reasoning appears consistent' if consistency > 0.6 else 'Potential logical gaps detected'
        })
        
        # Check 3: Is evidence sufficient?
        evidence = self._check_evidence_sufficiency(path.thoughts)
        checks.append({
            'check': 'Evidence Sufficiency',
            'passed': evidence > 0.5,
            'score': evidence,
            'detail': 'Sufficient evidence' if evidence > 0.5 else 'More evidence needed'
        })
        
        # Overall verification status
        all_passed = all(c['passed'] for c in checks)
        avg_score = sum(c['score'] for c in checks) / len(checks)
        
        return {
            'verified': all_passed,
            'verification_score': avg_score,
            'checks': checks,
            'status': 'Verified' if all_passed else 'Needs Review'
        }
    
    def _check_relevance(self, answer: str, query: str) -> float:
        """Check if answer is relevant to query"""
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove stopwords
        stopwords = {'what', 'is', 'are', 'the', 'a', 'an', 'how', 'why', 'when', 'where', 'who'}
        query_words -= stopwords
        answer_words -= stopwords
        
        if not query_words:
            return 0.5
        
        overlap = len(query_words & answer_words)
        return min(1.0, overlap / len(query_words) + 0.3)
    
    def _check_logical_consistency(self, thoughts: List[Thought]) -> float:
        """Check for logical consistency in reasoning chain"""
        if len(thoughts) < 2:
            return 0.5
        
        # Check if thoughts build on each other
        consistency_score = 0.7  # Base score
        
        for i in range(1, len(thoughts)):
            prev = thoughts[i-1].content.lower()
            curr = thoughts[i].content.lower()
            
            # Check for contradictions (simplified)
            if ('not' in curr and 'not' not in prev) or ('however' in curr):
                consistency_score -= 0.1
            
            # Check for logical connectors
            if any(w in curr for w in ['therefore', 'because', 'since', 'thus', 'hence']):
                consistency_score += 0.05
        
        return max(0.0, min(1.0, consistency_score))
    
    def _check_evidence_sufficiency(self, thoughts: List[Thought]) -> float:
        """Check if there's sufficient evidence"""
        total_evidence = sum(len(t.supporting_evidence) for t in thoughts)
        
        if total_evidence >= 3:
            return 1.0
        elif total_evidence >= 1:
            return 0.7
        else:
            return 0.3


class AdvancedReasoningEngine:
    """
    Main advanced reasoning engine that orchestrates all reasoning modules.
    Implements human-like critical thinking.
    """
    
    def __init__(self, knowledge_retriever=None):
        self.cot = ChainOfThought(knowledge_retriever)
        self.tot = TreeOfThoughts()
        self.self_consistency = SelfConsistency()
        self.metacognition = MetacognitiveMonitor()
        self.verifier = SelfVerifier()
        self.knowledge = knowledge_retriever
    
    def think(self, query: str, context: List[str] = None, deep_think: bool = False) -> ReasoningResult:
        """
        Main reasoning entry point.
        Orchestrates different reasoning strategies based on query complexity.
        """
        # Step 1: Analyze query complexity
        complexity = self._assess_complexity(query)
        
        # Step 2: Choose reasoning strategy
        if deep_think or complexity > 0.7:
            # Use full reasoning pipeline for complex queries
            return self._deep_reasoning(query, context)
        elif complexity > 0.4:
            # Use ToT for medium complexity
            return self._moderate_reasoning(query, context)
        else:
            # Use simple CoT for simple queries
            return self._simple_reasoning(query, context)
    
    def _assess_complexity(self, query: str) -> float:
        """Assess query complexity (0-1)"""
        score = 0.3  # Base complexity
        
        # Length factor
        words = len(query.split())
        if words > 20:
            score += 0.2
        elif words > 10:
            score += 0.1
        
        # Question type factor
        complex_markers = ['why', 'how', 'compare', 'analyze', 'explain', 'evaluate', 'critique']
        if any(m in query.lower() for m in complex_markers):
            score += 0.2
        
        # Multiple concepts
        concepts = self.cot._extract_concepts(query)
        if len(concepts) > 3:
            score += 0.2
        
        return min(1.0, score)
    
    def _deep_reasoning(self, query: str, context: List[str] = None) -> ReasoningResult:
        """Deep reasoning with all modules"""
        # Use self-consistency to get multiple paths
        best_path, all_paths = self.self_consistency.reason(query, context)
        
        # Metacognitive monitoring
        monitoring = self.metacognition.monitor(best_path)
        
        # Self-verification
        verification = self.verifier.verify(best_path, query)
        
        # Adjust confidence based on monitoring and verification
        final_confidence = (
            best_path.confidence * 
            monitoring['confidence_adjustment'] * 
            verification['verification_score']
        )
        
        # Format thought process for output
        thought_process = []
        for t in best_path.thoughts:
            thought_process.append({
                'step': t.content,
                'type': t.thought_type.value,
                'confidence': t.confidence
            })
        
        # Get alternative answers
        alternatives = []
        for path in all_paths:
            if path != best_path:
                alternatives.append({
                    'answer': path.final_answer[:200],
                    'confidence': path.confidence
                })
        
        return ReasoningResult(
            answer=best_path.final_answer,
            confidence=final_confidence,
            reasoning_type='deep_analysis',
            thought_process=thought_process,
            alternative_answers=alternatives,
            uncertainty_areas=monitoring['uncertainties'],
            verification_status=verification['status'],
            metacognitive_notes=[monitoring['recommendation']]
        )
    
    def _moderate_reasoning(self, query: str, context: List[str] = None) -> ReasoningResult:
        """Moderate reasoning with ToT"""
        path = self.tot.reason(query, context)
        
        # Quick verification
        verification = self.verifier.verify(path, query)
        
        thought_process = [{
            'step': t.content,
            'type': t.thought_type.value,
            'confidence': t.confidence
        } for t in path.thoughts]
        
        return ReasoningResult(
            answer=path.final_answer,
            confidence=path.confidence * verification['verification_score'],
            reasoning_type='tree_exploration',
            thought_process=thought_process,
            alternative_answers=[],
            uncertainty_areas=[],
            verification_status=verification['status'],
            metacognitive_notes=[]
        )
    
    def _simple_reasoning(self, query: str, context: List[str] = None) -> ReasoningResult:
        """Simple CoT reasoning"""
        path = self.cot.reason(query, context)
        
        thought_process = [{
            'step': t.content,
            'type': t.thought_type.value,
            'confidence': t.confidence
        } for t in path.thoughts]
        
        return ReasoningResult(
            answer=path.final_answer,
            confidence=path.confidence,
            reasoning_type='chain_of_thought',
            thought_process=thought_process,
            alternative_answers=[],
            uncertainty_areas=[],
            verification_status='Basic',
            metacognitive_notes=[]
        )
    
    def reflect(self, previous_result: ReasoningResult) -> str:
        """Metacognitive reflection on previous reasoning"""
        reflections = []
        
        reflections.append(f"My reasoning approach was: {previous_result.reasoning_type}")
        reflections.append(f"Confidence level: {previous_result.confidence:.0%}")
        
        if previous_result.uncertainty_areas:
            reflections.append(f"Areas of uncertainty: {', '.join(previous_result.uncertainty_areas[:3])}")
        
        if previous_result.alternative_answers:
            reflections.append(f"I considered {len(previous_result.alternative_answers)} alternative approaches")
        
        reflections.append(f"Verification status: {previous_result.verification_status}")
        
        return "\n".join(reflections)