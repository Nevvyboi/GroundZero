"""
GroundZero Reasoning Engine - Built From Scratch
================================================
A reasoning system that can actually THINK, not just retrieve.

This combines:
1. Knowledge Graph - Store relationships between concepts
2. Inference Engine - Derive new facts from existing ones
3. Pattern Matching - Understand question types
4. Working Memory - Multi-step reasoning
5. Response Generation - Build coherent answers

NO external LLM dependencies!
"""

import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import json


# ============================================================
# PART 1: KNOWLEDGE REPRESENTATION
# ============================================================

class RelationType(Enum):
    """Types of relationships between concepts"""
    IS_A = "is_a"                    # Cat IS_A Animal
    HAS_PROPERTY = "has_property"    # Fire HAS_PROPERTY hot
    PART_OF = "part_of"              # Wheel PART_OF Car
    LOCATED_IN = "located_in"        # Paris LOCATED_IN France
    CAPITAL_OF = "capital_of"        # Paris CAPITAL_OF France
    CREATED_BY = "created_by"        # Python CREATED_BY Guido
    BORN_IN = "born_in"              # Einstein BORN_IN Germany
    DIED_IN = "died_in"              # Einstein DIED_IN USA
    OCCURRED_ON = "occurred_on"      # WW2 OCCURRED_ON 1939-1945
    CAUSES = "causes"                # Rain CAUSES wet
    USED_FOR = "used_for"            # Hammer USED_FOR nailing
    OPPOSITE_OF = "opposite_of"      # Hot OPPOSITE_OF cold
    SIMILAR_TO = "similar_to"        # Car SIMILAR_TO vehicle
    INSTANCE_OF = "instance_of"      # Eiffel Tower INSTANCE_OF landmark
    DEFINED_AS = "defined_as"        # Photosynthesis DEFINED_AS ...


@dataclass
class Fact:
    """A single fact/triple in the knowledge graph"""
    subject: str
    relation: RelationType
    obj: str
    confidence: float = 1.0
    source: str = ""
    
    def __hash__(self):
        return hash((self.subject.lower(), self.relation, self.obj.lower()))
    
    def __eq__(self, other):
        return (self.subject.lower() == other.subject.lower() and 
                self.relation == other.relation and 
                self.obj.lower() == other.obj.lower())
    
    def inverse(self) -> 'Fact':
        """Get inverse relationship"""
        inverse_map = {
            RelationType.IS_A: RelationType.IS_A,  # No direct inverse
            RelationType.PART_OF: RelationType.HAS_PROPERTY,
            RelationType.CAPITAL_OF: RelationType.LOCATED_IN,
            RelationType.LOCATED_IN: RelationType.LOCATED_IN,
        }
        if self.relation in inverse_map:
            return Fact(self.obj, inverse_map[self.relation], self.subject, self.confidence)
        return None


class KnowledgeGraph:
    """
    Graph-based knowledge representation.
    
    Stores facts as triples: (subject, relation, object)
    Enables reasoning by traversing relationships.
    """
    
    def __init__(self):
        # Main storage: subject -> [(relation, object, confidence)]
        self.graph: Dict[str, List[Tuple[RelationType, str, float]]] = defaultdict(list)
        
        # Reverse index: object -> [(relation, subject, confidence)]
        self.reverse_graph: Dict[str, List[Tuple[RelationType, str, float]]] = defaultdict(list)
        
        # All facts for deduplication
        self.facts: Set[Fact] = set()
        
        # Concept definitions
        self.definitions: Dict[str, str] = {}
        
        # Concept properties
        self.properties: Dict[str, Dict[str, Any]] = defaultdict(dict)
    
    def add_fact(self, fact: Fact) -> bool:
        """Add a fact to the knowledge graph"""
        if fact in self.facts:
            return False
        
        self.facts.add(fact)
        
        # Add to main graph
        self.graph[fact.subject.lower()].append(
            (fact.relation, fact.obj, fact.confidence)
        )
        
        # Add to reverse graph
        self.reverse_graph[fact.obj.lower()].append(
            (fact.relation, fact.subject, fact.confidence)
        )
        
        return True
    
    def add_definition(self, concept: str, definition: str) -> None:
        """Store a concept's definition"""
        self.definitions[concept.lower()] = definition
    
    def get_facts_about(self, subject: str) -> List[Fact]:
        """Get all facts where subject is the subject"""
        subject = subject.lower()
        facts = []
        for rel, obj, conf in self.graph.get(subject, []):
            facts.append(Fact(subject, rel, obj, conf))
        return facts
    
    def get_facts_involving(self, entity: str) -> List[Fact]:
        """Get all facts involving an entity (as subject or object)"""
        entity = entity.lower()
        facts = self.get_facts_about(entity)
        
        # Also get facts where entity is the object
        for rel, subj, conf in self.reverse_graph.get(entity, []):
            facts.append(Fact(subj, rel, entity, conf))
        
        return facts
    
    def query(self, subject: str = None, relation: RelationType = None, 
              obj: str = None) -> List[Fact]:
        """Query the knowledge graph with optional filters"""
        results = []
        
        for fact in self.facts:
            if subject and fact.subject.lower() != subject.lower():
                continue
            if relation and fact.relation != relation:
                continue
            if obj and fact.obj.lower() != obj.lower():
                continue
            results.append(fact)
        
        return results
    
    def get_definition(self, concept: str) -> Optional[str]:
        """Get definition of a concept"""
        return self.definitions.get(concept.lower())


# ============================================================
# PART 2: NATURAL LANGUAGE UNDERSTANDING
# ============================================================

class QuestionType(Enum):
    """Types of questions we can handle"""
    WHAT_IS = "what_is"              # What is X?
    WHO_IS = "who_is"                # Who is X?
    WHERE_IS = "where_is"            # Where is X?
    WHEN_DID = "when_did"            # When did X happen?
    WHY_IS = "why_is"                # Why is X?
    HOW_DOES = "how_does"            # How does X work?
    IS_IT_TRUE = "is_it_true"        # Is X true?
    WHAT_IS_RELATION = "relation"    # What is the capital of X?
    COMPARE = "compare"              # Compare X and Y
    LIST = "list"                    # List all X
    COUNT = "count"                  # How many X?
    UNKNOWN = "unknown"


@dataclass
class ParsedQuestion:
    """Parsed representation of a question"""
    original: str
    question_type: QuestionType
    subject: Optional[str] = None
    relation: Optional[RelationType] = None
    object: Optional[str] = None
    keywords: List[str] = field(default_factory=list)


class LanguageParser:
    """
    Parses natural language into structured representations.
    No ML models - pure pattern matching and rules.
    """
    
    # Question patterns
    PATTERNS = {
        # Relation queries FIRST (more specific)
        QuestionType.WHAT_IS_RELATION: [
            r"^what is the (.+?) of (?:the )?(.+?)[\?]?$",
            r"^what'?s the (.+?) of (?:the )?(.+?)[\?]?$",
            r"^who is the (.+?) of (?:the )?(.+?)[\?]?$",
        ],
        QuestionType.WHERE_IS: [
            r"^where is (?:the )?(.+?) located[\?]?$",
            r"^where is (?:the )?(.+?)[\?]?$",
            r"^where was (.+?) born[\?]?$",
            r"^(?:in )?what (?:country|city|place|location) is (?:the )?(.+?)[\?]?$",
        ],
        QuestionType.WHEN_DID: [
            r"^when (?:did|was|were|is) (?:the )?(.+?)[\?]?$",
        ],
        QuestionType.WHO_IS: [
            r"^who (?:is|was|were) (?:the )?(.+?)[\?]?$",
        ],
        QuestionType.WHY_IS: [
            r"^why (?:is|are|was|were|do|does|did) (?:the )?(.+?)[\?]?$",
        ],
        QuestionType.HOW_DOES: [
            r"^how (?:does|do|did|is|are|can) (?:the )?(.+?)[\?]?$",
        ],
        QuestionType.IS_IT_TRUE: [
            r"^is (.+?) (?:in|at|from|part of) (.+?)[\?]?$",
            r"^(?:is|are|was|were|do|does|did|can|could) (?:the )?(.+?)[\?]?$",
        ],
        QuestionType.COMPARE: [
            r"^compare (?:the )?(.+?) (?:and|with|to|vs\.?) (?:the )?(.+?)[\?]?$",
            r"^what(?:'s| is) the difference between (?:the )?(.+?) and (?:the )?(.+?)[\?]?$",
        ],
        QuestionType.LIST: [
            r"^list (?:all )?(?:the )?(.+?)[\?]?$",
        ],
        QuestionType.COUNT: [
            r"^how many (.+?)[\?]?$",
        ],
        # Generic what is LAST (catch-all)
        QuestionType.WHAT_IS: [
            r"^what (?:is|are) (?:a |an |the )?(.+?)[\?]?$",
            r"^define (?:a |an |the )?(.+?)[\?]?$",
            r"^explain (?:what )?(?:is |are )?(?:a |an |the )?(.+?)[\?]?$",
            r"^tell me about (?:a |an |the )?(.+?)[\?]?$",
        ],
    }
    
    # Relation extraction patterns
    RELATION_PATTERNS = {
        RelationType.CAPITAL_OF: [r"capital"],
        RelationType.LOCATED_IN: [r"located", r"found in", r"in what"],
        RelationType.CREATED_BY: [r"created", r"made", r"invented", r"founded"],
        RelationType.BORN_IN: [r"born", r"birth"],
        RelationType.PART_OF: [r"part of", r"component", r"belongs to"],
        RelationType.IS_A: [r"type of", r"kind of", r"is a"],
    }
    
    def parse(self, text: str) -> ParsedQuestion:
        """Parse a question into structured form"""
        text = text.strip().lower()
        original = text
        
        # Try each question type
        for q_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                match = re.match(pattern, text, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    
                    parsed = ParsedQuestion(
                        original=original,
                        question_type=q_type,
                        keywords=self._extract_keywords(text)
                    )
                    
                    # Extract subject/object based on question type
                    if q_type == QuestionType.WHAT_IS_RELATION and len(groups) >= 2:
                        parsed.subject = groups[0].strip()  # e.g., "capital"
                        parsed.relation = self._identify_relation(groups[0])
                        parsed.object = groups[1].strip()   # e.g., "france"
                    elif q_type == QuestionType.IS_IT_TRUE and len(groups) >= 2:
                        parsed.subject = groups[0].strip()
                        parsed.object = groups[1].strip()
                    elif q_type == QuestionType.COMPARE and len(groups) >= 2:
                        parsed.subject = groups[0].strip()
                        parsed.object = groups[1].strip()
                    elif groups:
                        parsed.subject = groups[0].strip()
                    
                    return parsed
        
        # Unknown question type - extract keywords
        return ParsedQuestion(
            original=original,
            question_type=QuestionType.UNKNOWN,
            keywords=self._extract_keywords(text)
        )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Remove common words
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'what', 'who', 'where', 'when', 'why', 'how', 'which',
            'do', 'does', 'did', 'can', 'could', 'would', 'should',
            'of', 'in', 'on', 'at', 'to', 'for', 'with', 'about',
            'this', 'that', 'these', 'those', 'it', 'its',
            'and', 'or', 'but', 'if', 'then', 'so', 'because',
            'tell', 'me', 'please', 'know', 'explain', 'describe'
        }
        
        words = re.findall(r'\b[a-z]+\b', text.lower())
        return [w for w in words if w not in stopwords and len(w) > 2]
    
    def _identify_relation(self, text: str) -> Optional[RelationType]:
        """Identify relation type from text"""
        text = text.lower()
        for rel_type, patterns in self.RELATION_PATTERNS.items():
            for pattern in patterns:
                if pattern in text:
                    return rel_type
        return None


# ============================================================
# PART 3: INFERENCE ENGINE
# ============================================================

class InferenceEngine:
    """
    Derives new facts from existing knowledge.
    
    Uses rules like:
    - Transitivity: If A is_a B and B is_a C, then A is_a C
    - Inheritance: If A is_a B and B has_property P, then A has_property P
    - Symmetry: If A similar_to B, then B similar_to A
    """
    
    def __init__(self, knowledge: KnowledgeGraph):
        self.knowledge = knowledge
        self.inferred_facts: Set[Fact] = set()
    
    def infer_all(self, max_depth: int = 3) -> List[Fact]:
        """Run all inference rules"""
        new_facts = []
        
        for _ in range(max_depth):
            round_facts = []
            round_facts.extend(self._infer_transitivity())
            round_facts.extend(self._infer_inheritance())
            round_facts.extend(self._infer_symmetry())
            round_facts.extend(self._infer_inverse())
            
            if not round_facts:
                break
            
            new_facts.extend(round_facts)
        
        return new_facts
    
    def _infer_transitivity(self) -> List[Fact]:
        """If A rel B and B rel C, then A rel C (for transitive relations)"""
        new_facts = []
        transitive_relations = {RelationType.IS_A, RelationType.PART_OF, RelationType.LOCATED_IN}
        
        for fact1 in list(self.knowledge.facts):  # Convert to list
            if fact1.relation not in transitive_relations:
                continue
            
            # Find facts where fact1.object is the subject
            for rel, obj, conf in self.knowledge.graph.get(fact1.obj.lower(), []):
                if rel == fact1.relation:
                    new_fact = Fact(
                        fact1.subject, 
                        fact1.relation, 
                        obj,
                        confidence=fact1.confidence * conf * 0.9,  # Reduce confidence
                        source="inferred:transitivity"
                    )
                    if new_fact not in self.knowledge.facts and new_fact not in self.inferred_facts:
                        self.inferred_facts.add(new_fact)
                        self.knowledge.add_fact(new_fact)
                        new_facts.append(new_fact)
        
        return new_facts
    
    def _infer_inheritance(self) -> List[Fact]:
        """If A is_a B and B has_property P, then A has_property P"""
        new_facts = []
        
        # Find all is_a relationships
        for fact in list(self.knowledge.facts):  # Convert to list
            if fact.relation != RelationType.IS_A:
                continue
            
            # Get properties of the parent
            parent_facts = self.knowledge.get_facts_about(fact.obj)
            for parent_fact in parent_facts:
                if parent_fact.relation == RelationType.HAS_PROPERTY:
                    new_fact = Fact(
                        fact.subject,
                        RelationType.HAS_PROPERTY,
                        parent_fact.obj,
                        confidence=fact.confidence * parent_fact.confidence * 0.8,
                        source="inferred:inheritance"
                    )
                    if new_fact not in self.knowledge.facts and new_fact not in self.inferred_facts:
                        self.inferred_facts.add(new_fact)
                        self.knowledge.add_fact(new_fact)
                        new_facts.append(new_fact)
        
        return new_facts
    
    def _infer_symmetry(self) -> List[Fact]:
        """If A similar_to B, then B similar_to A"""
        new_facts = []
        symmetric_relations = {RelationType.SIMILAR_TO, RelationType.OPPOSITE_OF}
        
        for fact in list(self.knowledge.facts):  # Convert to list
            if fact.relation not in symmetric_relations:
                continue
            
            new_fact = Fact(
                fact.obj,
                fact.relation,
                fact.subject,
                confidence=fact.confidence,
                source="inferred:symmetry"
            )
            if new_fact not in self.knowledge.facts and new_fact not in self.inferred_facts:
                self.inferred_facts.add(new_fact)
                self.knowledge.add_fact(new_fact)
                new_facts.append(new_fact)
        
        return new_facts
    
    def _infer_inverse(self) -> List[Fact]:
        """Infer inverse relationships"""
        new_facts = []
        
        # capital_of -> located_in
        for fact in list(self.knowledge.facts):  # Convert to list to avoid modification during iteration
            if fact.relation == RelationType.CAPITAL_OF:
                new_fact = Fact(
                    fact.subject,
                    RelationType.LOCATED_IN,
                    fact.obj,
                    confidence=fact.confidence,
                    source="inferred:inverse"
                )
                if new_fact not in self.knowledge.facts and new_fact not in self.inferred_facts:
                    self.inferred_facts.add(new_fact)
                    self.knowledge.add_fact(new_fact)
                    new_facts.append(new_fact)
        
        return new_facts
    
    def answer_query(self, subject: str = None, relation: RelationType = None, 
                     obj: str = None) -> List[Fact]:
        """Answer a query, running inference if needed"""
        # First try direct lookup
        results = self.knowledge.query(subject, relation, obj)
        
        if not results:
            # Run inference and try again
            self.infer_all(max_depth=2)
            results = self.knowledge.query(subject, relation, obj)
        
        return results


# ============================================================
# PART 4: WORKING MEMORY & REASONING
# ============================================================

@dataclass
class ReasoningStep:
    """A single step in the reasoning process"""
    action: str
    input_data: Any
    output_data: Any
    confidence: float


class WorkingMemory:
    """
    Holds context during multi-step reasoning.
    Like human short-term memory.
    """
    
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.items: List[Any] = []
        self.focus: Optional[str] = None  # Current topic of focus
        self.reasoning_chain: List[ReasoningStep] = []
    
    def add(self, item: Any) -> None:
        """Add item to working memory"""
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items.pop(0)  # Remove oldest
    
    def set_focus(self, topic: str) -> None:
        """Set current focus of attention"""
        self.focus = topic
    
    def add_reasoning_step(self, step: ReasoningStep) -> None:
        """Record a reasoning step"""
        self.reasoning_chain.append(step)
    
    def get_reasoning_trace(self) -> str:
        """Get human-readable reasoning trace"""
        if not self.reasoning_chain:
            return "No reasoning steps recorded."
        
        lines = ["Reasoning process:"]
        for i, step in enumerate(self.reasoning_chain, 1):
            lines.append(f"  {i}. {step.action}")
            if step.output_data:
                lines.append(f"     → {step.output_data}")
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear working memory"""
        self.items = []
        self.focus = None
        self.reasoning_chain = []


# ============================================================
# PART 5: RESPONSE GENERATION
# ============================================================

class ResponseGenerator:
    """
    Generates natural language responses from reasoning results.
    Templates + simple NLG.
    """
    
    TEMPLATES = {
        QuestionType.WHAT_IS: [
            "{subject} is {definition}",
            "{subject} refers to {definition}",
            "A {subject} is {definition}",
        ],
        QuestionType.WHO_IS: [
            "{subject} is {definition}",
            "{subject} was {definition}",
        ],
        QuestionType.WHERE_IS: [
            "{subject} is located in {location}",
            "{subject} can be found in {location}",
            "You can find {subject} in {location}",
        ],
        QuestionType.WHAT_IS_RELATION: [
            "The {relation} of {object} is {answer}",
            "{answer} is the {relation} of {object}",
        ],
        QuestionType.IS_IT_TRUE: [
            "Yes, {fact}",
            "No, that's not correct. {correction}",
            "Based on my knowledge, {answer}",
        ],
        QuestionType.COMPARE: [
            "{subject} and {object} are similar in that {similarities}. They differ in {differences}.",
            "Comparing {subject} and {object}: {comparison}",
        ],
    }
    
    def generate(self, question: ParsedQuestion, facts: List[Fact], 
                 definition: str = None, working_memory: WorkingMemory = None) -> str:
        """Generate a response based on reasoning results"""
        
        if not facts and not definition:
            return self._generate_unknown_response(question)
        
        q_type = question.question_type
        
        # What is the X of Y? (relation query)
        if q_type == QuestionType.WHAT_IS_RELATION:
            if facts:
                # Find the most relevant fact
                fact = facts[0]
                rel_name = question.subject if question.subject else fact.relation.value.replace("_", " ")
                
                # Determine what to return based on the relation
                if fact.relation == RelationType.CAPITAL_OF:
                    return f"The {rel_name} of {question.object.title()} is {fact.subject.title()}."
                elif fact.relation == RelationType.LOCATED_IN:
                    return f"{fact.subject.title()} is located in {fact.obj.title()}."
                elif fact.relation == RelationType.BORN_IN:
                    return f"{fact.subject.title()} was born in {fact.obj}."
                else:
                    return f"The {rel_name} of {question.object.title()} is {fact.subject.title()}."
            return self._generate_unknown_response(question)
        
        # What is X? / Who is X?
        if q_type in [QuestionType.WHAT_IS, QuestionType.WHO_IS]:
            if definition:
                return f"{question.subject.title()} is {definition}."
            elif facts:
                # Build description from facts
                descriptions = []
                for fact in facts[:3]:
                    descriptions.append(self._fact_to_text(fact))
                return " ".join(descriptions)
        
        # Where is X?
        elif q_type == QuestionType.WHERE_IS:
            for fact in facts:
                if fact.relation == RelationType.BORN_IN:
                    return f"{fact.subject.title()} was born in {fact.obj}."
                elif fact.relation in [RelationType.LOCATED_IN, RelationType.CAPITAL_OF]:
                    return f"{fact.subject.title()} is located in {fact.obj.title()}."
            # If no location facts, show what we have
            if facts:
                return f"I found: {self._fact_to_text(facts[0])}"
            return self._generate_unknown_response(question)
        
        # Is X in Y? (verification)
        elif q_type == QuestionType.IS_IT_TRUE:
            if facts:
                return f"Yes! {self._fact_to_text(facts[0])}"
            return f"I couldn't verify that {question.subject} is related to {question.object}."
        
        # Compare X and Y
        elif q_type == QuestionType.COMPARE:
            return self._generate_comparison(question, facts)
        
        # Default: list facts
        if facts:
            responses = [self._fact_to_text(f) for f in facts[:5]]
            return " ".join(responses)
        
        return self._generate_unknown_response(question)
    
    def _fact_to_text(self, fact: Fact) -> str:
        """Convert a fact to natural language"""
        rel_templates = {
            RelationType.IS_A: "{subject} is a {object}",
            RelationType.HAS_PROPERTY: "{subject} is {object}",
            RelationType.LOCATED_IN: "{subject} is located in {object}",
            RelationType.CAPITAL_OF: "{subject} is the capital of {object}",
            RelationType.CREATED_BY: "{subject} was created by {object}",
            RelationType.BORN_IN: "{subject} was born in {object}",
            RelationType.PART_OF: "{subject} is part of {object}",
            RelationType.CAUSES: "{subject} causes {object}",
            RelationType.USED_FOR: "{subject} is used for {object}",
            RelationType.SIMILAR_TO: "{subject} is similar to {object}",
            RelationType.DEFINED_AS: "{subject} is defined as {object}",
        }
        
        template = rel_templates.get(fact.relation, "{subject} {relation} {object}")
        return template.format(
            subject=fact.subject.title(),
            object=fact.obj,
            relation=fact.relation.value.replace("_", " ")
        )
    
    def _generate_comparison(self, question: ParsedQuestion, facts: List[Fact]) -> str:
        """Generate comparison response"""
        subj_facts = [f for f in facts if f.subject.lower() == question.subject.lower()]
        obj_facts = [f for f in facts if f.subject.lower() == question.object.lower()]
        
        response_parts = []
        
        if subj_facts:
            response_parts.append(f"{question.subject.title()}: {self._fact_to_text(subj_facts[0])}")
        if obj_facts:
            response_parts.append(f"{question.object.title()}: {self._fact_to_text(obj_facts[0])}")
        
        if response_parts:
            return " | ".join(response_parts)
        return f"I don't have enough information to compare {question.subject} and {question.object}."
    
    def _generate_unknown_response(self, question: ParsedQuestion) -> str:
        """Generate response when we don't know the answer"""
        if question.subject:
            return f"I don't have enough information about {question.subject} to answer that question."
        return "I don't have enough information to answer that question."


# ============================================================
# PART 6: KNOWLEDGE EXTRACTION (from text)
# ============================================================

class KnowledgeExtractor:
    """
    Extracts structured knowledge from unstructured text.
    Converts sentences into facts.
    """
    
    # Patterns to extract facts - ORDER MATTERS (more specific first)
    EXTRACTION_PATTERNS = [
        # X is the capital of Y
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) is the capital of ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", RelationType.CAPITAL_OF),
        # X is located in Y
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) is (?:located |situated )?in ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", RelationType.LOCATED_IN),
        # X was born in Y
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) was born (?:in )?([A-Z][a-z]+(?:[\s,]+[A-Z]?[a-z]+)*)", RelationType.BORN_IN),
        # X was created/invented by Y
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) was (?:created|invented|founded|developed) by ([A-Z][a-z]+(?:\s+[A-Za-z]+)*)", RelationType.CREATED_BY),
        # X is part of Y
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) is (?:a )?part of ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", RelationType.PART_OF),
        # X causes Y
        (r"([A-Za-z]+(?:\s+[a-z]+)*) causes ([a-z]+(?:\s+[a-z]+)*)", RelationType.CAUSES),
        # X is used for Y
        (r"([A-Z][a-z]+) is used for ([a-z]+(?:\s+[a-z]+)*)", RelationType.USED_FOR),
        # X is a Y (catch-all for is_a)
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) (?:is|are) (?:a |an )?([a-z]+(?:\s+[a-z]+)?)", RelationType.IS_A),
        # A X is a Y
        (r"[Aa]n? ([a-z]+) is (?:a |an )?([a-z]+)", RelationType.IS_A),
    ]
    
    def extract_from_text(self, text: str, source: str = "") -> List[Fact]:
        """Extract facts from text"""
        facts = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            # Try each extraction pattern
            for pattern, rel_type in self.EXTRACTION_PATTERNS:
                matches = re.finditer(pattern, sentence)
                for match in matches:
                    subject = match.group(1).strip()
                    obj = match.group(2).strip()
                    
                    # Clean up - remove trailing punctuation
                    subject = re.sub(r'[,;:\s]+$', '', subject)
                    obj = re.sub(r'[,;:\s]+$', '', obj)
                    
                    # Skip if too short or same
                    if len(subject) < 2 or len(obj) < 2:
                        continue
                    if subject.lower() == obj.lower():
                        continue
                    
                    facts.append(Fact(
                        subject=subject,
                        relation=rel_type,
                        obj=obj,
                        confidence=0.8,
                        source=source
                    ))
        
        return facts
    
    def extract_definition(self, text: str, topic: str) -> Optional[str]:
        """Extract definition of a topic from text"""
        # Look for patterns like "X is ..." at the start
        patterns = [
            rf"{re.escape(topic)} is (.+?)[.]",
            rf"{re.escape(topic)} are (.+?)[.]",
            rf"{re.escape(topic)}, (.+?), is",
            rf"{re.escape(topic)} refers to (.+?)[.]",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                definition = match.group(1).strip()
                if len(definition) > 10:
                    return definition
        
        # Fallback: first sentence containing the topic
        sentences = text.split('.')
        for sent in sentences[:3]:
            if topic.lower() in sent.lower():
                return sent.strip()
        
        return None


# ============================================================
# PART 7: MAIN REASONER CLASS
# ============================================================

class GroundZeroReasoner:
    """
    Main reasoning engine combining all components.
    
    This is a FROM-SCRATCH reasoner that can:
    1. Learn from text (extract facts)
    2. Build a knowledge graph
    3. Infer new facts
    4. Answer questions through reasoning
    5. Generate natural responses
    """
    
    def __init__(self):
        self.knowledge = KnowledgeGraph()
        self.inference = InferenceEngine(self.knowledge)
        self.parser = LanguageParser()
        self.extractor = KnowledgeExtractor()
        self.generator = ResponseGenerator()
        self.memory = WorkingMemory()
        
        print("✅ GroundZero Reasoner initialized")
    
    def learn(self, text: str, source: str = "") -> Dict[str, Any]:
        """Learn from text - extract and store knowledge"""
        # Extract facts
        facts = self.extractor.extract_from_text(text, source)
        added = 0
        
        for fact in facts:
            if self.knowledge.add_fact(fact):
                added += 1
        
        # Extract definition if source title given
        if source:
            definition = self.extractor.extract_definition(text, source)
            if definition:
                self.knowledge.add_definition(source, definition)
        
        # Run inference to derive new facts
        inferred = self.inference.infer_all(max_depth=2)
        
        return {
            'facts_extracted': len(facts),
            'facts_added': added,
            'facts_inferred': len(inferred),
            'total_facts': len(self.knowledge.facts)
        }
    
    def reason(self, question: str) -> Dict[str, Any]:
        """Answer a question through reasoning"""
        # Clear working memory for new question
        self.memory.clear()
        self.memory.set_focus(question)
        
        # Step 1: Parse the question
        parsed = self.parser.parse(question)
        self.memory.add_reasoning_step(ReasoningStep(
            action=f"Parsed question as: {parsed.question_type.value}",
            input_data=question,
            output_data=f"subject={parsed.subject}, relation={parsed.relation}, object={parsed.object}",
            confidence=0.9
        ))
        
        # Step 2: Handle different question types
        facts = []
        definition = None
        
        # Handle "What is the X of Y?" (e.g., capital of France)
        if parsed.question_type == QuestionType.WHAT_IS_RELATION:
            # Find facts where object matches and relation matches
            relation = self._identify_relation_from_text(parsed.subject) if parsed.subject else None
            target = parsed.object
            
            self.memory.add_reasoning_step(ReasoningStep(
                action=f"Looking for {parsed.subject} of {target}",
                input_data=f"relation={relation}, target={target}",
                output_data=None,
                confidence=0.8
            ))
            
            # Search for facts about the target
            if target:
                target_facts = self.knowledge.get_facts_involving(target)
                
                # Filter by relation if we identified one
                if relation:
                    facts = [f for f in target_facts if f.relation == relation]
                else:
                    # Try to match by relation name in subject
                    if parsed.subject:
                        subj_lower = parsed.subject.lower()
                        for f in target_facts:
                            if subj_lower in f.relation.value:
                                facts.append(f)
                
                # If still no facts, return all facts about target
                if not facts:
                    facts = target_facts
                    
        # Handle "Where is X?" or "Where was X born?"
        elif parsed.question_type == QuestionType.WHERE_IS:
            subject = parsed.subject
            if subject:
                # Check if asking about birth location
                if 'born' in question.lower():
                    all_facts = self.knowledge.get_facts_involving(subject)
                    facts = [f for f in all_facts if f.relation == RelationType.BORN_IN]
                else:
                    all_facts = self.knowledge.get_facts_involving(subject)
                    facts = [f for f in all_facts if f.relation in 
                            [RelationType.LOCATED_IN, RelationType.CAPITAL_OF, RelationType.BORN_IN]]
                    if not facts:
                        facts = all_facts
        
        # Handle "Is X in Y?" type questions
        elif parsed.question_type == QuestionType.IS_IT_TRUE:
            if parsed.subject and parsed.object:
                # Check if there's a relationship between subject and object
                subj_facts = self.knowledge.get_facts_involving(parsed.subject)
                facts = [f for f in subj_facts if parsed.object.lower() in f.obj.lower() or 
                        parsed.object.lower() in f.subject.lower()]
        
        # Handle "What is X?" or "Who is X?"
        elif parsed.question_type in [QuestionType.WHAT_IS, QuestionType.WHO_IS]:
            if parsed.subject:
                facts = self.knowledge.get_facts_involving(parsed.subject)
                definition = self.knowledge.get_definition(parsed.subject)
                
                self.memory.add_reasoning_step(ReasoningStep(
                    action=f"Found {len(facts)} facts about '{parsed.subject}'",
                    input_data=parsed.subject,
                    output_data=[f"{f.relation.value}: {f.obj}" for f in facts[:3]],
                    confidence=0.8
                ))
        
        # Default: search by subject
        else:
            if parsed.subject:
                facts = self.knowledge.get_facts_involving(parsed.subject)
                definition = self.knowledge.get_definition(parsed.subject)
        
        # Step 3: If no facts found, try keyword search
        if not facts and not definition and parsed.keywords:
            for keyword in parsed.keywords:
                kw_facts = self.knowledge.get_facts_involving(keyword)
                facts.extend(kw_facts)
            
            self.memory.add_reasoning_step(ReasoningStep(
                action=f"Keyword search for: {parsed.keywords}",
                input_data=parsed.keywords,
                output_data=f"Found {len(facts)} facts",
                confidence=0.6
            ))
        
        # Step 4: Generate response
        response = self.generator.generate(parsed, facts, definition, self.memory)
        
        # Calculate confidence
        confidence = 0.1
        if definition:
            confidence = 0.9
        elif facts:
            confidence = min(0.85, 0.5 + len(facts) * 0.1)
        
        return {
            'answer': response,
            'confidence': confidence,
            'facts_used': len(facts),
            'reasoning_trace': self.memory.get_reasoning_trace(),
            'question_type': parsed.question_type.value
        }
    
    def _identify_relation_from_text(self, text: str) -> Optional[RelationType]:
        """Identify relation type from text like 'capital'"""
        text = text.lower()
        mapping = {
            'capital': RelationType.CAPITAL_OF,
            'location': RelationType.LOCATED_IN,
            'born': RelationType.BORN_IN,
            'birth': RelationType.BORN_IN,
            'creator': RelationType.CREATED_BY,
            'created': RelationType.CREATED_BY,
            'inventor': RelationType.CREATED_BY,
            'part': RelationType.PART_OF,
        }
        for key, rel in mapping.items():
            if key in text:
                return rel
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        return {
            'total_facts': len(self.knowledge.facts),
            'total_definitions': len(self.knowledge.definitions),
            'inferred_facts': len(self.inference.inferred_facts)
        }


# ============================================================
# TESTING
# ============================================================

def test_reasoner():
    """Test the reasoning engine"""
    print("=" * 60)
    print("🧪 Testing GroundZero Reasoner")
    print("=" * 60)
    
    reasoner = GroundZeroReasoner()
    
    # Learn some facts
    texts = [
        ("Paris is the capital of France. France is a country in Europe. "
         "The Eiffel Tower is located in Paris. France is known for wine and cheese.",
         "France"),
        
        ("Albert Einstein was a German-born physicist. Einstein was born in Ulm, Germany in 1879. "
         "He developed the theory of relativity. Einstein is famous for E=mc².",
         "Albert Einstein"),
        
        ("Python is a programming language. Python was created by Guido van Rossum. "
         "Python is used for web development. Python is easy to learn.",
         "Python"),
        
        ("Dogs are mammals. Dogs are used for companionship. A labrador is a dog. "
         "Dogs are loyal animals.",
         "Dogs"),
        
        ("The Sun is a star. The Sun is located in the Solar System. "
         "Earth orbits the Sun. The Sun provides light and heat.",
         "Sun"),
    ]
    
    print("\n📚 Learning from texts...")
    for text, source in texts:
        result = reasoner.learn(text, source)
        print(f"  ✅ {source}: {result['facts_added']} facts, {result['facts_inferred']} inferred")
    
    print(f"\n📊 Knowledge Base Stats: {reasoner.get_stats()}")
    
    # Test questions
    questions = [
        "What is the capital of France?",
        "Where is Paris located?",
        "Who is Albert Einstein?",
        "What is Python?",
        "Where was Einstein born?",
        "What is a dog?",
        "What is the Sun?",
        "Is Paris in France?",
    ]
    
    print("\n" + "=" * 60)
    print("🔍 Testing Questions")
    print("=" * 60)
    
    for q in questions:
        print(f"\n❓ {q}")
        result = reasoner.reason(q)
        print(f"💡 {result['answer']}")
        print(f"   Confidence: {result['confidence']:.0%}")
        print(f"   Facts used: {result['facts_used']}")
    
    # Show reasoning trace for one question
    print("\n" + "=" * 60)
    print("🧠 Reasoning Trace Example")
    print("=" * 60)
    result = reasoner.reason("What is the capital of France?")
    print(result['reasoning_trace'])
    
    print("\n✅ Test complete!")
    return reasoner


if __name__ == "__main__":
    test_reasoner()