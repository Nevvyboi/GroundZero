"""
GroundZero Advanced Reasoning System
=====================================
A sophisticated reasoning engine that goes beyond basic pattern matching.

CAPABILITIES:
1. Semantic Similarity - Find related concepts using word embeddings
2. Flexible Pattern Matching - Handle questions not explicitly programmed
3. Multi-hop Reasoning - Chain facts together (A→B→C)
4. Analogical Reasoning - "X is to Y as A is to B"
5. Common Sense Rules - Built-in world knowledge
6. Fuzzy Matching - Handle typos and variations

This builds on the persistent graph but adds neural-symbolic hybrid reasoning.
"""

import re
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path

# Import the persistent graph
try:
    from .persistent_graph import (
        PersistentKnowledgeGraph, 
        PersistentReasoner,
        Fact, 
        RelationType,
        RobustKnowledgeExtractor
    )
except ImportError:
    from persistent_graph import (
        PersistentKnowledgeGraph, 
        PersistentReasoner,
        Fact, 
        RelationType,
        RobustKnowledgeExtractor
    )


# ============================================================
# SEMANTIC SIMILARITY ENGINE
# ============================================================

class SemanticSimilarity:
    """
    Find semantically similar concepts using word embeddings.
    
    This allows the system to:
    - Find related concepts even if not explicitly linked
    - Handle synonyms and variations
    - Support fuzzy matching
    """
    
    def __init__(self, embeddings_path: Optional[Path] = None):
        # Simple word vectors (will be populated from knowledge base)
        self.word_vectors: Dict[str, np.ndarray] = {}
        self.vector_dim = 256
        
        # Common word similarities (built-in)
        self.synonyms = {
            'big': ['large', 'huge', 'enormous', 'giant', 'massive'],
            'small': ['tiny', 'little', 'miniature', 'compact', 'mini'],
            'fast': ['quick', 'rapid', 'speedy', 'swift'],
            'slow': ['sluggish', 'gradual', 'leisurely'],
            'good': ['great', 'excellent', 'fine', 'positive'],
            'bad': ['poor', 'terrible', 'negative', 'awful'],
            'hot': ['warm', 'heated', 'burning', 'scorching'],
            'cold': ['cool', 'freezing', 'chilly', 'icy'],
            'happy': ['joyful', 'pleased', 'content', 'glad'],
            'sad': ['unhappy', 'depressed', 'sorrowful', 'melancholy'],
            'start': ['begin', 'commence', 'initiate', 'launch'],
            'end': ['finish', 'conclude', 'terminate', 'stop'],
            'make': ['create', 'build', 'construct', 'produce', 'manufacture'],
            'country': ['nation', 'state', 'land'],
            'city': ['town', 'metropolis', 'urban area'],
            'person': ['human', 'individual', 'people', 'man', 'woman'],
            'animal': ['creature', 'beast', 'organism'],
            'place': ['location', 'area', 'region', 'site'],
            'thing': ['object', 'item', 'entity'],
        }
        
        # Build reverse synonym map
        self.reverse_synonyms: Dict[str, str] = {}
        for key, syns in self.synonyms.items():
            for syn in syns:
                self.reverse_synonyms[syn] = key
        
        print("✅ Semantic Similarity Engine initialized")
    
    def get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word"""
        word = word.lower()
        
        # Direct synonyms
        if word in self.synonyms:
            return self.synonyms[word]
        
        # Reverse lookup
        if word in self.reverse_synonyms:
            base = self.reverse_synonyms[word]
            return [base] + [s for s in self.synonyms[base] if s != word]
        
        return []
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        if vec1 is None or vec2 is None:
            return 0.0
        
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        Uses multiple methods and combines them.
        """
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # Exact match
        if text1 == text2:
            return 1.0
        
        # Word overlap (Jaccard)
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        jaccard = len(intersection) / len(union)
        
        # Synonym expansion
        expanded1 = set()
        for w in words1:
            expanded1.add(w)
            expanded1.update(self.get_synonyms(w))
        
        expanded2 = set()
        for w in words2:
            expanded2.add(w)
            expanded2.update(self.get_synonyms(w))
        
        expanded_intersection = expanded1 & expanded2
        expanded_union = expanded1 | expanded2
        expanded_jaccard = len(expanded_intersection) / len(expanded_union) if expanded_union else 0
        
        # Character-level similarity (for typos)
        char_sim = self._char_similarity(text1, text2)
        
        # Combine scores
        return max(jaccard, expanded_jaccard * 0.9, char_sim * 0.7)
    
    def _char_similarity(self, s1: str, s2: str) -> float:
        """Character-level similarity using bigrams"""
        if not s1 or not s2:
            return 0.0
        
        def get_bigrams(s):
            return set(s[i:i+2] for i in range(len(s)-1))
        
        bg1 = get_bigrams(s1)
        bg2 = get_bigrams(s2)
        
        if not bg1 or not bg2:
            return 0.0
        
        intersection = bg1 & bg2
        union = bg1 | bg2
        
        return len(intersection) / len(union)
    
    def find_similar_entities(self, entity: str, all_entities: List[str], 
                             threshold: float = 0.5, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find entities similar to the given one"""
        similarities = []
        
        for other in all_entities:
            if other.lower() == entity.lower():
                continue
            
            sim = self.text_similarity(entity, other)
            if sim >= threshold:
                similarities.append((other, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


# ============================================================
# COMMON SENSE RULES
# ============================================================

class CommonSenseRules:
    """
    Built-in common sense knowledge about the world.
    
    This provides basic reasoning capabilities like:
    - Size comparisons (elephant > dog > ant)
    - Time ordering (morning < afternoon < evening)
    - Spatial relationships
    - Category hierarchies
    """
    
    def __init__(self):
        # Size ordering (smaller to larger)
        self.size_order = [
            'atom', 'molecule', 'cell', 'bacteria', 'ant', 'bee', 'mouse', 
            'rat', 'cat', 'dog', 'human', 'horse', 'car', 'elephant', 
            'whale', 'building', 'mountain', 'country', 'continent', 
            'planet', 'star', 'galaxy', 'universe'
        ]
        
        # Time periods (shorter to longer)
        self.time_order = [
            'second', 'minute', 'hour', 'day', 'week', 'month', 'year',
            'decade', 'century', 'millennium', 'era', 'eon'
        ]
        
        # Category hierarchies
        self.hierarchies = {
            'animal': ['mammal', 'bird', 'fish', 'reptile', 'insect', 'amphibian'],
            'mammal': ['dog', 'cat', 'horse', 'elephant', 'whale', 'human', 'lion', 'tiger'],
            'bird': ['eagle', 'sparrow', 'penguin', 'owl', 'parrot'],
            'fish': ['salmon', 'tuna', 'shark', 'goldfish'],
            'reptile': ['snake', 'lizard', 'crocodile', 'turtle'],
            'insect': ['ant', 'bee', 'butterfly', 'mosquito', 'beetle'],
            'vehicle': ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'airplane', 'boat'],
            'car': ['sedan', 'suv', 'sports car', 'truck'],
            'fruit': ['apple', 'banana', 'orange', 'grape', 'mango', 'strawberry'],
            'vegetable': ['carrot', 'potato', 'tomato', 'broccoli', 'spinach'],
            'food': ['fruit', 'vegetable', 'meat', 'bread', 'dairy'],
            'color': ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'black', 'white'],
            'shape': ['circle', 'square', 'triangle', 'rectangle', 'oval'],
            'continent': ['africa', 'asia', 'europe', 'north america', 'south america', 'australia', 'antarctica'],
            'planet': ['mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune'],
        }
        
        # Properties by category
        self.category_properties = {
            'mammal': ['warm-blooded', 'has fur', 'gives birth to live young', 'produces milk'],
            'bird': ['has feathers', 'has wings', 'lays eggs', 'warm-blooded'],
            'fish': ['has gills', 'lives in water', 'has fins', 'cold-blooded'],
            'reptile': ['cold-blooded', 'has scales', 'lays eggs'],
            'insect': ['has six legs', 'has exoskeleton', 'cold-blooded'],
            'vehicle': ['can move', 'transports people or goods', 'man-made'],
            'fruit': ['edible', 'contains seeds', 'grows on plants'],
            'planet': ['orbits a star', 'has gravity', 'spherical'],
        }
        
        # Opposites
        self.opposites = {
            'hot': 'cold', 'cold': 'hot',
            'big': 'small', 'small': 'big',
            'fast': 'slow', 'slow': 'fast',
            'good': 'bad', 'bad': 'good',
            'up': 'down', 'down': 'up',
            'left': 'right', 'right': 'left',
            'in': 'out', 'out': 'in',
            'on': 'off', 'off': 'on',
            'open': 'closed', 'closed': 'open',
            'light': 'dark', 'dark': 'light',
            'young': 'old', 'old': 'young',
            'rich': 'poor', 'poor': 'rich',
            'happy': 'sad', 'sad': 'happy',
            'love': 'hate', 'hate': 'love',
            'life': 'death', 'death': 'life',
            'day': 'night', 'night': 'day',
            'summer': 'winter', 'winter': 'summer',
            'north': 'south', 'south': 'north',
            'east': 'west', 'west': 'east',
        }
        
        print("✅ Common Sense Rules loaded")
    
    def is_bigger_than(self, a: str, b: str) -> Optional[bool]:
        """Check if a is bigger than b"""
        a, b = a.lower(), b.lower()
        
        try:
            idx_a = self.size_order.index(a)
            idx_b = self.size_order.index(b)
            return idx_a > idx_b
        except ValueError:
            return None
    
    def is_longer_than(self, a: str, b: str) -> Optional[bool]:
        """Check if time period a is longer than b"""
        a, b = a.lower(), b.lower()
        
        try:
            idx_a = self.time_order.index(a)
            idx_b = self.time_order.index(b)
            return idx_a > idx_b
        except ValueError:
            return None
    
    def is_type_of(self, specific: str, general: str) -> bool:
        """Check if specific is a type of general"""
        specific, general = specific.lower(), general.lower()
        
        if general in self.hierarchies:
            if specific in self.hierarchies[general]:
                return True
            # Check subcategories
            for subcat in self.hierarchies[general]:
                if subcat in self.hierarchies and specific in self.hierarchies[subcat]:
                    return True
        
        return False
    
    def get_category(self, entity: str) -> Optional[str]:
        """Get the category of an entity"""
        entity = entity.lower()
        
        for category, members in self.hierarchies.items():
            if entity in members:
                return category
            # Check if entity matches category name
            if entity == category:
                # Find parent category
                for parent, children in self.hierarchies.items():
                    if category in children:
                        return parent
        
        return None
    
    def get_properties(self, entity: str) -> List[str]:
        """Get common sense properties of an entity"""
        entity = entity.lower()
        properties = []
        
        # Direct properties
        category = self.get_category(entity)
        if category and category in self.category_properties:
            properties.extend(self.category_properties[category])
        
        # Also check if entity IS a category
        if entity in self.category_properties:
            properties.extend(self.category_properties[entity])
        
        return list(set(properties))
    
    def get_opposite(self, word: str) -> Optional[str]:
        """Get the opposite of a word"""
        return self.opposites.get(word.lower())
    
    def compare(self, a: str, b: str, dimension: str) -> Optional[str]:
        """Compare two things along a dimension"""
        dimension = dimension.lower()
        
        if dimension in ['size', 'big', 'large']:
            result = self.is_bigger_than(a, b)
            if result is True:
                return f"{a} is bigger than {b}"
            elif result is False:
                return f"{b} is bigger than {a}"
        
        elif dimension in ['time', 'duration', 'long']:
            result = self.is_longer_than(a, b)
            if result is True:
                return f"{a} is longer than {b}"
            elif result is False:
                return f"{b} is longer than {a}"
        
        return None


# ============================================================
# ADVANCED PATTERN MATCHING
# ============================================================

class FlexiblePatternMatcher:
    """
    Flexible question understanding that goes beyond rigid patterns.
    
    Can handle:
    - Questions phrased in many different ways
    - Missing words or typos
    - Implicit questions
    """
    
    def __init__(self, semantic: SemanticSimilarity):
        self.semantic = semantic
        
        # Question templates with semantic slots
        self.templates = [
            # Capital questions
            {
                'patterns': [
                    r"(?:what|which) (?:is|was) (?:the )?capital (?:of|for) (.+)",
                    r"capital (?:of|for) (.+)",
                    r"(.+?)(?:'s| 's) capital",
                    r"(?:what|which) city is (?:the )?capital of (.+)",
                ],
                'type': 'capital_of',
                'extract': 'object'
            },
            # Location questions
            {
                'patterns': [
                    r"where is (.+?)(?: located)?",
                    r"(?:what|which) (?:country|continent|place) is (.+?) in",
                    r"location of (.+)",
                    r"(.+?) is (?:located )?(?:in|at) (?:what|which)",
                ],
                'type': 'location',
                'extract': 'subject'
            },
            # Birth questions
            {
                'patterns': [
                    r"where was (.+?) born",
                    r"(.+?)(?:'s| 's) birthplace",
                    r"birthplace of (.+)",
                    r"where (?:did|does) (.+?) come from",
                ],
                'type': 'birthplace',
                'extract': 'subject'
            },
            # Creator questions
            {
                'patterns': [
                    r"who (?:created|made|invented|founded|built|developed) (.+)",
                    r"(?:creator|inventor|founder|maker) of (.+)",
                    r"(.+?) was (?:created|made|invented|founded) by (?:who|whom)",
                    r"who is (?:the )?(?:creator|inventor|founder) of (.+)",
                ],
                'type': 'creator',
                'extract': 'object'
            },
            # Definition questions
            {
                'patterns': [
                    r"what is (?:a |an |the )?(.+)",
                    r"define (.+)",
                    r"explain (.+)",
                    r"tell me about (.+)",
                    r"who is (.+)",
                    r"describe (.+)",
                ],
                'type': 'definition',
                'extract': 'subject'
            },
            # Comparison questions
            {
                'patterns': [
                    r"is (?:a |an )?(\w+) (?:bigger|larger|smaller|faster|slower|taller|shorter) than (?:a |an )?(\w+)",
                    r"compare (?:a |an )?(.+?) (?:and|to|with|vs) (?:a |an )?(.+)",
                    r"(.+?) vs (?:a |an )?(.+)",
                    r"difference between (?:a |an )?(.+?) and (?:a |an )?(.+)",
                    r"which is (?:bigger|larger|smaller|faster) (?:a |an )?(\w+) or (?:a |an )?(\w+)",
                ],
                'type': 'comparison',
                'extract': 'both'
            },
            # Yes/No questions
            {
                'patterns': [
                    r"is (.+?) (?:a |an |the )?(.+)",
                    r"(?:is|are|was|were) (.+?) in (.+)",
                    r"does (.+?) (?:have|contain) (.+)",
                    r"can (.+?) (.+)",
                ],
                'type': 'verification',
                'extract': 'both'
            },
            # Type/Category questions
            {
                'patterns': [
                    r"what type of (.+?) is (.+)",
                    r"what kind of (.+?) is (.+)",
                    r"is (.+?) (?:a |an )(.+)",
                ],
                'type': 'category',
                'extract': 'both'
            },
        ]
    
    def parse(self, question: str) -> Dict[str, Any]:
        """Parse a question into structured form"""
        question = question.lower().strip().rstrip('?')
        
        for template in self.templates:
            for pattern in template['patterns']:
                match = re.search(pattern, question, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    
                    result = {
                        'type': template['type'],
                        'original': question,
                        'subject': None,
                        'object': None,
                        'matched_pattern': pattern
                    }
                    
                    if template['extract'] == 'subject' and groups:
                        result['subject'] = groups[0].strip()
                    elif template['extract'] == 'object' and groups:
                        result['object'] = groups[0].strip()
                    elif template['extract'] == 'both' and len(groups) >= 2:
                        result['subject'] = groups[0].strip()
                        result['object'] = groups[1].strip()
                    elif groups:
                        result['subject'] = groups[0].strip()
                    
                    return result
        
        # Fallback: extract keywords
        keywords = self._extract_keywords(question)
        return {
            'type': 'unknown',
            'original': question,
            'subject': keywords[0] if keywords else None,
            'object': keywords[1] if len(keywords) > 1 else None,
            'keywords': keywords
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
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


# ============================================================
# MULTI-HOP REASONING
# ============================================================

class MultiHopReasoner:
    """
    Chain facts together to answer complex questions.
    
    Example:
    Q: "What continent is the capital of France in?"
    Path: capital_of(?, France) → Paris
          located_in(Paris, ?) → France  
          located_in(France, ?) → Europe
    A: "Europe"
    """
    
    def __init__(self, graph: PersistentKnowledgeGraph):
        self.graph = graph
        self.max_hops = 3
    
    def find_path(self, start: str, end: str, max_depth: int = 3) -> Optional[List[Fact]]:
        """Find a path of facts connecting start to end"""
        start = start.lower()
        end = end.lower()
        
        # BFS to find shortest path
        visited = {start}
        queue = [(start, [])]
        
        while queue and len(queue[0][1]) < max_depth:
            current, path = queue.pop(0)
            
            # Get all facts involving current entity
            facts = self.graph.get_facts_involving(current)
            
            for fact in facts:
                # Determine the "other" entity in this fact
                other = fact.obj if fact.subject.lower() == current else fact.subject
                other = other.lower()
                
                if other == end:
                    return path + [fact]
                
                if other not in visited:
                    visited.add(other)
                    queue.append((other, path + [fact]))
        
        return None
    
    def multi_hop_query(self, question: str, parsed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Answer questions that require multiple reasoning steps"""
        
        # Example: "What continent is the capital of France in?"
        # Step 1: Find capital of France → Paris
        # Step 2: Find what France is in → Europe
        
        if parsed['type'] == 'capital_of' and parsed.get('object'):
            # First find the capital
            country = parsed['object']
            capital_facts = self.graph.query(
                relation=RelationType.CAPITAL_OF,
                obj=country
            )
            
            if capital_facts:
                capital = capital_facts[0].subject
                
                # Now find what the country is in
                location_facts = self.graph.query(
                    subject=country,
                    relation=RelationType.LOCATED_IN
                )
                
                if location_facts:
                    continent = location_facts[0].obj
                    
                    return {
                        'answer': f"The capital of {country.title()} is {capital.title()}, "
                                  f"which is in {continent.title()}.",
                        'confidence': 0.85,
                        'path': [capital_facts[0], location_facts[0]],
                        'hops': 2
                    }
        
        return None


# ============================================================
# ANALOGICAL REASONING
# ============================================================

class AnalogicalReasoner:
    """
    Reason by analogy: "X is to Y as A is to B"
    
    Examples:
    - Paris is to France as Tokyo is to Japan
    - Dog is to mammal as eagle is to bird
    """
    
    def __init__(self, graph: PersistentKnowledgeGraph, semantic: SemanticSimilarity):
        self.graph = graph
        self.semantic = semantic
    
    def complete_analogy(self, a: str, b: str, c: str) -> Optional[Tuple[str, float]]:
        """
        Complete analogy: A is to B as C is to ?
        
        Example: Paris is to France as Tokyo is to ?
        """
        a, b, c = a.lower(), b.lower(), c.lower()
        
        # Find the relationship between A and B
        a_facts = self.graph.get_facts_about(a)
        
        for fact in a_facts:
            if fact.obj.lower() == b:
                # Found relationship: A -[rel]-> B
                relation = fact.relation
                
                # Now find C -[rel]-> ?
                c_facts = self.graph.query(subject=c, relation=relation)
                
                if c_facts:
                    return (c_facts[0].obj, 0.9)
        
        # Try reverse: B -[rel]-> A
        b_facts = self.graph.get_facts_about(b)
        
        for fact in b_facts:
            if fact.obj.lower() == a:
                relation = fact.relation
                
                # Find ? -[rel]-> C  (need to search by object)
                # This is trickier...
                all_facts = self.graph.query(relation=relation)
                for f in all_facts:
                    if f.obj.lower() == c:
                        return (f.subject, 0.85)
        
        return None
    
    def find_analogous_pairs(self, a: str, b: str, limit: int = 5) -> List[Tuple[str, str, float]]:
        """
        Find pairs (C, D) that have the same relationship as (A, B)
        
        Example: Given (Paris, France), find (Tokyo, Japan), (Berlin, Germany), etc.
        """
        a, b = a.lower(), b.lower()
        analogous = []
        
        # Find relationship between A and B
        a_facts = self.graph.get_facts_about(a)
        
        for fact in a_facts:
            if fact.obj.lower() == b:
                relation = fact.relation
                
                # Find all pairs with same relationship
                similar_facts = self.graph.query(relation=relation)
                
                for sf in similar_facts:
                    if sf.subject.lower() != a:
                        analogous.append((sf.subject, sf.obj, sf.confidence))
        
        return analogous[:limit]


# ============================================================
# ADVANCED REASONER (MAIN CLASS)
# ============================================================

class AdvancedReasoner:
    """
    Complete advanced reasoning system combining all capabilities.
    
    Features:
    1. Persistent Knowledge Graph (facts & relationships)
    2. Semantic Similarity (find related concepts)
    3. Flexible Pattern Matching (understand any question)
    4. Multi-hop Reasoning (chain facts)
    5. Analogical Reasoning (A:B as C:D)
    6. Common Sense Rules (built-in world knowledge)
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        
        # Core components
        self.graph = PersistentKnowledgeGraph(data_dir)
        self.semantic = SemanticSimilarity()
        self.common_sense = CommonSenseRules()
        self.extractor = RobustKnowledgeExtractor()
        
        # Advanced reasoning
        self.pattern_matcher = FlexiblePatternMatcher(self.semantic)
        self.multi_hop = MultiHopReasoner(self.graph)
        self.analogical = AnalogicalReasoner(self.graph, self.semantic)
        
        print("✅ Advanced Reasoner initialized with all capabilities")
    
    def learn(self, text: str, source: str = "") -> Dict[str, Any]:
        """Learn from text - extract facts and add to graph"""
        # Extract facts
        facts = self.extractor.extract_facts(text, source)
        
        # Add to graph
        result = self.graph.add_facts_batch(facts)
        
        # Extract and add definition
        if source:
            definition = self.extractor.extract_definition(text, source)
            if definition:
                self.graph.add_definition(source, definition, source)
        
        # Run inference
        inference_result = self.graph.run_inference(max_iterations=2)
        
        return {
            'facts_extracted': len(facts),
            'facts_added': result['added'],
            'facts_skipped': result['skipped'],
            'facts_inferred': inference_result['total'],
            'total_facts': self.graph.get_stats()['total_facts']
        }
    
    def reason(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using all available reasoning methods.
        
        Tries in order:
        1. Direct graph lookup
        2. Multi-hop reasoning
        3. Semantic similarity search
        4. Common sense rules
        5. Analogical reasoning
        """
        # Parse the question
        parsed = self.pattern_matcher.parse(question)
        
        reasoning_steps = [f"Parsed as: {parsed['type']}"]
        
        # ==================== METHOD 1: Direct Graph Lookup ====================
        answer = self._try_direct_lookup(parsed)
        if answer:
            return {
                'answer': answer,
                'confidence': 0.9,
                'method': 'direct_lookup',
                'reasoning': reasoning_steps + ['Direct graph lookup succeeded']
            }
        
        reasoning_steps.append('Direct lookup failed, trying multi-hop...')
        
        # ==================== METHOD 2: Multi-hop Reasoning ====================
        multi_hop_result = self.multi_hop.multi_hop_query(question, parsed)
        if multi_hop_result:
            return {
                'answer': multi_hop_result['answer'],
                'confidence': multi_hop_result['confidence'],
                'method': 'multi_hop',
                'hops': multi_hop_result.get('hops', 0),
                'reasoning': reasoning_steps + [f"Multi-hop reasoning with {multi_hop_result.get('hops', 0)} steps"]
            }
        
        reasoning_steps.append('Multi-hop failed, trying semantic similarity...')
        
        # ==================== METHOD 3: Semantic Similarity ====================
        answer = self._try_semantic_search(parsed)
        if answer:
            return {
                'answer': answer,
                'confidence': 0.7,
                'method': 'semantic_similarity',
                'reasoning': reasoning_steps + ['Found via semantic similarity']
            }
        
        reasoning_steps.append('Semantic search failed, trying common sense...')
        
        # ==================== METHOD 4: Common Sense Rules ====================
        answer = self._try_common_sense(parsed, question)
        if answer:
            return {
                'answer': answer,
                'confidence': 0.75,
                'method': 'common_sense',
                'reasoning': reasoning_steps + ['Answered using common sense rules']
            }
        
        reasoning_steps.append('Common sense failed, trying analogy...')
        
        # ==================== METHOD 5: Analogical Reasoning ====================
        answer = self._try_analogical(parsed)
        if answer:
            return {
                'answer': answer,
                'confidence': 0.65,
                'method': 'analogical',
                'reasoning': reasoning_steps + ['Found via analogical reasoning']
            }
        
        # ==================== FALLBACK: Keyword Search ====================
        answer = self._fallback_keyword_search(parsed)
        if answer:
            return {
                'answer': answer,
                'confidence': 0.5,
                'method': 'keyword_search',
                'reasoning': reasoning_steps + ['Fallback keyword search']
            }
        
        return {
            'answer': "I don't have enough information to answer that question.",
            'confidence': 0.1,
            'method': 'none',
            'reasoning': reasoning_steps + ['All methods failed']
        }
    
    def _try_direct_lookup(self, parsed: Dict[str, Any]) -> Optional[str]:
        """Try direct knowledge graph lookup"""
        q_type = parsed['type']
        
        if q_type == 'capital_of' and parsed.get('object'):
            facts = self.graph.query(relation=RelationType.CAPITAL_OF, obj=parsed['object'])
            if facts:
                return f"The capital of {parsed['object'].title()} is {facts[0].subject.title()}."
        
        elif q_type == 'location' and parsed.get('subject'):
            facts = self.graph.query(subject=parsed['subject'], relation=RelationType.LOCATED_IN)
            if facts:
                return f"{parsed['subject'].title()} is located in {facts[0].obj.title()}."
        
        elif q_type == 'birthplace' and parsed.get('subject'):
            facts = self.graph.query(subject=parsed['subject'], relation=RelationType.BORN_IN)
            if facts:
                return f"{parsed['subject'].title()} was born in {facts[0].obj}."
        
        elif q_type == 'creator' and parsed.get('object'):
            facts = self.graph.query(subject=parsed['object'], relation=RelationType.CREATED_BY)
            if facts:
                return f"{parsed['object'].title()} was created by {facts[0].obj.title()}."
        
        elif q_type == 'definition' and parsed.get('subject'):
            # Try definition first
            definition = self.graph.get_definition(parsed['subject'])
            if definition:
                return f"{parsed['subject'].title()} is {definition}."
            
            # Try facts
            facts = self.graph.get_facts_about(parsed['subject'])
            if facts:
                descriptions = []
                for f in facts[:3]:
                    if f.relation == RelationType.IS_A:
                        descriptions.append(f"a {f.obj}")
                    elif f.relation == RelationType.LOCATED_IN:
                        descriptions.append(f"located in {f.obj.title()}")
                    elif f.relation == RelationType.CAPITAL_OF:
                        descriptions.append(f"the capital of {f.obj.title()}")
                
                if descriptions:
                    return f"{parsed['subject'].title()} is {', '.join(descriptions)}."
        
        elif q_type == 'verification' and parsed.get('subject') and parsed.get('object'):
            # Check if subject is in object
            facts = self.graph.query(
                subject=parsed['subject'],
                relation=RelationType.LOCATED_IN,
                obj=parsed['object']
            )
            if facts:
                return f"Yes, {parsed['subject'].title()} is in {parsed['object'].title()}."
            
            # Check capital
            facts = self.graph.query(
                subject=parsed['subject'],
                relation=RelationType.CAPITAL_OF,
                obj=parsed['object']
            )
            if facts:
                return f"Yes, {parsed['subject'].title()} is the capital of {parsed['object'].title()}."
        
        return None
    
    def _try_semantic_search(self, parsed: Dict[str, Any]) -> Optional[str]:
        """Try finding answer via semantic similarity"""
        target = parsed.get('subject') or parsed.get('object')
        if not target:
            return None
        
        # Get all entities we know about
        # This is a simplified version - in production you'd have a proper entity index
        facts = self.graph.get_facts_involving(target)
        
        if not facts:
            # Try synonyms
            synonyms = self.semantic.get_synonyms(target)
            for syn in synonyms:
                facts = self.graph.get_facts_involving(syn)
                if facts:
                    break
        
        if facts:
            # Return first relevant fact
            fact = facts[0]
            return self._fact_to_sentence(fact)
        
        return None
    
    def _try_common_sense(self, parsed: Dict[str, Any], question: str) -> Optional[str]:
        """Try answering using common sense rules"""
        q_type = parsed['type']
        question_lower = question.lower()
        
        # Check for "opposite of" questions first
        opposite_match = re.search(r'(?:what is )?(?:the )?opposite (?:of )?(\w+)', question_lower)
        if opposite_match or 'opposite' in question_lower:
            word = opposite_match.group(1) if opposite_match else None
            if not word:
                # Try to extract from keywords
                words = parsed.get('keywords', [])
                for w in words:
                    if w != 'opposite':
                        word = w
                        break
            
            if word:
                opposite = self.common_sense.get_opposite(word)
                if opposite:
                    return f"The opposite of {word} is {opposite}."
        
        # Check for size comparison
        bigger_match = re.search(r'is (?:a |an )?(\w+) (?:bigger|larger) than (?:a |an )?(\w+)', question_lower)
        if bigger_match:
            a, b = bigger_match.group(1), bigger_match.group(2)
            result = self.common_sense.is_bigger_than(a, b)
            if result is True:
                return f"Yes, a {a} is bigger than a {b}."
            elif result is False:
                return f"No, a {b} is bigger than a {a}."
        
        smaller_match = re.search(r'is (?:a |an )?(\w+) (?:smaller|tinier) than (?:a |an )?(\w+)', question_lower)
        if smaller_match:
            a, b = smaller_match.group(1), smaller_match.group(2)
            result = self.common_sense.is_bigger_than(b, a)  # Reversed
            if result is True:
                return f"Yes, a {a} is smaller than a {b}."
            elif result is False:
                return f"No, a {a} is bigger than a {b}."
        
        if q_type == 'comparison' and parsed.get('subject') and parsed.get('object'):
            # Size comparison
            result = self.common_sense.is_bigger_than(parsed['subject'], parsed['object'])
            if result is True:
                return f"Yes, a {parsed['subject']} is bigger than a {parsed['object']}."
            elif result is False:
                return f"No, a {parsed['object']} is bigger than a {parsed['subject']}."
        
        elif q_type == 'category' or q_type == 'verification':
            subj = parsed.get('subject', '')
            obj = parsed.get('object', '')
            
            if self.common_sense.is_type_of(subj, obj):
                return f"Yes, a {subj} is a type of {obj}."
            
            # Get properties
            props = self.common_sense.get_properties(subj)
            if props:
                return f"A {subj} is {', '.join(props[:3])}."
        
        elif q_type == 'definition':
            subj = parsed.get('subject', '')
            category = self.common_sense.get_category(subj)
            if category:
                props = self.common_sense.get_properties(subj)
                if props:
                    return f"A {subj} is a {category}. It is {props[0]}."
                return f"A {subj} is a type of {category}."
        
        return None
    
    def _try_analogical(self, parsed: Dict[str, Any]) -> Optional[str]:
        """Try analogical reasoning"""
        # This is most useful for "X is to Y as A is to ?" type questions
        # For now, find analogous pairs
        
        if parsed.get('subject') and parsed.get('object'):
            pairs = self.analogical.find_analogous_pairs(
                parsed['subject'], 
                parsed['object'],
                limit=3
            )
            
            if pairs:
                examples = [f"{p[0].title()} and {p[1].title()}" for p in pairs]
                return f"Similar to {parsed['subject'].title()} and {parsed['object'].title()}: {', '.join(examples)}."
        
        return None
    
    def _fallback_keyword_search(self, parsed: Dict[str, Any]) -> Optional[str]:
        """Fallback: search by keywords"""
        keywords = parsed.get('keywords', [])
        if not keywords:
            keywords = [parsed.get('subject'), parsed.get('object')]
            keywords = [k for k in keywords if k]
        
        for keyword in keywords:
            if not keyword:
                continue
            
            facts = self.graph.get_facts_involving(keyword)
            if facts:
                # Return most relevant facts
                responses = [self._fact_to_sentence(f) for f in facts[:2]]
                return " ".join(responses)
        
        return None
    
    def _fact_to_sentence(self, fact: Fact) -> str:
        """Convert a fact to a natural sentence"""
        templates = {
            RelationType.IS_A: "{subject} is a {object}.",
            RelationType.LOCATED_IN: "{subject} is located in {object}.",
            RelationType.CAPITAL_OF: "{subject} is the capital of {object}.",
            RelationType.BORN_IN: "{subject} was born in {object}.",
            RelationType.CREATED_BY: "{subject} was created by {object}.",
            RelationType.PART_OF: "{subject} is part of {object}.",
            RelationType.HAS_PROPERTY: "{subject} is {object}.",
        }
        
        template = templates.get(fact.relation, "{subject} {relation} {object}.")
        
        return template.format(
            subject=fact.subject.title(),
            object=fact.obj.title() if fact.obj[0].isupper() else fact.obj,
            relation=fact.relation.value.replace('_', ' ')
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        graph_stats = self.graph.get_stats()
        
        return {
            **graph_stats,
            'capabilities': [
                'direct_lookup',
                'multi_hop_reasoning', 
                'semantic_similarity',
                'common_sense_rules',
                'analogical_reasoning',
                'flexible_pattern_matching'
            ],
            'common_sense_categories': len(self.common_sense.hierarchies),
            'synonym_groups': len(self.semantic.synonyms)
        }
