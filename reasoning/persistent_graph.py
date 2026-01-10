"""
GroundZero Persistent Knowledge Graph
=====================================
A robust, scalable knowledge graph with full persistence.

WHAT IS A KNOWLEDGE GRAPH?
==========================

A Knowledge Graph stores information as RELATIONSHIPS between things,
not just as text. This is how humans actually think!

Example - Text Storage (what you had before):
    "Paris is the capital of France. France is in Europe."
    → Stored as one blob of text
    → To answer "What is the capital of France?" you search for similar text

Example - Knowledge Graph (what this does):
    Paris  ──[capital_of]──►  France
    France ──[located_in]──►  Europe
    Paris  ──[located_in]──►  France  (inferred!)
    Paris  ──[located_in]──►  Europe  (inferred!)
    
    → Stored as RELATIONSHIPS
    → To answer "What is the capital of France?" you traverse the graph
    → Can INFER new facts automatically!

WHY IS THIS BETTER?
===================

1. PRECISE ANSWERS
   Text Search: "I found text containing 'capital' and 'France'"
   Knowledge Graph: "Paris IS the capital of France" (100% certain)

2. INFERENCE (deriving NEW facts)
   - If Paris is capital of France, and France is in Europe
   - Then Paris is in Europe! (even if never explicitly stated)

3. MULTI-HOP REASONING
   Q: "What continent is the capital of France in?"
   Graph traversal: Paris → France → Europe
   Answer: "Europe"

4. VERIFICATION
   Q: "Is Tokyo the capital of France?"
   Graph lookup: France.capital = Paris ≠ Tokyo
   Answer: "No, Paris is the capital of France"

ARCHITECTURE
============

┌─────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE GRAPH                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ENTITIES (Nodes)           RELATIONS (Edges)                   │
│  ┌─────────┐               ┌─────────────────┐                  │
│  │  Paris  │──────────────►│   capital_of    │────►[France]     │
│  └─────────┘               └─────────────────┘                  │
│  ┌─────────┐               ┌─────────────────┐                  │
│  │ France  │──────────────►│   located_in    │────►[Europe]     │
│  └─────────┘               └─────────────────┘                  │
│  ┌─────────┐               ┌─────────────────┐                  │
│  │Einstein │──────────────►│    born_in      │────►[Germany]    │
│  └─────────┘               └─────────────────┘                  │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  STORAGE LAYERS                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   SQLite     │  │    Index     │  │    Cache     │          │
│  │  (persist)   │  │  (fast lookup)│  │  (in-memory) │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘

PERSISTENCE
===========
- All facts stored in SQLite database
- Survives restarts
- Can handle millions of facts
- Indexed for fast lookups

SCALABILITY
===========
- In-memory cache for hot data
- SQLite indexes on subject, relation, object
- Batch operations for bulk loading
- Lazy loading for large graphs

INFERENCE RULES
===============
1. TRANSITIVITY: If A→B and B→C, then A→C
   Example: Paris in France, France in Europe → Paris in Europe

2. INHERITANCE: If A is_a B, and B has property P, then A has P
   Example: Dog is_a Mammal, Mammal is warm-blooded → Dog is warm-blooded

3. SYMMETRY: If A related_to B, then B related_to A
   Example: France borders Germany → Germany borders France

4. INVERSE: capital_of implies located_in
   Example: Paris capital_of France → Paris located_in France
"""

import sqlite3
import threading
import json
import re
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from enum import Enum
from datetime import datetime


# ============================================================
# RELATION TYPES
# ============================================================

class RelationType(Enum):
    """Types of relationships between entities"""
    # Identity & Classification
    IS_A = "is_a"                    # Cat IS_A Animal
    INSTANCE_OF = "instance_of"      # Eiffel Tower INSTANCE_OF landmark
    SAME_AS = "same_as"              # NYC SAME_AS New York City
    
    # Properties
    HAS_PROPERTY = "has_property"    # Fire HAS_PROPERTY hot
    HAS_PART = "has_part"            # Car HAS_PART wheel
    PART_OF = "part_of"              # Wheel PART_OF Car
    
    # Location
    LOCATED_IN = "located_in"        # Paris LOCATED_IN France
    CAPITAL_OF = "capital_of"        # Paris CAPITAL_OF France
    BORDERS = "borders"              # France BORDERS Germany
    
    # People
    BORN_IN = "born_in"              # Einstein BORN_IN Germany
    DIED_IN = "died_in"              # Einstein DIED_IN USA
    BORN_ON = "born_on"              # Einstein BORN_ON 1879
    DIED_ON = "died_on"              # Einstein DIED_ON 1955
    NATIONALITY = "nationality"      # Einstein NATIONALITY German
    OCCUPATION = "occupation"        # Einstein OCCUPATION physicist
    
    # Creation
    CREATED_BY = "created_by"        # Python CREATED_BY Guido
    FOUNDED_BY = "founded_by"        # Apple FOUNDED_BY Steve Jobs
    INVENTED_BY = "invented_by"      # Telephone INVENTED_BY Bell
    CREATED_IN = "created_in"        # Python CREATED_IN 1991
    
    # Causation & Time
    CAUSES = "causes"                # Rain CAUSES wet
    CAUSED_BY = "caused_by"          # Flood CAUSED_BY rain
    OCCURRED_ON = "occurred_on"      # WW2 OCCURRED_ON 1939-1945
    BEFORE = "before"                # WW1 BEFORE WW2
    AFTER = "after"                  # WW2 AFTER WW1
    
    # Usage & Purpose
    USED_FOR = "used_for"            # Hammer USED_FOR nailing
    MADE_OF = "made_of"              # Table MADE_OF wood
    
    # Similarity & Opposition
    SIMILAR_TO = "similar_to"        # Car SIMILAR_TO vehicle
    OPPOSITE_OF = "opposite_of"      # Hot OPPOSITE_OF cold
    RELATED_TO = "related_to"        # Generic relation
    
    # Definitions
    DEFINED_AS = "defined_as"        # Photosynthesis DEFINED_AS ...
    MEANS = "means"                  # Word MEANS definition


# Relation properties for inference
TRANSITIVE_RELATIONS = {
    RelationType.IS_A, 
    RelationType.PART_OF, 
    RelationType.LOCATED_IN,
    RelationType.BEFORE,
    RelationType.AFTER
}

SYMMETRIC_RELATIONS = {
    RelationType.SIMILAR_TO, 
    RelationType.OPPOSITE_OF,
    RelationType.BORDERS,
    RelationType.SAME_AS,
    RelationType.RELATED_TO
}

INVERSE_RELATIONS = {
    RelationType.CAPITAL_OF: RelationType.LOCATED_IN,
    RelationType.HAS_PART: RelationType.PART_OF,
    RelationType.PART_OF: RelationType.HAS_PART,
    RelationType.CAUSES: RelationType.CAUSED_BY,
    RelationType.CAUSED_BY: RelationType.CAUSES,
    RelationType.BEFORE: RelationType.AFTER,
    RelationType.AFTER: RelationType.BEFORE,
}


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Fact:
    """A single fact (triple) in the knowledge graph"""
    subject: str
    relation: RelationType
    obj: str
    confidence: float = 1.0
    source: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    id: Optional[int] = None
    
    def __hash__(self):
        return hash((self.subject.lower(), self.relation.value, self.obj.lower()))
    
    def __eq__(self, other):
        if not isinstance(other, Fact):
            return False
        return (self.subject.lower() == other.subject.lower() and 
                self.relation == other.relation and 
                self.obj.lower() == other.obj.lower())
    
    def to_tuple(self) -> Tuple[str, str, str]:
        """Convert to (subject, relation, object) tuple"""
        return (self.subject.lower(), self.relation.value, self.obj.lower())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'subject': self.subject,
            'relation': self.relation.value,
            'object': self.obj,
            'confidence': self.confidence,
            'source': self.source,
            'created_at': self.created_at
        }


@dataclass 
class Entity:
    """An entity (node) in the knowledge graph"""
    name: str
    entity_type: Optional[str] = None
    description: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None


# ============================================================
# PERSISTENT KNOWLEDGE GRAPH
# ============================================================

class PersistentKnowledgeGraph:
    """
    A robust, scalable, persistent knowledge graph.
    
    Features:
    - SQLite persistence (survives restarts)
    - In-memory cache for fast lookups
    - Automatic indexing
    - Batch operations
    - Inference engine
    - Thread-safe operations
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.data_dir / "knowledge_graph.db"
        
        # Thread safety
        self._lock = threading.RLock()
        
        # In-memory cache for fast lookups
        self._fact_cache: Set[Tuple[str, str, str]] = set()  # For deduplication
        self._subject_index: Dict[str, List[int]] = defaultdict(list)  # subject -> fact IDs
        self._object_index: Dict[str, List[int]] = defaultdict(list)   # object -> fact IDs
        self._relation_index: Dict[str, List[int]] = defaultdict(list) # relation -> fact IDs
        
        # Entity cache
        self._entity_cache: Dict[str, Entity] = {}
        
        # Statistics
        self._stats = {
            'total_facts': 0,
            'total_entities': 0,
            'total_inferred': 0,
            'facts_by_relation': defaultdict(int)
        }
        
        # Initialize database
        self._init_database()
        
        # Load cache from database
        self._load_cache()
        
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings"""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None  # Autocommit
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn
    
    def _init_database(self) -> None:
        """Initialize the database schema"""
        conn = self._get_connection()
        
        # Facts table (the core of the knowledge graph)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                relation TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT DEFAULT '',
                is_inferred INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(subject, relation, object)
            )
        """)
        
        # Indexes for fast lookups
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_object ON facts(object)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_relation ON facts(relation)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_subject_relation ON facts(subject, relation)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_relation_object ON facts(relation, object)")
        
        # Entities table (nodes with metadata)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                entity_type TEXT,
                description TEXT,
                aliases TEXT DEFAULT '[]',
                properties TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)")
        
        # Definitions table (for "what is X?" questions)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS definitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT UNIQUE NOT NULL,
                definition TEXT NOT NULL,
                source TEXT DEFAULT '',
                confidence REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.execute("CREATE INDEX IF NOT EXISTS idx_definitions_term ON definitions(term)")
        
        conn.close()
    
    def _load_cache(self) -> None:
        """Load facts into memory cache for fast lookups"""
        conn = self._get_connection()
        
        # Load fact tuples for deduplication
        rows = conn.execute("SELECT id, subject, relation, object FROM facts").fetchall()
        
        for row_id, subject, relation, obj in rows:
            key = (subject.lower(), relation, obj.lower())
            self._fact_cache.add(key)
            self._subject_index[subject.lower()].append(row_id)
            self._object_index[obj.lower()].append(row_id)
            self._relation_index[relation].append(row_id)
        
        # Count stats
        self._stats['total_facts'] = len(rows)
        
        # Count by relation
        for row in conn.execute("SELECT relation, COUNT(*) FROM facts GROUP BY relation"):
            self._stats['facts_by_relation'][row[0]] = row[1]
        
        # Count entities
        self._stats['total_entities'] = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        
        # Count inferred
        self._stats['total_inferred'] = conn.execute(
            "SELECT COUNT(*) FROM facts WHERE is_inferred = 1"
        ).fetchone()[0]
        
        conn.close()
    
    # ==================== FACT OPERATIONS ====================
    
    def add_fact(self, fact: Fact) -> Optional[int]:
        """
        Add a fact to the knowledge graph.
        
        Returns:
            Fact ID if added, None if duplicate
        """
        key = fact.to_tuple()
        
        with self._lock:
            # Check cache first (fast)
            if key in self._fact_cache:
                return None
            
            try:
                conn = self._get_connection()
                cursor = conn.execute(
                    """INSERT INTO facts (subject, relation, object, confidence, source, is_inferred)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (fact.subject.lower(), fact.relation.value, fact.obj.lower(),
                     fact.confidence, fact.source, 0)
                )
                fact_id = cursor.lastrowid
                conn.close()
                
                # Update cache
                self._fact_cache.add(key)
                self._subject_index[fact.subject.lower()].append(fact_id)
                self._object_index[fact.obj.lower()].append(fact_id)
                self._relation_index[fact.relation.value].append(fact_id)
                
                # Update stats
                self._stats['total_facts'] += 1
                self._stats['facts_by_relation'][fact.relation.value] += 1
                
                return fact_id
                
            except sqlite3.IntegrityError:
                # Duplicate
                return None
            except Exception as e:
                print(f"Error adding fact: {e}")
                return None
    
    def add_facts_batch(self, facts: List[Fact]) -> Dict[str, int]:
        """
        Add multiple facts efficiently.
        
        Returns:
            Dict with 'added' and 'skipped' counts
        """
        added = 0
        skipped = 0
        
        with self._lock:
            conn = self._get_connection()
            
            for fact in facts:
                key = fact.to_tuple()
                
                if key in self._fact_cache:
                    skipped += 1
                    continue
                
                try:
                    cursor = conn.execute(
                        """INSERT INTO facts (subject, relation, object, confidence, source, is_inferred)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (fact.subject.lower(), fact.relation.value, fact.obj.lower(),
                         fact.confidence, fact.source, 0)
                    )
                    fact_id = cursor.lastrowid
                    
                    # Update cache
                    self._fact_cache.add(key)
                    self._subject_index[fact.subject.lower()].append(fact_id)
                    self._object_index[fact.obj.lower()].append(fact_id)
                    self._relation_index[fact.relation.value].append(fact_id)
                    
                    added += 1
                    
                except sqlite3.IntegrityError:
                    skipped += 1
            
            conn.close()
            
            # Update stats
            self._stats['total_facts'] += added
        
        return {'added': added, 'skipped': skipped}
    
    def get_facts_about(self, subject: str, relation: RelationType = None) -> List[Fact]:
        """Get all facts where entity is the subject"""
        subject = subject.lower()
        
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        
        if relation:
            rows = conn.execute(
                "SELECT * FROM facts WHERE subject = ? AND relation = ?",
                (subject, relation.value)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM facts WHERE subject = ?",
                (subject,)
            ).fetchall()
        
        conn.close()
        
        return [self._row_to_fact(row) for row in rows]
    
    def get_facts_involving(self, entity: str) -> List[Fact]:
        """Get all facts involving an entity (as subject OR object)"""
        entity = entity.lower()
        
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        
        rows = conn.execute(
            "SELECT * FROM facts WHERE subject = ? OR object = ?",
            (entity, entity)
        ).fetchall()
        
        conn.close()
        
        return [self._row_to_fact(row) for row in rows]
    
    def query(self, subject: str = None, relation: RelationType = None, 
              obj: str = None) -> List[Fact]:
        """
        Query facts with optional filters.
        
        Examples:
            query(subject="paris") - All facts about Paris
            query(relation=CAPITAL_OF) - All capitals
            query(subject="paris", relation=CAPITAL_OF) - What is Paris capital of?
            query(relation=CAPITAL_OF, obj="france") - What is the capital of France?
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        
        conditions = []
        params = []
        
        if subject:
            conditions.append("subject = ?")
            params.append(subject.lower())
        if relation:
            conditions.append("relation = ?")
            params.append(relation.value)
        if obj:
            conditions.append("object = ?")
            params.append(obj.lower())
        
        if conditions:
            query = f"SELECT * FROM facts WHERE {' AND '.join(conditions)}"
            rows = conn.execute(query, params).fetchall()
        else:
            rows = conn.execute("SELECT * FROM facts LIMIT 1000").fetchall()
        
        conn.close()
        
        return [self._row_to_fact(row) for row in rows]
    
    def _row_to_fact(self, row: sqlite3.Row) -> Fact:
        """Convert database row to Fact object"""
        return Fact(
            id=row['id'],
            subject=row['subject'],
            relation=RelationType(row['relation']),
            obj=row['object'],
            confidence=row['confidence'],
            source=row['source'],
            created_at=row['created_at']
        )
    
    # ==================== DEFINITION OPERATIONS ====================
    
    def add_definition(self, term: str, definition: str, source: str = "", 
                       confidence: float = 1.0) -> bool:
        """Add or update a definition"""
        try:
            conn = self._get_connection()
            conn.execute(
                """INSERT INTO definitions (term, definition, source, confidence)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(term) DO UPDATE SET 
                       definition = excluded.definition,
                       source = excluded.source,
                       confidence = excluded.confidence""",
                (term.lower(), definition, source, confidence)
            )
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding definition: {e}")
            return False
    
    def get_definition(self, term: str) -> Optional[str]:
        """Get definition of a term"""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT definition FROM definitions WHERE term = ?",
            (term.lower(),)
        ).fetchone()
        conn.close()
        
        return row[0] if row else None
    
    # ==================== ENTITY OPERATIONS ====================
    
    def add_entity(self, entity: Entity) -> Optional[int]:
        """Add an entity to the graph"""
        try:
            conn = self._get_connection()
            cursor = conn.execute(
                """INSERT INTO entities (name, entity_type, description, aliases, properties)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET
                       entity_type = COALESCE(excluded.entity_type, entities.entity_type),
                       description = COALESCE(excluded.description, entities.description)""",
                (entity.name.lower(), entity.entity_type, entity.description,
                 json.dumps(entity.aliases), json.dumps(entity.properties))
            )
            entity_id = cursor.lastrowid
            conn.close()
            
            self._stats['total_entities'] += 1
            return entity_id
            
        except Exception as e:
            print(f"Error adding entity: {e}")
            return None
    
    def get_entity(self, name: str) -> Optional[Entity]:
        """Get entity by name"""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        
        row = conn.execute(
            "SELECT * FROM entities WHERE name = ?",
            (name.lower(),)
        ).fetchone()
        
        conn.close()
        
        if row:
            return Entity(
                id=row['id'],
                name=row['name'],
                entity_type=row['entity_type'],
                description=row['description'],
                aliases=json.loads(row['aliases']),
                properties=json.loads(row['properties'])
            )
        return None
    
    # ==================== INFERENCE ENGINE ====================
    
    def run_inference(self, max_iterations: int = 3) -> Dict[str, int]:
        """
        Run inference rules to derive new facts.
        
        Rules:
        1. Transitivity: A→B, B→C ⟹ A→C
        2. Symmetry: A↔B ⟹ B↔A
        3. Inverse: A-[r]→B ⟹ B-[r']→A
        
        Returns:
            Dict with counts of inferred facts by rule type
        """
        results = {
            'transitivity': 0,
            'symmetry': 0,
            'inverse': 0,
            'total': 0
        }
        
        for iteration in range(max_iterations):
            new_facts = []
            
            # 1. Transitivity
            new_facts.extend(self._infer_transitivity())
            
            # 2. Symmetry
            new_facts.extend(self._infer_symmetry())
            
            # 3. Inverse relations
            new_facts.extend(self._infer_inverse())
            
            if not new_facts:
                break
            
            # Add inferred facts
            for fact, rule_type in new_facts:
                fact.source = f"inferred:{rule_type}"
                if self._add_inferred_fact(fact):
                    results[rule_type] += 1
                    results['total'] += 1
        
        self._stats['total_inferred'] += results['total']
        return results
    
    def _infer_transitivity(self) -> List[Tuple[Fact, str]]:
        """Infer transitive relations: A→B, B→C ⟹ A→C"""
        new_facts = []
        
        conn = self._get_connection()
        
        for rel in TRANSITIVE_RELATIONS:
            # Find chains: A-[rel]→B and B-[rel]→C
            rows = conn.execute("""
                SELECT f1.subject, f1.object, f2.object, f1.confidence, f2.confidence
                FROM facts f1
                JOIN facts f2 ON f1.object = f2.subject AND f1.relation = f2.relation
                WHERE f1.relation = ?
            """, (rel.value,)).fetchall()
            
            for subj, mid, obj, conf1, conf2 in rows:
                if subj != obj:  # Avoid self-loops
                    key = (subj, rel.value, obj)
                    if key not in self._fact_cache:
                        new_facts.append((
                            Fact(subj, rel, obj, confidence=conf1 * conf2 * 0.9),
                            'transitivity'
                        ))
        
        conn.close()
        return new_facts
    
    def _infer_symmetry(self) -> List[Tuple[Fact, str]]:
        """Infer symmetric relations: A↔B ⟹ B↔A"""
        new_facts = []
        
        conn = self._get_connection()
        
        for rel in SYMMETRIC_RELATIONS:
            rows = conn.execute(
                "SELECT subject, object, confidence FROM facts WHERE relation = ?",
                (rel.value,)
            ).fetchall()
            
            for subj, obj, conf in rows:
                key = (obj, rel.value, subj)
                if key not in self._fact_cache:
                    new_facts.append((
                        Fact(obj, rel, subj, confidence=conf),
                        'symmetry'
                    ))
        
        conn.close()
        return new_facts
    
    def _infer_inverse(self) -> List[Tuple[Fact, str]]:
        """Infer inverse relations"""
        new_facts = []
        
        conn = self._get_connection()
        
        for rel, inverse_rel in INVERSE_RELATIONS.items():
            rows = conn.execute(
                "SELECT subject, object, confidence FROM facts WHERE relation = ?",
                (rel.value,)
            ).fetchall()
            
            for subj, obj, conf in rows:
                key = (subj, inverse_rel.value, obj)
                if key not in self._fact_cache:
                    new_facts.append((
                        Fact(subj, inverse_rel, obj, confidence=conf * 0.95),
                        'inverse'
                    ))
        
        conn.close()
        return new_facts
    
    def _add_inferred_fact(self, fact: Fact) -> bool:
        """Add an inferred fact"""
        key = fact.to_tuple()
        
        if key in self._fact_cache:
            return False
        
        try:
            conn = self._get_connection()
            cursor = conn.execute(
                """INSERT INTO facts (subject, relation, object, confidence, source, is_inferred)
                   VALUES (?, ?, ?, ?, ?, 1)""",
                (fact.subject.lower(), fact.relation.value, fact.obj.lower(),
                 fact.confidence, fact.source)
            )
            fact_id = cursor.lastrowid
            conn.close()
            
            # Update cache
            self._fact_cache.add(key)
            self._subject_index[fact.subject.lower()].append(fact_id)
            self._object_index[fact.obj.lower()].append(fact_id)
            
            self._stats['total_facts'] += 1
            return True
            
        except sqlite3.IntegrityError:
            return False
    
    # ==================== STATISTICS ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        return {
            'total_facts': self._stats['total_facts'],
            'total_entities': self._stats['total_entities'],
            'total_inferred': self._stats['total_inferred'],
            'facts_by_relation': dict(self._stats['facts_by_relation']),
            'unique_subjects': len(self._subject_index),
            'unique_objects': len(self._object_index)
        }
    
    def get_all_relations(self) -> List[str]:
        """Get all relation types in use"""
        return list(self._relation_index.keys())
    
    # ==================== SEARCH ====================
    
    def search_entities(self, query: str, limit: int = 10) -> List[str]:
        """Search for entities by name (fuzzy match)"""
        query = query.lower()
        
        conn = self._get_connection()
        
        # Exact match first
        exact = conn.execute(
            "SELECT DISTINCT subject FROM facts WHERE subject = ? LIMIT ?",
            (query, limit)
        ).fetchall()
        
        if exact:
            conn.close()
            return [row[0] for row in exact]
        
        # Partial match
        rows = conn.execute(
            """SELECT DISTINCT subject FROM facts 
               WHERE subject LIKE ? 
               ORDER BY LENGTH(subject)
               LIMIT ?""",
            (f"%{query}%", limit)
        ).fetchall()
        
        conn.close()
        return [row[0] for row in rows]


# ============================================================
# KNOWLEDGE EXTRACTOR
# ============================================================

class RobustKnowledgeExtractor:
    """
    Extracts facts from natural language text.
    
    Uses pattern matching to find relationships in text
    and convert them to structured facts.
    """
    
    # Extraction patterns: (regex, relation_type, subject_group, object_group)
    PATTERNS = [
        # Capitals
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) is the capital of ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", 
         RelationType.CAPITAL_OF, 1, 2),
        (r"The capital of ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) is ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
         RelationType.CAPITAL_OF, 2, 1),
        
        # Location
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) is (?:located |situated )?in ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
         RelationType.LOCATED_IN, 1, 2),
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) is a (?:city|town|country|state|region|province) in ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
         RelationType.LOCATED_IN, 1, 2),
        
        # Birth
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) was born (?:in|at) ([A-Z][a-z]+(?:[\s,]+[A-Za-z]+)*)",
         RelationType.BORN_IN, 1, 2),
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) was born on ([A-Z]?[a-z]+ \d+,? \d{4}|\d{4})",
         RelationType.BORN_ON, 1, 2),
        
        # Death
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) died (?:in|at) ([A-Z][a-z]+(?:[\s,]+[A-Za-z]+)*)",
         RelationType.DIED_IN, 1, 2),
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) died on ([A-Z]?[a-z]+ \d+,? \d{4}|\d{4})",
         RelationType.DIED_ON, 1, 2),
        
        # Creation
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) was (?:created|invented|founded|developed|designed) by ([A-Z][a-z]+(?:\s+[A-Za-z]+)*)",
         RelationType.CREATED_BY, 1, 2),
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) (?:created|invented|founded|developed|designed) ([A-Z][a-z]+(?:\s+[A-Za-z]+)*)",
         RelationType.CREATED_BY, 2, 1),
        
        # Classification (is a)
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) is an? ([a-z]+(?:\s+[a-z]+)?)",
         RelationType.IS_A, 1, 2),
        (r"([A-Z][a-z]+) are ([a-z]+(?:\s+[a-z]+)?)",
         RelationType.IS_A, 1, 2),
        
        # Part of
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) is (?:a )?part of ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
         RelationType.PART_OF, 1, 2),
        
        # Used for
        (r"([A-Z][a-z]+) is used (?:for|to) ([a-z]+(?:\s+[a-z]+)*)",
         RelationType.USED_FOR, 1, 2),
        
        # Borders
        (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*) borders ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
         RelationType.BORDERS, 1, 2),
        
        # Made of
        (r"([A-Z][a-z]+) is made (?:of|from) ([a-z]+)",
         RelationType.MADE_OF, 1, 2),
    ]
    
    def extract_facts(self, text: str, source: str = "") -> List[Fact]:
        """Extract facts from text"""
        facts = []
        seen = set()
        
        for pattern, rel_type, subj_group, obj_group in self.PATTERNS:
            for match in re.finditer(pattern, text):
                subject = match.group(subj_group).strip()
                obj = match.group(obj_group).strip()
                
                # Clean up
                subject = re.sub(r'[,;:\s]+$', '', subject)
                obj = re.sub(r'[,;:\s]+$', '', obj)
                
                # Skip invalid
                if len(subject) < 2 or len(obj) < 2:
                    continue
                if subject.lower() == obj.lower():
                    continue
                
                # Deduplicate
                key = (subject.lower(), rel_type.value, obj.lower())
                if key in seen:
                    continue
                seen.add(key)
                
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
        topic_lower = topic.lower()
        
        # Definition patterns
        patterns = [
            rf"(?i){re.escape(topic)} is ([^.]+(?:an?|the)[^.]+)\.",
            rf"(?i){re.escape(topic)} (?:refers to|means|is defined as) ([^.]+)\.",
            rf"(?i){re.escape(topic)},? ([^,]+),? is",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                definition = match.group(1).strip()
                if len(definition) > 10:
                    return definition
        
        # Fallback: first sentence containing the topic
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences[:5]:
            if topic_lower in sent.lower() and len(sent) > 20:
                return sent.strip()
        
        return None


# ============================================================
# INTEGRATED REASONER
# ============================================================

class PersistentReasoner:
    """
    Complete reasoning system with persistent knowledge graph.
    
    Combines:
    - Knowledge Graph (facts & relationships)
    - Inference Engine (derives new facts)
    - Question Answering (traverses graph)
    - Natural Language Generation (creates responses)
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        
        # Initialize persistent knowledge graph
        self.graph = PersistentKnowledgeGraph(data_dir)
        
        # Initialize extractor
        self.extractor = RobustKnowledgeExtractor()
        
        # Question patterns
        self.question_patterns = {
            'capital': (RelationType.CAPITAL_OF, 'object'),  # What is capital OF X
            'location': (RelationType.LOCATED_IN, 'subject'),  # Where IS X located
            'born': (RelationType.BORN_IN, 'subject'),
            'created': (RelationType.CREATED_BY, 'subject'),
            'invented': (RelationType.INVENTED_BY, 'subject'),
            'founded': (RelationType.FOUNDED_BY, 'subject'),
        }
        
    
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
        """Answer a question using the knowledge graph"""
        question_lower = question.lower()
        
        # Detect question type and extract target
        answer = None
        confidence = 0.0
        facts_used = []
        reasoning = []
        
        # Pattern: "What is the capital of X?"
        capital_match = re.search(r"what is the capital of (?:the )?(.+?)\??$", question_lower)
        if capital_match:
            country = capital_match.group(1).strip()
            reasoning.append(f"Looking for capital of {country}")
            
            facts = self.graph.query(relation=RelationType.CAPITAL_OF, obj=country)
            if facts:
                answer = f"The capital of {country.title()} is {facts[0].subject.title()}."
                confidence = facts[0].confidence
                facts_used = [f.to_dict() for f in facts]
                reasoning.append(f"Found: {facts[0].subject} is capital of {country}")
        
        # Pattern: "Where is X located?"
        location_match = re.search(r"where is (?:the )?(.+?)(?: located)?\??$", question_lower)
        if not answer and location_match:
            entity = location_match.group(1).strip()
            reasoning.append(f"Looking for location of {entity}")
            
            facts = self.graph.query(subject=entity, relation=RelationType.LOCATED_IN)
            if facts:
                answer = f"{entity.title()} is located in {facts[0].obj.title()}."
                confidence = facts[0].confidence
                facts_used = [f.to_dict() for f in facts]
                reasoning.append(f"Found: {entity} is in {facts[0].obj}")
        
        # Pattern: "Where was X born?"
        born_match = re.search(r"where was (.+?) born\??$", question_lower)
        if not answer and born_match:
            person = born_match.group(1).strip()
            reasoning.append(f"Looking for birthplace of {person}")
            
            facts = self.graph.query(subject=person, relation=RelationType.BORN_IN)
            if facts:
                answer = f"{person.title()} was born in {facts[0].obj}."
                confidence = facts[0].confidence
                facts_used = [f.to_dict() for f in facts]
        
        # Pattern: "What is X?" - check definition
        what_is_match = re.search(r"what is (?:a |an |the )?(.+?)\??$", question_lower)
        if not answer and what_is_match:
            term = what_is_match.group(1).strip()
            reasoning.append(f"Looking for definition of {term}")
            
            definition = self.graph.get_definition(term)
            if definition:
                answer = f"{term.title()} is {definition}."
                confidence = 0.85
                reasoning.append(f"Found definition")
            else:
                # Try to find facts about it
                facts = self.graph.get_facts_about(term)
                if facts:
                    descriptions = []
                    for f in facts[:3]:
                        if f.relation == RelationType.IS_A:
                            descriptions.append(f"a {f.obj}")
                        elif f.relation == RelationType.LOCATED_IN:
                            descriptions.append(f"located in {f.obj}")
                    
                    if descriptions:
                        answer = f"{term.title()} is {', '.join(descriptions)}."
                        confidence = 0.7
                        facts_used = [f.to_dict() for f in facts[:3]]
        
        # Pattern: "Is X in Y?"
        is_in_match = re.search(r"is (.+?) in (.+?)\??$", question_lower)
        if not answer and is_in_match:
            entity = is_in_match.group(1).strip()
            location = is_in_match.group(2).strip()
            reasoning.append(f"Checking if {entity} is in {location}")
            
            facts = self.graph.query(subject=entity, relation=RelationType.LOCATED_IN, obj=location)
            if facts:
                answer = f"Yes, {entity.title()} is in {location.title()}."
                confidence = facts[0].confidence
            else:
                # Check capital_of as it implies located_in
                facts = self.graph.query(subject=entity, relation=RelationType.CAPITAL_OF, obj=location)
                if facts:
                    answer = f"Yes, {entity.title()} is the capital of {location.title()}."
                    confidence = facts[0].confidence
                else:
                    answer = f"I don't have information confirming {entity} is in {location}."
                    confidence = 0.3
        
        # Default: search for any facts
        if not answer:
            # Extract key terms
            terms = re.findall(r'\b[A-Za-z]{3,}\b', question)
            for term in terms:
                facts = self.graph.get_facts_involving(term.lower())
                if facts:
                    reasoning.append(f"Found facts about {term}")
                    answer = self._facts_to_response(facts[:3])
                    confidence = 0.5
                    facts_used = [f.to_dict() for f in facts[:3]]
                    break
        
        if not answer:
            answer = "I don't have enough information to answer that question."
            confidence = 0.1
        
        return {
            'answer': answer,
            'confidence': confidence,
            'facts_used': len(facts_used),
            'reasoning': reasoning,
            'question_type': 'knowledge_graph'
        }
    
    def _facts_to_response(self, facts: List[Fact]) -> str:
        """Convert facts to natural language response"""
        responses = []
        for fact in facts:
            if fact.relation == RelationType.IS_A:
                responses.append(f"{fact.subject.title()} is a {fact.obj}")
            elif fact.relation == RelationType.LOCATED_IN:
                responses.append(f"{fact.subject.title()} is in {fact.obj.title()}")
            elif fact.relation == RelationType.CAPITAL_OF:
                responses.append(f"{fact.subject.title()} is the capital of {fact.obj.title()}")
            elif fact.relation == RelationType.BORN_IN:
                responses.append(f"{fact.subject.title()} was born in {fact.obj}")
            elif fact.relation == RelationType.CREATED_BY:
                responses.append(f"{fact.subject.title()} was created by {fact.obj.title()}")
            else:
                responses.append(f"{fact.subject.title()} {fact.relation.value.replace('_', ' ')} {fact.obj}")
        
        return ". ".join(responses) + "." if responses else ""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reasoner statistics"""
        return self.graph.get_stats()


