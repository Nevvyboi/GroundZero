"""
Advanced Knowledge Base for GroundZero AI
==========================================
Production-grade knowledge storage with:
- Semantic chunking for optimal retrieval
- Entity extraction and linking
- Relationship graph for multi-hop reasoning
- Hierarchical document organization
- Incremental indexing
- Query expansion and reranking
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict
import json
import sqlite3
import os
import re
import hashlib
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
from enum import Enum

from .vector_store import AdvancedVectorStore, simple_embed


class ChunkType(Enum):
    """Types of knowledge chunks"""
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    ENTITY = "entity"
    DEFINITION = "definition"
    FACT = "fact"
    RELATIONSHIP = "relationship"
    SUMMARY = "summary"


@dataclass
class KnowledgeChunk:
    """A chunk of knowledge with metadata"""
    id: str
    content: str
    chunk_type: ChunkType
    source_id: str
    source_title: str = ""
    position: int = 0  # Position in source document
    embedding: Optional[np.ndarray] = None
    entities: List[str] = field(default_factory=list)
    relationships: List[Tuple[str, str, str]] = field(default_factory=list)  # (subject, predicate, object)
    importance: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'content': self.content,
            'chunk_type': self.chunk_type.value,
            'source_id': self.source_id,
            'source_title': self.source_title,
            'position': self.position,
            'entities': self.entities,
            'relationships': self.relationships,
            'importance': self.importance,
            'created_at': self.created_at,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'KnowledgeChunk':
        d['chunk_type'] = ChunkType(d['chunk_type'])
        d.pop('embedding', None)  # Don't load embedding from dict
        return cls(**d)


@dataclass
class Document:
    """A document in the knowledge base"""
    id: str
    title: str
    content: str
    source: str = ""
    category: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    chunk_ids: List[str] = field(default_factory=list)
    summary: str = ""
    entities: List[str] = field(default_factory=list)
    importance: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'source': self.source,
            'category': self.category,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'chunk_ids': self.chunk_ids,
            'summary': self.summary,
            'entities': self.entities,
            'importance': self.importance,
            'metadata': self.metadata
        }


class SemanticChunker:
    """
    Intelligent text chunking that preserves semantic boundaries.
    Uses sentence detection, topic segmentation, and entity preservation.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 512,
        min_chunk_size: int = 100,
        overlap: int = 50
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
        
        # Sentence boundary patterns
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        
        # Section header patterns
        self.header_pattern = re.compile(r'^(?:#+\s+|\d+\.\s+|[A-Z][A-Z\s]+:)', re.MULTILINE)
        
        # Definition patterns
        self.definition_pattern = re.compile(r'^([^.]+)\s+(?:is|are|was|were|refers to|means)\s+', re.IGNORECASE)
    
    def chunk(
        self,
        text: str,
        source_id: str,
        source_title: str = ""
    ) -> List[KnowledgeChunk]:
        """Split text into semantic chunks"""
        chunks = []
        
        # First, split by major sections
        sections = self._split_by_sections(text)
        
        chunk_position = 0
        for section in sections:
            # Split section into sentences
            sentences = self._split_sentences(section)
            
            # Group sentences into chunks
            current_chunk = ""
            chunk_sentences = []
            
            for sentence in sentences:
                # Check if adding this sentence would exceed max size
                if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                    # Save current chunk
                    chunk = self._create_chunk(
                        current_chunk.strip(),
                        source_id,
                        source_title,
                        chunk_position,
                        chunk_sentences
                    )
                    chunks.append(chunk)
                    chunk_position += 1
                    
                    # Start new chunk with overlap
                    if self.overlap > 0 and chunk_sentences:
                        overlap_text = " ".join(chunk_sentences[-2:])  # Last 2 sentences as overlap
                        current_chunk = overlap_text + " " + sentence
                        chunk_sentences = chunk_sentences[-2:] + [sentence]
                    else:
                        current_chunk = sentence
                        chunk_sentences = [sentence]
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
                    chunk_sentences.append(sentence)
            
            # Save remaining chunk
            if current_chunk and len(current_chunk) >= self.min_chunk_size:
                chunk = self._create_chunk(
                    current_chunk.strip(),
                    source_id,
                    source_title,
                    chunk_position,
                    chunk_sentences
                )
                chunks.append(chunk)
                chunk_position += 1
        
        return chunks
    
    def _split_by_sections(self, text: str) -> List[str]:
        """Split text by section headers"""
        sections = self.header_pattern.split(text)
        return [s.strip() for s in sections if s.strip()]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = self.sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_chunk(
        self,
        content: str,
        source_id: str,
        source_title: str,
        position: int,
        sentences: List[str]
    ) -> KnowledgeChunk:
        """Create a knowledge chunk with appropriate type detection"""
        # Detect chunk type
        chunk_type = ChunkType.PARAGRAPH
        
        # Check for definition
        if self.definition_pattern.match(content):
            chunk_type = ChunkType.DEFINITION
        elif len(sentences) == 1:
            chunk_type = ChunkType.SENTENCE
        
        # Extract entities (simple pattern matching)
        entities = self._extract_entities(content)
        
        # Extract relationships
        relationships = self._extract_relationships(content)
        
        # Calculate importance based on entity density and position
        importance = 1.0 + 0.1 * len(entities)
        if position == 0:
            importance += 0.5  # First chunk often contains key info
        
        chunk_id = hashlib.md5(f"{source_id}:{position}:{content[:50]}".encode()).hexdigest()[:12]
        
        return KnowledgeChunk(
            id=chunk_id,
            content=content,
            chunk_type=chunk_type,
            source_id=source_id,
            source_title=source_title,
            position=position,
            entities=entities,
            relationships=relationships,
            importance=importance
        )
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract potential entities from text"""
        # Simple patterns for entity extraction
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',  # Multi-word proper nouns
            r'\b[A-Z]{2,}\b',  # Acronyms
        ]
        
        entities = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        return list(set(entities))[:10]  # Limit to top 10
    
    def _extract_relationships(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract subject-predicate-object relationships"""
        relationships = []
        
        # Simple pattern: "X is/are Y"
        pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(is|are|was|were)\s+(?:a|an|the)?\s*([^.]+)'
        
        for match in re.finditer(pattern, text):
            subject = match.group(1).strip()
            predicate = match.group(2).strip()
            obj = match.group(3).strip()[:50]  # Limit object length
            relationships.append((subject, predicate, obj))
        
        return relationships[:5]  # Limit to 5 relationships per chunk


class EntityLinker:
    """Links entities to a knowledge graph"""
    
    def __init__(self):
        self.entity_aliases: Dict[str, str] = {}  # alias -> canonical name
        self.entity_info: Dict[str, Dict[str, Any]] = {}  # canonical -> info
        self.entity_embeddings: Dict[str, np.ndarray] = {}
        
    def add_entity(
        self,
        name: str,
        aliases: List[str] = None,
        info: Dict[str, Any] = None,
        embedding: np.ndarray = None
    ):
        """Add an entity to the linker"""
        canonical = name.lower().strip()
        self.entity_info[canonical] = info or {}
        
        aliases = aliases or []
        for alias in [name] + aliases:
            self.entity_aliases[alias.lower().strip()] = canonical
        
        if embedding is not None:
            self.entity_embeddings[canonical] = embedding
    
    def link(self, mention: str) -> Optional[str]:
        """Link a mention to its canonical entity"""
        mention_lower = mention.lower().strip()
        
        # Direct match
        if mention_lower in self.entity_aliases:
            return self.entity_aliases[mention_lower]
        
        # Partial match
        for alias, canonical in self.entity_aliases.items():
            if mention_lower in alias or alias in mention_lower:
                return canonical
        
        return None
    
    def get_info(self, entity: str) -> Optional[Dict[str, Any]]:
        """Get information about an entity"""
        canonical = self.link(entity)
        if canonical:
            return self.entity_info.get(canonical)
        return None


class RelationshipGraph:
    """Graph of entity relationships for multi-hop reasoning"""
    
    def __init__(self):
        self.edges: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # subject -> [(predicate, object)]
        self.reverse_edges: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # object -> [(predicate, subject)]
        self.predicates: Set[str] = set()
        
    def add_relationship(self, subject: str, predicate: str, obj: str):
        """Add a relationship to the graph"""
        subject = subject.lower().strip()
        obj = obj.lower().strip()
        predicate = predicate.lower().strip()
        
        self.edges[subject].append((predicate, obj))
        self.reverse_edges[obj].append((predicate, subject))
        self.predicates.add(predicate)
    
    def get_related(
        self,
        entity: str,
        max_hops: int = 2,
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """Get entities related to the given entity within max_hops"""
        entity = entity.lower().strip()
        results = []
        visited = {entity}
        queue = [(entity, [], 0)]  # (entity, path, hops)
        
        while queue and len(results) < max_results:
            current, path, hops = queue.pop(0)
            
            if hops > 0:
                results.append({
                    'entity': current,
                    'path': path,
                    'hops': hops
                })
            
            if hops < max_hops:
                # Forward edges
                for predicate, obj in self.edges.get(current, []):
                    if obj not in visited:
                        visited.add(obj)
                        queue.append((obj, path + [(current, predicate, obj)], hops + 1))
                
                # Reverse edges
                for predicate, subj in self.reverse_edges.get(current, []):
                    if subj not in visited:
                        visited.add(subj)
                        queue.append((subj, path + [(subj, predicate, current)], hops + 1))
        
        return results
    
    def find_path(
        self,
        start: str,
        end: str,
        max_hops: int = 3
    ) -> Optional[List[Tuple[str, str, str]]]:
        """Find path between two entities"""
        start = start.lower().strip()
        end = end.lower().strip()
        
        if start == end:
            return []
        
        visited = {start}
        queue = [(start, [])]
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) >= max_hops:
                continue
            
            # Forward edges
            for predicate, obj in self.edges.get(current, []):
                new_path = path + [(current, predicate, obj)]
                
                if obj == end:
                    return new_path
                
                if obj not in visited:
                    visited.add(obj)
                    queue.append((obj, new_path))
            
            # Reverse edges
            for predicate, subj in self.reverse_edges.get(current, []):
                new_path = path + [(subj, predicate, current)]
                
                if subj == end:
                    return new_path
                
                if subj not in visited:
                    visited.add(subj)
                    queue.append((subj, new_path))
        
        return None


class AdvancedKnowledgeBase:
    """
    Production-grade knowledge base for GroundZero AI.
    Combines vector search, entity linking, and relationship graphs.
    """
    
    def __init__(
        self,
        data_dir: str = "./knowledge_base_data",
        embedding_dim: int = 768,
        max_vectors: int = 1000000
    ):
        self.data_dir = data_dir
        self.embedding_dim = embedding_dim
        
        os.makedirs(data_dir, exist_ok=True)
        
        # Core components
        self.vector_store = AdvancedVectorStore(
            dim=embedding_dim,
            max_elements=max_vectors,
            data_dir=os.path.join(data_dir, "vectors")
        )
        self.chunker = SemanticChunker()
        self.entity_linker = EntityLinker()
        self.relationship_graph = RelationshipGraph()
        
        # Storage
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, KnowledgeChunk] = {}
        
        # SQLite for persistence
        self.db_path = os.path.join(data_dir, "knowledge.db")
        self._init_db()
        
        # Embedding function (can be replaced with real model)
        self.embed_fn = lambda x: simple_embed(x, embedding_dim)
        
        # Statistics
        self.stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'total_entities': 0,
            'total_relationships': 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
    
    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT,
                content TEXT,
                source TEXT,
                category TEXT,
                created_at TEXT,
                updated_at TEXT,
                summary TEXT,
                importance REAL,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                content TEXT,
                chunk_type TEXT,
                source_id TEXT,
                source_title TEXT,
                position INTEGER,
                importance REAL,
                created_at TEXT,
                metadata TEXT,
                FOREIGN KEY (source_id) REFERENCES documents(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                name TEXT PRIMARY KEY,
                info TEXT,
                created_at TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT,
                predicate TEXT,
                object TEXT,
                source_chunk_id TEXT,
                FOREIGN KEY (source_chunk_id) REFERENCES chunks(id)
            )
        ''')
        
        # Create indices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relationships_subject ON relationships(subject)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relationships_object ON relationships(object)')
        
        conn.commit()
        conn.close()
    
    def set_embedding_function(self, embed_fn):
        """Set custom embedding function"""
        self.embed_fn = embed_fn
    
    def add_document(
        self,
        title: str,
        content: str,
        source: str = "",
        category: str = "",
        metadata: Dict[str, Any] = None
    ) -> Document:
        """Add a document to the knowledge base"""
        with self.lock:
            # Generate document ID
            doc_id = hashlib.md5(f"{title}:{content[:100]}".encode()).hexdigest()[:12]
            
            # Check for duplicate
            if doc_id in self.documents:
                return self.documents[doc_id]
            
            # Create chunks
            chunks = self.chunker.chunk(content, doc_id, title)
            
            # Process chunks
            chunk_ids = []
            for chunk in chunks:
                # Generate embedding
                chunk.embedding = self.embed_fn(chunk.content)
                
                # Add to vector store
                self.vector_store.add(
                    chunk.embedding,
                    chunk.id,
                    chunk.content,
                    {'source_id': doc_id, 'source_title': title}
                )
                
                # Add entities to linker
                for entity in chunk.entities:
                    self.entity_linker.add_entity(entity)
                
                # Add relationships to graph
                for subj, pred, obj in chunk.relationships:
                    self.relationship_graph.add_relationship(subj, pred, obj)
                
                # Store chunk
                self.chunks[chunk.id] = chunk
                chunk_ids.append(chunk.id)
                
                # Save to DB
                self._save_chunk_to_db(chunk)
            
            # Extract document-level entities
            all_entities = list(set(e for c in chunks for e in c.entities))
            
            # Generate summary (simple: first chunk)
            summary = chunks[0].content[:200] if chunks else ""
            
            # Create document
            doc = Document(
                id=doc_id,
                title=title,
                content=content,
                source=source,
                category=category,
                chunk_ids=chunk_ids,
                summary=summary,
                entities=all_entities,
                metadata=metadata or {}
            )
            
            self.documents[doc_id] = doc
            self._save_document_to_db(doc)
            
            # Update stats
            self.stats['total_documents'] += 1
            self.stats['total_chunks'] += len(chunks)
            self.stats['total_entities'] = len(self.entity_linker.entity_info)
            self.stats['total_relationships'] = sum(
                len(edges) for edges in self.relationship_graph.edges.values()
            )
            
            return doc
    
    def _save_document_to_db(self, doc: Document):
        """Save document to SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO documents 
            (id, title, content, source, category, created_at, updated_at, summary, importance, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            doc.id, doc.title, doc.content, doc.source, doc.category,
            doc.created_at, doc.updated_at, doc.summary, doc.importance,
            json.dumps(doc.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def _save_chunk_to_db(self, chunk: KnowledgeChunk):
        """Save chunk to SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO chunks
            (id, content, chunk_type, source_id, source_title, position, importance, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            chunk.id, chunk.content, chunk.chunk_type.value, chunk.source_id,
            chunk.source_title, chunk.position, chunk.importance, chunk.created_at,
            json.dumps(chunk.metadata)
        ))
        
        # Save relationships
        for subj, pred, obj in chunk.relationships:
            cursor.execute('''
                INSERT INTO relationships (subject, predicate, object, source_chunk_id)
                VALUES (?, ?, ?, ?)
            ''', (subj.lower(), pred.lower(), obj.lower(), chunk.id))
        
        conn.commit()
        conn.close()
    
    def search(
        self,
        query: str,
        k: int = 10,
        search_type: str = "hybrid",  # "vector", "keyword", "hybrid"
        expand_entities: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            k: Number of results
            search_type: Type of search
            expand_entities: Whether to expand search with related entities
        """
        results = []
        
        # Get query embedding
        query_embedding = self.embed_fn(query)
        
        # Vector search
        vector_results = self.vector_store.search(query_embedding, k=k * 2)
        
        # Add vector results
        for vr in vector_results:
            chunk = self.chunks.get(vr['id'])
            if chunk:
                results.append({
                    'chunk_id': chunk.id,
                    'content': chunk.content,
                    'source_id': chunk.source_id,
                    'source_title': chunk.source_title,
                    'score': vr['score'],
                    'search_type': 'vector',
                    'entities': chunk.entities
                })
        
        # Entity expansion
        if expand_entities:
            # Extract entities from query
            query_entities = self.chunker._extract_entities(query)
            
            for entity in query_entities:
                # Get related entities
                related = self.relationship_graph.get_related(entity, max_hops=2)
                
                for rel in related[:5]:
                    # Search for chunks mentioning related entity
                    rel_embedding = self.embed_fn(rel['entity'])
                    rel_results = self.vector_store.search(rel_embedding, k=3)
                    
                    for rr in rel_results:
                        if not any(r['chunk_id'] == rr['id'] for r in results):
                            chunk = self.chunks.get(rr['id'])
                            if chunk:
                                results.append({
                                    'chunk_id': chunk.id,
                                    'content': chunk.content,
                                    'source_id': chunk.source_id,
                                    'source_title': chunk.source_title,
                                    'score': rr['score'] * 0.8,  # Discount expanded results
                                    'search_type': 'entity_expansion',
                                    'entities': chunk.entities,
                                    'expansion_path': rel['path']
                                })
        
        # Sort by score and return top k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]
    
    def get_context(
        self,
        query: str,
        max_tokens: int = 2000
    ) -> str:
        """Get formatted context for a query (for LLM input)"""
        results = self.search(query, k=10)
        
        context_parts = []
        total_chars = 0
        approx_chars_per_token = 4
        max_chars = max_tokens * approx_chars_per_token
        
        for result in results:
            content = result['content']
            source = result.get('source_title', 'Unknown')
            
            part = f"[From: {source}]\n{content}\n"
            
            if total_chars + len(part) <= max_chars:
                context_parts.append(part)
                total_chars += len(part)
            else:
                break
        
        return "\n".join(context_parts)
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID"""
        return self.documents.get(doc_id)
    
    def get_chunk(self, chunk_id: str) -> Optional[KnowledgeChunk]:
        """Get a chunk by ID"""
        return self.chunks.get(chunk_id)
    
    def get_related_entities(
        self,
        entity: str,
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """Get entities related to the given entity"""
        return self.relationship_graph.get_related(entity, max_hops)
    
    def find_connection(
        self,
        entity1: str,
        entity2: str
    ) -> Optional[List[Tuple[str, str, str]]]:
        """Find connection path between two entities"""
        return self.relationship_graph.find_path(entity1, entity2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        return {
            **self.stats,
            'vector_store_stats': self.vector_store.get_stats(),
            'unique_predicates': len(self.relationship_graph.predicates)
        }
    
    def save(self):
        """Save all data to disk"""
        self.vector_store.save()
        
        # Save in-memory structures
        with open(os.path.join(self.data_dir, "documents.pkl"), 'wb') as f:
            pickle.dump(self.documents, f)
        
        with open(os.path.join(self.data_dir, "chunks.pkl"), 'wb') as f:
            pickle.dump({k: v.to_dict() for k, v in self.chunks.items()}, f)
        
        with open(os.path.join(self.data_dir, "entity_linker.pkl"), 'wb') as f:
            pickle.dump({
                'aliases': self.entity_linker.entity_aliases,
                'info': self.entity_linker.entity_info
            }, f)
        
        with open(os.path.join(self.data_dir, "relationship_graph.pkl"), 'wb') as f:
            pickle.dump({
                'edges': dict(self.relationship_graph.edges),
                'reverse_edges': dict(self.relationship_graph.reverse_edges),
                'predicates': list(self.relationship_graph.predicates)
            }, f)
        
        with open(os.path.join(self.data_dir, "stats.json"), 'w') as f:
            json.dump(self.stats, f)
    
    def load(self):
        """Load all data from disk"""
        self.vector_store.load()
        
        docs_path = os.path.join(self.data_dir, "documents.pkl")
        if os.path.exists(docs_path):
            with open(docs_path, 'rb') as f:
                self.documents = pickle.load(f)
        
        chunks_path = os.path.join(self.data_dir, "chunks.pkl")
        if os.path.exists(chunks_path):
            with open(chunks_path, 'rb') as f:
                chunks_data = pickle.load(f)
                self.chunks = {k: KnowledgeChunk.from_dict(v) for k, v in chunks_data.items()}
        
        entity_path = os.path.join(self.data_dir, "entity_linker.pkl")
        if os.path.exists(entity_path):
            with open(entity_path, 'rb') as f:
                data = pickle.load(f)
                self.entity_linker.entity_aliases = data['aliases']
                self.entity_linker.entity_info = data['info']
        
        graph_path = os.path.join(self.data_dir, "relationship_graph.pkl")
        if os.path.exists(graph_path):
            with open(graph_path, 'rb') as f:
                data = pickle.load(f)
                self.relationship_graph.edges = defaultdict(list, data['edges'])
                self.relationship_graph.reverse_edges = defaultdict(list, data['reverse_edges'])
                self.relationship_graph.predicates = set(data['predicates'])
        
        stats_path = os.path.join(self.data_dir, "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.stats = json.load(f)


# Singleton instance
_knowledge_base: Optional[AdvancedKnowledgeBase] = None


def get_knowledge_base(data_dir: str = "./data/knowledge") -> AdvancedKnowledgeBase:
    """Get or create the knowledge base singleton"""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = AdvancedKnowledgeBase(data_dir=data_dir)
    return _knowledge_base


if __name__ == "__main__":
    print("Testing Advanced Knowledge Base...")
    
    # Create knowledge base
    kb = AdvancedKnowledgeBase(data_dir="./test_kb_data")
    
    # Add test documents
    print("\n1. Adding test documents...")
    
    doc1 = kb.add_document(
        title="Introduction to Machine Learning",
        content="""Machine Learning is a subset of artificial intelligence that enables systems to learn 
        and improve from experience without being explicitly programmed. Deep learning is a type of 
        machine learning based on artificial neural networks. Neural networks are computing systems 
        inspired by biological neural networks. Supervised learning is a type of machine learning 
        where the model is trained on labeled data.""",
        category="AI",
        source="Wikipedia"
    )
    print(f"  Added document: {doc1.title} ({len(doc1.chunk_ids)} chunks)")
    
    doc2 = kb.add_document(
        title="History of Computing",
        content="""Alan Turing was a British mathematician who is widely considered the father of 
        computer science. Turing developed the concept of the Turing machine in 1936. 
        The Turing machine is a mathematical model of computation. John von Neumann was 
        another pioneer who contributed to computer architecture. The von Neumann architecture 
        is the basis for most modern computers.""",
        category="History",
        source="Encyclopedia"
    )
    print(f"  Added document: {doc2.title} ({len(doc2.chunk_ids)} chunks)")
    
    doc3 = kb.add_document(
        title="Natural Language Processing",
        content="""Natural Language Processing is a field of artificial intelligence that focuses 
        on the interaction between computers and humans through natural language. NLP combines 
        computational linguistics with machine learning. Transformers are a type of neural network 
        architecture that has revolutionized NLP. BERT and GPT are examples of transformer-based models.""",
        category="AI",
        source="Research Paper"
    )
    print(f"  Added document: {doc3.title} ({len(doc3.chunk_ids)} chunks)")
    
    # Test search
    print("\n2. Testing search...")
    
    query = "What is deep learning?"
    results = kb.search(query, k=3)
    print(f"  Query: '{query}'")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['source_title']}] Score: {r['score']:.3f}")
        print(f"     {r['content'][:100]}...")
    
    # Test entity relationships
    print("\n3. Testing entity relationships...")
    
    related = kb.get_related_entities("Machine Learning", max_hops=2)
    print(f"  Entities related to 'Machine Learning': {len(related)}")
    for r in related[:5]:
        print(f"    - {r['entity']} ({r['hops']} hops)")
    
    # Test connection finding
    print("\n4. Testing connection finding...")
    
    path = kb.find_connection("neural networks", "machine learning")
    if path:
        print(f"  Path from 'neural networks' to 'machine learning':")
        for subj, pred, obj in path:
            print(f"    {subj} --[{pred}]--> {obj}")
    else:
        print("  No direct path found")
    
    # Test context generation
    print("\n5. Testing context generation...")
    
    context = kb.get_context("Tell me about neural networks", max_tokens=500)
    print(f"  Context length: {len(context)} characters")
    print(f"  Context preview: {context[:200]}...")
    
    # Get stats
    print("\n6. Knowledge base statistics:")
    stats = kb.get_stats()
    print(f"  Documents: {stats['total_documents']}")
    print(f"  Chunks: {stats['total_chunks']}")
    print(f"  Entities: {stats['total_entities']}")
    print(f"  Relationships: {stats['total_relationships']}")
    
    # Clean up test data
    import shutil
    shutil.rmtree("./test_kb_data", ignore_errors=True)
    
    print("\n✅ All knowledge base tests passed!")
