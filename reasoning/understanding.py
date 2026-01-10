"""
Understanding Engine
====================
True understanding from learned knowledge using multiple AI techniques.

This module implements several approaches to go from retrieval → understanding:

APPROACH 1: Word-Level Embeddings + Attention (Transformer-like)
- Each word gets its own embedding
- Attention mechanism finds relevant words
- Good for: Question answering, entity extraction

APPROACH 2: Semantic Chunking + Reasoning
- Split knowledge into semantic chunks
- Build a reasoning graph
- Good for: Complex questions, multi-hop reasoning

APPROACH 3: Neural Concept Network
- Build concept → concept relationships
- Traverse the network to find answers
- Good for: "What is X", definitions, relationships

All approaches work WITHOUT external APIs - pure local inference.
Future-ready for: voice transcription vectors, image embeddings
"""

import numpy as np
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import sqlite3
from pathlib import Path


@dataclass
class Understanding:
    """Result of understanding process"""
    answer: str
    confidence: float
    reasoning_path: List[str]  # How we arrived at the answer
    concepts_used: List[str]   # Key concepts involved
    sources: List[Dict[str, str]]


class WordEmbeddings:
    """
    Word-level embeddings for fine-grained understanding.
    
    Unlike document embeddings (one vector per article),
    this creates vectors for individual words, allowing:
    - "What is the capital of France?" → finds "Paris" directly
    - Understanding relationships between words
    """
    
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.word_vectors: Dict[str, np.ndarray] = {}
        self.word_contexts: Dict[str, List[str]] = defaultdict(list)  # word → [contexts]
        self.word_frequency: Dict[str, int] = defaultdict(int)
        self.cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # For building embeddings
        self._vocab_size = 0
        self._initialized = False
    
    def add_text(self, text: str, source_title: str = "") -> None:
        """Process text to build word embeddings"""
        words = self._tokenize(text)
        
        # Update frequencies
        for word in words:
            self.word_frequency[word] += 1
        
        # Build co-occurrence (words that appear near each other)
        window_size = 5
        for i, word in enumerate(words):
            # Get context words
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            context = words[start:i] + words[i+1:end]
            
            # Store context
            self.word_contexts[word].extend(context[:10])  # Limit stored contexts
            
            # Update co-occurrence
            for ctx_word in context:
                if word != ctx_word:
                    pair = tuple(sorted([word, ctx_word]))
                    self.cooccurrence[pair] += 1
    
    def build_embeddings(self) -> None:
        """Build word vectors from co-occurrence data"""
        if not self.word_frequency:
            return
        
        # Get vocabulary (most frequent words)
        vocab = sorted(self.word_frequency.items(), key=lambda x: -x[1])[:10000]
        word_to_idx = {w: i for i, (w, _) in enumerate(vocab)}
        
        self._vocab_size = len(vocab)
        
        # Build co-occurrence matrix
        cooc_matrix = np.zeros((self._vocab_size, self._vocab_size), dtype=np.float32)
        
        for (w1, w2), count in self.cooccurrence.items():
            if w1 in word_to_idx and w2 in word_to_idx:
                i, j = word_to_idx[w1], word_to_idx[w2]
                cooc_matrix[i, j] = count
                cooc_matrix[j, i] = count
        
        # Apply PPMI (Positive Pointwise Mutual Information)
        row_sum = cooc_matrix.sum(axis=1, keepdims=True) + 1e-10
        col_sum = cooc_matrix.sum(axis=0, keepdims=True) + 1e-10
        total = cooc_matrix.sum() + 1e-10
        
        pmi = np.log2((cooc_matrix * total) / (row_sum * col_sum) + 1e-10)
        ppmi = np.maximum(pmi, 0)
        
        # SVD to reduce dimensions
        try:
            U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)
            embeddings = U[:, :self.dimension] * np.sqrt(S[:self.dimension])
        except:
            # Fallback: random projection
            np.random.seed(42)
            proj = np.random.randn(self._vocab_size, self.dimension)
            embeddings = ppmi @ proj
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        embeddings = embeddings / norms
        
        # Store
        for word, idx in word_to_idx.items():
            self.word_vectors[word] = embeddings[idx].astype(np.float32)
        
        self._initialized = True
        pass
    
    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """Get embedding for a word"""
        word = word.lower()
        return self.word_vectors.get(word)
    
    def find_similar_words(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find words with similar meaning"""
        vec = self.get_embedding(word)
        if vec is None:
            return []
        
        similarities = []
        for other_word, other_vec in self.word_vectors.items():
            if other_word != word:
                sim = float(np.dot(vec, other_vec))
                similarities.append((other_word, sim))
        
        similarities.sort(key=lambda x: -x[1])
        return similarities[:top_k]
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = text.lower()
        words = re.findall(r'\b[a-z]{2,}\b', text)
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                     'from', 'as', 'into', 'through', 'and', 'but', 'or', 'if',
                     'this', 'that', 'these', 'those', 'it', 'its'}
        return [w for w in words if w not in stopwords]


class ConceptNetwork:
    """
    Neural Concept Network - builds relationships between concepts.
    
    Example:
    - "Paris" → is_capital_of → "France"
    - "France" → is_a → "country"
    - "France" → located_in → "Europe"
    
    This allows answering:
    - "What is the capital of France?" → traverse to find "Paris"
    - "Tell me about France" → gather all connected concepts
    """
    
    def __init__(self):
        self.concepts: Dict[str, Dict[str, Any]] = {}  # concept → {definition, properties}
        self.relations: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)  # concept → [(relation, target, weight)]
        self.reverse_relations: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
        
        # Relation patterns to extract
        self.relation_patterns = [
            (r'(.+?) is (?:a|an|the) (.+)', 'is_a'),
            (r'(.+?) (?:is|are) located in (.+)', 'located_in'),
            (r'(.+?) is the capital of (.+)', 'capital_of'),
            (r'(.+?) was born (?:in|on) (.+)', 'born_in'),
            (r'(.+?) (?:is|was) (?:a|an) (.+?) (?:who|that|which)', 'is_a'),
            (r'(.+?) (?:founded|created|invented) (.+)', 'created'),
            (r'(.+?) (?:contains|includes|has) (.+)', 'contains'),
        ]
    
    def extract_concepts(self, text: str, source_title: str = "") -> List[str]:
        """Extract concepts and relationships from text"""
        extracted = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            # Try to extract relationships
            for pattern, relation_type in self.relation_patterns:
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    subject = self._normalize_concept(match.group(1))
                    obj = self._normalize_concept(match.group(2))
                    
                    if subject and obj and len(subject) > 2 and len(obj) > 2:
                        self.add_relation(subject, relation_type, obj)
                        extracted.append(f"{subject} → {relation_type} → {obj}")
            
            # Extract definitions (first sentence about a topic)
            if source_title:
                title_norm = self._normalize_concept(source_title)
                # First substantive sentence often defines the topic
                if title_norm.lower() in sentence.lower()[:50]:
                    self.add_concept(title_norm, sentence[:500])
                    extracted.append(f"DEFINED: {title_norm}")
        
        return extracted
    
    def add_concept(self, name: str, definition: str, properties: Dict = None) -> None:
        """Add or update a concept"""
        name = self._normalize_concept(name)
        if name not in self.concepts:
            self.concepts[name] = {'definition': definition, 'properties': properties or {}}
        else:
            # Merge definitions
            existing = self.concepts[name]['definition']
            if len(definition) > len(existing):
                self.concepts[name]['definition'] = definition
    
    def add_relation(self, subject: str, relation: str, obj: str, weight: float = 1.0) -> None:
        """Add a relationship between concepts"""
        subject = self._normalize_concept(subject)
        obj = self._normalize_concept(obj)
        
        # Add forward relation
        self.relations[subject].append((relation, obj, weight))
        
        # Add reverse relation
        reverse_rel = f"reverse_{relation}"
        self.reverse_relations[obj].append((reverse_rel, subject, weight))
    
    def query_concept(self, query: str) -> Dict[str, Any]:
        """Query the concept network"""
        query_norm = self._normalize_concept(query)
        
        result = {
            'concept': query_norm,
            'definition': None,
            'relations': [],
            'related_concepts': []
        }
        
        # Direct lookup
        if query_norm in self.concepts:
            result['definition'] = self.concepts[query_norm]['definition']
        
        # Get relations
        if query_norm in self.relations:
            result['relations'] = self.relations[query_norm]
        
        # Get reverse relations (things that relate TO this concept)
        if query_norm in self.reverse_relations:
            result['relations'].extend(self.reverse_relations[query_norm])
        
        # Find related concepts (fuzzy match)
        query_words = set(query_norm.lower().split())
        for concept in self.concepts:
            concept_words = set(concept.lower().split())
            overlap = len(query_words & concept_words)
            if overlap > 0 and concept != query_norm:
                result['related_concepts'].append((concept, overlap))
        
        result['related_concepts'].sort(key=lambda x: -x[1])
        result['related_concepts'] = result['related_concepts'][:5]
        
        return result
    
    def _normalize_concept(self, text: str) -> str:
        """Normalize concept name"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove leading articles
        text = re.sub(r'^(the|a|an)\s+', '', text, flags=re.IGNORECASE)
        # Capitalize properly
        return text.strip()[:100]


class UnderstandingEngine:
    """
    Main understanding engine combining multiple approaches.
    
    Architecture:
    1. Word embeddings - for fine-grained word-level understanding
    2. Concept network - for relationship-based reasoning
    3. Attention mechanism - to focus on relevant parts
    4. Answer synthesis - to generate coherent responses
    
    Future extensions:
    - Voice: Add audio embeddings (same vector space)
    - Images: Add image embeddings (CLIP-style)
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.word_embeddings = WordEmbeddings(dimension=128)
        self.concept_network = ConceptNetwork()
        
        # Knowledge store
        self.knowledge_chunks: List[Dict[str, Any]] = []  # [{text, source, embedding}]
        self.chunk_embeddings: Optional[np.ndarray] = None
        
        # Database for persistence
        self.db_path = self.data_dir / "understanding.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize understanding database"""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                source_title TEXT,
                source_url TEXT,
                chunk_type TEXT DEFAULT 'paragraph',
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                definition TEXT,
                properties TEXT DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                relation TEXT NOT NULL,
                object TEXT NOT NULL,
                weight REAL DEFAULT 1.0
            )
        """)
        conn.commit()
        conn.close()
    
    def learn(self, content: str, source_title: str = "", source_url: str = "") -> Dict[str, Any]:
        """
        Learn from content - builds understanding, not just storage.
        
        Process:
        1. Chunk content into semantic units
        2. Extract word embeddings
        3. Extract concepts and relationships
        4. Store for retrieval and reasoning
        """
        results = {
            'chunks_added': 0,
            'concepts_extracted': [],
            'relations_found': 0
        }
        
        # 1. Chunk content into paragraphs/sentences
        chunks = self._chunk_content(content)
        
        for chunk in chunks:
            if len(chunk) < 50:
                continue
            
            # Add to word embeddings
            self.word_embeddings.add_text(chunk, source_title)
            
            # Extract concepts
            extracted = self.concept_network.extract_concepts(chunk, source_title)
            results['concepts_extracted'].extend(extracted)
            
            # Store chunk
            self.knowledge_chunks.append({
                'text': chunk,
                'source_title': source_title,
                'source_url': source_url
            })
            results['chunks_added'] += 1
        
        results['relations_found'] = len(results['concepts_extracted'])
        
        return results
    
    def build_understanding(self) -> None:
        """Build the understanding model from learned data"""
        pass
        
        # Build word embeddings FIRST
        self.word_embeddings.build_embeddings()
        
        # NOW build chunk embeddings (after word embeddings exist)
        if self.knowledge_chunks:
            embeddings = []
            for chunk in self.knowledge_chunks:
                emb = self._embed_text(chunk['text'])
                embeddings.append(emb)
                chunk['embedding'] = emb
            
            self.chunk_embeddings = np.array(embeddings, dtype=np.float32)
            
            # Verify embeddings are valid
            valid_count = sum(1 for e in embeddings if np.linalg.norm(e) > 0.1)
            pass
        
        pass
    
    def understand(self, query: str) -> Understanding:
        """
        Understand and answer a query using multiple reasoning paths.
        
        Process:
        1. Parse query to identify intent and key concepts
        2. Search concept network for direct answers
        3. Use word embeddings to find semantically similar content
        4. Apply attention to focus on relevant parts
        5. Synthesize answer from multiple sources
        """
        reasoning_path = []
        concepts_used = []
        sources = []
        
        # 1. Parse query
        query_type, query_concepts = self._parse_query(query)
        reasoning_path.append(f"Query type: {query_type}")
        reasoning_path.append(f"Key concepts: {query_concepts}")
        
        # 2. Try concept network first (for definitions and relationships)
        concept_answer = None
        for concept in query_concepts:
            result = self.concept_network.query_concept(concept)
            if result['definition']:
                concept_answer = result['definition']
                concepts_used.append(concept)
                reasoning_path.append(f"Found definition for '{concept}'")
                break
            if result['relations']:
                # Build answer from relations
                rel_parts = []
                for rel, obj, _ in result['relations'][:3]:
                    rel_parts.append(f"{rel.replace('_', ' ')}: {obj}")
                if rel_parts:
                    concept_answer = f"{concept}: " + "; ".join(rel_parts)
                    concepts_used.extend([concept] + [r[1] for r in result['relations'][:3]])
                    reasoning_path.append(f"Found relations for '{concept}'")
                    break
        
        # 3. Use semantic search on chunks
        semantic_results = self._semantic_search(query, top_k=5)
        reasoning_path.append(f"Found {len(semantic_results)} relevant chunks")
        
        # 4. Apply attention to find most relevant sentences
        relevant_sentences = self._attention_extract(query, semantic_results)
        reasoning_path.append(f"Extracted {len(relevant_sentences)} relevant sentences")
        
        # 5. Synthesize answer
        if concept_answer:
            answer = concept_answer
            confidence = 0.85
        elif relevant_sentences:
            answer = self._synthesize_answer(query, relevant_sentences)
            confidence = 0.7
            sources = [{'title': r['source_title'], 'url': r.get('source_url', '')} 
                      for r in semantic_results[:3] if r.get('source_title')]
        else:
            answer = None
            confidence = 0.1
        
        return Understanding(
            answer=answer,
            confidence=confidence,
            reasoning_path=reasoning_path,
            concepts_used=concepts_used,
            sources=sources
        )
    
    def _parse_query(self, query: str) -> Tuple[str, List[str]]:
        """Parse query to determine type and extract key concepts"""
        query_lower = query.lower()
        
        # Determine query type
        if re.match(r'^(what is|who is|define|explain)\b', query_lower):
            query_type = 'definition'
        elif re.match(r'^(when|where|which)\b', query_lower):
            query_type = 'factual'
        elif re.match(r'^(why|how come)\b', query_lower):
            query_type = 'causal'
        elif re.match(r'^(how to|how do|how can)\b', query_lower):
            query_type = 'procedural'
        else:
            query_type = 'general'
        
        # Extract key concepts (nouns and noun phrases)
        # Remove question words
        clean_query = re.sub(r'^(what|who|when|where|why|how|is|are|was|were|the|a|an|tell|me|about)\s+', 
                           '', query_lower)
        
        # Split into potential concepts
        concepts = []
        # Try to find multi-word concepts first
        words = clean_query.split()
        if len(words) >= 2:
            concepts.append(' '.join(words[:3]))  # First 3 words as phrase
        concepts.extend([w for w in words if len(w) > 3])
        
        return query_type, concepts[:5]
    
    def _semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search knowledge chunks using semantic similarity"""
        if self.chunk_embeddings is None or len(self.chunk_embeddings) == 0:
            return []
        
        query_emb = self._embed_text(query)
        
        # Check if query embedding is valid
        if np.linalg.norm(query_emb) < 0.1:
            # Fall back to keyword search
            return self._keyword_search(query, top_k)
        
        # Calculate similarities
        similarities = np.dot(self.chunk_embeddings, query_emb)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            chunk = self.knowledge_chunks[idx]
            results.append({
                'text': chunk['text'],
                'source_title': chunk.get('source_title', ''),
                'source_url': chunk.get('source_url', ''),
                'similarity': float(similarities[idx])
            })
        
        return results
    
    def _keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Fallback keyword-based search"""
        query_words = set(self.word_embeddings._tokenize(query))
        
        results = []
        for chunk in self.knowledge_chunks:
            chunk_words = set(self.word_embeddings._tokenize(chunk['text']))
            overlap = len(query_words & chunk_words)
            
            if overlap > 0:
                results.append({
                    'text': chunk['text'],
                    'source_title': chunk.get('source_title', ''),
                    'source_url': chunk.get('source_url', ''),
                    'similarity': overlap / max(len(query_words), 1)
                })
        
        results.sort(key=lambda x: -x['similarity'])
        return results[:top_k]
    
    def _attention_extract(self, query: str, chunks: List[Dict]) -> List[str]:
        """Use attention mechanism to find relevant sentences"""
        query_words = set(self.word_embeddings._tokenize(query))
        
        # Get query embedding
        query_emb = np.zeros(self.word_embeddings.dimension, dtype=np.float32)
        query_word_count = 0
        for word in query_words:
            word_emb = self.word_embeddings.get_embedding(word)
            if word_emb is not None and len(word_emb) == self.word_embeddings.dimension:
                query_emb += word_emb
                query_word_count += 1
        
        if query_word_count > 0:
            query_emb /= query_word_count
        
        relevant = []
        for chunk in chunks:
            sentences = re.split(r'[.!?]', chunk['text'])
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                
                # Calculate attention score
                sent_words = self.word_embeddings._tokenize(sentence)
                
                # Word overlap (keyword matching)
                overlap = len(set(sent_words) & query_words)
                
                # Semantic similarity
                sent_emb = np.zeros(self.word_embeddings.dimension, dtype=np.float32)
                sent_word_count = 0
                for word in sent_words:
                    word_emb = self.word_embeddings.get_embedding(word)
                    if word_emb is not None and len(word_emb) == self.word_embeddings.dimension:
                        sent_emb += word_emb
                        sent_word_count += 1
                
                if sent_word_count > 0:
                    sent_emb /= sent_word_count
                    semantic_sim = float(np.dot(query_emb, sent_emb))
                else:
                    semantic_sim = 0
                
                # Combined attention score
                attention = overlap * 0.4 + semantic_sim * 0.6
                
                if attention > 0.2 or overlap > 0:
                    relevant.append((sentence, attention))
        
        # Sort by attention and return top sentences
        relevant.sort(key=lambda x: -x[1])
        return [s[0] for s in relevant[:5]]
    
    def _synthesize_answer(self, query: str, sentences: List[str]) -> str:
        """Synthesize a coherent answer from relevant sentences"""
        if not sentences:
            return None
        
        # For now, combine top sentences
        # Future: Use a local language model for better synthesis
        answer = ' '.join(sentences[:3])
        
        # Clean up
        answer = re.sub(r'\s+', ' ', answer).strip()
        if len(answer) > 600:
            answer = answer[:600] + '...'
        
        return answer
    
    def _embed_text(self, text: str) -> np.ndarray:
        """Create embedding for text using word vectors"""
        words = self.word_embeddings._tokenize(text)
        
        if not words:
            return np.zeros(self.word_embeddings.dimension, dtype=np.float32)
        
        embeddings = []
        for word in words:
            emb = self.word_embeddings.get_embedding(word)
            if emb is not None and len(emb) == self.word_embeddings.dimension:
                embeddings.append(emb)
        
        if not embeddings:
            return np.zeros(self.word_embeddings.dimension, dtype=np.float32)
        
        # Average word embeddings
        avg_emb = np.mean(embeddings, axis=0)
        
        # Normalize
        norm = np.linalg.norm(avg_emb)
        if norm > 0:
            avg_emb /= norm
        
        return avg_emb.astype(np.float32)
    
    def _chunk_content(self, content: str) -> List[str]:
        """Split content into semantic chunks"""
        # Split by paragraphs first
        paragraphs = re.split(r'\n\n+', content)
        
        chunks = []
        for para in paragraphs:
            para = para.strip()
            if len(para) < 50:
                continue
            
            # If paragraph is too long, split by sentences
            if len(para) > 500:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""
                for sent in sentences:
                    if len(current_chunk) + len(sent) < 500:
                        current_chunk += " " + sent
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sent
                if current_chunk:
                    chunks.append(current_chunk.strip())
            else:
                chunks.append(para)
        
        return chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get understanding engine statistics"""
        return {
            'word_embeddings': len(self.word_embeddings.word_vectors),
            'concepts': len(self.concept_network.concepts),
            'relations': sum(len(r) for r in self.concept_network.relations.values()),
            'knowledge_chunks': len(self.knowledge_chunks),
            'embedding_dimension': self.word_embeddings.dimension
        }

