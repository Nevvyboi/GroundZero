"""
Database Schema Definitions
===========================
SQLite table schemas for persistent storage.
"""

# Schema version for migrations
SCHEMA_VERSION = 1

# Table creation SQL statements
SCHEMAS = {
    "schema_version": """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    
    "vocabulary": """
        CREATE TABLE IF NOT EXISTS vocabulary (
            id INTEGER PRIMARY KEY,
            word TEXT UNIQUE NOT NULL,
            frequency INTEGER DEFAULT 1,
            confidence REAL DEFAULT 0.0,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    
    "vocabulary_index": """
        CREATE INDEX IF NOT EXISTS idx_vocabulary_word ON vocabulary(word)
    """,
    
    "knowledge": """
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_hash TEXT UNIQUE NOT NULL,
            content TEXT NOT NULL,
            summary TEXT,
            source_url TEXT,
            source_title TEXT,
            confidence REAL DEFAULT 0.5,
            access_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    
    "knowledge_index": """
        CREATE INDEX IF NOT EXISTS idx_knowledge_hash ON knowledge(content_hash)
    """,
    
    "learned_sources": """
        CREATE TABLE IF NOT EXISTS learned_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
            title TEXT,
            content_length INTEGER,
            chunks_learned INTEGER DEFAULT 0,
            words_learned INTEGER DEFAULT 0,
            learned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            success BOOLEAN DEFAULT TRUE
        )
    """,
    
    "learned_sources_index": """
        CREATE INDEX IF NOT EXISTS idx_learned_sources_url ON learned_sources(url)
    """,
    
    "concepts": """
        CREATE TABLE IF NOT EXISTS concepts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            confidence REAL DEFAULT 0.5,
            mention_count INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    
    "concept_relations": """
        CREATE TABLE IF NOT EXISTS concept_relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            concept_a_id INTEGER NOT NULL,
            concept_b_id INTEGER NOT NULL,
            relation_type TEXT NOT NULL,
            strength REAL DEFAULT 0.5,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (concept_a_id) REFERENCES concepts(id),
            FOREIGN KEY (concept_b_id) REFERENCES concepts(id),
            UNIQUE(concept_a_id, concept_b_id, relation_type)
        )
    """,
    
    "learning_sessions": """
        CREATE TABLE IF NOT EXISTS learning_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            sites_learned INTEGER DEFAULT 0,
            words_learned INTEGER DEFAULT 0,
            knowledge_added INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active'
        )
    """,
    
    "conversation_history": """
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            reasoning_type TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    
    "model_state": """
        CREATE TABLE IF NOT EXISTS model_state (
            key TEXT PRIMARY KEY,
            value_type TEXT NOT NULL,
            value_int INTEGER,
            value_real REAL,
            value_text TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
}


def get_all_schemas() -> list:
    """Return all schema creation statements in order"""
    return list(SCHEMAS.values())


def get_schema_names() -> list:
    """Return all schema names"""
    return list(SCHEMAS.keys())
