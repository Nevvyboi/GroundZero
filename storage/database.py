"""
Database Connection Manager
===========================
SQLite connection handling with thread safety.
"""

import sqlite3
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager

from .schemas import get_all_schemas, SCHEMA_VERSION


class Database:
    """Thread-safe SQLite database manager"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()
        self._initialize_schema()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable foreign keys
            self._local.connection.execute("PRAGMA foreign_keys = ON")
        return self._local.connection
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions"""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
    
    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a single query"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor
    
    def execute_many(self, query: str, params_list: List[tuple]) -> None:
        """Execute query with multiple parameter sets"""
        with self._lock:
            conn = self._get_connection()
            conn.executemany(query, params_list)
            conn.commit()
    
    def fetch_one(self, query: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        """Fetch a single row"""
        cursor = self.execute(query, params)
        return cursor.fetchone()
    
    def fetch_all(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Fetch all rows"""
        cursor = self.execute(query, params)
        return cursor.fetchall()
    
    def fetch_value(self, query: str, params: tuple = ()) -> Any:
        """Fetch a single value"""
        row = self.fetch_one(query, params)
        return row[0] if row else None
    
    def _initialize_schema(self) -> None:
        """Create database tables if they don't exist"""
        conn = self._get_connection()
        
        for schema in get_all_schemas():
            conn.execute(schema)
        
        # Check/set schema version
        existing_version = conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()
        
        if existing_version is None:
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,)
            )
        
        conn.commit()
    
    def close(self) -> None:
        """Close the database connection"""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
    
    # Convenience methods for common operations
    
    def insert(self, table: str, data: Dict[str, Any]) -> int:
        """Insert a row and return the ID"""
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        cursor = self.execute(query, tuple(data.values()))
        return cursor.lastrowid
    
    def insert_or_ignore(self, table: str, data: Dict[str, Any]) -> int:
        """Insert a row if it doesn't exist"""
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        query = f"INSERT OR IGNORE INTO {table} ({columns}) VALUES ({placeholders})"
        cursor = self.execute(query, tuple(data.values()))
        return cursor.lastrowid
    
    def update(self, table: str, data: Dict[str, Any], where: str, where_params: tuple) -> int:
        """Update rows and return affected count"""
        set_clause = ", ".join([f"{k} = ?" for k in data.keys()])
        query = f"UPDATE {table} SET {set_clause} WHERE {where}"
        cursor = self.execute(query, tuple(data.values()) + where_params)
        return cursor.rowcount
    
    def count(self, table: str, where: str = "1=1", params: tuple = ()) -> int:
        """Count rows in a table"""
        query = f"SELECT COUNT(*) FROM {table} WHERE {where}"
        return self.fetch_value(query, params)
    
    def exists(self, table: str, where: str, params: tuple) -> bool:
        """Check if a row exists"""
        query = f"SELECT 1 FROM {table} WHERE {where} LIMIT 1"
        return self.fetch_one(query, params) is not None
