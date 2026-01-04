#!/usr/bin/env python3
"""
GroundZero v3 - FastAPI Edition
===============================
An AI that learns from scratch using vector databases.

Features:
- Semantic search using vector embeddings
- FAISS-based vector storage (or brute force fallback)
- Continuous learning from Wikipedia
- URL ingestion
- Intelligent question answering
- Persistent storage (survives restarts)

Usage:
    python main.py
    
Then open http://localhost:5000 in your browser.

API Docs available at:
    http://localhost:5000/docs
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Start GroundZero server"""
    import uvicorn
    from config import Settings
    
    settings = Settings()
    
    # Run with uvicorn
    uvicorn.run(
        "api.server:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="warning"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
