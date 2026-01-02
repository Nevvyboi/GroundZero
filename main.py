"""
NeuralMind - Scalable AI Learning System
=========================================

Architecture Overview:
----------------------
neuralmind/
├── core/                    # Core neural network & embeddings
│   ├── __init__.py
│   ├── tokenizer.py         # Vocabulary & tokenization
│   ├── embeddings.py        # Token & positional embeddings
│   ├── transformer.py       # Transformer blocks & attention
│   └── model.py             # Main neural model orchestrator
│
├── storage/                 # Persistence layer (SQLite + binary)
│   ├── __init__.py
│   ├── database.py          # SQLite connection & schema
│   ├── memory_store.py      # Knowledge & memory persistence
│   ├── model_store.py       # Model weights & state (binary)
│   └── schemas.py           # Database table definitions
│
├── learning/                # Knowledge acquisition
│   ├── __init__.py
│   ├── knowledge_processor.py  # Text processing & summarization
│   ├── continuous_learner.py   # Background learning orchestrator
│   └── knowledge_graph.py      # Concept relationships
│
├── reasoning/               # Reasoning & inference
│   ├── __init__.py
│   ├── engine.py            # Main reasoning orchestrator
│   ├── logic.py             # Logical reasoning
│   ├── math_solver.py       # Mathematical operations
│   ├── code_analyzer.py     # Code understanding
│   └── metacognition.py     # Self-awareness & introspection
│
├── web/                     # Web crawling & search
│   ├── __init__.py
│   ├── crawler.py           # Web page fetching & parsing
│   ├── search.py            # Wikipedia & web search
│   └── content_extractor.py # Clean text extraction
│
├── dialogue/                # Conversation management
│   ├── __init__.py
│   ├── conversation.py      # Dialogue state & history
│   ├── response_generator.py # Response composition
│   └── clarification.py     # Clarifying questions
│
├── api/                     # Web API layer
│   ├── __init__.py
│   ├── server.py            # Flask/SocketIO server
│   ├── routes.py            # API endpoints
│   └── websocket.py         # Real-time updates
│
├── config/                  # Configuration
│   ├── __init__.py
│   └── settings.py          # Global settings
│
├── static/                  # Frontend assets
│   ├── index.html
│   ├── app.js
│   └── styles.css
│
└── run.py                   # Application entry point

Design Principles:
------------------
1. Single Responsibility - Each module does one thing well
2. Dependency Injection - Components receive dependencies, don't create them
3. Interface Segregation - Small, focused interfaces
4. Open/Closed - Extend without modifying existing code
5. Persistence Agnostic - Storage abstracted behind interfaces
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from api.server import create_app, run_server
from config.settings import Settings


def main():
    """Main entry point for NeuralMind"""
    settings = Settings()
    
    print("\n" + "=" * 60)
    print("🧠 NeuralMind AI - Scalable Learning System")
    print("=" * 60)
    print(f"📁 Data Directory: {settings.DATA_DIR}")
    print(f"🌐 Server: http://localhost:{settings.PORT}")
    print("=" * 60 + "\n")
    
    app, socketio = create_app(settings)
    run_server(app, socketio, settings)


if __name__ == "__main__":
    main()
