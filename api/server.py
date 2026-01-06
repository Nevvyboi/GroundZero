"""
API Server - FastAPI Edition
============================
FastAPI server with WebSocket support and proper startup/shutdown.
Now with Knowledge Graph reasoning!
"""

import sys
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from config import Settings
from storage import KnowledgeBase
from learning import LearningEngine
from reasoning import ResponseGenerator

# Try to import Advanced Knowledge Graph Reasoner
try:
    from reasoning.advanced_reasoner import AdvancedReasoner
    REASONER_AVAILABLE = True
except ImportError:
    REASONER_AVAILABLE = False
    print("⚠️ Advanced Knowledge Graph not available")

# Global components
knowledge_base: Optional[KnowledgeBase] = None
learning_engine: Optional[LearningEngine] = None
response_generator: Optional[ResponseGenerator] = None
graph_reasoner: Optional['AdvancedReasoner'] = None
settings: Optional[Settings] = None

# WebSocket connections for real-time updates
active_connections: List[WebSocket] = []


def get_components():
    """Get initialized components"""
    return {
        'kb': knowledge_base,
        'learner': learning_engine,
        'response_generator': response_generator,
        'graph_reasoner': graph_reasoner
    }


def print_banner():
    """Print startup banner"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║      ██████╗ ██████╗  ██████╗ ██╗   ██╗███╗   ██╗██████╗     ║
║     ██╔════╝ ██╔══██╗██╔═══██╗██║   ██║████╗  ██║██╔══██╗    ║
║     ██║  ███╗██████╔╝██║   ██║██║   ██║██╔██╗ ██║██║  ██║    ║
║     ██║   ██║██╔══██╗██║   ██║██║   ██║██║╚██╗██║██║  ██║    ║
║     ╚██████╔╝██║  ██║╚██████╔╝╚██████╔╝██║ ╚████║██████╔╝    ║
║      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═════╝     ║
║                                                               ║
║         🧠 GroundZero - AI Built From Scratch 🧠              ║
║                                                               ║
║   Vector Search + Knowledge Graph + Common Sense Reasoning    ║
║   Multi-hop • Analogies • Semantic Similarity • Inference     ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
""")


def print_step(message: str, done: bool = False):
    """Print startup step"""
    icon = "✅" if done else "🔄"
    print(f"  {icon} {message}")


async def broadcast_to_websockets(message: dict):
    """Broadcast message to all connected WebSocket clients"""
    disconnected = []
    for ws in active_connections:
        try:
            await ws.send_json(message)
        except:
            disconnected.append(ws)
    
    # Remove disconnected
    for ws in disconnected:
        if ws in active_connections:
            active_connections.remove(ws)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup initialization and shutdown cleanup.
    """
    global knowledge_base, learning_engine, response_generator, graph_reasoner, settings
    
    print_banner()
    print("\n📦 Initializing Components...\n")
    
    settings = Settings()
    
    # Step 1: Initialize Knowledge Base (Vector Database + SQLite)
    print_step("Initializing Knowledge Base (Vector Database)")
    knowledge_base = KnowledgeBase(
        data_dir=settings.data_dir,
        dimension=settings.embedding_dimension
    )
    print_step("Knowledge Base ready", done=True)
    
    # Step 2: Initialize Advanced Knowledge Graph (with common sense, analogies, multi-hop)
    if REASONER_AVAILABLE:
        print_step("Initializing Advanced Knowledge Graph (Common Sense + Reasoning)")
        graph_reasoner = AdvancedReasoner(settings.data_dir)
        print_step("Advanced Knowledge Graph ready", done=True)
    else:
        graph_reasoner = None
        print_step("Knowledge Graph not available (optional)", done=True)
    
    # Step 3: Initialize Response Generator (connects to Knowledge Graph)
    print_step("Initializing Response Generator")
    response_generator = ResponseGenerator(knowledge_base, data_dir=settings.data_dir)
    
    # Connect the graph reasoner to the response generator
    if graph_reasoner and response_generator:
        response_generator.graph_reasoner = graph_reasoner
        print_step("Response Generator connected to Knowledge Graph", done=True)
    else:
        print_step("Response Generator ready (vector search only)", done=True)
    
    # Step 4: Initialize Learning Engine (connects to Knowledge Graph)
    print_step("Initializing Learning Engine")
    learning_engine = LearningEngine(knowledge_base, graph_reasoner=graph_reasoner)
    
    # Setup WebSocket callbacks
    def on_article_start(title, url):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(broadcast_to_websockets({
                    'type': 'article_start',
                    'title': title,
                    'url': url
                }))
        except:
            pass
    
    def on_article_complete(title, word_count):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(broadcast_to_websockets({
                    'type': 'article_complete',
                    'title': title,
                    'word_count': word_count,
                    'stats': learning_engine.get_stats()
                }))
        except:
            pass
    
    learning_engine.on_article_start = on_article_start
    learning_engine.on_article_complete = on_article_complete
    print_step("Learning Engine ready", done=True)
    
    # Print loaded statistics
    stats = knowledge_base.get_statistics()
    print("\n" + "=" * 55)
    print("📊 Knowledge Base Statistics (Loaded from Disk):")
    print("=" * 55)
    print(f"  📚 Knowledge entries:  {stats['total_knowledge']:,}")
    print(f"  📖 Sources learned:    {stats['total_sources']:,}")
    print(f"  📝 Vocabulary size:    {stats['vocabulary_size']:,}")
    print(f"  💬 Total words:        {stats['total_words']:,}")
    print(f"  🔢 Vector dimension:   {stats['embeddings']['dimension']}")
    print(f"  🗃️  Index type:         {stats['vectors']['index_type']}")
    print(f"  💾 Data directory:     {settings.data_dir.absolute()}")
    if graph_reasoner:
        gr_stats = graph_reasoner.get_stats()
        print(f"  🧠 Knowledge Graph:    {gr_stats['total_facts']} facts, {gr_stats['unique_subjects']} entities")
    print("=" * 55)
    
    if stats['total_knowledge'] > 0:
        print(f"\n✅ Loaded {stats['total_knowledge']:,} knowledge entries from disk!")
    else:
        print("\n📝 Starting with empty knowledge base. Start learning to add knowledge!")
    
    print("\n✅ All components initialized successfully!\n")
    
    yield  # Application runs here
    
    # === SHUTDOWN ===
    print("\n👋 Shutting down GroundZero...")
    
    # Stop learning
    if learning_engine and learning_engine.is_running:
        print("  🛑 Stopping learning engine...")
        learning_engine.stop()
    
    # Save all data
    if knowledge_base:
        print("  💾 Saving knowledge base to disk...")
        knowledge_base.save()
        print("  ✅ Data saved!")
    
    print("  👋 Goodbye!\n")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    app = FastAPI(
        title="GroundZero",
        description="AI built from scratch - Vector Search + Knowledge Graph + Symbolic Reasoning",
        version="3.1.0",
        lifespan=lifespan
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Import and include routes
    from .routes import router
    app.include_router(router)
    
    # Mount static files
    static_path = Path(__file__).parent.parent / "static"
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    
    return app


# Create app instance
app = create_app()