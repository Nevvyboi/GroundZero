"""
GroundZero API Server v2.0
==========================
FastAPI server with enhanced features:
- Voice transcription WebSocket streaming
- Model timeline tracking
- Model size change with backfilling
- Improved component lifecycle management
- Real-time learning progress updates
"""

import sys
import asyncio
import threading
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from datetime import datetime
import tempfile
import io

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Import configuration
try:
    from config import Settings
except ImportError:
    from config.settings import Settings

# Import VectorStore first (needed by KnowledgeBase)
AdvancedVectorStore = None
simple_embed = None
HNSWIndex = None
try:
    from storage.vector_store import HybridVectorStore as AdvancedVectorStore
    from storage.vector_store import simple_embed, HNSWIndex
except ImportError as e:
    print(f"  ○ VectorStore import error: {e}")
except Exception as e:
    print(f"  ✗ VectorStore failed: {e}")

# Now import KnowledgeBase (depends on vector_store)
KnowledgeBase = None
try:
    from storage.knowledge_base import AdvancedKnowledgeBase as KnowledgeBase
except ImportError as e:
    print(f"  ○ KnowledgeBase import error: {e}")
except Exception as e:
    print(f"  ✗ KnowledgeBase failed: {e}")

# Import learning components - try both old and new styles
LearningEngine = None
try:
    from learning import LearningEngine
except ImportError:
    try:
        from learning.engine import LearningEngine
    except ImportError:
        try:
            from learning.engine import AdvancedLearningEngine as LearningEngine
        except ImportError:
            pass

try:
    from learning import StrategicPlanner
except ImportError:
    try:
        from learning.strategic import StrategicPlanner
    except ImportError:
        StrategicPlanner = None

# Import reasoning components
REASONING_AVAILABLE = False
ResponseGenerator = None
AdvancedReasoner = None
PersistentReasoner = None  # SQLite-backed knowledge graph
ContextBrain = None

try:
    from reasoning import ResponseGenerator, AdvancedReasoner, PersistentReasoner
    REASONING_AVAILABLE = True
except ImportError:
    try:
        from reasoning.engine import ResponseGenerator
        from reasoning.advanced_reasoner import AdvancedReasoner
        from reasoning.persistent_graph import PersistentReasoner
        REASONING_AVAILABLE = True
        print("  ✓ Imported reasoning components (individual)")
    except ImportError as e:
        print(f"  ○ Reasoning not available: {e}")
    except Exception as e:
        print(f"  ✗ Reasoning import error: {e}")

try:
    from reasoning import ContextBrain
except ImportError:
    try:
        from reasoning.context_brain import ContextBrain
    except ImportError:
        pass

# Import neural components
NEURAL_AVAILABLE = False
NeuralBrain = None

try:
    from neural import NeuralBrain, NEURAL_AVAILABLE
except ImportError:
    try:
        from neural.brain import NeuralBrain
        NEURAL_AVAILABLE = True
    except ImportError:
        pass

# Import voice transcription (optional)
VOICE_AVAILABLE = False
try:
    import whisper
    VOICE_AVAILABLE = True
except ImportError:
    pass

# Global components dictionary
_components: Dict[str, Any] = {}

# WebSocket connections for real-time updates
active_connections: List[WebSocket] = []
voice_connections: List[WebSocket] = []

# Voice transcription model (lazy loaded)
_whisper_model = None


def get_components() -> Dict[str, Any]:
    """Get initialized components"""
    return _components


def get_whisper_model():
    """Lazy load whisper model"""
    global _whisper_model
    if _whisper_model is None and VOICE_AVAILABLE:
        print("   📢 Loading Whisper model (first use)...")
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def print_banner():
    """Print startup banner"""
    print("\n" + "=" * 70)
    print(r"""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║   🧠  GroundZero v2.0 - Advanced Neural AI System  🧠            ║
    ║                                                                   ║
    ║   Vector Search • Knowledge Graph • Transformer Neural Network    ║
    ║   Strategic Learning • Spaced Repetition • Voice Input           ║
    ║   Model Timeline Tracking • Dynamic Size Scaling                 ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
""")
    print("=" * 70)


async def broadcast_to_websockets(message: dict):
    """Broadcast message to all connected WebSocket clients"""
    disconnected = []
    for ws in active_connections:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in active_connections:
            active_connections.remove(ws)


async def broadcast_learning_progress(stats: dict):
    """Broadcast learning progress update"""
    await broadcast_to_websockets({
        'type': 'learning_progress',
        'stats': stats,
        'timestamp': datetime.now().isoformat()
    })


async def transcribe_audio_chunk(audio_data: bytes) -> dict:
    """Transcribe audio data using Whisper"""
    if not VOICE_AVAILABLE:
        return {'error': 'Voice transcription not available', 'text': ''}
    
    try:
        model = get_whisper_model()
        
        # Write audio to temp file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
            f.write(audio_data)
            temp_path = f.name
        
        # Transcribe
        result = model.transcribe(temp_path, fp16=False)
        
        # Clean up
        Path(temp_path).unlink(missing_ok=True)
        
        return {
            'text': result['text'].strip(),
            'language': result.get('language', 'en'),
            'segments': result.get('segments', [])
        }
    except Exception as e:
        return {'error': str(e), 'text': ''}


def setup_learning_callbacks(learning_engine):
    """Setup callbacks for learning engine to broadcast progress"""
    
    def on_article_start(title, url):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(broadcast_to_websockets({
                    'type': 'article_start',
                    'title': title,
                    'url': url,
                    'timestamp': datetime.now().isoformat()
                }))
        except Exception:
            pass
    
    def on_article_complete(title, word_count):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                stats = safe_get_stats(learning_engine)
                asyncio.create_task(broadcast_to_websockets({
                    'type': 'article_complete',
                    'title': title,
                    'word_count': word_count,
                    'stats': stats,
                    'timestamp': datetime.now().isoformat()
                }))
        except Exception:
            pass
    
    def on_progress(current, total, stats):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(broadcast_to_websockets({
                    'type': 'learning_progress',
                    'current': current,
                    'total': total,
                    'stats': stats,
                    'timestamp': datetime.now().isoformat()
                }))
        except Exception:
            pass
    
    if hasattr(learning_engine, 'on_article_start'):
        learning_engine.on_article_start = on_article_start
    if hasattr(learning_engine, 'on_article_complete'):
        learning_engine.on_article_complete = on_article_complete
    if hasattr(learning_engine, 'on_progress'):
        learning_engine.on_progress = on_progress


def safe_get_stats(obj, default=None):
    """Safely get stats from an object, trying multiple method names"""
    if obj is None:
        return default or {}
    
    for method_name in ['get_stats', 'get_statistics', 'stats']:
        if hasattr(obj, method_name):
            try:
                method = getattr(obj, method_name)
                if callable(method):
                    return method()
                else:
                    return method
            except Exception:
                pass
    
    return default or {}


def try_init(cls, args_list):
    """Try initializing a class with different argument combinations"""
    for item in args_list:
        if item is None:
            continue
        args, kwargs = item
        try:
            return cls(*args, **kwargs)
        except TypeError:
            continue
        except Exception:
            continue
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with comprehensive component initialization"""
    global _components
    
    print_banner()
    
    # Load settings
    settings = Settings()
    _components['settings'] = settings
    
    # ═══════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ═══════════════════════════════════════════════════════════════
    print("\n🚀 INITIALIZING COMPONENTS\n")
    
    # 1. Vector Store - Try FAISS first, then fall back to JSON-based
    print("   📊 Vector Store")
    vector_store = None
    
    # Try HybridVectorStore first (automatically loads FAISS + SQLite)
    # Note: vectors.faiss and vectors.db are in data/ directly, not data/vectors/
    if AdvancedVectorStore:
        try:
            vector_store = AdvancedVectorStore(
                data_dir=str(settings.data_dir),  # data/ contains vectors.faiss
                dim=settings.embedding_dimension
            )
            vs_stats = safe_get_stats(vector_store)
            faiss_count = vs_stats.get('faiss_vectors', vs_stats.get('total_vectors', 0))
            print(f"      ├─ Type: {vs_stats.get('index_type', 'FAISS + SQLite')}")
            print(f"      ├─ Vectors: {faiss_count:,}")
            print(f"      ├─ Dimension: {vs_stats.get('dimension', settings.embedding_dimension)}")
            print(f"      └─ ✓ Ready")
            _components['vector_store'] = vector_store
            _components['faiss_index'] = getattr(vector_store, 'index', None)
        except Exception as e:
            print(f"      ├─ HybridVectorStore failed: {e}")
    
    # Fallback: Try loading FAISS directly if HybridVectorStore failed
    if not vector_store:
        try:
            import faiss
            faiss_path = settings.data_dir / "vectors.faiss"
            if faiss_path.exists():
                faiss_index = faiss.read_index(str(faiss_path))
                print(f"      ├─ Type: FAISS ({type(faiss_index).__name__})")
                print(f"      ├─ Vectors: {faiss_index.ntotal:,}")
                print(f"      ├─ Dimension: {faiss_index.d}")
                print(f"      └─ ✓ Ready (direct FAISS)")
                _components['faiss_index'] = faiss_index
        except ImportError:
            print("      ├─ FAISS not installed (pip install faiss-cpu)")
        except Exception as e:
            print(f"      └─ ○ No vector store available: {e}")
    
    # 2. Knowledge Base
    print("\n   📦 Knowledge Base")
    knowledge_base = None
    try:
        if KnowledgeBase:
            # Try different initialization patterns with verbose errors
            init_attempts = [
                ([], {'data_dir': str(settings.data_dir), 'embedding_dim': settings.embedding_dimension}),
                ([], {'data_dir': str(settings.data_dir)}),
                ([str(settings.data_dir), settings.embedding_dimension], {}),
                ([str(settings.data_dir)], {}),
            ]
            
            for args, kwargs in init_attempts:
                try:
                    knowledge_base = KnowledgeBase(*args, **kwargs)
                    break
                except Exception as init_err:
                    print(f"      ├─ Init failed: {init_err}")
                    continue
            
            if knowledge_base:
                kb_stats = safe_get_stats(knowledge_base)
                faiss_vectors = kb_stats.get('vectors', {}).get('total_vectors', kb_stats.get('total_vectors', 0))
                print(f"      ├─ Documents: {kb_stats.get('total_documents', 0):,}")
                print(f"      ├─ Chunks: {kb_stats.get('total_chunks', 0):,}")
                print(f"      ├─ Entities: {kb_stats.get('total_entities', 0):,}")
                print(f"      ├─ Relationships: {kb_stats.get('total_relationships', 0):,}")
                print(f"      ├─ Vectors: {faiss_vectors:,}")
                print(f"      └─ ✓ Ready")
                _components['kb'] = knowledge_base
            else:
                print(f"      └─ ✗ All init attempts failed")
        else:
            print(f"      └─ ○ KnowledgeBase class not available")
    except Exception as e:
        print(f"      └─ ✗ Failed: {e}")
    
    # 3. Knowledge Graph / Advanced Reasoner
    print("\n   🗺️  Knowledge Graph & Reasoning")
    graph_reasoner = None
    
    # First try PersistentReasoner (loads from SQLite - has your 18,765 facts!)
    if PersistentReasoner:
        try:
            graph_reasoner = PersistentReasoner(settings.data_dir)
            gr_stats = safe_get_stats(graph_reasoner)
            relations_count = len(gr_stats.get('facts_by_relation', {}))
            print(f"      ├─ Facts: {gr_stats.get('total_facts', 0):,}")
            print(f"      ├─ Entities: {gr_stats.get('unique_subjects', 0):,}")
            print(f"      ├─ Relations: {relations_count}")
            print(f"      └─ ✓ Ready (SQLite)")
            _components['graph_reasoner'] = graph_reasoner
        except Exception as e:
            print(f"      ├─ PersistentReasoner failed: {e}")
    
    # Fallback to in-memory AdvancedReasoner
    if graph_reasoner is None and AdvancedReasoner:
        try:
            graph_reasoner = AdvancedReasoner()  # In-memory, no data_dir
            _components['graph_reasoner'] = graph_reasoner
            print(f"      └─ ✓ Ready (in-memory)")
        except Exception as e:
            print(f"      └─ ○ AdvancedReasoner failed: {e}")
    
    if graph_reasoner is None:
        print(f"      └─ ○ Not available")
    
    # 4. Context Brain
    print("\n   🎯 Context Brain")
    if ContextBrain:
        try:
            context_brain = try_init(ContextBrain, [
                ([], {'embedding_dim': 256, 'max_memories': 100000}),
                ([], {'data_dir': settings.data_dir}),
                ([], {'data_dir': str(settings.data_dir)}),
                ([], {}),
            ])
            
            if context_brain:
                print(f"      ├─ Working Memory: Active")
                print(f"      ├─ Attention: Multi-head")
                print(f"      └─ ✓ Ready")
                _components['context_brain'] = context_brain
            else:
                print(f"      └─ ○ Not available")
        except Exception as e:
            print(f"      └─ ○ Not available: {e}")
    else:
        print(f"      └─ ○ Not available")
    
    # 5. Neural Network
    print("\n   🧠 Neural Network")
    if NEURAL_AVAILABLE and NeuralBrain:
        try:
            neural_brain = NeuralBrain(settings.data_dir, model_size="medium")
            nr_stats = safe_get_stats(neural_brain)
            params = nr_stats.get('model_params', 0)
            tokens = nr_stats.get('total_tokens_trained', 0)
            vocab = nr_stats.get('vocab_size', 0)
            device = nr_stats.get('device', 'cpu')
            model_size = nr_stats.get('model_size', 'medium')
            print(f"      ├─ Size: {model_size}")
            print(f"      ├─ Parameters: {params:,}")
            print(f"      ├─ Tokens trained: {tokens:,}")
            print(f"      ├─ Vocabulary: {vocab:,}")
            print(f"      ├─ Device: {device}")
            
            if hasattr(neural_brain, 'get_timeline'):
                timeline = neural_brain.get_timeline()
                print(f"      ├─ Timeline events: {len(timeline)}")
            
            print(f"      └─ ✓ Ready")
            _components['neural_brain'] = neural_brain
        except Exception as e:
            print(f"      └─ ✗ Failed: {e}")
    else:
        print(f"      └─ ○ Not available (pip install torch)")
    
    # 6. Strategic Planner
    print("\n   🎓 Strategic Planner")
    if StrategicPlanner:
        try:
            strategic_planner = try_init(StrategicPlanner, [
                ([], {'data_dir': settings.data_dir}),
                ([], {'data_dir': str(settings.data_dir)}),
                ([settings.data_dir], {}),
                ([], {}),
            ])
            
            if strategic_planner:
                sp_stats = safe_get_stats(strategic_planner)
                print(f"      ├─ Topics: {sp_stats.get('total_topics', 0)}")
                print(f"      ├─ Mastered: {sp_stats.get('mastered_topics', 0)}")
                print(f"      └─ ✓ Ready")
                _components['strategic_planner'] = strategic_planner
            else:
                print(f"      └─ ○ Not available")
        except Exception as e:
            print(f"      └─ ○ Not available: {e}")
    else:
        print(f"      └─ ○ Not available")
    
    # 7. Response Generator (requires KnowledgeBase)
    print("\n   💬 Response Generator")
    if ResponseGenerator:
        if knowledge_base:
            try:
                response_generator = ResponseGenerator(knowledge_base, data_dir=str(settings.data_dir))
                
                if _components.get('graph_reasoner'):
                    response_generator.graph_reasoner = _components['graph_reasoner']
                if _components.get('neural_brain'):
                    response_generator.neural_brain = _components['neural_brain']
                if _components.get('context_brain'):
                    response_generator.context_brain = _components['context_brain']
                
                print(f"      └─ ✓ Ready")
                _components['response_generator'] = response_generator
            except Exception as e:
                print(f"      └─ ○ Init failed: {e}")
        else:
            print(f"      └─ ○ Requires KnowledgeBase (not available)")
    else:
        print(f"      └─ ○ ResponseGenerator class not available")
    
    # 8. Learning Engine
    print("\n   ⚡ Learning Engine")
    if LearningEngine:
        try:
            # Try different initialization patterns based on old vs new API
            learning_engine = try_init(LearningEngine, [
                # New API: AdvancedLearningEngine(data_dir=, neural_brain=, knowledge_base=)
                ([], {
                    'data_dir': str(settings.data_dir / "learning"),
                    'neural_brain': _components.get('neural_brain'),
                    'knowledge_base': knowledge_base
                }),
                # Just data_dir
                ([], {'data_dir': str(settings.data_dir / "learning")}),
                # Old API: LearningEngine(kb, graph_reasoner=, neural_brain=)
                ([knowledge_base], {
                    'graph_reasoner': _components.get('graph_reasoner'),
                    'neural_brain': _components.get('neural_brain')
                }) if knowledge_base else None,
                # Minimal with kb
                ([knowledge_base], {}) if knowledge_base else None,
                ([], {'knowledge_base': knowledge_base}) if knowledge_base else None,
            ])
            
            if learning_engine:
                setup_learning_callbacks(learning_engine)
                
                # Get learning stats
                le_stats = safe_get_stats(learning_engine)
                mode = "Strategic (curriculum + spaced repetition)" if hasattr(learning_engine, 'strategic') else "Standard"
                articles = le_stats.get('total_articles_learned', le_stats.get('total_articles', 0))
                tokens = le_stats.get('total_tokens_trained', le_stats.get('total_tokens', 0))
                vital = le_stats.get('vital_articles', {})
                vital_total = vital.get('total', 0) if isinstance(vital, dict) else 0
                vital_learned = vital.get('learned', 0) if isinstance(vital, dict) else 0
                
                print(f"      ├─ Mode: {mode}")
                print(f"      ├─ Articles learned: {articles:,}")
                print(f"      ├─ Tokens trained: {tokens:,}")
                print(f"      ├─ Vital articles: {vital_total} ({vital_learned} learned)")
                if _components.get('graph_reasoner'):
                    print(f"      ├─ Knowledge Graph: Connected")
                if _components.get('neural_brain'):
                    print(f"      ├─ Neural Network: Connected")
                print(f"      └─ ✓ Ready")
                _components['learner'] = learning_engine
            else:
                print(f"      └─ ✗ Failed to initialize")
        except Exception as e:
            print(f"      └─ ✗ Failed: {e}")
    else:
        print(f"      └─ ○ Not available")
    
    # 9. Voice Transcription
    print("\n   🎤 Voice Transcription")
    if VOICE_AVAILABLE:
        print(f"      ├─ Backend: Whisper")
        print(f"      ├─ Model: base (lazy loaded)")
        print(f"      └─ ✓ Available")
    else:
        print(f"      └─ ○ Not available (pip install openai-whisper)")
    
    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("📊 SYSTEM SUMMARY")
    print("─" * 70)
    
    # Gather stats from all available components
    kb_stats = safe_get_stats(_components.get('kb'))
    graph_stats = safe_get_stats(_components.get('graph_reasoner'))
    learner_stats = safe_get_stats(_components.get('learner'))
    faiss_index = _components.get('faiss_index')
    
    # Vector count - from FAISS or vector_store
    vector_count = 0
    if faiss_index is not None:
        vector_count = faiss_index.ntotal
    elif _components.get('vector_store'):
        vs_stats = safe_get_stats(_components['vector_store'])
        vector_count = vs_stats.get('total_vectors', 0)
    
    # Facts from knowledge graph
    facts_count = graph_stats.get('total_facts', 0)
    entities_count = graph_stats.get('total_entities', graph_stats.get('unique_subjects', 0))
    
    # Learning stats
    articles_learned = learner_stats.get('total_articles_learned', learner_stats.get('total_articles', 0))
    tokens_trained = learner_stats.get('total_tokens_trained', learner_stats.get('total_tokens', 0))
    
    print(f"   Vectors stored:     {vector_count:,}")
    print(f"   Knowledge facts:    {facts_count:,}")
    print(f"   Entities:           {entities_count:,}")
    print(f"   Articles learned:   {articles_learned:,}")
    print(f"   Tokens trained:     {tokens_trained:,}")
    print(f"   Data directory:     {settings.data_dir.absolute()}")
    
    print("\n   Components:")
    components_status = [
        ("Knowledge Base", 'kb' in _components),
        ("Vector Store", 'vector_store' in _components or 'faiss_index' in _components),
        ("Knowledge Graph", 'graph_reasoner' in _components),
        ("Context Brain", 'context_brain' in _components),
        ("Neural Network", 'neural_brain' in _components),
        ("Strategic Planner", 'strategic_planner' in _components),
        ("Response Generator", 'response_generator' in _components),
        ("Learning Engine", 'learner' in _components),
        ("Voice Transcription", VOICE_AVAILABLE),
    ]
    
    for name, available in components_status:
        status = "✓" if available else "○"
        print(f"      {status} {name}")
    
    print("─" * 70)
    
    if vector_count > 0 or facts_count > 0:
        print(f"\n✅ Loaded {vector_count:,} vectors, {facts_count:,} facts")
    else:
        print("\n📝 Starting fresh - begin learning to add knowledge!")
    
    # Note: Uvicorn runs on port 8000 by default
    print(f"\n🌐 Server ready at http://localhost:8000")
    print(f"📖 API docs at http://localhost:8000/docs")
    print("─" * 70 + "\n")
    
    yield  # Application runs here
    
    # ═══════════════════════════════════════════════════════════════
    # SHUTDOWN
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("👋 SHUTTING DOWN")
    print("─" * 70)
    
    if _components.get('learner'):
        learner = _components['learner']
        if hasattr(learner, 'is_running') and learner.is_running:
            print("   Stopping learning...")
            if hasattr(learner, 'stop'):
                learner.stop()
    
    if _components.get('neural_brain'):
        print("   Saving neural network...")
        try:
            _components['neural_brain'].save()
        except Exception:
            pass
    
    if _components.get('kb'):
        print("   Saving knowledge base...")
        try:
            _components['kb'].save()
        except Exception:
            pass
    
    if _components.get('vector_store'):
        print("   Saving vector store...")
        try:
            _components['vector_store'].save()
        except Exception:
            pass
    
    if _components.get('strategic_planner'):
        print("   Saving strategic planner...")
        try:
            sp = _components['strategic_planner']
            if hasattr(sp, 'save_state'):
                sp.save_state()
            elif hasattr(sp, 'save'):
                sp.save()
        except Exception:
            pass
    
    print("   ✓ Goodbye!\n")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    app = FastAPI(
        title="GroundZero",
        description="Advanced Neural AI System - Built From Scratch",
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    from .routes import router
    app.include_router(router)
    
    static_path = Path(__file__).parent.parent / "static"
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    
    return app


app = create_app()


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the server from command line"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GroundZero API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()
    
    run_server(args.host, args.port)