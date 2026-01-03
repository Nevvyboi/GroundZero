"""
API Server
==========
Flask server with WebSocket support.
"""

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO

from config import Settings
from storage import Database, MemoryStore, ModelStore
from core import NeuralModel
from reasoning import ReasoningEngine, Metacognition
from learning import ContinuousLearner
from dialogue import ResponseGenerator

from .routes import register_routes
from .websocket import register_socket_handlers


# Global instances
db: Database = None
memory: MemoryStore = None
model_store: ModelStore = None
neural_model: NeuralModel = None
reasoning: ReasoningEngine = None
metacognition: Metacognition = None
learner: ContinuousLearner = None
response_generator: ResponseGenerator = None


def create_app(settings: Settings):
    """Create and configure the Flask application"""
    global db, memory, model_store, neural_model, reasoning, metacognition, learner, response_generator
    
    app = Flask(__name__, static_folder='../static', static_url_path='')
    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    
    # Initialize storage
    print("📦 Initializing storage...")
    db = Database(settings.database_path)
    memory = MemoryStore(db)
    model_store = ModelStore(settings.weights_path, settings.embeddings_path, settings.state_path)
    
    # Initialize knowledge index (indexes existing knowledge)
    print("🔍 Initializing knowledge index...")
    try:
        from core.knowledge_index import get_knowledge_index
        index = get_knowledge_index(memory)
        if index:
            stats = index.get_statistics()
            print(f"   📚 Indexed {stats['total_documents']} documents with {stats['total_terms']} terms")
    except Exception as e:
        print(f"   ⚠️ Knowledge index initialization failed: {e}")
    
    # Initialize neural model
    print("🧠 Initializing neural model...")
    neural_model = NeuralModel(settings.model, memory, model_store)
    
    # Initialize reasoning
    print("🔍 Initializing reasoning engine...")
    reasoning = ReasoningEngine(memory)
    metacognition = Metacognition(memory, neural_model)
    reasoning.set_metacognition(metacognition)
    
    # Initialize learner
    print("📚 Initializing continuous learner...")
    learner = ContinuousLearner(
        memory_store=memory,
        neural_model=neural_model,
        seed_urls=settings.SEED_URLS,
        target_sites=settings.learning.target_sites,
        chunk_size=settings.learning.chunk_size,
        request_delay=settings.learning.request_delay
    )
    
    # Learner callbacks
    learner.on_progress = lambda stats: socketio.emit('learning_progress', {**stats, 'model': neural_model.get_stats()})
    learner.on_content = lambda content: socketio.emit('learning_content', content)
    learner.on_complete = lambda stats: (neural_model.save(), socketio.emit('learning_complete', {'message': '🎉 Complete!', 'stats': stats}))
    learner.on_error = lambda error: socketio.emit('learning_error', {'error': error})
    
    # Initialize response generator
    print("💬 Initializing response generator...")
    response_generator = ResponseGenerator(memory, neural_model, reasoning, metacognition, learner)
    
    # Register routes and handlers
    register_routes(app)
    register_socket_handlers(socketio)
    
    stats = memory.get_statistics()
    print(f"\n📊 Loaded: {stats['vocabulary_size']:,} words | {stats['knowledge_count']:,} knowledge | {stats['sources_learned']:,} sources")
    
    return app, socketio


def run_server(app, socketio, settings: Settings):
    """Run the server"""
    socketio.run(app, host=settings.HOST, port=settings.PORT, debug=settings.DEBUG, allow_unsafe_werkzeug=True)


def get_components():
    """Get global component instances"""
    return {
        'db': db, 'memory': memory, 'model_store': model_store,
        'neural_model': neural_model, 'reasoning': reasoning,
        'metacognition': metacognition, 'learner': learner,
        'response_generator': response_generator
    }