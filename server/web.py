from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import json
import os
from datetime import datetime
from pathlib import Path

# Import our custom modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.model import NeuralMind
from core.webLearner import WebLearner, SearchEngine

app = Flask(__name__, static_folder='../static', static_url_path='')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global instances
MODEL_PATH = Path(__file__).parent.parent / "savedModel"
model: NeuralMind = None
web_learner: WebLearner = None

def init_model():
    """Initialize or load the model"""
    global model, web_learner
    
    if MODEL_PATH.exists():
        print("Loading existing model...")
        model = NeuralMind.load(str(MODEL_PATH))
    else:
        print("Creating new model...")
        model = NeuralMind(
            vocab_size=50000,
            d_model=256,
            n_heads=8,
            n_layers=6,
            d_ff=1024
        )
        
    web_learner = WebLearner(model)
    
    # Set up callbacks for real-time updates
    web_learner.on_progress = lambda stats: socketio.emit('learning_progress', stats)
    web_learner.on_content = lambda content: socketio.emit('learning_content', content)
    web_learner.on_error = lambda error: socketio.emit('learning_error', {'error': error})
    
    print(f"Model initialized with {model.tokenizer.vocab_count()} vocabulary words")
    print(f"Memory contains {model.memory.memory_count()} knowledge entries")


# ============= API Routes =============

@app.route('/')
def index():
    """Serve the main interface"""
    return send_from_directory('../static', 'index.html')


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current model and learning status"""
    model_stats = model.get_stats()
    learner_stats = web_learner.get_stats()
    
    return jsonify({
        "status": "online",
        "model": model_stats,
        "learner": learner_stats,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat with the AI"""
    data = request.json
    message = data.get('message', '')
    show_reasoning = data.get('show_reasoning', True)
    
    if not message:
        return jsonify({"error": "No message provided"}), 400
    
    if show_reasoning:
        response_data = model.generate_response_with_reasoning(message)
        return jsonify({
            **response_data,
            "stats": model.get_stats()
        })
    else:
        response = model.generate_response(message)
        return jsonify({
            "response": response,
            "stats": model.get_stats()
        })


@app.route('/api/teach', methods=['POST'])
def teach():
    """Teach the AI something new"""
    data = request.json
    content = data.get('content', '')
    source = data.get('source', 'user_teaching')
    
    if not content:
        return jsonify({"error": "No content provided"}), 400
        
    stats = model.learn_from_text(content, source)
    
    # Save model after learning
    model.save(str(MODEL_PATH))
    
    socketio.emit('model_updated', stats)
    
    return jsonify({
        "status": "learned",
        "message": "Knowledge acquired!",
        "stats": stats
    })


@app.route('/api/reason', methods=['POST'])
def reason():
    """Direct reasoning endpoint for logic, math, and code problems"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    result = model.reasoning.solve_with_steps(query)
    
    return jsonify({
        **result,
        "stats": model.get_stats()
    })


@app.route('/api/reason/math', methods=['POST'])
def reason_math():
    """Solve math problems with step-by-step solutions"""
    data = request.json
    expression = data.get('expression', '')
    
    if not expression:
        return jsonify({"error": "No expression provided"}), 400
    
    result = model.reasoning.math.solve(expression)
    
    return jsonify(result.to_dict())


@app.route('/api/reason/logic', methods=['POST'])
def reason_logic():
    """Solve logical reasoning problems"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    result = model.reasoning.logic.solve(query)
    
    return jsonify(result.to_dict())


@app.route('/api/reason/code', methods=['POST'])
def reason_code():
    """Analyze and debug code"""
    data = request.json
    query = data.get('query', '')
    code = data.get('code', '')
    
    if not query and not code:
        return jsonify({"error": "No query or code provided"}), 400
    
    result = model.reasoning.code.solve(query or "Analyze this code", code)
    
    return jsonify(result.to_dict())


@app.route('/api/correct', methods=['POST'])
def correct():
    """Correct the AI and teach proper information"""
    data = request.json
    wrong_response = data.get('wrong_response', '')
    correct_info = data.get('correct_info', '')
    search_for_answer = data.get('search', False)
    
    if search_for_answer:
        # Search for correct information
        search_results = SearchEngine.search_wikipedia(correct_info)
        
        if search_results and 'error' not in search_results[0]:
            # Learn from search results
            for result in search_results[:2]:
                content = SearchEngine.get_wikipedia_content(result['title'])
                if content:
                    model.learn_from_text(content, source=result['url'])
                    
            model.save(str(MODEL_PATH))
            
            return jsonify({
                "status": "searched_and_learned",
                "sources": search_results[:2],
                "stats": model.get_stats()
            })
    else:
        # Direct correction
        result = model.correct_and_learn(wrong_response, correct_info)
        model.save(str(MODEL_PATH))
        
        return jsonify({
            **result,
            "stats": model.get_stats()
        })
        
    return jsonify({"status": "no_results_found"})


@app.route('/api/learn/start', methods=['POST'])
def start_learning():
    """Start autonomous learning"""
    result = web_learner.start_learning()
    return jsonify(result)


@app.route('/api/learn/stop', methods=['POST'])
def stop_learning():
    """Stop autonomous learning"""
    result = web_learner.stop_learning()
    
    # Save model after learning session
    model.save(str(MODEL_PATH))
    
    return jsonify({
        **result,
        "stats": model.get_stats()
    })


@app.route('/api/learn/url', methods=['POST'])
def learn_from_url():
    """Learn from a specific URL"""
    data = request.json
    url = data.get('url', '')
    
    if not url:
        return jsonify({"error": "No URL provided"}), 400
        
    result = web_learner.learn_from_url(url)
    
    # Save model after learning
    model.save(str(MODEL_PATH))
    
    return jsonify(result)


@app.route('/api/learn/search', methods=['POST'])
def search_and_learn():
    """Search for a topic and learn about it"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
        
    result = web_learner.search_and_learn(query)
    
    # Save model after learning
    model.save(str(MODEL_PATH))
    
    return jsonify({
        **result,
        "model_stats": model.get_stats()
    })


@app.route('/api/learn/history', methods=['GET'])
def learning_history():
    """Get learning history"""
    n = request.args.get('n', 20, type=int)
    history = web_learner.get_recent_sites(n)
    
    return jsonify({
        "history": history,
        "total_sites": web_learner.sites_learned
    })


@app.route('/api/model/stats', methods=['GET'])
def model_stats():
    """Get detailed model statistics"""
    return jsonify({
        "model": model.get_stats(),
        "top_words": dict(sorted(
            model.tokenizer.word_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )[:50]),
        "memory_samples": [
            {
                "key": m["key"][:50],
                "source": m["source"],
                "access_count": m["access_count"]
            }
            for m in list(model.memory.memories.values())[-10:]
        ]
    })


@app.route('/api/model/save', methods=['POST'])
def save_model():
    """Manually save the model"""
    result = model.save(str(MODEL_PATH))
    return jsonify(result)


@app.route('/api/model/reset', methods=['POST'])
def reset_model():
    """Reset the model to fresh state"""
    global model, web_learner
    
    # Create fresh model
    model = NeuralMind(
        vocab_size=50000,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024
    )
    
    web_learner.set_model(model)
    
    return jsonify({
        "status": "reset",
        "message": "Model reset to fresh state",
        "stats": model.get_stats()
    })


# ============= WebSocket Events =============

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {
        'status': 'connected',
        'model_stats': model.get_stats(),
        'learner_stats': web_learner.get_stats()
    })


@socketio.on('request_status')
def handle_status_request():
    """Send current status to client"""
    emit('status_update', {
        'model': model.get_stats(),
        'learner': web_learner.get_stats()
    })


# ============= Main =============

if __name__ == '__main__':
    init_model()
    print("\n" + "="*50)
    print("🧠 NeuralMind AI Server Starting...")
    print("="*50)
    print(f"📊 Model loaded with {model.tokenizer.vocab_count()} words")
    print(f"🧮 Memory entries: {model.memory.memory_count()}")
    print(f"🌐 Server running at http://localhost:5000")
    print("="*50 + "\n")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)