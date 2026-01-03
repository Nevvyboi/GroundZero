"""
API Routes
==========
REST API endpoints with auto-search support.
"""

from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime


def register_routes(app: Flask):
    """Register all API routes"""
    
    @app.route('/')
    def index():
        return send_from_directory('../static', 'index.html')
    
    @app.route('/api/status')
    def status():
        from .server import get_components
        c = get_components()
        
        return jsonify({
            'status': 'online',
            'timestamp': datetime.now().isoformat(),
            'model': c['neural_model'].get_stats() if c['neural_model'] else {},
            'memory': c['memory'].get_statistics() if c['memory'] else {},
            'learner': c['learner'].get_stats() if c['learner'] else {}
        })
    
    @app.route('/api/chat', methods=['POST'])
    def chat():
        """Main chat endpoint with auto-search detection"""
        from .server import get_components
        c = get_components()
        
        data = request.json
        message = data.get('message', '').strip()
        auto_search = data.get('auto_search', True)
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        try:
            result = c['response_generator'].generate(message)
            
            # Check if we should auto-search
            if auto_search and result.get('needs_search', False):
                return jsonify({
                    'needs_search': True,
                    'response': result.get('response', ''),
                    'query': message,
                    'stats': c['neural_model'].get_stats()
                })
            
            return jsonify({
                **result,
                'needs_search': False,
                'stats': c['neural_model'].get_stats()
            })
            
        except Exception as e:
            print(f"Chat error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'response': "I encountered an issue. Please try again.",
                'needs_search': False,
                'confidence': 0,
                'sources': [],
                'error': str(e)
            })
    
    @app.route('/api/chat/search-and-respond', methods=['POST'])
    def search_and_respond():
        """Search for info, learn from it, then respond"""
        from .server import get_components
        c = get_components()
        
        query = request.json.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        try:
            # Search and learn
            learn_result = c['learner'].search_and_learn(query, max_articles=3)
            
            # Now generate response with new knowledge
            result = c['response_generator'].generate(query)
            
            return jsonify({
                **result,
                'searched': True,
                'sources_count': learn_result.get('count', 0),
                'sources': learn_result.get('learned_from', []),
                'stats': c['neural_model'].get_stats()
            })
            
        except Exception as e:
            print(f"Search-respond error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'response': f"I searched but couldn't find good information about that. Try asking something else!",
                'searched': True,
                'sources_count': 0,
                'sources': [],
                'error': str(e)
            })
    
    @app.route('/api/teach', methods=['POST'])
    def teach():
        from .server import get_components
        c = get_components()
        
        data = request.json
        content = data.get('content', '').strip()
        source = data.get('source', 'user_teaching')
        
        if not content:
            return jsonify({'error': 'No content provided'}), 400
        
        # Learn from content
        learn_result = c['neural_model'].learn_from_text(content, source)
        
        # Store as knowledge
        c['memory'].store_knowledge(
            content=content,
            summary=content[:200],
            source_url=source,
            source_title='User Teaching',
            confidence=0.9
        )
        
        # Save model
        c['neural_model'].save()
        
        return jsonify({
            'status': 'learned',
            'message': 'Knowledge acquired!',
            'stats': learn_result
        })
    
    @app.route('/api/learn/start', methods=['POST'])
    def start_learning():
        from .server import get_components
        c = get_components()
        return jsonify(c['learner'].start())
    
    @app.route('/api/learn/stop', methods=['POST'])
    def stop_learning():
        from .server import get_components
        c = get_components()
        result = c['learner'].stop()
        return jsonify({**result, 'stats': c['neural_model'].get_stats()})
    
    @app.route('/api/learn/reset', methods=['POST'])
    def reset_learning():
        from .server import get_components
        c = get_components()
        return jsonify(c['learner'].reset())
    
    @app.route('/api/learn/search', methods=['POST'])
    def search_and_learn():
        from .server import get_components
        c = get_components()
        
        query = request.json.get('query', '').strip()
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        result = c['learner'].search_and_learn(query)
        return jsonify({**result, 'model_stats': c['neural_model'].get_stats()})
    
    @app.route('/api/learn/url', methods=['POST'])
    def learn_url():
        from .server import get_components
        c = get_components()
        
        url = request.json.get('url', '').strip()
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        result = c['learner'].learn_from_any_url(url)
        return jsonify({**result, 'stats': c['neural_model'].get_stats()})
    
    @app.route('/api/learn/web-search', methods=['POST'])
    def web_search_and_learn():
        from .server import get_components
        c = get_components()
        
        query = request.json.get('query', '').strip()
        max_results = request.json.get('max_results', 5)
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        result = c['learner'].web_search_and_learn(query, max_results=max_results)
        return jsonify({**result, 'model_stats': c['neural_model'].get_stats()})
    
    @app.route('/api/learn/history')
    def learning_history():
        from .server import get_components
        c = get_components()
        
        n = request.args.get('n', 20, type=int)
        history = c['memory'].get_recent_sources(n)
        
        return jsonify({
            'history': history,
            'total': c['memory'].get_learned_sources_count()
        })
    
    @app.route('/api/knowledge/search')
    def search_knowledge():
        from .server import get_components
        c = get_components()
        
        query = request.args.get('q', '')
        limit = request.args.get('limit', 10, type=int)
        
        results = c['memory'].search_knowledge(query, limit)
        return jsonify({'results': results, 'query': query})
    
    @app.route('/api/knowledge/stats')
    def knowledge_stats():
        from .server import get_components
        c = get_components()
        
        return jsonify({
            'statistics': c['memory'].get_statistics(),
            'top_words': c['memory'].get_top_words(30),
            'recent_words': c['memory'].get_recent_words(30),
            'recent_sources': c['memory'].get_recent_sources(10),
            'top_concepts': c['memory'].get_top_concepts(20),
            'top_knowledge': c['memory'].get_top_knowledge(10)
        })
    
    @app.route('/api/knowledge/rebuild-index', methods=['POST'])
    def rebuild_index():
        """Rebuild the knowledge index from all existing knowledge"""
        from .server import get_components
        c = get_components()
        
        try:
            from core.knowledge_index import rebuild_knowledge_index
            stats = rebuild_knowledge_index(c['memory'])
            return jsonify({
                'status': 'success',
                'message': 'Knowledge index rebuilt',
                'statistics': stats
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @app.route('/api/knowledge/index-stats')
    def index_stats():
        """Get knowledge index statistics"""
        from .server import get_components
        c = get_components()
        
        try:
            from core.knowledge_index import get_knowledge_index
            index = get_knowledge_index(c['memory'])
            if index:
                return jsonify({
                    'status': 'success',
                    'statistics': index.get_statistics()
                })
            else:
                return jsonify({
                    'status': 'not_initialized',
                    'statistics': {}
                })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @app.route('/api/model/stats')
    def model_stats():
        from .server import get_components
        c = get_components()
        
        return jsonify({
            'model': c['neural_model'].get_stats(),
            'storage': c['model_store'].get_storage_info()
        })
    
    @app.route('/api/model/save', methods=['POST'])
    def save_model():
        from .server import get_components
        c = get_components()
        
        result = c['neural_model'].save()
        return jsonify(result)