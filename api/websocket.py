"""
WebSocket Handlers
==================
Real-time event handlers.
"""

from flask_socketio import SocketIO, emit


def register_socket_handlers(socketio: SocketIO):
    """Register WebSocket event handlers"""
    
    @socketio.on('connect')
    def handle_connect():
        from .server import get_components
        c = get_components()
        
        emit('connected', {
            'status': 'connected',
            'model_stats': c['neural_model'].get_stats() if c['neural_model'] else {},
            'learner_stats': c['learner'].get_stats() if c['learner'] else {},
            'memory_stats': c['memory'].get_statistics() if c['memory'] else {}
        })
    
    @socketio.on('disconnect')
    def handle_disconnect():
        pass
    
    @socketio.on('request_status')
    def handle_status_request():
        from .server import get_components
        c = get_components()
        
        emit('status_update', {
            'model': c['neural_model'].get_stats() if c['neural_model'] else {},
            'learner': c['learner'].get_stats() if c['learner'] else {},
            'memory': c['memory'].get_statistics() if c['memory'] else {}
        })
    
    @socketio.on('request_knowledge_stats')
    def handle_knowledge_stats():
        from .server import get_components
        c = get_components()
        
        if c['memory']:
            emit('knowledge_stats', {
                'statistics': c['memory'].get_statistics(),
                'top_words': c['memory'].get_top_words(30),
                'top_concepts': c['memory'].get_top_concepts(20),
                'top_knowledge': c['memory'].get_top_knowledge(10)
            })
