import sys
import os
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    from server.web import app, socketio, init_model
    
    init_model()
    
    print("\n" + "═"*65)
    print("  🌐 Server starting at: http://localhost:5000")
    print("  📊 Open this URL in your browser to access the interface")
    print("═"*65 + "\n")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()