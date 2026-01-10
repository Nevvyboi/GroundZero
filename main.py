#!/usr/bin/env python3
"""
GroundZero v2.0 - Advanced Neural AI System
============================================
Main entry point for the application.

Usage:
    python main.py                    # Start web server
    python main.py --cli              # Start CLI mode
    python main.py --learn 100        # Learn 100 articles via CLI
    python main.py --port 8080        # Custom port

Features:
    - Web UI with voice input
    - CLI for command-line learning
    - REST API for integration
    - Model timeline tracking
    - Dynamic model size scaling
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the web server"""
    try:
        import uvicorn
        from api.server import app
        
        print(f"\n🚀 Starting GroundZero server...")
        uvicorn.run(app, host=host, port=port, log_level="info")
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Run: pip install uvicorn fastapi")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


def start_cli():
    """Start CLI mode"""
    try:
        from cli import main as cli_main
        cli_main()
    except ImportError as e:
        print(f"Error: CLI module not found - {e}")
        sys.exit(1)


def run_learning(num_articles: int, topics: str = None):
    """Run learning from command line"""
    try:
        from config.settings import Settings
        from storage.knowledge_base import AdvancedKnowledgeBase
        from learning.engine import AdvancedLearningEngine
        
        settings = Settings()
        
        print(f"\n🧠 GroundZero Learning Mode")
        print(f"   Articles: {num_articles}")
        if topics:
            print(f"   Topics: {topics}")
        print()
        
        # Initialize components
        kb = AdvancedKnowledgeBase(
            data_dir=settings.data_dir,
            embedding_dim=settings.embedding_dimension
        )
        
        # Try to load neural brain
        neural_brain = None
        try:
            from neural.brain import NeuralBrain
            neural_brain = NeuralBrain(settings.data_dir, model_size="medium")
        except ImportError:
            print("   Note: Neural network not available (torch not installed)")
        
        # Initialize learning engine
        learner = AdvancedLearningEngine(
            kb,
            neural_brain=neural_brain
        )
        
        # Progress callback
        def on_progress(current, total, stats):
            if current % 50 == 0 or current == total:
                print(f"\n📊 Progress: {current}/{total} articles")
                print(f"   Tokens learned: {stats.get('total_tokens', 0):,}")
                print(f"   Articles/min: {stats.get('articles_per_minute', 0):.1f}")
        
        learner.on_progress = on_progress
        
        # Start learning
        import asyncio
        
        async def learn():
            await learner.learn_articles(num_articles)
        
        asyncio.run(learn())
        
        # Save
        kb.save()
        if neural_brain:
            neural_brain.save()
        
        print(f"\n✅ Learning complete!")
        
    except Exception as e:
        print(f"Error during learning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def check_dependencies():
    """Check and report on dependencies"""
    print("\n🔍 Checking dependencies...\n")
    
    dependencies = [
        ("fastapi", "Web framework"),
        ("uvicorn", "ASGI server"),
        ("torch", "Neural network (PyTorch)"),
        ("numpy", "Numerical computing"),
        ("aiohttp", "Async HTTP client"),
        ("rich", "Terminal formatting"),
        ("whisper", "Voice transcription (optional)"),
    ]
    
    all_good = True
    for module, description in dependencies:
        try:
            __import__(module)
            print(f"   ✓ {module}: {description}")
        except ImportError:
            print(f"   ✗ {module}: {description} - NOT INSTALLED")
            all_good = False
    
    print()
    if all_good:
        print("✅ All dependencies available!")
    else:
        print("⚠️  Some dependencies missing. Install with:")
        print("   pip install fastapi uvicorn torch numpy aiohttp rich")
    print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="GroundZero v2.0 - Advanced Neural AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                    Start web server on default port
    python main.py --port 8080        Start on custom port
    python main.py --cli              Interactive CLI mode
    python main.py --learn 100        Learn 100 articles
    python main.py --check            Check dependencies
        """
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Start in CLI mode instead of web server"
    )
    parser.add_argument(
        "--learn",
        type=int,
        metavar="N",
        help="Learn N articles and exit"
    )
    parser.add_argument(
        "--topics",
        type=str,
        help="Comma-separated list of topics to learn"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check dependencies and exit"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if args.check:
        check_dependencies()
        return
    
    # Learning mode
    if args.learn:
        run_learning(args.learn, args.topics)
        return
    
    # CLI mode
    if args.cli:
        start_cli()
        return
    
    # Default: web server
    start_server(args.host, args.port)


if __name__ == "__main__":
    main()
