#!/usr/bin/env python3
"""
GroundZero AI v3.0 - Main Entry Point
=====================================

A neurosymbolic AI system with neural network capabilities.

Usage:
    python main.py test       # Run verification tests
    python main.py chat       # Start smart chat
    python main.py train      # Train with knowledge
    python main.py neural     # Train neural network
    python main.py status     # View progress status
    python main.py dashboard  # Launch web dashboard with API

Author: GroundZero AI
Version: 3.0.0
"""

import sys
import os
import json
import time
import threading
import http.server
import socketserver
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.knowledge_graph import KnowledgeGraph
from src.causal_graph import CausalGraph
from src.question_detector import QuestionTypeDetector, QuestionType, ThinkingMode
from src.metacognition import MetacognitiveController
from src.reasoning import ChainOfThoughtReasoner
from src.constitutional import Constitution
from src.chat_engine import SmartChatEngine
from src.progress_tracker import ProgressTracker
from src.neural_engine import NeuralEngine, NeuralStats

# Configuration
DATA_DIR = Path(__file__).parent / "data"
DASHBOARD_DIR = Path(__file__).parent / "dashboard"
NEURAL_MODEL_PATH = DATA_DIR / "neural_model.pkl"


# =============================================================================
# GLOBAL STATE (for API access)
# =============================================================================

class GlobalState:
    """Shared state for the dashboard API"""
    Engine: SmartChatEngine = None
    Neural: NeuralEngine = None
    LossHistory: list = []
    
    @classmethod
    def Initialize(cls):
        """Initialize the global state"""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        cls.Engine = SmartChatEngine(str(DATA_DIR))
        cls.Neural = NeuralEngine(EmbeddingDim=100)
        
        # Load neural model if exists
        if NEURAL_MODEL_PATH.exists():
            try:
                cls.Neural.Load(str(NEURAL_MODEL_PATH))
                print(f"✅ Loaded neural model: {cls.Neural.Stats.TotalTriples} triples")
            except Exception as e:
                print(f"⚠️ Could not load neural model: {e}")
        
        # Sync knowledge to neural network
        cls.SyncKnowledgeToNeural()
    
    @classmethod
    def SyncKnowledgeToNeural(cls):
        """Sync facts from knowledge graph to neural network"""
        if cls.Engine and cls.Neural:
            count = 0
            for triples in cls.Engine.Knowledge.BySubject.values():
                for triple in triples:
                    cls.Neural.AddTriple(triple.Subject, triple.Predicate, triple.Object)
                    count += 1
            
            if count > 0:
                print(f"📊 Synced {count} facts to neural network")
    
    @classmethod
    def GetStats(cls) -> dict:
        """Get combined stats for dashboard"""
        stats = {
            "Knowledge": {},
            "Causal": {},
            "Chat": {},
            "Neural": {},
            "Progress": {},
            "LossHistory": cls.Neural.LossHistory if cls.Neural else []
        }
        
        if cls.Engine:
            engine_stats = cls.Engine.GetStats()
            stats["Knowledge"] = engine_stats.get("Knowledge", {})
            stats["Causal"] = engine_stats.get("Causal", {})
            stats["Chat"] = engine_stats.get("Chat", {})
            
            # Progress
            tracker = ProgressTracker(cls.Engine.Knowledge, cls.Engine.Causal)
            level_info = tracker.GetCurrentLevel()
            stats["Progress"] = {
                "Level": level_info["CurrentLevel"].get("Level", 0),
                "LevelName": level_info["CurrentLevel"].get("Name", "Unknown"),
                "Description": level_info["CurrentLevel"].get("Description", ""),
                "ProgressPercent": level_info.get("ProgressPercent", 0),
                "CurrentFacts": level_info.get("CurrentFacts", 0),
                "NextMilestone": level_info.get("NextMilestone", 100)
            }
        
        if cls.Neural:
            stats["Neural"] = cls.Neural.GetStatsDict()
        
        return stats
    
    @classmethod
    def Save(cls):
        """Save all state"""
        if cls.Engine:
            cls.Engine.Save()
        if cls.Neural:
            cls.Neural.Save(str(NEURAL_MODEL_PATH))


# =============================================================================
# COMBINED DASHBOARD + API HANDLER (Single Port)
# =============================================================================

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler serving both static files AND API endpoints on same port"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests - API or static files"""
        parsed = urlparse(self.path)
        
        # API endpoints
        if parsed.path == '/api/stats':
            self.send_json(GlobalState.GetStats())
        elif parsed.path == '/api/neural/predict':
            params = parse_qs(parsed.query)
            self.handle_predict(params)
        elif parsed.path == '/api/neural/similar':
            params = parse_qs(parsed.query)
            self.handle_similar(params)
        elif parsed.path == '/api/neural/hypotheses':
            self.handle_hypotheses()
        else:
            # Serve static files from dashboard folder
            super().do_GET()
    
    def do_POST(self):
        """Handle POST requests"""
        parsed = urlparse(self.path)
        
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        
        try:
            data = json.loads(body) if body else {}
        except:
            data = {}
        
        if parsed.path == '/api/chat':
            self.handle_chat(data)
        elif parsed.path == '/api/learn':
            self.handle_learn(data)
        elif parsed.path == '/api/neural/train':
            self.handle_neural_train(data)
        elif parsed.path == '/api/neural/add':
            self.handle_neural_add(data)
        else:
            self.send_error(404)
    
    def send_json(self, data, status=200):
        """Send JSON response with CORS headers"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def handle_chat(self, data):
        """Handle chat message"""
        message = data.get('message', '')
        if GlobalState.Engine and message:
            response = GlobalState.Engine.Process(message)
            self.send_json({
                "answer": response.Answer,
                "confidence": response.Confidence,
                "questionType": response.QuestionType.name,
                "thinkingMode": response.ThinkingMode.name
            })
        else:
            self.send_json({"error": "No message provided"}, 400)
    
    def handle_learn(self, data):
        """Handle learning new knowledge"""
        text = data.get('text', '')
        if GlobalState.Engine and text:
            result = GlobalState.Engine.Learn(text)
            GlobalState.SyncKnowledgeToNeural()
            self.send_json(result)
        else:
            self.send_json({"error": "No text provided"}, 400)
    
    def handle_neural_train(self, data):
        """Handle neural network training"""
        epochs = data.get('epochs', 50)
        if GlobalState.Neural:
            # Train in background thread
            def train():
                GlobalState.Neural.Train(Epochs=epochs, Verbose=True)
                GlobalState.Save()
            
            thread = threading.Thread(target=train)
            thread.start()
            
            self.send_json({"status": "training_started", "epochs": epochs})
        else:
            self.send_json({"error": "Neural engine not initialized"}, 500)
    
    def handle_neural_add(self, data):
        """Add triple to neural network"""
        head = data.get('head')
        relation = data.get('relation')
        tail = data.get('tail')
        
        if GlobalState.Neural and head and relation and tail:
            GlobalState.Neural.AddTriple(head, relation, tail)
            self.send_json({"status": "added"})
        else:
            self.send_json({"error": "Invalid triple"}, 400)
    
    def handle_predict(self, params):
        """Handle prediction request"""
        head = params.get('head', [''])[0]
        relation = params.get('relation', [''])[0]
        
        if GlobalState.Neural and head and relation:
            predictions = GlobalState.Neural.PredictTail(head, relation, TopK=5)
            self.send_json({
                "predictions": [p.ToDict() for p in predictions]
            })
        else:
            self.send_json({"error": "Missing parameters"}, 400)
    
    def handle_similar(self, params):
        """Handle similarity search"""
        entity = params.get('entity', [''])[0]
        
        if GlobalState.Neural and entity:
            similar = GlobalState.Neural.FindSimilar(entity, TopK=5)
            self.send_json({
                "similar": [{"entity": e, "similarity": s} for e, s in similar]
            })
        else:
            self.send_json({"error": "Missing entity"}, 400)
    
    def handle_hypotheses(self):
        """Generate hypotheses"""
        if GlobalState.Neural:
            hypotheses = GlobalState.Neural.GenerateHypotheses(MinConfidence=0.3, MaxHypotheses=10)
            self.send_json({
                "hypotheses": [h.ToDict() for h in hypotheses]
            })
        else:
            self.send_json({"error": "Neural engine not initialized"}, 500)
    
    def log_message(self, format, *args):
        """Custom logging"""
        if '/api/' in args[0]:
            print(f"🔌 API: {args[0]}")
        # Suppress other logs


# =============================================================================
# COMMANDS
# =============================================================================

def RunTests():
    """Run verification tests on all components"""
    print("\n" + "=" * 70)
    print("🔬 GroundZero AI v3.0 - Verification Tests")
    print("=" * 70 + "\n")
    
    Results = {}
    
    # Test neural engine
    print("1. Testing Neural Engine...")
    try:
        neural = NeuralEngine(EmbeddingDim=50)
        neural.AddTriples([
            ("dog", "is_a", "mammal"),
            ("cat", "is_a", "mammal"),
            ("mammal", "is_a", "animal"),
        ])
        losses = neural.Train(Epochs=20, Verbose=False)
        predictions = neural.PredictTail("dog", "is_a", TopK=3)
        
        Results["NeuralEngine"] = len(predictions) > 0 and losses[-1] < losses[0]
        print(f"   ✓ Neural Engine: {len(losses)} epochs, final loss {losses[-1]:.4f}")
        print(f"     Top prediction: {predictions[0].Tail} ({predictions[0].Confidence:.0%})\n")
    except Exception as E:
        Results["NeuralEngine"] = False
        print(f"   ✗ Error: {E}\n")
    
    # Test other components...
    print("2. Testing Knowledge Graph...")
    try:
        KG = KnowledgeGraph()
        KG.Add("dog", "is_a", "animal")
        Results["KnowledgeGraph"] = KG.Size() > 0
        print(f"   ✓ Knowledge Graph: {KG.Size()} facts\n")
    except Exception as E:
        Results["KnowledgeGraph"] = False
        print(f"   ✗ Error: {E}\n")
    
    print("3. Testing Causal Graph...")
    try:
        CG = CausalGraph()
        CG.AddCause("rain", "wet_ground", Strength=0.9)
        Results["CausalGraph"] = CG.Size() > 0
        print(f"   ✓ Causal Graph: {CG.Size()} relations\n")
    except Exception as E:
        Results["CausalGraph"] = False
        print(f"   ✗ Error: {E}\n")
    
    # Summary
    print("=" * 70)
    print("📊 Test Summary")
    print("=" * 70)
    
    Passed = sum(1 for r in Results.values() if r)
    for Test, Result in Results.items():
        Status = "✓ PASS" if Result else "✗ FAIL"
        print(f"  {Test:25} {Status}")
    
    print(f"\n  Total: {Passed}/{len(Results)} tests passed")
    print("=" * 70 + "\n")
    
    return Passed == len(Results)


def StartChat():
    """Start interactive chat session"""
    print("\n" + "=" * 70)
    print("🧠 GroundZero AI v3.0 - Neural Chat")
    print("=" * 70)
    print("Commands: /neural (train), /predict <entity>, /stats, /quit")
    print("=" * 70 + "\n")
    
    GlobalState.Initialize()
    
    try:
        while True:
            try:
                UserInput = input("You: ").strip()
            except EOFError:
                break
            
            if not UserInput:
                continue
            
            if UserInput.lower() == "/quit":
                break
            elif UserInput.lower() == "/neural":
                print("\n🧠 Training neural network...")
                GlobalState.Neural.Train(Epochs=50, Verbose=True)
                print()
                continue
            elif UserInput.lower().startswith("/predict "):
                entity = UserInput[9:].strip()
                print(f"\n🔮 Predicting for '{entity}':")
                for pred in GlobalState.Neural.PredictTail(entity, "is_a", TopK=5):
                    print(f"   → {pred.Tail}: {pred.Confidence:.0%}")
                print()
                continue
            elif UserInput.lower() == "/stats":
                stats = GlobalState.GetStats()
                print(f"\n📊 Neural Stats:")
                for k, v in stats["Neural"].items():
                    print(f"   {k}: {v}")
                print()
                continue
            
            # Normal chat
            Response = GlobalState.Engine.Process(UserInput)
            print(f"\n🧠 GroundZero: {Response.Answer}")
            print(f"   [{Response.QuestionType.name} | {Response.Confidence:.0%}]\n")
    
    finally:
        GlobalState.Save()
        print("\n👋 Goodbye!\n")


def TrainNeural():
    """Train neural network with existing knowledge"""
    print("\n" + "=" * 70)
    print("🧠 GroundZero AI - Neural Network Training")
    print("=" * 70 + "\n")
    
    GlobalState.Initialize()
    
    print(f"Knowledge triples: {GlobalState.Neural.Stats.TotalTriples}")
    print(f"Entities: {GlobalState.Neural.Stats.TotalEntities}")
    print(f"Relations: {GlobalState.Neural.Stats.TotalRelations}")
    
    try:
        epochs = int(input("\nTraining epochs [100]: ").strip() or "100")
    except:
        epochs = 100
    
    GlobalState.Neural.Train(Epochs=epochs, Verbose=True)
    GlobalState.Save()
    
    # Test predictions
    print("\n📊 Testing predictions...")
    entities = list(GlobalState.Neural.EntityEmbeddings.keys())[:5]
    for entity in entities:
        preds = GlobalState.Neural.PredictTail(entity, "is_a", TopK=2)
        if preds:
            print(f"   {entity} is_a → {preds[0].Tail} ({preds[0].Confidence:.0%})")
    
    print("\n✅ Training complete!")


def StartDashboard(Port: int = 8080):
    """Start web dashboard with API on SAME port"""
    print("\n" + "=" * 70)
    print("🌐 GroundZero AI v3.0 - Neural Dashboard")
    print("=" * 70)
    
    GlobalState.Initialize()
    
    print(f"\n📊 Dashboard + API: http://localhost:{Port}")
    print("   API endpoints:   /api/stats, /api/chat, /api/neural/*")
    print("\nPress Ctrl+C to stop\n")
    
    # Single server handles BOTH dashboard files AND API
    class ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True
    
    with ThreadedServer(("", Port), DashboardHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            GlobalState.Save()
            print("\n\n👋 Dashboard stopped\n")


def ShowStatus():
    """Show current status"""
    GlobalState.Initialize()
    
    stats = GlobalState.GetStats()
    
    print("\n" + "=" * 70)
    print("📊 GroundZero AI v3.0 - Status")
    print("=" * 70)
    
    print("\n📚 Knowledge:")
    print(f"   Facts: {stats['Knowledge'].get('TotalFacts', 0)}")
    print(f"   Causal: {stats['Causal'].get('TotalRelations', 0)}")
    
    print("\n🧠 Neural Network:")
    for k, v in stats['Neural'].items():
        print(f"   {k}: {v}")
    
    print("\n📈 Progress:")
    print(f"   Level: {stats['Progress'].get('Level', 0)} - {stats['Progress'].get('LevelName', 'Unknown')}")
    print(f"   {stats['Progress'].get('CurrentFacts', 0)} / {stats['Progress'].get('NextMilestone', 100)} facts")
    
    print("\n" + "=" * 70 + "\n")


def PrintHelp():
    """Print help message"""
    print("""
🧠 GroundZero AI v3.0 - Neurosymbolic Intelligence
==================================================

Usage:
    python main.py <command>

Commands:
    test        Run verification tests
    chat        Start interactive neural chat
    train       Train system with knowledge
    neural      Train neural network specifically
    status      View current status
    dashboard   Launch web dashboard with live neural stats
    help        Show this help message

Examples:
    python main.py test
    python main.py chat
    python main.py neural
    python main.py dashboard

Neural Capabilities:
    ✓ UNDERSTAND - Learn semantic embeddings
    ✓ GENERALIZE - Predict unseen facts
    ✓ LEARN      - Continuously improve
    ✓ REASON     - Infer relationships
    ✓ CREATE     - Generate hypotheses
""")


def Main():
    """Main entry point"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if len(sys.argv) < 2:
        PrintHelp()
        return
    
    Command = sys.argv[1].lower()
    
    if Command == "test":
        Success = RunTests()
        sys.exit(0 if Success else 1)
    elif Command == "chat":
        StartChat()
    elif Command == "train":
        from src.auto_learner import AutoLearner
        GlobalState.Initialize()
        
        ArticleCount = [0]  # Use list for mutable closure
        
        def OnArticleLearned(Title, Facts, Causal):
            """Callback after each article - trains neural every 10 articles"""
            ArticleCount[0] += 1
            
            # Sync new facts to neural engine
            GlobalState.SyncKnowledgeToNeural()
            
            # Train neural every 10 articles
            if ArticleCount[0] % 10 == 0:
                print(f"\n🧠 Auto-training neural network ({GlobalState.Neural.Stats.TotalTriples} triples)...")
                GlobalState.Neural.Train(Epochs=20, Verbose=False)
                loss = GlobalState.Neural.LossHistory[-1] if GlobalState.Neural.LossHistory else 0
                print(f"   ✓ Trained! Loss: {loss:.4f}")
                
                # Save progress
                GlobalState.Save()
                print(f"   💾 Saved checkpoint\n")
            
            return True  # Continue learning
        
        Learner = AutoLearner(GlobalState.Engine)
        
        try:
            Learner.LearnContinuously(Callback=OnArticleLearned)
        finally:
            # Final neural training when stopping
            if GlobalState.Neural.Stats.TotalTriples > 0:
                print(f"\n🧠 Final neural training ({GlobalState.Neural.Stats.TotalTriples} triples)...")
                GlobalState.Neural.Train(Epochs=50, Verbose=True)
                GlobalState.Save()
                print("✅ Training complete and saved!")
    elif Command == "neural":
        TrainNeural()
    elif Command == "status":
        ShowStatus()
    elif Command == "dashboard":
        Port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
        StartDashboard(Port)
    elif Command == "help":
        PrintHelp()
    else:
        print(f"Unknown command: {Command}")
        PrintHelp()
        sys.exit(1)


if __name__ == "__main__":
    Main()