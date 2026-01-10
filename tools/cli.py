#!/usr/bin/env python3
"""
GroundZero CLI v2.0 - Command Line Interface
=============================================
Control the learning process from the command line.

FEATURES:
- Start/stop/pause learning
- View progress statistics every N articles
- Change model size with backfilling
- Query the model directly
- View timeline/history
- Export/import data

Usage:
    python cli.py learn --articles 100 --show-progress 50
    python cli.py chat "What is machine learning?"
    python cli.py model --size medium --backfill
    python cli.py timeline
    python cli.py stats
"""

import argparse
import sys
import time
import signal
import json
from pathlib import Path
from typing import Optional
import threading


# Rich console for pretty output (optional)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


class CLIController:
    """Controls the GroundZero system from CLI"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Components (lazy loaded)
        self._kb = None
        self._learner = None
        self._neural_brain = None
        self._graph_reasoner = None
        
        # Learning state
        self.is_learning = False
        self._stop_requested = False
        self._articles_learned = 0
        
        # Stats display interval
        self.show_progress_every = 50
    
    def _init_components(self):
        """Initialize all components"""
        if self._kb is not None:
            return
        
        self._print("🔧 Initializing GroundZero...", style="bold cyan")
        
        try:
            # Import components
            sys.path.insert(0, str(Path(__file__).parent))
            
            from storage import KnowledgeBase
            from learning import LearningEngine
            from neural import NeuralBrain
            from reasoning import AdvancedReasoner, PersistentKnowledgeGraph
            
            # Knowledge Base
            self._kb = KnowledgeBase(self.data_dir / "knowledge", dimension=256)
            self._print("   ├─ ✓ Knowledge Base", style="green")
            
            # Knowledge Graph
            try:
                graph = PersistentKnowledgeGraph(self.data_dir / "graph")
                self._graph_reasoner = AdvancedReasoner(graph)
                self._print("   ├─ ✓ Knowledge Graph", style="green")
            except Exception as e:
                self._print(f"   ├─ ⚠ Knowledge Graph: {e}", style="yellow")
                self._graph_reasoner = None
            
            # Neural Brain
            try:
                self._neural_brain = NeuralBrain(self.data_dir, model_size="small")
                self._print("   ├─ ✓ Neural Brain", style="green")
            except Exception as e:
                self._print(f"   ├─ ⚠ Neural Brain: {e}", style="yellow")
                self._neural_brain = None
            
            # Learning Engine
            self._learner = LearningEngine(
                self._kb,
                graph_reasoner=self._graph_reasoner,
                neural_brain=self._neural_brain
            )
            self._print("   └─ ✓ Learning Engine", style="green")
            
            self._print("✅ GroundZero initialized!", style="bold green")
            
        except Exception as e:
            self._print(f"❌ Initialization failed: {e}", style="bold red")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def _print(self, message: str, style: str = None):
        """Print with optional rich styling"""
        if RICH_AVAILABLE and console:
            console.print(message, style=style)
        else:
            print(message)
    
    def _create_stats_table(self) -> str:
        """Create statistics table"""
        if not self._learner:
            return "Not initialized"
        
        stats = self._learner.get_stats()
        
        if RICH_AVAILABLE:
            table = Table(title="📊 Learning Statistics", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            # Current session
            session = stats.get('current_session', {})
            table.add_row("Articles (Session)", str(session.get('articles_read', 0)))
            table.add_row("Words (Session)", f"{session.get('words_learned', 0):,}")
            table.add_row("Facts Extracted", str(session.get('facts_extracted', 0)))
            
            # Total
            total = stats.get('total', {})
            table.add_row("Total Knowledge", str(total.get('total_knowledge', 0)))
            table.add_row("Total Words", f"{total.get('total_words', 0):,}")
            table.add_row("Vocabulary Size", f"{total.get('vocabulary_size', 0):,}")
            
            # Neural
            if self._neural_brain and self._neural_brain.is_available:
                neural = self._neural_brain.get_stats()
                table.add_row("Neural Parameters", f"{neural.get('model_params', 0):,}")
                table.add_row("Tokens Trained", f"{neural.get('total_tokens_trained', 0):,}")
                recent_losses = neural.get('recent_losses', [])
                if recent_losses:
                    table.add_row("Recent Loss", f"{recent_losses[-1]:.4f}")
            
            # Strategic
            strategic = stats.get('strategic', {})
            if strategic.get('available'):
                table.add_row("Vital Progress", str(strategic.get('vital_progress', {})))
                table.add_row("Next Source", strategic.get('next_source', 'N/A'))
            
            return table
        else:
            # Plain text stats
            lines = ["=" * 50, "📊 Learning Statistics", "=" * 50]
            session = stats.get('current_session', {})
            lines.append(f"Articles (Session): {session.get('articles_read', 0)}")
            lines.append(f"Words (Session): {session.get('words_learned', 0):,}")
            lines.append(f"Facts Extracted: {session.get('facts_extracted', 0)}")
            
            total = stats.get('total', {})
            lines.append(f"Total Knowledge: {total.get('total_knowledge', 0)}")
            lines.append(f"Vocabulary Size: {total.get('vocabulary_size', 0):,}")
            
            if self._neural_brain:
                neural = self._neural_brain.get_stats()
                lines.append(f"Neural Parameters: {neural.get('model_params', 0):,}")
            
            lines.append("=" * 50)
            return "\n".join(lines)
    
    def cmd_learn(self, articles: int = 100, show_progress: int = 50, 
              topics: Optional[str] = None, forever: bool = False,
              save_every: int = 100):
        """Start learning process
        
        Args:
            articles: Number of articles to learn (ignored if forever=True)
            show_progress: Show stats every N articles
            topics: Comma-separated topics to focus on
            forever: Learn continuously until stopped
            save_every: Save progress every N articles in forever mode
        """
        self._init_components()
        
        self.show_progress_every = show_progress
        self._articles_learned = 0
        self._stop_requested = False
        
        if forever:
            self._print(f"\n🔄 Starting CONTINUOUS learning (Ctrl+C to stop)", style="bold magenta")
            self._print(f"   Save every {save_every} articles", style="dim")
            target_articles = float('inf')
        else:
            self._print(f"\n🎓 Starting learning: {articles} articles", style="bold blue")
            target_articles = articles
        
        self._print(f"   Progress every {show_progress} articles", style="dim")
        
        # Set up signal handler for graceful stop
        def signal_handler(sig, frame):
            self._print("\n⏹️  Stop requested...", style="yellow")
            self._stop_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Add topics if specified
        if topics:
            for topic in topics.split(','):
                topic = topic.strip()
                if topic:
                    self._learner.add_category_to_learn(topic)
                    self._print(f"   Added topic: {topic}", style="cyan")
        
        # Start learning
        result = self._learner.start()
        self._print(f"   Session ID: {result.get('session_id')}", style="dim")
        self._print(f"   Mode: {result.get('mode', 'unknown')}", style="dim")
        
        # Progress tracking
        start_time = time.time()
        last_progress_count = 0
        last_save_count = 0
        
        if RICH_AVAILABLE and not forever:
            # Finite mode with progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Learning...", total=target_articles if not forever else None)
                
                while self._articles_learned < target_articles and not self._stop_requested:
                    time.sleep(1)
                    
                    stats = self._learner.get_stats()
                    current = stats.get('current_session', {}).get('articles_read', 0)
                    
                    if current > self._articles_learned:
                        self._articles_learned = current
                        progress.update(task, completed=current)
                        
                        # Show detailed stats every N articles
                        if current >= last_progress_count + show_progress:
                            last_progress_count = current
                            progress.stop()
                            console.print(self._create_stats_table())
                            progress.start()
                    
                    # Check if learner is still running
                    if not self._learner.is_running:
                        break
        else:
            # Forever mode or plain text progress
            self._print("\n" + "=" * 50, style="dim")
            self._print("📚 Learning in progress... (Ctrl+C to stop)", style="cyan")
            self._print("=" * 50 + "\n", style="dim")
            
            while not self._stop_requested:
                time.sleep(2)
                
                stats = self._learner.get_stats()
                current = stats.get('current_session', {}).get('articles_read', 0)
                
                if current > self._articles_learned:
                    self._articles_learned = current
                    
                    # Show progress
                    elapsed = time.time() - start_time
                    rate = current / elapsed if elapsed > 0 else 0
                    
                    if forever:
                        # Forever mode: show running totals
                        total_kb = stats.get('total', {}).get('total_knowledge', 0)
                        print(f"\r🔄 Articles: {current:,} | Rate: {rate:.1f}/min | KB Items: {total_kb:,}   ", 
                              end="", flush=True)
                    else:
                        print(f"\r📚 Progress: {current}/{target_articles} ({rate:.1f} articles/min)", 
                              end="", flush=True)
                    
                    # Show detailed stats every N articles
                    if current >= last_progress_count + show_progress:
                        last_progress_count = current
                        print()
                        if RICH_AVAILABLE:
                            console.print(self._create_stats_table())
                        else:
                            print(self._create_stats_table())
                    
                    # Save periodically in forever mode
                    if forever and current >= last_save_count + save_every:
                        last_save_count = current
                        self._save_all()
                        self._print(f"\n💾 Auto-saved at {current} articles", style="green")
                
                # Check if we've reached target (for non-forever mode)
                if not forever and self._articles_learned >= target_articles:
                    break
                    
                if not self._learner.is_running:
                    break
        
        # Stop learning
        result = self._learner.stop()
        
        # Final save
        self._save_all()
        
        # Final stats
        elapsed = time.time() - start_time
        self._print(f"\n✅ Learning complete!", style="bold green")
        self._print(f"   Articles: {self._articles_learned:,}", style="cyan")
        self._print(f"   Time: {elapsed/60:.1f} minutes", style="cyan")
        self._print(f"   Rate: {self._articles_learned/elapsed*60:.1f} articles/min", style="cyan")
        
        # Show final stats table
        if RICH_AVAILABLE:
            console.print(self._create_stats_table())
        else:
            print(self._create_stats_table())
    
    def _save_all(self):
        """Save all components"""
        try:
            if self._kb:
                self._kb.save()
            if self._neural_brain and self._neural_brain.is_available:
                self._neural_brain.save()
            if hasattr(self._learner, 'strategic_planner'):
                self._learner.strategic_planner.save()
        except Exception as e:
            self._print(f"⚠️ Save error: {e}", style="yellow")
    
    def cmd_chat(self, message: str, use_neural: bool = True):
        """Chat with the model"""
        self._init_components()
        
        self._print(f"\n💬 You: {message}", style="bold")
        
        if use_neural and self._neural_brain and self._neural_brain.is_available:
            result = self._neural_brain.answer(message, use_reasoning=True)
            self._print(f"\n🤖 GroundZero: {result.get('answer', 'No answer')}", style="green")
            self._print(f"   (Confidence: {result.get('confidence', 0):.1%}, Method: {result.get('method', 'unknown')})", style="dim")
        else:
            # Use knowledge base search
            results = self._kb.search(message, top_k=3)
            if results:
                self._print(f"\n🔍 Found {len(results)} relevant entries:", style="cyan")
                for i, r in enumerate(results, 1):
                    content = r.get('content', '')[:200]
                    self._print(f"\n{i}. {content}...", style="white")
            else:
                self._print("\n🤷 No relevant knowledge found.", style="yellow")
    
    def cmd_model(self, size: str = None, backfill: bool = True, info: bool = False):
        """Manage neural model"""
        self._init_components()
        
        if not self._neural_brain or not self._neural_brain.is_available:
            self._print("❌ Neural brain not available", style="red")
            return
        
        if info or size is None:
            # Show model info
            stats = self._neural_brain.get_stats()
            
            if RICH_AVAILABLE:
                table = Table(title="🧠 Neural Model Info")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="green")
                
                table.add_row("Model Size", stats.get('model_size', 'unknown'))
                table.add_row("Parameters", f"{stats.get('model_params', 0):,}")
                table.add_row("Layers", str(stats.get('n_layers', 0)))
                table.add_row("Attention Heads", str(stats.get('n_heads', 0)))
                table.add_row("Embedding Dim", str(stats.get('d_model', 0)))
                table.add_row("Max Sequence", str(stats.get('max_seq_len', 0)))
                table.add_row("Vocab Size", str(stats.get('vocab_size', 0)))
                table.add_row("Tokens Trained", f"{stats.get('total_tokens_trained', 0):,}")
                table.add_row("Device", stats.get('device', 'unknown'))
                table.add_row("Available Sizes", ", ".join(stats.get('available_sizes', [])))
                
                console.print(table)
            else:
                self._print("🧠 Neural Model Info:")
                self._print(f"   Size: {stats.get('model_size')}")
                self._print(f"   Parameters: {stats.get('model_params', 0):,}")
                self._print(f"   Tokens Trained: {stats.get('total_tokens_trained', 0):,}")
        
        elif size:
            # Change model size
            self._print(f"\n🔄 Changing model size to: {size}", style="bold blue")
            self._print(f"   Backfill: {backfill}", style="dim")
            
            result = self._neural_brain.change_model_size(size, backfill=backfill)
            
            if 'error' in result:
                self._print(f"❌ Error: {result['error']}", style="red")
            else:
                self._print(f"✅ Model changed!", style="green")
                self._print(f"   Old: {result.get('old_size')} ({result.get('old_params', 0):,} params)")
                self._print(f"   New: {result.get('new_size')} ({result.get('new_params', 0):,} params)")
                if result.get('backfilled'):
                    self._print(f"   Backfilled: {result.get('backfill_steps', 0)} steps", style="cyan")
    
    def cmd_timeline(self, limit: int = 20):
        """Show model evolution timeline"""
        self._init_components()
        
        if not self._neural_brain:
            self._print("❌ Neural brain not available", style="red")
            return
        
        events = self._neural_brain.get_timeline(limit)
        
        if not events:
            self._print("📅 No timeline events yet", style="yellow")
            return
        
        self._print(f"\n📅 Model Timeline (last {len(events)} events)", style="bold blue")
        
        if RICH_AVAILABLE:
            table = Table()
            table.add_column("Time", style="dim")
            table.add_column("Event", style="cyan")
            table.add_column("Details", style="white")
            
            for event in reversed(events):
                timestamp = event.get('timestamp', '')[:19]  # Trim to readable
                event_type = event.get('event_type', 'unknown')
                details = json.dumps(event.get('details', {}), indent=None)[:60]
                
                # Color based on event type
                type_colors = {
                    'created': 'green',
                    'trained': 'blue',
                    'size_changed': 'yellow',
                    'milestone': 'magenta',
                    'checkpoint': 'dim'
                }
                color = type_colors.get(event_type, 'white')
                
                table.add_row(timestamp, f"[{color}]{event_type}[/{color}]", details)
            
            console.print(table)
        else:
            for event in reversed(events):
                timestamp = event.get('timestamp', '')[:19]
                event_type = event.get('event_type', 'unknown')
                details = event.get('details', {})
                print(f"{timestamp} | {event_type:15} | {details}")
    
    def cmd_stats(self):
        """Show comprehensive statistics"""
        self._init_components()
        
        if RICH_AVAILABLE:
            console.print(self._create_stats_table())
        else:
            print(self._create_stats_table())
    
    def cmd_generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.8):
        """Generate text from prompt"""
        self._init_components()
        
        if not self._neural_brain or not self._neural_brain.is_available:
            self._print("❌ Neural brain not available", style="red")
            return
        
        self._print(f"\n📝 Prompt: {prompt}", style="bold")
        self._print(f"   Max tokens: {max_tokens}, Temperature: {temperature}", style="dim")
        
        output = self._neural_brain.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        
        self._print(f"\n🤖 Generated:", style="cyan")
        self._print(output, style="white")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="GroundZero CLI - Control your neural AI from the command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s learn --articles 100 --show-progress 50
  %(prog)s learn --topics "Machine Learning,Physics"
  %(prog)s chat "What is the capital of France?"
  %(prog)s model --info
  %(prog)s model --size medium --backfill
  %(prog)s timeline
  %(prog)s stats
  %(prog)s generate "The future of AI is"
        """
    )
    
    parser.add_argument('--data-dir', default='data', help='Data directory path')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Learn command
    learn_parser = subparsers.add_parser('learn', help='Start learning process')
    learn_parser.add_argument('--articles', '-a', type=int, default=100,
                              help='Number of articles to learn (ignored if --forever)')
    learn_parser.add_argument('--forever', '-f', action='store_true',
                              help='Learn continuously until stopped (Ctrl+C)')
    learn_parser.add_argument('--show-progress', '-p', type=int, default=50,
                              help='Show stats every N articles')
    learn_parser.add_argument('--topics', '-t', type=str,
                              help='Comma-separated topics to focus on')
    learn_parser.add_argument('--save-every', '-s', type=int, default=100,
                              help='Save progress every N articles (for --forever mode)')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Chat with the model')
    chat_parser.add_argument('message', help='Your message')
    chat_parser.add_argument('--no-neural', action='store_true',
                             help='Skip neural model, use knowledge base only')
    
    # Model command
    model_parser = subparsers.add_parser('model', help='Manage neural model')
    model_parser.add_argument('--size', '-s', type=str,
                              choices=['nano', 'tiny', 'small', 'medium', 'large', 'xl', 'xxl'],
                              help='New model size')
    model_parser.add_argument('--backfill', '-b', action='store_true', default=True,
                              help='Backfill data when changing size')
    model_parser.add_argument('--no-backfill', action='store_true',
                              help='Skip backfilling')
    model_parser.add_argument('--info', '-i', action='store_true',
                              help='Show model information')
    
    # Timeline command
    timeline_parser = subparsers.add_parser('timeline', help='Show model evolution timeline')
    timeline_parser.add_argument('--limit', '-l', type=int, default=20,
                                 help='Number of events to show')
    
    # Stats command
    subparsers.add_parser('stats', help='Show statistics')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate text')
    gen_parser.add_argument('prompt', help='Starting prompt')
    gen_parser.add_argument('--max-tokens', '-m', type=int, default=100,
                            help='Maximum tokens to generate')
    gen_parser.add_argument('--temperature', '-t', type=float, default=0.8,
                            help='Sampling temperature')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create controller
    controller = CLIController(args.data_dir)
    
    # Execute command
    if args.command == 'learn':
        controller.cmd_learn(
            articles=args.articles,
            show_progress=args.show_progress,
            topics=args.topics,
            forever=args.forever,
            save_every=args.save_every
        )
    
    elif args.command == 'chat':
        controller.cmd_chat(args.message, use_neural=not args.no_neural)
    
    elif args.command == 'model':
        backfill = not args.no_backfill if args.no_backfill else args.backfill
        controller.cmd_model(size=args.size, backfill=backfill, info=args.info)
    
    elif args.command == 'timeline':
        controller.cmd_timeline(limit=args.limit)
    
    elif args.command == 'stats':
        controller.cmd_stats()
    
    elif args.command == 'generate':
        controller.cmd_generate(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )


if __name__ == "__main__":
    main()
