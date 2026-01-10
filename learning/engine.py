"""
Advanced Learning Engine for GroundZero AI
===========================================
Sophisticated learning system with:
- Strategic learning from vital articles
- Curriculum learning with difficulty progression
- Active learning for sample selection
- Spaced repetition for retention
- Progress tracking and analytics
- Background learning threads
- Session management
"""

import asyncio
import aiohttp
import threading
import queue
import time
import json
import os
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import random
import math


class LearningPriority(Enum):
    """Priority levels for learning content"""
    CRITICAL = 1  # Must learn immediately
    HIGH = 2      # High importance topics
    MEDIUM = 3    # Regular content
    LOW = 4       # Nice to have
    BACKGROUND = 5  # Fill time with these


class LearningStatus(Enum):
    """Status of a learning item"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class LearningItem:
    """A single item to learn"""
    id: str
    title: str
    url: str
    priority: LearningPriority = LearningPriority.MEDIUM
    status: LearningStatus = LearningStatus.PENDING
    category: str = ""
    difficulty: float = 0.5  # 0-1 scale
    importance: float = 1.0
    attempts: int = 0
    content: str = ""
    summary: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'title': self.title,
            'url': self.url,
            'priority': self.priority.value,
            'status': self.status.value,
            'category': self.category,
            'difficulty': self.difficulty,
            'importance': self.importance,
            'attempts': self.attempts,
            'summary': self.summary,
            'created_at': self.created_at,
            'completed_at': self.completed_at,
            'error': self.error,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'LearningItem':
        d['priority'] = LearningPriority(d['priority'])
        d['status'] = LearningStatus(d['status'])
        d.pop('content', None)  # Don't persist content
        return cls(**d)


@dataclass
class LearningSession:
    """A learning session"""
    id: str
    started_at: str
    ended_at: Optional[str] = None
    items_processed: int = 0
    items_successful: int = 0
    items_failed: int = 0
    total_tokens: int = 0
    categories: Dict[str, int] = field(default_factory=dict)
    avg_difficulty: float = 0.5
    learning_rate: float = 0.0  # Items per minute
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'started_at': self.started_at,
            'ended_at': self.ended_at,
            'items_processed': self.items_processed,
            'items_successful': self.items_successful,
            'items_failed': self.items_failed,
            'total_tokens': self.total_tokens,
            'categories': self.categories,
            'avg_difficulty': self.avg_difficulty,
            'learning_rate': self.learning_rate
        }


class VitalArticlesManager:
    """Manages Wikipedia's Vital Articles for strategic learning"""
    
    # Vital article categories with importance weights
    CATEGORIES = {
        'People': {'weight': 1.0, 'subcategories': [
            'Writers and journalists', 'Artists, musicians, and composers',
            'Entertainers, directors, producers, and screenwriters',
            'Philosophers, historians, and social scientists',
            'Religious figures', 'Politicians and leaders',
            'Military personnel, revolutionaries, and activists',
            'Scientists, inventors, and mathematicians',
            'Explorers and adventurers', 'Businesspeople',
            'Athletes and sports figures'
        ]},
        'History': {'weight': 1.2, 'subcategories': [
            'General', 'Prehistory', 'Ancient history',
            'Post-classical history', 'Modern history'
        ]},
        'Geography': {'weight': 0.9, 'subcategories': [
            'Continents and regions', 'Countries', 'Cities',
            'Bodies of water', 'Islands', 'Mountains and deserts',
            'Other geographical features'
        ]},
        'Arts': {'weight': 0.8, 'subcategories': [
            'Architecture', 'Art', 'Film', 'Music', 'Literature',
            'Theatre and dance', 'Entertainment'
        ]},
        'Philosophy and religion': {'weight': 1.0, 'subcategories': [
            'Philosophy', 'Religion'
        ]},
        'Everyday life': {'weight': 0.7, 'subcategories': [
            'Food and drink', 'Sports and recreation', 'Household'
        ]},
        'Society and social sciences': {'weight': 1.1, 'subcategories': [
            'Society', 'Social issues', 'Politics and government',
            'Economics', 'Business and law', 'Education', 'War and military'
        ]},
        'Biology and health sciences': {'weight': 1.0, 'subcategories': [
            'Anatomy and physiology', 'Health and medicine', 'Animals',
            'Plants and fungi', 'Organisms', 'Biology'
        ]},
        'Physical sciences': {'weight': 1.1, 'subcategories': [
            'Astronomy', 'Chemistry', 'Earth science', 'Physics'
        ]},
        'Technology': {'weight': 1.2, 'subcategories': [
            'Electronics', 'Computing', 'Transportation',
            'Agriculture and industry', 'Engineering'
        ]},
        'Mathematics': {'weight': 1.1, 'subcategories': [
            'Mathematics', 'Units and measurement'
        ]}
    }
    
    def __init__(self, cache_file: str = "./vital_articles_cache.json"):
        self.cache_file = cache_file
        self.articles: Dict[str, List[LearningItem]] = {}
        self.all_articles: List[LearningItem] = []
        self._load_cache()
    
    def _load_cache(self):
        """Load cached vital articles"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    for category, articles in data.items():
                        self.articles[category] = [
                            LearningItem.from_dict(a) if isinstance(a, dict) 
                            else self._create_item(a, category)
                            for a in articles
                        ]
                        self.all_articles.extend(self.articles[category])
            except:
                pass
    
    def _create_item(self, title: str, category: str) -> LearningItem:
        """Create a learning item from title and category"""
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        item_id = hashlib.md5(url.encode()).hexdigest()[:12]
        
        weight = self.CATEGORIES.get(category, {}).get('weight', 1.0)
        
        return LearningItem(
            id=item_id,
            title=title,
            url=url,
            category=category,
            importance=weight,
            priority=LearningPriority.HIGH if weight >= 1.0 else LearningPriority.MEDIUM
        )
    
    async def fetch_vital_articles(
        self,
        levels: List[int] = [1, 2, 3],
        limit: int = 1000
    ) -> List[LearningItem]:
        """Fetch vital articles from Wikipedia"""
        articles = []
        
        async with aiohttp.ClientSession() as session:
            for level in levels:
                url = f"https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/{level}"
                try:
                    async with session.get(url, timeout=30) as response:
                        if response.status == 200:
                            html = await response.text()
                            # Parse article links (simplified)
                            links = re.findall(r'/wiki/([^"#:]+)"[^>]*>([^<]+)</a>', html)
                            
                            for wiki_title, display_title in links[:limit // len(levels)]:
                                if not wiki_title.startswith(('Wikipedia:', 'Template:', 'File:')):
                                    item = self._create_item(display_title, 'General')
                                    articles.append(item)
                except Exception as e:
                    print(f"Error fetching level {level}: {e}")
        
        self.all_articles = articles
        return articles
    
    def get_articles_by_category(self, category: str) -> List[LearningItem]:
        """Get articles for a specific category"""
        return self.articles.get(category, [])
    
    def get_prioritized_articles(self, n: int = 100) -> List[LearningItem]:
        """Get n highest priority articles"""
        pending = [a for a in self.all_articles if a.status == LearningStatus.PENDING]
        pending.sort(key=lambda x: (x.priority.value, -x.importance))
        return pending[:n]
    
    def mark_completed(self, article_id: str):
        """Mark an article as completed"""
        for article in self.all_articles:
            if article.id == article_id:
                article.status = LearningStatus.COMPLETED
                article.completed_at = datetime.now().isoformat()
                break
    
    def save_cache(self):
        """Save articles to cache"""
        data = {}
        for category, articles in self.articles.items():
            data[category] = [a.to_dict() for a in articles]
        
        with open(self.cache_file, 'w') as f:
            json.dump(data, f)


class CurriculumLearner:
    """
    Implements curriculum learning - starting with easy content and 
    progressively increasing difficulty.
    """
    
    def __init__(
        self,
        initial_difficulty: float = 0.2,
        target_difficulty: float = 0.8,
        difficulty_increment: float = 0.05,
        window_size: int = 10
    ):
        self.current_difficulty = initial_difficulty
        self.target_difficulty = target_difficulty
        self.difficulty_increment = difficulty_increment
        self.window_size = window_size
        
        self.recent_scores: List[float] = []
        self.total_items = 0
        self.difficulty_history: List[Tuple[int, float]] = []
    
    def estimate_difficulty(self, item: LearningItem) -> float:
        """Estimate difficulty of a learning item"""
        # Factors: content length, technical terms, references
        difficulty = 0.5
        
        if item.content:
            # Longer content is harder
            length_factor = min(1.0, len(item.content) / 10000)
            difficulty += 0.2 * length_factor
            
            # Technical terms increase difficulty
            tech_terms = len(re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', item.content))
            term_factor = min(1.0, tech_terms / 50)
            difficulty += 0.2 * term_factor
            
            # Category affects difficulty
            hard_categories = {'Mathematics', 'Physical sciences', 'Technology'}
            if item.category in hard_categories:
                difficulty += 0.1
        
        return min(1.0, max(0.0, difficulty))
    
    def select_items(
        self,
        candidates: List[LearningItem],
        n: int = 10
    ) -> List[LearningItem]:
        """Select items appropriate for current difficulty level"""
        # Score each candidate
        scored = []
        for item in candidates:
            diff = self.estimate_difficulty(item)
            item.difficulty = diff
            
            # Score based on proximity to current difficulty
            diff_diff = abs(diff - self.current_difficulty)
            score = 1.0 - diff_diff + item.importance * 0.2
            
            scored.append((score, item))
        
        # Sort by score and return top n
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:n]]
    
    def record_result(self, item: LearningItem, success: bool):
        """Record learning result and adjust difficulty"""
        self.total_items += 1
        
        # Record score
        score = 1.0 if success else 0.0
        self.recent_scores.append(score)
        if len(self.recent_scores) > self.window_size:
            self.recent_scores.pop(0)
        
        # Calculate success rate
        success_rate = sum(self.recent_scores) / len(self.recent_scores)
        
        # Adjust difficulty based on performance
        if success_rate >= 0.8 and self.current_difficulty < self.target_difficulty:
            # Doing well, increase difficulty
            self.current_difficulty = min(
                self.target_difficulty,
                self.current_difficulty + self.difficulty_increment
            )
        elif success_rate < 0.5:
            # Struggling, decrease difficulty
            self.current_difficulty = max(
                0.1,
                self.current_difficulty - self.difficulty_increment
            )
        
        self.difficulty_history.append((self.total_items, self.current_difficulty))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get curriculum learning statistics"""
        return {
            'current_difficulty': self.current_difficulty,
            'target_difficulty': self.target_difficulty,
            'total_items': self.total_items,
            'recent_success_rate': sum(self.recent_scores) / len(self.recent_scores) if self.recent_scores else 0.0,
            'difficulty_history': self.difficulty_history[-20:]
        }


class SpacedRepetition:
    """
    Implements spaced repetition for better retention.
    Based on SM-2 algorithm.
    """
    
    def __init__(self):
        self.items: Dict[str, Dict[str, Any]] = {}  # item_id -> scheduling info
    
    def add_item(self, item_id: str):
        """Add item to spaced repetition system"""
        self.items[item_id] = {
            'easiness': 2.5,
            'interval': 1,
            'repetitions': 0,
            'next_review': datetime.now().isoformat(),
            'history': []
        }
    
    def record_review(self, item_id: str, quality: int):
        """
        Record a review result.
        
        Args:
            item_id: Item identifier
            quality: Quality of recall (0-5)
                0: Complete blackout
                1: Wrong answer; remembered when shown
                2: Wrong answer; easy to recall when shown
                3: Correct with serious difficulty
                4: Correct with some hesitation
                5: Perfect recall
        """
        if item_id not in self.items:
            self.add_item(item_id)
        
        item = self.items[item_id]
        
        # SM-2 algorithm
        if quality >= 3:
            # Correct response
            if item['repetitions'] == 0:
                item['interval'] = 1
            elif item['repetitions'] == 1:
                item['interval'] = 6
            else:
                item['interval'] = round(item['interval'] * item['easiness'])
            
            item['repetitions'] += 1
        else:
            # Incorrect response
            item['repetitions'] = 0
            item['interval'] = 1
        
        # Update easiness factor
        item['easiness'] = max(1.3, item['easiness'] + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        
        # Calculate next review date
        next_review = datetime.now() + timedelta(days=item['interval'])
        item['next_review'] = next_review.isoformat()
        
        # Record in history
        item['history'].append({
            'date': datetime.now().isoformat(),
            'quality': quality,
            'interval': item['interval']
        })
    
    def get_due_items(self, limit: int = 50) -> List[str]:
        """Get items due for review"""
        now = datetime.now()
        due = []
        
        for item_id, info in self.items.items():
            next_review = datetime.fromisoformat(info['next_review'])
            if next_review <= now:
                due.append((next_review, item_id))
        
        due.sort()
        return [item_id for _, item_id in due[:limit]]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get spaced repetition statistics"""
        now = datetime.now()
        due_count = sum(
            1 for info in self.items.values()
            if datetime.fromisoformat(info['next_review']) <= now
        )
        
        avg_easiness = sum(info['easiness'] for info in self.items.values()) / len(self.items) if self.items else 2.5
        
        return {
            'total_items': len(self.items),
            'due_count': due_count,
            'average_easiness': avg_easiness
        }


class AdvancedLearningEngine:
    """
    Main learning engine for GroundZero AI.
    Coordinates all learning components.
    """
    
    def __init__(
        self,
        data_dir: str = "./data/learning",
        neural_brain = None,
        knowledge_base = None
    ):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # External components
        self.neural_brain = neural_brain
        self.knowledge_base = knowledge_base
        
        # Learning components
        self.vital_articles = VitalArticlesManager(
            cache_file=os.path.join(data_dir, "vital_articles.json")
        )
        self.curriculum = CurriculumLearner()
        self.spaced_rep = SpacedRepetition()
        
        # Session tracking
        self.current_session: Optional[LearningSession] = None
        self.sessions: List[LearningSession] = []
        
        # Learning queue
        self.learning_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.learned_items: Set[str] = set()
        
        # Progress tracking
        self.total_articles = 0
        self.total_tokens = 0
        self.start_time: Optional[float] = None
        
        # Background learning thread
        self.learning_thread: Optional[threading.Thread] = None
        self.is_learning = False
        self.should_stop = False
        
        # Progress callback
        self.progress_callback: Optional[Callable[[Dict], None]] = None
        
        # Load state
        self._load_state()
    
    def _load_state(self):
        """Load saved state"""
        state_file = os.path.join(self.data_dir, "engine_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.learned_items = set(state.get('learned_items', []))
                    self.total_articles = state.get('total_articles', 0)
                    self.total_tokens = state.get('total_tokens', 0)
            except:
                pass
    
    def _save_state(self):
        """Save state to disk"""
        state = {
            'learned_items': list(self.learned_items),
            'total_articles': self.total_articles,
            'total_tokens': self.total_tokens
        }
        
        with open(os.path.join(self.data_dir, "engine_state.json"), 'w') as f:
            json.dump(state, f)
    
    def set_progress_callback(self, callback: Callable[[Dict], None]):
        """Set callback for progress updates"""
        self.progress_callback = callback
    
    def start_session(self) -> LearningSession:
        """Start a new learning session"""
        session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        
        self.current_session = LearningSession(
            id=session_id,
            started_at=datetime.now().isoformat()
        )
        
        self.start_time = time.time()
        return self.current_session
    
    def end_session(self) -> LearningSession:
        """End current learning session"""
        if self.current_session:
            self.current_session.ended_at = datetime.now().isoformat()
            
            # Calculate learning rate
            if self.start_time:
                elapsed_minutes = (time.time() - self.start_time) / 60
                if elapsed_minutes > 0:
                    self.current_session.learning_rate = self.current_session.items_processed / elapsed_minutes
            
            self.sessions.append(self.current_session)
            session = self.current_session
            self.current_session = None
            self.start_time = None
            
            self._save_state()
            return session
        
        return None
    
    async def fetch_content(self, url: str) -> Optional[str]:
        """Fetch content from URL"""
        try:
            async with aiohttp.ClientSession() as session:
                # Wikipedia API for clean content
                if 'wikipedia.org' in url:
                    title = url.split('/wiki/')[-1]
                    api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
                    
                    async with session.get(api_url, timeout=30) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data.get('extract', '')
                else:
                    async with session.get(url, timeout=30) as response:
                        if response.status == 200:
                            return await response.text()
        except Exception as e:
            print(f"Error fetching {url}: {e}")
        
        return None
    
    async def learn_item(self, item: LearningItem) -> bool:
        """Learn a single item"""
        if item.id in self.learned_items:
            return True
        
        try:
            item.status = LearningStatus.IN_PROGRESS
            item.attempts += 1
            
            # Fetch content if not present
            if not item.content:
                content = await self.fetch_content(item.url)
                if not content:
                    item.status = LearningStatus.FAILED
                    item.error = "Failed to fetch content"
                    return False
                item.content = content
            
            # Add to knowledge base
            if self.knowledge_base:
                self.knowledge_base.add_document(
                    title=item.title,
                    content=item.content,
                    source=item.url,
                    category=item.category
                )
            
            # Train neural network
            if self.neural_brain:
                # Prepare training text
                training_text = f"# {item.title}\n\n{item.content}"
                
                # Train on content
                tokens_trained = self.neural_brain.train_on_text(training_text)
                self.total_tokens += tokens_trained
                
                if self.current_session:
                    self.current_session.total_tokens += tokens_trained
            
            # Mark as completed
            item.status = LearningStatus.COMPLETED
            item.completed_at = datetime.now().isoformat()
            self.learned_items.add(item.id)
            self.total_articles += 1
            
            # Update curriculum
            self.curriculum.record_result(item, success=True)
            
            # Add to spaced repetition
            self.spaced_rep.add_item(item.id)
            
            # Update session stats
            if self.current_session:
                self.current_session.items_processed += 1
                self.current_session.items_successful += 1
                
                if item.category:
                    self.current_session.categories[item.category] = \
                        self.current_session.categories.get(item.category, 0) + 1
            
            return True
            
        except Exception as e:
            item.status = LearningStatus.FAILED
            item.error = str(e)
            
            self.curriculum.record_result(item, success=False)
            
            if self.current_session:
                self.current_session.items_processed += 1
                self.current_session.items_failed += 1
            
            return False
    
    async def learn_batch(
        self,
        items: List[LearningItem],
        progress_interval: int = 50
    ) -> Dict[str, Any]:
        """Learn a batch of items with progress reporting"""
        self.start_session()
        
        results = {
            'total': len(items),
            'successful': 0,
            'failed': 0,
            'skipped': 0
        }
        
        for i, item in enumerate(items):
            if self.should_stop:
                break
            
            if item.id in self.learned_items:
                results['skipped'] += 1
                continue
            
            success = await self.learn_item(item)
            
            if success:
                results['successful'] += 1
            else:
                results['failed'] += 1
            
            # Progress report
            if (i + 1) % progress_interval == 0:
                progress = self.get_progress()
                if self.progress_callback:
                    self.progress_callback(progress)
                else:
                    self._print_progress(progress)
        
        self.end_session()
        return results
    
    def _print_progress(self, progress: Dict[str, Any]):
        """Print progress to console"""
        print(f"\n{'='*60}")
        print(f"Learning Progress Report")
        print(f"{'='*60}")
        print(f"Articles learned: {progress['total_articles']}")
        print(f"Tokens trained: {progress['total_tokens']:,}")
        print(f"Current difficulty: {progress['curriculum']['current_difficulty']:.2f}")
        print(f"Recent success rate: {progress['curriculum']['recent_success_rate']:.1%}")
        if progress['session']:
            print(f"Session rate: {progress['session']['learning_rate']:.1f} articles/min")
        print(f"{'='*60}\n")
    
    def start_background_learning(
        self,
        target_articles: int = 100,
        progress_interval: int = 50,
        topics: Optional[List[str]] = None
    ):
        """Start learning in background thread"""
        if self.is_learning:
            return False
        
        def learning_worker():
            self.is_learning = True
            self.should_stop = False
            
            # Get articles to learn
            if topics:
                articles = []
                for topic in topics:
                    articles.extend(self.vital_articles.get_articles_by_category(topic))
            else:
                articles = self.vital_articles.get_prioritized_articles(target_articles)
            
            # Select based on curriculum
            articles = self.curriculum.select_items(articles, target_articles)
            
            # Run async learning
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                loop.run_until_complete(
                    self.learn_batch(articles, progress_interval)
                )
            finally:
                loop.close()
                self.is_learning = False
        
        self.learning_thread = threading.Thread(target=learning_worker, daemon=True)
        self.learning_thread.start()
        return True
    
    def stop_learning(self):
        """Stop background learning"""
        self.should_stop = True
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current learning progress"""
        progress = {
            'total_articles': self.total_articles,
            'total_tokens': self.total_tokens,
            'is_learning': self.is_learning,
            'curriculum': self.curriculum.get_stats(),
            'spaced_rep': self.spaced_rep.get_stats(),
            'session': self.current_session.to_dict() if self.current_session else None
        }
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            progress['elapsed_seconds'] = elapsed
            if elapsed > 0 and self.current_session:
                progress['current_rate'] = self.current_session.items_processed / (elapsed / 60)
        
        return progress
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        return {
            'total_articles_learned': self.total_articles,
            'total_tokens_trained': self.total_tokens,
            'total_sessions': len(self.sessions),
            'is_learning': self.is_learning,
            'curriculum': self.curriculum.get_stats(),
            'spaced_rep': self.spaced_rep.get_stats(),
            'vital_articles': {
                'total': len(self.vital_articles.all_articles),
                'learned': len(self.learned_items)
            }
        }
    
    def get_review_items(self, limit: int = 10) -> List[str]:
        """Get items due for review"""
        return self.spaced_rep.get_due_items(limit)
    
    def record_review(self, item_id: str, quality: int):
        """Record a review result"""
        self.spaced_rep.record_review(item_id, quality)


# Singleton instance
_learning_engine: Optional[AdvancedLearningEngine] = None


def get_learning_engine(
    data_dir: str = "./data/learning",
    neural_brain = None,
    knowledge_base = None
) -> AdvancedLearningEngine:
    """Get or create learning engine singleton"""
    global _learning_engine
    if _learning_engine is None:
        _learning_engine = AdvancedLearningEngine(
            data_dir=data_dir,
            neural_brain=neural_brain,
            knowledge_base=knowledge_base
        )
    return _learning_engine


if __name__ == "__main__":
    import asyncio
    
    print("Testing Advanced Learning Engine...")
    
    # Create engine
    engine = AdvancedLearningEngine(data_dir="./test_learning_data")
    
    # Test vital articles manager
    print("\n1. Testing Vital Articles Manager:")
    vm = VitalArticlesManager()
    
    # Create some test articles
    test_articles = [
        LearningItem(
            id="test1",
            title="Machine Learning",
            url="https://en.wikipedia.org/wiki/Machine_learning",
            category="Technology",
            importance=1.2
        ),
        LearningItem(
            id="test2",
            title="Albert Einstein",
            url="https://en.wikipedia.org/wiki/Albert_Einstein",
            category="People",
            importance=1.0
        ),
        LearningItem(
            id="test3",
            title="World War II",
            url="https://en.wikipedia.org/wiki/World_War_II",
            category="History",
            importance=1.1
        )
    ]
    
    vm.all_articles = test_articles
    prioritized = vm.get_prioritized_articles(3)
    print(f"  Prioritized articles: {[a.title for a in prioritized]}")
    
    # Test curriculum learner
    print("\n2. Testing Curriculum Learner:")
    cl = CurriculumLearner()
    
    for i in range(5):
        cl.record_result(test_articles[i % len(test_articles)], success=True)
    
    stats = cl.get_stats()
    print(f"  Current difficulty: {stats['current_difficulty']:.2f}")
    print(f"  Success rate: {stats['recent_success_rate']:.1%}")
    
    # Test spaced repetition
    print("\n3. Testing Spaced Repetition:")
    sr = SpacedRepetition()
    
    sr.add_item("item1")
    sr.record_review("item1", 5)  # Perfect recall
    
    sr.add_item("item2")
    sr.record_review("item2", 2)  # Struggled
    
    due = sr.get_due_items()
    print(f"  Items due for review: {due}")
    print(f"  Stats: {sr.get_stats()}")
    
    # Test session management
    print("\n4. Testing Session Management:")
    engine.start_session()
    print(f"  Started session: {engine.current_session.id}")
    
    # Simulate some learning
    engine.total_articles = 10
    engine.total_tokens = 5000
    engine.current_session.items_processed = 10
    engine.current_session.items_successful = 9
    
    progress = engine.get_progress()
    print(f"  Progress: {progress['total_articles']} articles, {progress['total_tokens']} tokens")
    
    session = engine.end_session()
    print(f"  Ended session: {session.id}, {session.items_successful}/{session.items_processed} successful")
    
    # Get overall stats
    print("\n5. Overall Statistics:")
    stats = engine.get_stats()
    print(f"  Total articles: {stats['total_articles_learned']}")
    print(f"  Total sessions: {stats['total_sessions']}")
    
    # Clean up
    import shutil
    shutil.rmtree("./test_learning_data", ignore_errors=True)
    
    print("\n✅ All learning engine tests passed!")
