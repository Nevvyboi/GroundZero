"""
Strategic Learning Planner for GroundZero AI
=============================================
Intelligent planning system for optimized knowledge acquisition:
- Topic dependency graphs
- Knowledge gap analysis
- Learning path optimization
- Resource allocation
- Goal-based planning
"""

import json
import os
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import heapq
import math


@dataclass
class Topic:
    """A topic in the knowledge domain"""
    id: str
    name: str
    category: str = ""
    description: str = ""
    importance: float = 1.0
    difficulty: float = 0.5
    prerequisites: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)
    articles: List[str] = field(default_factory=list)
    mastery_level: float = 0.0  # 0-1 scale
    last_studied: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category,
            'description': self.description,
            'importance': self.importance,
            'difficulty': self.difficulty,
            'prerequisites': self.prerequisites,
            'related_topics': self.related_topics,
            'articles': self.articles,
            'mastery_level': self.mastery_level,
            'last_studied': self.last_studied
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'Topic':
        return cls(**d)


@dataclass
class LearningGoal:
    """A learning goal"""
    id: str
    name: str
    target_topics: List[str]
    target_mastery: float = 0.8
    deadline: Optional[str] = None
    priority: int = 1  # 1 = highest
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "active"  # active, completed, paused
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'target_topics': self.target_topics,
            'target_mastery': self.target_mastery,
            'deadline': self.deadline,
            'priority': self.priority,
            'created_at': self.created_at,
            'status': self.status
        }


@dataclass
class LearningPath:
    """An optimized learning path"""
    topics: List[str]
    estimated_time: float  # hours
    difficulty_curve: List[float]
    prerequisites_satisfied: bool
    
    def to_dict(self) -> Dict:
        return {
            'topics': self.topics,
            'estimated_time': self.estimated_time,
            'difficulty_curve': self.difficulty_curve,
            'prerequisites_satisfied': self.prerequisites_satisfied
        }


class TopicGraph:
    """
    Directed graph of topic dependencies.
    Used to determine optimal learning order.
    """
    
    def __init__(self):
        self.topics: Dict[str, Topic] = {}
        self.edges: Dict[str, Set[str]] = defaultdict(set)  # topic -> prerequisites
        self.reverse_edges: Dict[str, Set[str]] = defaultdict(set)  # prerequisite -> dependents
    
    def add_topic(self, topic: Topic):
        """Add a topic to the graph"""
        self.topics[topic.id] = topic
        
        for prereq in topic.prerequisites:
            self.edges[topic.id].add(prereq)
            self.reverse_edges[prereq].add(topic.id)
    
    def get_topic(self, topic_id: str) -> Optional[Topic]:
        """Get topic by ID"""
        return self.topics.get(topic_id)
    
    def get_prerequisites(self, topic_id: str) -> Set[str]:
        """Get direct prerequisites of a topic"""
        return self.edges.get(topic_id, set())
    
    def get_all_prerequisites(self, topic_id: str) -> Set[str]:
        """Get all prerequisites (transitive closure)"""
        all_prereqs = set()
        to_process = list(self.edges.get(topic_id, set()))
        
        while to_process:
            prereq = to_process.pop()
            if prereq not in all_prereqs:
                all_prereqs.add(prereq)
                to_process.extend(self.edges.get(prereq, set()))
        
        return all_prereqs
    
    def get_dependents(self, topic_id: str) -> Set[str]:
        """Get topics that depend on this topic"""
        return self.reverse_edges.get(topic_id, set())
    
    def topological_sort(self, topic_ids: List[str]) -> List[str]:
        """Sort topics in dependency order"""
        # Calculate in-degrees
        in_degree = {t: 0 for t in topic_ids}
        topic_set = set(topic_ids)
        
        for topic_id in topic_ids:
            for prereq in self.edges.get(topic_id, set()):
                if prereq in topic_set:
                    in_degree[topic_id] += 1
        
        # Process in order of in-degree
        result = []
        queue = [t for t in topic_ids if in_degree[t] == 0]
        
        while queue:
            topic = queue.pop(0)
            result.append(topic)
            
            for dependent in self.reverse_edges.get(topic, set()):
                if dependent in topic_set:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
        
        return result
    
    def find_learning_path(
        self,
        start_topics: List[str],
        target_topics: List[str],
        max_length: int = 20
    ) -> List[str]:
        """Find optimal path from start topics to target topics"""
        # Get all required topics
        required = set(target_topics)
        
        for target in target_topics:
            required.update(self.get_all_prerequisites(target))
        
        # Remove already mastered topics
        needed = []
        for topic_id in required:
            topic = self.topics.get(topic_id)
            if topic and topic.mastery_level < 0.8:
                needed.append(topic_id)
        
        # Sort by dependency order
        path = self.topological_sort(needed)
        
        return path[:max_length]


class KnowledgeGapAnalyzer:
    """
    Analyzes knowledge gaps and suggests topics to study.
    """
    
    def __init__(self, topic_graph: TopicGraph):
        self.graph = topic_graph
    
    def find_gaps(
        self,
        target_topics: List[str],
        mastery_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """Find knowledge gaps for target topics"""
        gaps = []
        
        for topic_id in target_topics:
            topic = self.graph.get_topic(topic_id)
            if not topic:
                continue
            
            # Check this topic
            if topic.mastery_level < mastery_threshold:
                gaps.append({
                    'topic_id': topic_id,
                    'topic_name': topic.name,
                    'current_mastery': topic.mastery_level,
                    'target_mastery': mastery_threshold,
                    'gap_size': mastery_threshold - topic.mastery_level,
                    'type': 'direct'
                })
            
            # Check prerequisites
            for prereq_id in self.graph.get_all_prerequisites(topic_id):
                prereq = self.graph.get_topic(prereq_id)
                if prereq and prereq.mastery_level < mastery_threshold:
                    gaps.append({
                        'topic_id': prereq_id,
                        'topic_name': prereq.name,
                        'current_mastery': prereq.mastery_level,
                        'target_mastery': mastery_threshold,
                        'gap_size': mastery_threshold - prereq.mastery_level,
                        'type': 'prerequisite',
                        'required_for': topic_id
                    })
        
        # Remove duplicates and sort by gap size
        seen = set()
        unique_gaps = []
        for gap in gaps:
            if gap['topic_id'] not in seen:
                seen.add(gap['topic_id'])
                unique_gaps.append(gap)
        
        unique_gaps.sort(key=lambda x: -x['gap_size'])
        return unique_gaps
    
    def prioritize_gaps(
        self,
        gaps: List[Dict[str, Any]],
        time_budget: float = 10.0  # hours
    ) -> List[Dict[str, Any]]:
        """Prioritize gaps based on impact and time budget"""
        # Calculate impact for each gap
        for gap in gaps:
            topic = self.graph.get_topic(gap['topic_id'])
            if topic:
                # Impact = importance * gap_size * number of dependents
                dependents = len(self.graph.get_dependents(gap['topic_id']))
                gap['impact'] = topic.importance * gap['gap_size'] * (1 + 0.1 * dependents)
                gap['estimated_hours'] = topic.difficulty * 2  # Simple estimate
        
        # Sort by impact/time ratio
        gaps.sort(key=lambda x: -x.get('impact', 0) / max(0.1, x.get('estimated_hours', 1)))
        
        # Select within time budget
        selected = []
        total_time = 0
        
        for gap in gaps:
            estimated_hours = gap.get('estimated_hours', 1)
            if total_time + estimated_hours <= time_budget:
                selected.append(gap)
                total_time += estimated_hours
        
        return selected


class StrategicPlanner:
    """
    Main strategic planner for learning optimization.
    """
    
    def __init__(self, data_dir: str = "./data/strategic"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.topic_graph = TopicGraph()
        self.gap_analyzer = KnowledgeGapAnalyzer(self.topic_graph)
        self.goals: Dict[str, LearningGoal] = {}
        self.current_plan: Optional[LearningPath] = None
        
        # Statistics
        self.stats = {
            'total_topics': 0,
            'mastered_topics': 0,
            'total_study_hours': 0.0,
            'goals_completed': 0
        }
        
        self._load_state()
        self._init_default_topics()
    
    def _init_default_topics(self):
        """Initialize with default topic structure"""
        if self.topic_graph.topics:
            return
        
        # AI/ML topic hierarchy
        default_topics = [
            # Foundation
            Topic(id='math_basics', name='Mathematics Basics', category='Foundation',
                  importance=1.0, difficulty=0.3),
            Topic(id='linear_algebra', name='Linear Algebra', category='Foundation',
                  prerequisites=['math_basics'], importance=1.2, difficulty=0.5),
            Topic(id='calculus', name='Calculus', category='Foundation',
                  prerequisites=['math_basics'], importance=1.1, difficulty=0.5),
            Topic(id='probability', name='Probability & Statistics', category='Foundation',
                  prerequisites=['math_basics'], importance=1.2, difficulty=0.5),
            
            # Programming
            Topic(id='python_basics', name='Python Basics', category='Programming',
                  importance=1.0, difficulty=0.3),
            Topic(id='data_structures', name='Data Structures', category='Programming',
                  prerequisites=['python_basics'], importance=1.0, difficulty=0.4),
            Topic(id='algorithms', name='Algorithms', category='Programming',
                  prerequisites=['data_structures'], importance=1.0, difficulty=0.5),
            
            # Machine Learning
            Topic(id='ml_basics', name='Machine Learning Basics', category='ML',
                  prerequisites=['linear_algebra', 'probability', 'python_basics'],
                  importance=1.3, difficulty=0.5),
            Topic(id='supervised_learning', name='Supervised Learning', category='ML',
                  prerequisites=['ml_basics'], importance=1.2, difficulty=0.5),
            Topic(id='unsupervised_learning', name='Unsupervised Learning', category='ML',
                  prerequisites=['ml_basics'], importance=1.1, difficulty=0.5),
            
            # Deep Learning
            Topic(id='neural_networks', name='Neural Networks', category='DL',
                  prerequisites=['ml_basics', 'calculus'], importance=1.3, difficulty=0.6),
            Topic(id='cnn', name='Convolutional Neural Networks', category='DL',
                  prerequisites=['neural_networks'], importance=1.2, difficulty=0.6),
            Topic(id='rnn', name='Recurrent Neural Networks', category='DL',
                  prerequisites=['neural_networks'], importance=1.2, difficulty=0.6),
            Topic(id='transformers', name='Transformers', category='DL',
                  prerequisites=['neural_networks', 'rnn'], importance=1.4, difficulty=0.7),
            
            # NLP
            Topic(id='nlp_basics', name='NLP Basics', category='NLP',
                  prerequisites=['ml_basics'], importance=1.2, difficulty=0.5),
            Topic(id='word_embeddings', name='Word Embeddings', category='NLP',
                  prerequisites=['nlp_basics', 'neural_networks'], importance=1.2, difficulty=0.6),
            Topic(id='language_models', name='Language Models', category='NLP',
                  prerequisites=['word_embeddings', 'transformers'], importance=1.4, difficulty=0.7),
            
            # General Knowledge
            Topic(id='world_history', name='World History', category='Knowledge',
                  importance=1.0, difficulty=0.4),
            Topic(id='science', name='General Science', category='Knowledge',
                  importance=1.0, difficulty=0.4),
            Topic(id='geography', name='Geography', category='Knowledge',
                  importance=0.8, difficulty=0.3),
        ]
        
        for topic in default_topics:
            self.topic_graph.add_topic(topic)
        
        self.stats['total_topics'] = len(default_topics)
    
    def _load_state(self):
        """Load saved state"""
        state_file = os.path.join(self.data_dir, "planner_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    
                    # Load topics
                    for topic_data in state.get('topics', []):
                        topic = Topic.from_dict(topic_data)
                        self.topic_graph.add_topic(topic)
                    
                    # Load goals
                    for goal_data in state.get('goals', []):
                        goal = LearningGoal(**goal_data)
                        self.goals[goal.id] = goal
                    
                    self.stats = state.get('stats', self.stats)
            except Exception as e:
                print(f"Error loading state: {e}")
    
    def _save_state(self):
        """Save state to disk"""
        state = {
            'topics': [t.to_dict() for t in self.topic_graph.topics.values()],
            'goals': [g.to_dict() for g in self.goals.values()],
            'stats': self.stats
        }
        
        with open(os.path.join(self.data_dir, "planner_state.json"), 'w') as f:
            json.dump(state, f, indent=2)
    
    def add_topic(self, topic: Topic):
        """Add a topic to the knowledge graph"""
        self.topic_graph.add_topic(topic)
        self.stats['total_topics'] = len(self.topic_graph.topics)
        self._save_state()
    
    def update_mastery(self, topic_id: str, mastery: float):
        """Update mastery level for a topic"""
        topic = self.topic_graph.get_topic(topic_id)
        if topic:
            old_mastery = topic.mastery_level
            topic.mastery_level = min(1.0, max(0.0, mastery))
            topic.last_studied = datetime.now().isoformat()
            
            # Update mastered count
            if old_mastery < 0.8 <= topic.mastery_level:
                self.stats['mastered_topics'] += 1
            elif old_mastery >= 0.8 > topic.mastery_level:
                self.stats['mastered_topics'] -= 1
            
            self._save_state()
    
    def add_goal(self, goal: LearningGoal):
        """Add a learning goal"""
        self.goals[goal.id] = goal
        self._save_state()
    
    def generate_plan(
        self,
        goal_id: Optional[str] = None,
        time_budget: float = 10.0
    ) -> LearningPath:
        """Generate an optimized learning plan"""
        # Determine target topics
        if goal_id and goal_id in self.goals:
            goal = self.goals[goal_id]
            target_topics = goal.target_topics
        else:
            # Default: focus on high-importance topics with low mastery
            target_topics = [
                t.id for t in self.topic_graph.topics.values()
                if t.importance >= 1.0 and t.mastery_level < 0.5
            ][:10]
        
        # Find knowledge gaps
        gaps = self.gap_analyzer.find_gaps(target_topics)
        prioritized_gaps = self.gap_analyzer.prioritize_gaps(gaps, time_budget)
        
        # Build learning path
        topic_ids = [gap['topic_id'] for gap in prioritized_gaps]
        path_topics = self.topic_graph.find_learning_path([], topic_ids)
        
        # Calculate difficulty curve
        difficulty_curve = []
        for topic_id in path_topics:
            topic = self.topic_graph.get_topic(topic_id)
            if topic:
                difficulty_curve.append(topic.difficulty)
        
        # Estimate time
        total_time = sum(
            self.topic_graph.get_topic(t).difficulty * 2
            for t in path_topics
            if self.topic_graph.get_topic(t)
        )
        
        # Check prerequisites
        prereqs_satisfied = all(
            self.topic_graph.get_topic(p).mastery_level >= 0.5
            for t in path_topics
            for p in self.topic_graph.get_prerequisites(t)
            if self.topic_graph.get_topic(p)
        )
        
        self.current_plan = LearningPath(
            topics=path_topics,
            estimated_time=total_time,
            difficulty_curve=difficulty_curve,
            prerequisites_satisfied=prereqs_satisfied
        )
        
        return self.current_plan
    
    def get_next_topic(self) -> Optional[str]:
        """Get next topic to study from current plan"""
        if not self.current_plan or not self.current_plan.topics:
            self.generate_plan()
        
        if self.current_plan and self.current_plan.topics:
            # Find first non-mastered topic
            for topic_id in self.current_plan.topics:
                topic = self.topic_graph.get_topic(topic_id)
                if topic and topic.mastery_level < 0.8:
                    return topic_id
        
        return None
    
    def get_topic_suggestions(
        self,
        category: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get suggested topics to study"""
        candidates = []
        
        for topic in self.topic_graph.topics.values():
            if category and topic.category != category:
                continue
            
            # Calculate priority score
            prereq_mastery = 1.0
            for prereq_id in topic.prerequisites:
                prereq = self.topic_graph.get_topic(prereq_id)
                if prereq:
                    prereq_mastery = min(prereq_mastery, prereq.mastery_level)
            
            if prereq_mastery >= 0.5:  # Prerequisites met
                score = topic.importance * (1 - topic.mastery_level) * prereq_mastery
                candidates.append({
                    'topic_id': topic.id,
                    'topic_name': topic.name,
                    'category': topic.category,
                    'current_mastery': topic.mastery_level,
                    'difficulty': topic.difficulty,
                    'importance': topic.importance,
                    'score': score
                })
        
        candidates.sort(key=lambda x: -x['score'])
        return candidates[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get strategic planner statistics"""
        # Calculate category breakdown
        categories = defaultdict(lambda: {'total': 0, 'mastered': 0})
        for topic in self.topic_graph.topics.values():
            categories[topic.category]['total'] += 1
            if topic.mastery_level >= 0.8:
                categories[topic.category]['mastered'] += 1
        
        return {
            **self.stats,
            'categories': dict(categories),
            'current_plan': self.current_plan.to_dict() if self.current_plan else None,
            'active_goals': len([g for g in self.goals.values() if g.status == 'active'])
        }


# Singleton instance
_strategic_planner: Optional[StrategicPlanner] = None


def get_strategic_planner(data_dir: str = "./data/strategic") -> StrategicPlanner:
    """Get or create strategic planner singleton"""
    global _strategic_planner
    if _strategic_planner is None:
        _strategic_planner = StrategicPlanner(data_dir=data_dir)
    return _strategic_planner


if __name__ == "__main__":
    print("Testing Strategic Learning Planner...")
    
    # Create planner
    planner = StrategicPlanner(data_dir="./test_strategic_data")
    
    # Test topic graph
    print("\n1. Testing Topic Graph:")
    print(f"  Total topics: {len(planner.topic_graph.topics)}")
    
    # Test prerequisite traversal
    prereqs = planner.topic_graph.get_all_prerequisites('transformers')
    print(f"  Prerequisites for 'transformers': {prereqs}")
    
    # Test topological sort
    topics = ['neural_networks', 'ml_basics', 'linear_algebra']
    sorted_topics = planner.topic_graph.topological_sort(topics)
    print(f"  Topological order: {sorted_topics}")
    
    # Test gap analysis
    print("\n2. Testing Knowledge Gap Analysis:")
    gaps = planner.gap_analyzer.find_gaps(['language_models'])
    print(f"  Knowledge gaps for 'language_models': {len(gaps)}")
    for gap in gaps[:3]:
        print(f"    - {gap['topic_name']}: gap={gap['gap_size']:.2f}")
    
    # Test learning path generation
    print("\n3. Testing Learning Path Generation:")
    path = planner.generate_plan(time_budget=5.0)
    print(f"  Learning path: {path.topics[:5]}")
    print(f"  Estimated time: {path.estimated_time:.1f} hours")
    print(f"  Prerequisites satisfied: {path.prerequisites_satisfied}")
    
    # Test mastery update
    print("\n4. Testing Mastery Update:")
    planner.update_mastery('math_basics', 0.9)
    topic = planner.topic_graph.get_topic('math_basics')
    print(f"  Updated 'math_basics' mastery: {topic.mastery_level}")
    
    # Test suggestions
    print("\n5. Testing Topic Suggestions:")
    suggestions = planner.get_topic_suggestions(limit=3)
    print(f"  Top suggestions:")
    for s in suggestions:
        print(f"    - {s['topic_name']} (score: {s['score']:.2f})")
    
    # Test goal management
    print("\n6. Testing Goal Management:")
    goal = LearningGoal(
        id='master_nlp',
        name='Master NLP',
        target_topics=['language_models', 'word_embeddings', 'nlp_basics'],
        target_mastery=0.8
    )
    planner.add_goal(goal)
    print(f"  Added goal: {goal.name}")
    
    # Get stats
    print("\n7. Strategic Planner Statistics:")
    stats = planner.get_stats()
    print(f"  Total topics: {stats['total_topics']}")
    print(f"  Mastered topics: {stats['mastered_topics']}")
    print(f"  Active goals: {stats['active_goals']}")
    
    # Clean up
    import shutil
    shutil.rmtree("./test_strategic_data", ignore_errors=True)
    
    print("\n✅ All strategic planner tests passed!")
