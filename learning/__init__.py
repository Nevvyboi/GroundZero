"""
GroundZero Learning Module v2.0
================================
Advanced learning system for intelligent knowledge acquisition.
"""

from .engine import (
    AdvancedLearningEngine,
    LearningItem,
    LearningSession,
    LearningPriority,
    LearningStatus,
    VitalArticlesManager,
    CurriculumLearner,
    SpacedRepetition,
    get_learning_engine
)
from .strategic import (
    StrategicPlanner,
    Topic,
    LearningGoal,
    LearningPath,
    TopicGraph,
    KnowledgeGapAnalyzer,
    get_strategic_planner
)

from .data_manager import (
    LearningDataManager,
    VitalArticle,
    LearnedArticle,
    get_data_manager
)

__all__ = [
    'AdvancedLearningEngine',
    'LearningItem',
    'LearningSession',
    'LearningPriority',
    'LearningStatus',
    'VitalArticlesManager',
    'CurriculumLearner',
    'SpacedRepetition',
    'get_learning_engine',
    'StrategicPlanner',
    'Topic',
    'LearningGoal',
    'LearningPath',
    'TopicGraph',
    'KnowledgeGapAnalyzer',
    'get_strategic_planner'
    # Data Manager
    'LearningDataManager',
    'VitalArticle',
    'LearnedArticle',
    'get_data_manager'
]