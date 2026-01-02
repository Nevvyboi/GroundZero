"""
Reasoning Engine
================
Main orchestrator for all reasoning capabilities.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from .logic import LogicReasoner
from .math_solver import MathSolver
from .code_analyzer import CodeAnalyzer
from .metacognition import Metacognition


class ReasoningType(Enum):
    """Types of reasoning"""
    GENERAL = "general"
    LOGIC = "logic"
    MATH = "math"
    CODE = "code"
    METACOGNITION = "metacognition"


@dataclass
class ReasoningStep:
    """A single step in the reasoning process"""
    step_num: int
    description: str
    operation: str
    result: Any


@dataclass
class ReasoningResult:
    """Result of reasoning process"""
    reasoning_type: ReasoningType
    success: bool
    steps: List[ReasoningStep]
    final_answer: Optional[str]
    confidence: float


class ReasoningEngine:
    """
    Orchestrates different reasoning capabilities.
    Determines the type of reasoning needed and delegates appropriately.
    """
    
    def __init__(self, memory_store=None):
        self.memory = memory_store
        self.logic = LogicReasoner()
        self.math = MathSolver()
        self.code = CodeAnalyzer()
        self.metacognition = None  # Set later when model is available
    
    def set_metacognition(self, metacognition: Metacognition) -> None:
        """Set metacognition module"""
        self.metacognition = metacognition
    
    def reason(self, query: str) -> ReasoningResult:
        """
        Main reasoning entry point.
        Analyzes query and routes to appropriate reasoner.
        """
        query_lower = query.lower()
        
        # Detect reasoning type
        reasoning_type = self._detect_reasoning_type(query_lower)
        
        # Route to appropriate reasoner
        if reasoning_type == ReasoningType.MATH:
            return self._reason_math(query)
        elif reasoning_type == ReasoningType.LOGIC:
            return self._reason_logic(query)
        elif reasoning_type == ReasoningType.CODE:
            return self._reason_code(query)
        elif reasoning_type == ReasoningType.METACOGNITION:
            return self._reason_metacognition(query)
        else:
            return ReasoningResult(
                reasoning_type=ReasoningType.GENERAL,
                success=False,
                steps=[],
                final_answer=None,
                confidence=0.0
            )
    
    def _detect_reasoning_type(self, query: str) -> ReasoningType:
        """Detect what type of reasoning is needed"""
        # Math indicators
        math_indicators = [
            "calculate", "compute", "solve", "what is", "how much",
            "+", "-", "*", "/", "=", "^", "sqrt", "sum", "average",
            "multiply", "divide", "add", "subtract", "percent"
        ]
        if any(ind in query for ind in math_indicators):
            # Check if it looks like a math expression
            if self.math.can_solve(query):
                return ReasoningType.MATH
        
        # Logic indicators
        logic_indicators = [
            "if ", "then", "therefore", "implies", "logically",
            "true or false", "valid", "invalid", "conclude",
            "all ", "some ", "none ", "every "
        ]
        if any(ind in query for ind in logic_indicators):
            return ReasoningType.LOGIC
        
        # Code indicators
        code_indicators = [
            "def ", "function", "class ", "debug", "bug", "error",
            "code", "program", "script", "syntax", "compile",
            "python", "javascript", "java ", "c++", "sql"
        ]
        if any(ind in query for ind in code_indicators):
            return ReasoningType.CODE
        
        # Metacognition indicators
        meta_indicators = [
            "what do you know", "what have you learned", "what can you do",
            "your capabilities", "your knowledge", "about yourself",
            "who are you", "how confident", "what topics",
            "your limitations", "what don't you know"
        ]
        if any(ind in query for ind in meta_indicators):
            return ReasoningType.METACOGNITION
        
        return ReasoningType.GENERAL
    
    def _reason_math(self, query: str) -> ReasoningResult:
        """Mathematical reasoning"""
        try:
            result, steps = self.math.solve(query)
            
            reasoning_steps = [
                ReasoningStep(
                    step_num=i + 1,
                    description=step['description'],
                    operation=step.get('operation', 'compute'),
                    result=step['result']
                )
                for i, step in enumerate(steps)
            ]
            
            return ReasoningResult(
                reasoning_type=ReasoningType.MATH,
                success=True,
                steps=reasoning_steps,
                final_answer=f"The result is: {result}",
                confidence=0.95
            )
        except Exception as e:
            return ReasoningResult(
                reasoning_type=ReasoningType.MATH,
                success=False,
                steps=[ReasoningStep(1, "Error", "error", str(e))],
                final_answer=None,
                confidence=0.0
            )
    
    def _reason_logic(self, query: str) -> ReasoningResult:
        """Logical reasoning"""
        try:
            result, steps = self.logic.analyze(query)
            
            reasoning_steps = [
                ReasoningStep(
                    step_num=i + 1,
                    description=step['description'],
                    operation=step.get('operation', 'analyze'),
                    result=step['result']
                )
                for i, step in enumerate(steps)
            ]
            
            return ReasoningResult(
                reasoning_type=ReasoningType.LOGIC,
                success=True,
                steps=reasoning_steps,
                final_answer=result,
                confidence=0.85
            )
        except Exception as e:
            return ReasoningResult(
                reasoning_type=ReasoningType.LOGIC,
                success=False,
                steps=[],
                final_answer=None,
                confidence=0.0
            )
    
    def _reason_code(self, query: str) -> ReasoningResult:
        """Code analysis reasoning"""
        try:
            result, steps = self.code.analyze(query)
            
            reasoning_steps = [
                ReasoningStep(
                    step_num=i + 1,
                    description=step['description'],
                    operation=step.get('operation', 'analyze'),
                    result=step['result']
                )
                for i, step in enumerate(steps)
            ]
            
            return ReasoningResult(
                reasoning_type=ReasoningType.CODE,
                success=True,
                steps=reasoning_steps,
                final_answer=result,
                confidence=0.80
            )
        except Exception as e:
            return ReasoningResult(
                reasoning_type=ReasoningType.CODE,
                success=False,
                steps=[],
                final_answer=None,
                confidence=0.0
            )
    
    def _reason_metacognition(self, query: str) -> ReasoningResult:
        """Self-reflection reasoning"""
        if self.metacognition is None:
            return ReasoningResult(
                reasoning_type=ReasoningType.METACOGNITION,
                success=False,
                steps=[],
                final_answer="Metacognition module not initialized.",
                confidence=0.0
            )
        
        try:
            result = self.metacognition.reflect(query)
            
            return ReasoningResult(
                reasoning_type=ReasoningType.METACOGNITION,
                success=True,
                steps=[ReasoningStep(1, "Self-reflection", "introspect", "Analyzed internal state")],
                final_answer=result,
                confidence=1.0
            )
        except Exception as e:
            return ReasoningResult(
                reasoning_type=ReasoningType.METACOGNITION,
                success=False,
                steps=[],
                final_answer=None,
                confidence=0.0
            )
