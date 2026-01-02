"""
Math Solver
===========
Mathematical expression parsing and evaluation.
"""

import re
import math
from typing import Tuple, List, Dict, Any


class MathSolver:
    """Handles mathematical computations with step-by-step solutions"""
    
    SAFE_FUNCTIONS = {
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'log10': math.log10,
        'exp': math.exp,
        'abs': abs,
        'pow': pow,
        'round': round,
        'floor': math.floor,
        'ceil': math.ceil
    }
    
    SAFE_CONSTANTS = {
        'pi': math.pi,
        'e': math.e
    }
    
    def can_solve(self, query: str) -> bool:
        """Check if query contains a solvable math expression"""
        math_patterns = [
            r'\d+\s*[\+\-\*\/\^]\s*\d+',
            r'calculate|compute|solve|evaluate',
            r'what\s+is\s+\d+',
            r'sqrt|sin|cos|tan|log',
            r'\d+\s*%\s*of',
            r'average|mean|sum\s+of'
        ]
        return any(re.search(p, query.lower()) for p in math_patterns)
    
    def solve(self, query: str) -> Tuple[float, List[Dict[str, Any]]]:
        """Solve a mathematical expression. Returns (result, steps)."""
        steps = []
        
        expression = self._extract_expression(query)
        steps.append({'description': 'Extracted expression', 'operation': 'parse', 'result': expression})
        
        if 'average' in query.lower() or 'mean' in query.lower():
            return self._compute_average(query, steps)
        
        if '%' in query and 'of' in query.lower():
            return self._compute_percentage(query, steps)
        
        sanitized = self._sanitize_expression(expression)
        steps.append({'description': 'Sanitized expression', 'operation': 'sanitize', 'result': sanitized})
        
        try:
            result = self._safe_eval(sanitized)
            steps.append({'description': 'Computed result', 'operation': 'evaluate', 'result': result})
            return result, steps
        except Exception as e:
            raise ValueError(f"Could not evaluate: {e}")
    
    def _extract_expression(self, query: str) -> str:
        """Extract mathematical expression from text"""
        prefixes = [r'calculate\s*', r'compute\s*', r'solve\s*', r'evaluate\s*', r'what\s+is\s*', r'find\s*']
        result = query
        for prefix in prefixes:
            result = re.sub(prefix, '', result, flags=re.IGNORECASE)
        return re.sub(r'[?!.]+$', '', result.strip()).strip()
    
    def _sanitize_expression(self, expr: str) -> str:
        """Sanitize for safe evaluation"""
        expr = expr.replace('^', '**').replace('×', '*').replace('÷', '/')
        expr = re.sub(r'(\d)\s*\(', r'\1*(', expr)
        expr = re.sub(r'\bplus\b', '+', expr, flags=re.IGNORECASE)
        expr = re.sub(r'\bminus\b', '-', expr, flags=re.IGNORECASE)
        expr = re.sub(r'\btimes\b', '*', expr, flags=re.IGNORECASE)
        expr = re.sub(r'\bdivided\s+by\b', '/', expr, flags=re.IGNORECASE)
        return expr.strip()
    
    def _safe_eval(self, expression: str) -> float:
        """Safely evaluate expression"""
        namespace = {**self.SAFE_FUNCTIONS, **self.SAFE_CONSTANTS}
        return float(eval(expression, {"__builtins__": {}}, namespace))
    
    def _compute_average(self, query: str, steps: List[Dict]) -> Tuple[float, List[Dict]]:
        """Compute average"""
        numbers = [float(n) for n in re.findall(r'-?\d+\.?\d*', query)]
        if not numbers:
            raise ValueError("No numbers found")
        steps.append({'description': f'Found numbers: {numbers}', 'operation': 'extract', 'result': numbers})
        average = sum(numbers) / len(numbers)
        steps.append({'description': f'Average: {sum(numbers)}/{len(numbers)}', 'operation': 'compute', 'result': average})
        return average, steps
    
    def _compute_percentage(self, query: str, steps: List[Dict]) -> Tuple[float, List[Dict]]:
        """Compute percentage"""
        match = re.search(r'(\d+\.?\d*)\s*%\s*of\s*(\d+\.?\d*)', query)
        if not match:
            raise ValueError("Could not parse percentage")
        percent, value = float(match.group(1)), float(match.group(2))
        steps.append({'description': f'{percent}% of {value}', 'operation': 'parse', 'result': f'{percent}% × {value}'})
        result = (percent / 100) * value
        steps.append({'description': 'Result', 'operation': 'compute', 'result': result})
        return result, steps
