"""
NeuralMind Reasoning Engine
Adds logical reasoning, math solving, code debugging, and complex analysis
"""

import re
import ast
import operator
import traceback
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math


class ReasoningType(Enum):
    LOGIC = "logic"
    MATH = "math"
    CODE = "code"
    ANALYSIS = "analysis"
    GENERAL = "general"


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain"""
    step_num: int
    description: str
    operation: str
    result: Any
    confidence: float = 1.0


@dataclass
class ReasoningResult:
    """Complete reasoning result with chain of thought"""
    query: str
    reasoning_type: ReasoningType
    steps: List[ReasoningStep] = field(default_factory=list)
    final_answer: str = ""
    confidence: float = 1.0
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "type": self.reasoning_type.value,
            "steps": [
                {
                    "step": s.step_num,
                    "description": s.description,
                    "operation": s.operation,
                    "result": str(s.result)
                } for s in self.steps
            ],
            "answer": self.final_answer,
            "confidence": self.confidence,
            "success": self.success,
            "error": self.error
        }


class LogicEngine:
    """
    Handles logical reasoning: comparisons, transitive relations, 
    boolean logic, set operations, and syllogisms
    """
    
    def __init__(self):
        self.relations: Dict[str, List[Tuple[str, str]]] = {}
        self.facts: Dict[str, Any] = {}
        self.comparison_ops = {
            '>': operator.gt, '<': operator.lt,
            '>=': operator.ge, '<=': operator.le,
            '==': operator.eq, '!=': operator.ne,
            '=': operator.eq
        }
        
    def parse_relations(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract relations from text like 'A > B and B > C'"""
        relations = []
        
        # Pattern for comparisons: A > B, X < Y, etc.
        patterns = [
            r'(\w+)\s*(>|<|>=|<=|==|!=|=)\s*(\w+)',
            r'(\w+)\s+is\s+(greater|less|equal|bigger|smaller|larger|more|fewer)\s+than\s+(\w+)',
            r'(\w+)\s+is\s+(taller|shorter|older|younger|heavier|lighter|faster|slower)\s+than\s+(\w+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 3:
                    a, op, b = match
                    # Normalize operator
                    op_map = {
                        'greater': '>', 'less': '<', 'equal': '==',
                        'bigger': '>', 'smaller': '<', 'larger': '>',
                        'more': '>', 'fewer': '<',
                        'taller': '>', 'shorter': '<', 'older': '>',
                        'younger': '<', 'heavier': '>', 'lighter': '<',
                        'faster': '>', 'slower': '<'
                    }
                    op = op_map.get(op.lower(), op)
                    relations.append((a, op, b))
                    
        return relations
    
    def build_graph(self, relations: List[Tuple[str, str, str]]) -> Dict[str, Dict[str, List[str]]]:
        """Build a directed graph from relations"""
        graph = {'>' : {}, '<': {}, '==': {}}
        
        for a, op, b in relations:
            if op in ['>', '>=']:
                if a not in graph['>']:
                    graph['>'][a] = []
                graph['>'][a].append(b)
                if b not in graph['<']:
                    graph['<'][b] = []
                graph['<'][b].append(a)
            elif op in ['<', '<=']:
                if a not in graph['<']:
                    graph['<'][a] = []
                graph['<'][a].append(b)
                if b not in graph['>']:
                    graph['>'][b] = []
                graph['>'][b].append(a)
            elif op in ['==', '=']:
                if a not in graph['==']:
                    graph['=='][a] = []
                graph['=='][a].append(b)
                if b not in graph['==']:
                    graph['=='][b] = []
                graph['=='][b].append(a)
                
        return graph
    
    def find_transitive_relation(self, graph: Dict, a: str, b: str, op: str, visited: set = None) -> Optional[List[str]]:
        """Find path from a to b using transitive closure"""
        if visited is None:
            visited = set()
            
        if a in visited:
            return None
        visited.add(a)
        
        if a not in graph.get(op, {}):
            return None
            
        if b in graph[op].get(a, []):
            return [a, b]
            
        for intermediate in graph[op].get(a, []):
            path = self.find_transitive_relation(graph, intermediate, b, op, visited.copy())
            if path:
                return [a] + path
                
        return None
    
    def solve(self, query: str) -> ReasoningResult:
        """Solve a logical reasoning problem"""
        result = ReasoningResult(query=query, reasoning_type=ReasoningType.LOGIC)
        
        try:
            # Step 1: Parse relations from the query
            relations = self.parse_relations(query)
            result.steps.append(ReasoningStep(
                step_num=1,
                description="Extract relations from the problem",
                operation="parse",
                result=relations
            ))
            
            if not relations:
                result.final_answer = "I couldn't identify any relations in the query."
                result.success = False
                return result
            
            # Step 2: Build relation graph
            graph = self.build_graph(relations)
            result.steps.append(ReasoningStep(
                step_num=2,
                description="Build relation graph for transitive reasoning",
                operation="build_graph",
                result=f"Graph with {sum(len(v) for v in graph['>'].values())} '>' edges"
            ))
            
            # Step 3: Identify what we're solving for
            # Look for question patterns
            question_patterns = [
                r'what.*relationship.*between\s+(\w+)\s+and\s+(\w+)',
                r'compare\s+(\w+)\s+and\s+(\w+)',
                r'is\s+(\w+)\s*(>|<|>=|<=|==)\s*(\w+)',
                r'how.*(\w+).*relate.*(\w+)',
                r'(\w+)\s+vs\s+(\w+)',
            ]
            
            target_a, target_b = None, None
            for pattern in question_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    target_a, target_b = groups[0], groups[-1]
                    break
            
            # If no explicit question, find relationship between first and last mentioned entities
            if not target_a or not target_b:
                entities = set()
                for a, _, b in relations:
                    entities.add(a)
                    entities.add(b)
                entities = list(entities)
                if len(entities) >= 2:
                    target_a, target_b = entities[0], entities[-1]
            
            result.steps.append(ReasoningStep(
                step_num=3,
                description=f"Identify target comparison: {target_a} vs {target_b}",
                operation="identify_targets",
                result=f"Comparing {target_a} and {target_b}"
            ))
            
            # Step 4: Find transitive relations
            gt_path = self.find_transitive_relation(graph, target_a, target_b, '>')
            lt_path = self.find_transitive_relation(graph, target_a, target_b, '<')
            eq_path = self.find_transitive_relation(graph, target_a, target_b, '==')
            
            # Step 5: Determine relationship
            if gt_path:
                chain = ' > '.join(gt_path)
                result.steps.append(ReasoningStep(
                    step_num=4,
                    description=f"Found transitive chain: {chain}",
                    operation="transitive_closure",
                    result=f"{target_a} > {target_b}"
                ))
                result.final_answer = f"**{target_a} > {target_b}** (is greater than)\n\nReasoning chain: {chain}"
                
            elif lt_path:
                chain = ' < '.join(lt_path)
                result.steps.append(ReasoningStep(
                    step_num=4,
                    description=f"Found transitive chain: {chain}",
                    operation="transitive_closure",
                    result=f"{target_a} < {target_b}"
                ))
                result.final_answer = f"**{target_a} < {target_b}** (is less than)\n\nReasoning chain: {chain}"
                
            elif eq_path:
                result.steps.append(ReasoningStep(
                    step_num=4,
                    description=f"Found equality relation",
                    operation="transitive_closure",
                    result=f"{target_a} == {target_b}"
                ))
                result.final_answer = f"**{target_a} = {target_b}** (are equal)"
                
            else:
                # Try reverse
                gt_path_rev = self.find_transitive_relation(graph, target_b, target_a, '>')
                if gt_path_rev:
                    chain = ' > '.join(gt_path_rev)
                    result.steps.append(ReasoningStep(
                        step_num=4,
                        description=f"Found reverse transitive chain: {chain}",
                        operation="transitive_closure",
                        result=f"{target_b} > {target_a}, therefore {target_a} < {target_b}"
                    ))
                    result.final_answer = f"**{target_a} < {target_b}** (is less than)\n\nReasoning: Since {chain}, we know {target_a} < {target_b}"
                else:
                    result.final_answer = f"Cannot determine a definitive relationship between {target_a} and {target_b} from the given information."
                    result.confidence = 0.5
                    
        except Exception as e:
            result.success = False
            result.error = str(e)
            result.final_answer = f"Error in logical reasoning: {str(e)}"
            
        return result


class MathEngine:
    """
    Handles mathematical operations: arithmetic, algebra, 
    step-by-step problem solving, equations
    """
    
    def __init__(self):
        self.variables: Dict[str, float] = {}
        self.constants = {
            'pi': math.pi, 'e': math.e, 'phi': (1 + math.sqrt(5)) / 2,
            'tau': math.tau, 'inf': float('inf')
        }
        self.functions = {
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'sqrt': math.sqrt, 'abs': abs, 'log': math.log,
            'log10': math.log10, 'exp': math.exp, 'pow': pow,
            'floor': math.floor, 'ceil': math.ceil, 'round': round,
            'factorial': math.factorial, 'gcd': math.gcd,
        }
        
    def tokenize(self, expr: str) -> List[str]:
        """Tokenize a math expression"""
        # Add spaces around operators
        expr = re.sub(r'([+\-*/^()=,])', r' \1 ', expr)
        tokens = expr.split()
        return [t for t in tokens if t]
    
    def parse_word_problem(self, text: str) -> Tuple[str, List[ReasoningStep]]:
        """Convert word problem to mathematical expression"""
        steps = []
        
        # Extract numbers
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
        steps.append(ReasoningStep(
            step_num=1,
            description="Extract numbers from problem",
            operation="extract",
            result=numbers
        ))
        
        # Identify operation keywords
        operations = {
            'add|plus|sum|total|combined|together|more than|increased': '+',
            'subtract|minus|difference|less than|decreased|fewer': '-',
            'multiply|times|product|of': '*',
            'divide|quotient|split|per|ratio|shared': '/',
            'power|raised|squared|cubed|exponent': '^',
            'percent|percentage': '%',
            'square root|sqrt': 'sqrt',
        }
        
        detected_ops = []
        text_lower = text.lower()
        for pattern, op in operations.items():
            if re.search(pattern, text_lower):
                detected_ops.append(op)
                
        steps.append(ReasoningStep(
            step_num=2,
            description="Identify mathematical operations",
            operation="detect_ops",
            result=detected_ops
        ))
        
        # Build expression
        if len(numbers) >= 2 and detected_ops:
            op = detected_ops[0]
            if op == '%' or 'percent' in text_lower:
                # Handle "X% of Y" or "X percent of Y"
                expr = f"({numbers[0]} / 100) * {numbers[1]}"
            elif op == 'sqrt':
                expr = f"sqrt({numbers[0]})"
            else:
                expr = f" {op} ".join(numbers)
        elif len(numbers) == 1:
            expr = numbers[0]
        else:
            expr = text
            
        steps.append(ReasoningStep(
            step_num=3,
            description="Construct mathematical expression",
            operation="build_expr",
            result=expr
        ))
        
        return expr, steps
    
    def evaluate_expression(self, expr: str) -> Tuple[Any, List[ReasoningStep]]:
        """Evaluate a mathematical expression step by step"""
        steps = []
        original = expr
        
        # First, handle functions BEFORE any other transformations
        # This prevents log10(x) from becoming log10*(x)
        for func_name in self.functions.keys():
            pattern = rf'\b{func_name}\s*\(\s*([^)]+)\s*\)'
            matches = list(re.finditer(pattern, expr, re.IGNORECASE))
            for match in reversed(matches):
                try:
                    inner = match.group(1).strip()
                    func = self.functions[func_name]
                    
                    # Handle multi-argument functions
                    if ',' in inner:
                        args = [arg.strip() for arg in inner.split(',')]
                        evaluated_args = []
                        for arg in args:
                            val, _ = self.evaluate_expression(arg)
                            evaluated_args.append(val)
                        if func_name in ['pow', 'gcd']:
                            result = func(int(evaluated_args[0]) if func_name == 'gcd' else evaluated_args[0], 
                                         int(evaluated_args[1]) if func_name == 'gcd' else evaluated_args[1])
                        else:
                            result = func(*evaluated_args)
                    else:
                        # Single argument - recursively evaluate
                        inner_val, _ = self.evaluate_expression(inner)
                        if func_name == 'factorial':
                            result = func(int(inner_val))
                        else:
                            result = func(float(inner_val))
                    
                    expr = expr[:match.start()] + str(result) + expr[match.end():]
                    steps.append(ReasoningStep(
                        step_num=len(steps) + 1,
                        description=f"Evaluate {func_name}({inner})",
                        operation="function",
                        result=result
                    ))
                except Exception as e:
                    pass
        
        # Clean expression
        expr = expr.replace('^', '**').replace('×', '*').replace('÷', '/')
        
        # Handle implicit multiplication: 2x -> 2*x, 2(3) -> 2*(3)
        # But NOT for function names like log10, factorial5, etc.
        expr = re.sub(r'(\d)([a-zA-Z])(?![a-zA-Z0-9_])', r'\1*\2', expr)
        expr = re.sub(r'(\d)\(', r'\1*(', expr)
        expr = re.sub(r'\)(\d)', r')*\1', expr)
        expr = re.sub(r'\)\(', r')*(', expr)
        
        steps.append(ReasoningStep(
            step_num=len(steps) + 1,
            description="Clean and normalize expression",
            operation="normalize",
            result=expr
        ))
        
        # Replace constants
        for name, value in self.constants.items():
            expr = re.sub(rf'\b{name}\b', str(value), expr, flags=re.IGNORECASE)
        
        # Handle step-by-step arithmetic for visibility
        # First, handle parentheses
        paren_pattern = r'\(([^()]+)\)'
        while '(' in expr:
            match = re.search(paren_pattern, expr)
            if match:
                inner = match.group(1)
                inner_result, inner_steps = self.evaluate_simple(inner)
                steps.append(ReasoningStep(
                    step_num=len(steps) + 1,
                    description=f"Evaluate parentheses: ({inner})",
                    operation="parentheses",
                    result=inner_result
                ))
                expr = expr[:match.start()] + str(inner_result) + expr[match.end():]
            else:
                break
        
        # Final evaluation
        final_result, final_steps = self.evaluate_simple(expr)
        steps.extend(final_steps)
        
        return final_result, steps
    
    def evaluate_simple(self, expr: str) -> Tuple[float, List[ReasoningStep]]:
        """Evaluate a simple expression without parentheses"""
        steps = []
        
        try:
            # Handle exponentiation first
            while '**' in expr:
                match = re.search(r'([\d.]+)\s*\*\*\s*([\d.]+)', expr)
                if match:
                    a, b = float(match.group(1)), float(match.group(2))
                    result = a ** b
                    steps.append(ReasoningStep(
                        step_num=len(steps) + 1,
                        description=f"Exponentiation: {a}^{b}",
                        operation="power",
                        result=result
                    ))
                    expr = expr[:match.start()] + str(result) + expr[match.end():]
                else:
                    break
            
            # Handle multiplication and division (left to right)
            while re.search(r'[\d.]+\s*[*/]\s*[\d.]+', expr):
                match = re.search(r'([\d.]+)\s*([*/])\s*([\d.]+)', expr)
                if match:
                    a, op, b = float(match.group(1)), match.group(2), float(match.group(3))
                    if op == '*':
                        result = a * b
                        op_name = "Multiplication"
                    else:
                        result = a / b if b != 0 else float('inf')
                        op_name = "Division"
                    steps.append(ReasoningStep(
                        step_num=len(steps) + 1,
                        description=f"{op_name}: {a} {op} {b}",
                        operation=op_name.lower(),
                        result=result
                    ))
                    expr = expr[:match.start()] + str(result) + expr[match.end():]
                else:
                    break
            
            # Handle addition and subtraction (left to right)
            while re.search(r'[\d.]+\s*[+\-]\s*[\d.]+', expr):
                match = re.search(r'([\d.]+)\s*([+\-])\s*([\d.]+)', expr)
                if match:
                    a, op, b = float(match.group(1)), match.group(2), float(match.group(3))
                    if op == '+':
                        result = a + b
                        op_name = "Addition"
                    else:
                        result = a - b
                        op_name = "Subtraction"
                    steps.append(ReasoningStep(
                        step_num=len(steps) + 1,
                        description=f"{op_name}: {a} {op} {b}",
                        operation=op_name.lower(),
                        result=result
                    ))
                    expr = expr[:match.start()] + str(result) + expr[match.end():]
                else:
                    break
            
            # Try to get final number
            result = float(expr.strip())
            return result, steps
            
        except Exception as e:
            # Fallback to Python eval for complex expressions
            try:
                # Safe eval with only math operations
                allowed_names = {**self.constants, **self.functions}
                result = eval(expr, {"__builtins__": {}}, allowed_names)
                return float(result), steps
            except:
                return 0, steps
    
    def solve_equation(self, equation: str) -> Tuple[Dict[str, float], List[ReasoningStep]]:
        """Solve simple algebraic equations"""
        steps = []
        
        # Parse equation
        if '=' not in equation:
            return {}, [ReasoningStep(1, "No equation found", "error", "Missing '='")]
        
        left, right = equation.split('=', 1)
        left, right = left.strip(), right.strip()
        
        steps.append(ReasoningStep(
            step_num=1,
            description=f"Parse equation: {left} = {right}",
            operation="parse",
            result=f"LHS: {left}, RHS: {right}"
        ))
        
        # Find variable
        variables = set(re.findall(r'[a-zA-Z]\w*', equation)) - set(self.constants.keys()) - set(self.functions.keys())
        
        if not variables:
            return {}, steps
        
        var = list(variables)[0]
        steps.append(ReasoningStep(
            step_num=2,
            description=f"Identify variable to solve for",
            operation="identify_var",
            result=var
        ))
        
        # Simple linear equation solver: ax + b = c or ax = b
        # Try to isolate variable
        try:
            # Move everything to left side: ax + b - c = 0
            # Coefficient extraction for simple cases
            
            # Pattern: ax + b = c or ax - b = c
            pattern = rf'([\d.]*)\s*\*?\s*{var}\s*([+\-]\s*[\d.]+)?\s*=\s*([\d.]+)'
            match = re.match(pattern, equation.replace(' ', ''))
            
            if match:
                coef = float(match.group(1)) if match.group(1) else 1
                const = float(match.group(2).replace(' ', '')) if match.group(2) else 0
                rhs = float(match.group(3))
                
                steps.append(ReasoningStep(
                    step_num=3,
                    description=f"Extract coefficients: {coef}{var} + {const} = {rhs}",
                    operation="extract_coef",
                    result=f"a={coef}, b={const}, c={rhs}"
                ))
                
                # Solve: coef * var + const = rhs
                # var = (rhs - const) / coef
                solution = (rhs - const) / coef
                
                steps.append(ReasoningStep(
                    step_num=4,
                    description=f"Isolate variable: {var} = ({rhs} - {const}) / {coef}",
                    operation="solve",
                    result=solution
                ))
                
                return {var: solution}, steps
            
            # Pattern: just ax = b
            pattern2 = rf'([\d.]*)\s*\*?\s*{var}\s*=\s*([\d.]+)'
            match2 = re.match(pattern2, equation.replace(' ', ''))
            
            if match2:
                coef = float(match2.group(1)) if match2.group(1) else 1
                rhs = float(match2.group(2))
                solution = rhs / coef
                
                steps.append(ReasoningStep(
                    step_num=3,
                    description=f"Solve: {var} = {rhs} / {coef}",
                    operation="solve",
                    result=solution
                ))
                
                return {var: solution}, steps
                
        except Exception as e:
            steps.append(ReasoningStep(
                step_num=len(steps) + 1,
                description=f"Error solving equation",
                operation="error",
                result=str(e)
            ))
        
        return {}, steps
    
    def solve(self, query: str) -> ReasoningResult:
        """Main entry point for math problems"""
        result = ReasoningResult(query=query, reasoning_type=ReasoningType.MATH)
        
        try:
            # Detect if it's a word problem or direct expression
            has_words = bool(re.search(r'[a-zA-Z]{3,}', query.replace('sqrt', '').replace('log', '').replace('factorial', '').replace('sin', '').replace('cos', '').replace('tan', '')))
            has_equation = '=' in query and re.search(r'[a-zA-Z]', query.split('=')[0])
            has_function = bool(re.search(r'(sqrt|factorial|log|log10|sin|cos|tan|exp|abs|pow|floor|ceil|round|gcd)\s*\(', query, re.IGNORECASE))
            
            if has_equation and not has_function:
                # Solve equation
                solutions, steps = self.solve_equation(query)
                result.steps = steps
                if solutions:
                    result.final_answer = "**Solution:**\n" + "\n".join([f"{var} = {val}" for var, val in solutions.items()])
                else:
                    result.final_answer = "Could not solve the equation."
                    result.success = False
            
            elif has_function or re.match(r'^[\d\s+\-*/^().]+$', query.strip()) or re.search(r'\d\s*[+\-*/^]\s*\d', query):
                # Direct expression with function or pure arithmetic
                answer, steps = self.evaluate_expression(query)
                result.steps = steps
                
                # Format nicely
                if isinstance(answer, float) and answer == int(answer):
                    answer = int(answer)
                    
                result.final_answer = f"**{query} = {answer}**"
                    
            elif has_words:
                # Word problem
                expr, parse_steps = self.parse_word_problem(query)
                result.steps.extend(parse_steps)
                
                answer, eval_steps = self.evaluate_expression(expr)
                result.steps.extend(eval_steps)
                
                result.final_answer = f"**Answer: {answer}**\n\nExpression: {expr}"
                
            else:
                # Direct expression
                answer, steps = self.evaluate_expression(query)
                result.steps = steps
                
                # Format nicely
                if isinstance(answer, float) and answer == int(answer):
                    answer = int(answer)
                    
                result.final_answer = f"**{query} = {answer}**"
                
        except Exception as e:
            result.success = False
            result.error = str(e)
            result.final_answer = f"Error: {str(e)}"
            
        return result


class CodeAnalyzer:
    """
    Analyzes and debugs code: syntax checking, error detection,
    suggestions, and explanations
    """
    
    def __init__(self):
        self.common_errors = {
            'IndentationError': 'Check your indentation. Python uses spaces/tabs to define code blocks.',
            'SyntaxError': 'There\'s a syntax mistake. Check for missing colons, parentheses, or quotes.',
            'NameError': 'You\'re using a variable or function that hasn\'t been defined yet.',
            'TypeError': 'You\'re using the wrong type. Check if you\'re mixing strings and numbers.',
            'IndexError': 'You\'re trying to access an index that doesn\'t exist in the list.',
            'KeyError': 'The dictionary key you\'re looking for doesn\'t exist.',
            'ValueError': 'The value you passed is the right type but wrong value.',
            'AttributeError': 'The object doesn\'t have that attribute or method.',
            'ZeroDivisionError': 'You\'re trying to divide by zero.',
            'ImportError': 'The module you\'re trying to import doesn\'t exist or isn\'t installed.',
            'FileNotFoundError': 'The file path doesn\'t exist or is incorrect.',
        }
        
    def detect_language(self, code: str) -> str:
        """Detect programming language from code"""
        indicators = {
            'python': [r'\bdef\s+\w+\s*\(', r'\bimport\s+\w+', r'\bprint\s*\(', r':\s*$', r'\bclass\s+\w+:'],
            'javascript': [r'\bfunction\s+\w+\s*\(', r'\bconst\s+\w+', r'\blet\s+\w+', r'\bconsole\.log', r'=>'],
            'java': [r'\bpublic\s+class', r'\bSystem\.out\.print', r'\bvoid\s+main', r'\bimport\s+java\.'],
            'c': [r'#include\s*<', r'\bint\s+main\s*\(', r'\bprintf\s*\(', r'\bscanf\s*\('],
            'cpp': [r'#include\s*<iostream>', r'\bcout\s*<<', r'\bcin\s*>>', r'\busing\s+namespace\s+std'],
            'sql': [r'\bSELECT\b', r'\bFROM\b', r'\bWHERE\b', r'\bINSERT\s+INTO\b', r'\bCREATE\s+TABLE\b'],
        }
        
        scores = {lang: 0 for lang in indicators}
        for lang, patterns in indicators.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                    scores[lang] += 1
                    
        best_lang = max(scores, key=scores.get)
        return best_lang if scores[best_lang] > 0 else 'unknown'
    
    def analyze_python(self, code: str) -> List[Dict]:
        """Analyze Python code for errors and issues"""
        issues = []
        
        # Try to parse as AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            issues.append({
                'type': 'SyntaxError',
                'line': e.lineno,
                'column': e.offset,
                'message': str(e.msg),
                'suggestion': self.common_errors['SyntaxError']
            })
            return issues
        
        # Analyze AST for common issues
        class CodeVisitor(ast.NodeVisitor):
            def __init__(self):
                self.defined_names = set()
                self.used_names = set()
                self.issues = []
                
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    self.defined_names.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    self.used_names.add((node.id, node.lineno))
                self.generic_visit(node)
                
            def visit_FunctionDef(self, node):
                self.defined_names.add(node.name)
                # Check for missing return in functions that seem to need it
                has_return = any(isinstance(n, ast.Return) and n.value for n in ast.walk(node))
                if not has_return and not node.name.startswith('_'):
                    # Check if function has significant body
                    if len(node.body) > 1 or not isinstance(node.body[0], ast.Pass):
                        self.issues.append({
                            'type': 'Warning',
                            'line': node.lineno,
                            'message': f"Function '{node.name}' might be missing a return statement",
                            'suggestion': 'Consider adding a return statement if the function should return a value.'
                        })
                self.generic_visit(node)
                
            def visit_Compare(self, node):
                # Check for == True or == False
                for op, comp in zip(node.ops, node.comparators):
                    if isinstance(op, ast.Eq):
                        if isinstance(comp, ast.Constant) and comp.value in (True, False):
                            self.issues.append({
                                'type': 'Style',
                                'line': node.lineno,
                                'message': 'Comparing to True/False is redundant',
                                'suggestion': 'Use `if condition:` instead of `if condition == True:`'
                            })
                self.generic_visit(node)
                
        visitor = CodeVisitor()
        visitor.visit(tree)
        
        # Check for undefined names (basic check)
        builtins = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports.add(alias.asname or alias.name)
        
        for name, lineno in visitor.used_names:
            if name not in visitor.defined_names and name not in builtins and name not in imports:
                issues.append({
                    'type': 'NameError',
                    'line': lineno,
                    'message': f"'{name}' might not be defined",
                    'suggestion': f"Make sure '{name}' is defined before using it."
                })
        
        issues.extend(visitor.issues)
        return issues
    
    def suggest_fixes(self, code: str, error_msg: str) -> List[str]:
        """Suggest fixes based on error message"""
        suggestions = []
        
        # Common fix patterns
        fixes = {
            r'unexpected indent': ['Remove extra spaces/tabs at the beginning of the line'],
            r'expected an indented block': ['Add indentation (4 spaces) after the colon'],
            r"name '(\w+)' is not defined": [
                'Define the variable before using it',
                'Check for typos in the variable name',
                'Import the module if it\'s a library function'
            ],
            r'invalid syntax': [
                'Check for missing colons after if/for/while/def/class',
                'Check for unclosed parentheses, brackets, or quotes',
                'Make sure you\'re using = for assignment and == for comparison'
            ],
            r'list index out of range': [
                'Check that the index is less than len(list)',
                'Use try/except to handle missing indices',
                'Verify the list has been populated'
            ],
            r"can only concatenate str": [
                'Use str() to convert numbers to strings',
                'Use f-strings: f"text {variable}"',
                'Use .format() method'
            ],
            r'takes \d+ positional argument': [
                'Check the number of arguments in your function call',
                'Add/remove arguments to match the function definition',
                'Check if you forgot "self" in a method definition'
            ],
        }
        
        for pattern, fix_list in fixes.items():
            if re.search(pattern, error_msg, re.IGNORECASE):
                suggestions.extend(fix_list)
                
        if not suggestions:
            suggestions.append('Review the error message and check the indicated line')
            suggestions.append('Search for the error message online for more help')
            
        return suggestions
    
    def format_code(self, code: str) -> str:
        """Basic code formatting"""
        lines = code.split('\n')
        formatted = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Decrease indent for these keywords at start
            if stripped.startswith(('elif ', 'else:', 'except:', 'except ', 'finally:', 'elif:', 'else ')):
                indent_level = max(0, indent_level - 1)
            elif stripped.startswith(('return', 'break', 'continue', 'pass', 'raise')):
                pass  # Keep current indent
            
            # Add the line with proper indentation
            if stripped:
                formatted.append('    ' * indent_level + stripped)
            else:
                formatted.append('')
            
            # Increase indent after these
            if stripped.endswith(':') and not stripped.startswith('#'):
                indent_level += 1
            
            # Decrease after return/break at end of block
            if stripped.startswith(('return ', 'return,', 'break', 'continue')) and indent_level > 0:
                indent_level -= 1
                
        return '\n'.join(formatted)
    
    def explain_code(self, code: str) -> List[Dict]:
        """Generate line-by-line explanation of code"""
        explanations = []
        lines = code.split('\n')
        
        patterns = {
            r'^import\s+(\w+)': lambda m: f"Import the '{m.group(1)}' module",
            r'^from\s+(\w+)\s+import': lambda m: f"Import specific items from '{m.group(1)}'",
            r'^def\s+(\w+)\s*\(([^)]*)\)': lambda m: f"Define function '{m.group(1)}' with parameters: {m.group(2) or 'none'}",
            r'^class\s+(\w+)': lambda m: f"Define a class named '{m.group(1)}'",
            r'^if\s+(.+):': lambda m: f"Check condition: {m.group(1)}",
            r'^elif\s+(.+):': lambda m: f"Else, check condition: {m.group(1)}",
            r'^else:': lambda m: "Execute if no previous conditions were true",
            r'^for\s+(\w+)\s+in\s+(.+):': lambda m: f"Loop through {m.group(2)}, assigning each item to '{m.group(1)}'",
            r'^while\s+(.+):': lambda m: f"Loop while condition is true: {m.group(1)}",
            r'^return\s+(.+)': lambda m: f"Return the value: {m.group(1)}",
            r'^return$': lambda m: "Return None (exit function)",
            r'^print\s*\((.+)\)': lambda m: f"Output to console: {m.group(1)}",
            r'^(\w+)\s*=\s*(.+)': lambda m: f"Assign '{m.group(2)}' to variable '{m.group(1)}'",
            r'^#\s*(.+)': lambda m: f"Comment: {m.group(1)}",
            r'^try:': lambda m: "Start a try block to catch exceptions",
            r'^except\s*(\w+)?': lambda m: f"Handle {m.group(1) or 'any'} exception",
            r'^raise\s+(\w+)': lambda m: f"Raise a {m.group(1)} exception",
            r'^with\s+(.+)\s+as\s+(\w+)': lambda m: f"Open {m.group(1)} as '{m.group(2)}' with automatic cleanup",
            r'^assert\s+(.+)': lambda m: f"Assert that condition is true: {m.group(1)}",
            r'^pass$': lambda m: "Do nothing (placeholder)",
            r'^break$': lambda m: "Exit the current loop",
            r'^continue$': lambda m: "Skip to next iteration of loop",
        }
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped:
                continue
                
            explanation = None
            for pattern, explain_func in patterns.items():
                match = re.match(pattern, stripped)
                if match:
                    explanation = explain_func(match)
                    break
                    
            if explanation:
                explanations.append({
                    'line': i,
                    'code': stripped,
                    'explanation': explanation
                })
            else:
                explanations.append({
                    'line': i,
                    'code': stripped,
                    'explanation': 'Execute this statement'
                })
                
        return explanations
    
    def solve(self, query: str, code: str = None) -> ReasoningResult:
        """Main entry point for code analysis"""
        result = ReasoningResult(query=query, reasoning_type=ReasoningType.CODE)
        
        try:
            # Extract code from query if not provided separately
            if code is None:
                # Look for code blocks
                code_match = re.search(r'```(?:\w+)?\n?(.*?)```', query, re.DOTALL)
                if code_match:
                    code = code_match.group(1)
                else:
                    # Try to find inline code
                    code_match = re.search(r'`([^`]+)`', query)
                    if code_match:
                        code = code_match.group(1)
                    else:
                        # Check for code after certain keywords
                        for marker in ['code:', 'debug:', 'fix:', 'analyze:', 'explain:']:
                            if marker in query.lower():
                                code = query.lower().split(marker, 1)[1].strip()
                                break
                        
                        # If still no code, try to extract code-like content
                        if not code:
                            # Look for code patterns in the query itself
                            code_patterns = [
                                r'((?:def|class|for|while|if|import|from|print|return)\s+.*)',
                            ]
                            for pattern in code_patterns:
                                match = re.search(pattern, query)
                                if match:
                                    code = match.group(1)
                                    break
                                
            if not code:
                result.final_answer = "Please provide code to analyze. You can use code blocks (```) or paste the code directly."
                result.success = False
                return result
            
            # Step 1: Detect language
            language = self.detect_language(code)
            result.steps.append(ReasoningStep(
                step_num=1,
                description="Detect programming language",
                operation="detect_language",
                result=language
            ))
            
            # Step 2: Analyze code
            if language == 'python':
                issues = self.analyze_python(code)
            else:
                # Basic analysis for other languages
                issues = []
                if not code.strip():
                    issues.append({'type': 'Error', 'message': 'Empty code', 'suggestion': 'Provide some code to analyze'})
            
            result.steps.append(ReasoningStep(
                step_num=2,
                description=f"Analyze {language} code for issues",
                operation="analyze",
                result=f"Found {len(issues)} issue(s)"
            ))
            
            # Step 3: Generate explanations
            if 'explain' in query.lower() or 'what does' in query.lower():
                explanations = self.explain_code(code)
                result.steps.append(ReasoningStep(
                    step_num=3,
                    description="Generate code explanations",
                    operation="explain",
                    result=f"Explained {len(explanations)} lines"
                ))
                
                explanation_text = "\n".join([
                    f"**Line {e['line']}:** `{e['code']}`\n  → {e['explanation']}"
                    for e in explanations
                ])
                result.final_answer = f"## Code Explanation ({language})\n\n{explanation_text}"
                
            elif issues:
                # Format issues
                issue_text = "\n\n".join([
                    f"**{issue['type']}** (Line {issue.get('line', '?')})\n"
                    f"Message: {issue['message']}\n"
                    f"💡 Suggestion: {issue.get('suggestion', 'Review this line')}"
                    for issue in issues
                ])
                
                result.final_answer = f"## Code Analysis ({language})\n\n{issue_text}"
                
                if any(i['type'] in ['SyntaxError', 'Error'] for i in issues):
                    result.confidence = 0.9
                    
            else:
                result.final_answer = f"## Code Analysis ({language})\n\n✅ No obvious issues found!\n\nThe code appears to be syntactically correct."
                result.steps.append(ReasoningStep(
                    step_num=3,
                    description="Code passed basic analysis",
                    operation="pass",
                    result="No issues detected"
                ))
                
        except Exception as e:
            result.success = False
            result.error = str(e)
            result.final_answer = f"Error analyzing code: {str(e)}"
            
        return result


class ComplexAnalyzer:
    """
    Handles complex multi-step analysis, comparisons,
    and structured reasoning
    """
    
    def __init__(self):
        self.logic = LogicEngine()
        self.math = MathEngine()
        self.code = CodeAnalyzer()
        
    def analyze_question(self, query: str) -> Dict:
        """Analyze what type of reasoning is needed"""
        query_lower = query.lower()
        
        analysis = {
            'needs_logic': False,
            'needs_math': False,
            'needs_code': False,
            'needs_comparison': False,
            'needs_steps': False,
            'complexity': 'simple'
        }
        
        # Logic indicators
        logic_patterns = [
            r'if\s+\w+\s*(>|<|=)',
            r'relationship\s+between',
            r'therefore|thus|hence|so\s+\w+\s+must',
            r'implies|conclude|deduce',
            r'and\s+\w+\s*(>|<)',
            r'transitive|syllogism',
        ]
        
        # Math indicators
        math_patterns = [
            r'\d+\s*[+\-*/^]\s*\d+',
            r'calculate|compute|solve|evaluate',
            r'equation|formula|expression',
            r'sum|product|difference|quotient',
            r'percent|ratio|proportion',
            r'sqrt|square root|power|exponent',
        ]
        
        # Code indicators
        code_patterns = [
            r'```',
            r'debug|fix|error|bug',
            r'code|function|script|program',
            r'syntax|compile|runtime',
            r'def\s+\w+|class\s+\w+|import\s+\w+',
        ]
        
        for pattern in logic_patterns:
            if re.search(pattern, query_lower):
                analysis['needs_logic'] = True
                break
                
        for pattern in math_patterns:
            if re.search(pattern, query_lower):
                analysis['needs_math'] = True
                break
                
        for pattern in code_patterns:
            if re.search(pattern, query_lower):
                analysis['needs_code'] = True
                break
        
        # Complexity assessment
        word_count = len(query.split())
        sentence_count = len(re.split(r'[.!?]', query))
        
        if word_count > 50 or sentence_count > 3:
            analysis['complexity'] = 'complex'
            analysis['needs_steps'] = True
        elif word_count > 20:
            analysis['complexity'] = 'medium'
            
        return analysis
    
    def solve(self, query: str) -> ReasoningResult:
        """Main entry point for complex analysis"""
        result = ReasoningResult(query=query, reasoning_type=ReasoningType.ANALYSIS)
        
        try:
            # Analyze what we need to do
            analysis = self.analyze_question(query)
            result.steps.append(ReasoningStep(
                step_num=1,
                description="Analyze query requirements",
                operation="analyze",
                result=analysis
            ))
            
            sub_results = []
            
            # Route to appropriate engines
            if analysis['needs_logic']:
                logic_result = self.logic.solve(query)
                sub_results.append(('Logic', logic_result))
                result.steps.extend(logic_result.steps)
                
            if analysis['needs_math']:
                math_result = self.math.solve(query)
                sub_results.append(('Math', math_result))
                result.steps.extend(math_result.steps)
                
            if analysis['needs_code']:
                code_result = self.code.solve(query)
                sub_results.append(('Code', code_result))
                result.steps.extend(code_result.steps)
            
            # Combine results
            if sub_results:
                combined = []
                for name, sub_result in sub_results:
                    if sub_result.success:
                        combined.append(f"### {name} Analysis\n{sub_result.final_answer}")
                        
                result.final_answer = "\n\n".join(combined)
                result.confidence = min(r.confidence for _, r in sub_results)
            else:
                result.final_answer = "I analyzed your query but couldn't determine a specific reasoning approach. Could you rephrase or provide more details?"
                result.confidence = 0.5
                
        except Exception as e:
            result.success = False
            result.error = str(e)
            result.final_answer = f"Error in analysis: {str(e)}"
            
        return result


class ReasoningEngine:
    """
    Master reasoning engine that coordinates all reasoning capabilities
    """
    
    def __init__(self):
        self.logic = LogicEngine()
        self.math = MathEngine()
        self.code = CodeAnalyzer()
        self.analyzer = ComplexAnalyzer()
        
    def classify_query(self, query: str) -> ReasoningType:
        """Determine the type of reasoning needed"""
        query_lower = query.lower()
        
        # Check for code
        if '```' in query or re.search(r'\bdef\s+\w+|class\s+\w+|import\s+\w+', query):
            return ReasoningType.CODE
        if any(kw in query_lower for kw in ['debug', 'fix code', 'syntax error', 'code error', 'explain code', 'explain:', 'analyze code']):
            return ReasoningType.CODE
        # Check for code-like patterns
        if re.search(r'(for|while|if)\s+\w+.*:', query) or re.search(r'print\s*\(', query):
            return ReasoningType.CODE
        
        # Check for logic
        logic_indicators = [
            r'if\s+\w+\s*(>|<|>=|<=|=)',
            r'relationship\s+between',
            r'\w+\s*(>|<|>=|<=)\s*\w+.*and.*\w+\s*(>|<|>=|<=)\s*\w+',
            r'therefore|thus|conclude|deduce',
            r'is\s+\w+\s+(greater|less|equal|bigger|smaller)',
            r'is\s+(taller|shorter|older|younger|heavier|lighter|faster|slower)\s+than',
            r'(taller|shorter|older|younger|heavier|lighter|faster|slower)\s+than.*\.\s*\w+\s+is\s+(taller|shorter|older|younger)',
        ]
        for pattern in logic_indicators:
            if re.search(pattern, query_lower):
                return ReasoningType.LOGIC
        
        # Check for math
        if re.search(r'\d+\s*[+\-*/^%]\s*\d+', query):
            return ReasoningType.MATH
        if any(kw in query_lower for kw in ['calculate', 'compute', 'solve', 'evaluate', 'equation', '=']):
            if re.search(r'\d', query):
                return ReasoningType.MATH
        if any(kw in query_lower for kw in ['sum', 'product', 'difference', 'quotient', 'percent', 'sqrt', 'factorial']):
            return ReasoningType.MATH
        if re.search(r'\d+\s*%', query) or 'percent' in query_lower or 'percentage' in query_lower:
            return ReasoningType.MATH
        if re.search(r'what is \d+', query_lower) and re.search(r'of \d+', query_lower):
            return ReasoningType.MATH
        
        # Complex analysis for multi-part questions
        if len(query.split()) > 30 or query.count('?') > 1:
            return ReasoningType.ANALYSIS
        
        return ReasoningType.GENERAL
    
    def reason(self, query: str) -> ReasoningResult:
        """Main reasoning entry point"""
        reasoning_type = self.classify_query(query)
        
        if reasoning_type == ReasoningType.LOGIC:
            return self.logic.solve(query)
        elif reasoning_type == ReasoningType.MATH:
            return self.math.solve(query)
        elif reasoning_type == ReasoningType.CODE:
            return self.code.solve(query)
        elif reasoning_type == ReasoningType.ANALYSIS:
            return self.analyzer.solve(query)
        else:
            # Return a general result indicating no specific reasoning was applied
            return ReasoningResult(
                query=query,
                reasoning_type=ReasoningType.GENERAL,
                final_answer="",
                success=True,
                confidence=0.5
            )
    
    def solve_with_steps(self, query: str) -> Dict:
        """Solve and return structured response with steps"""
        result = self.reason(query)
        return result.to_dict()