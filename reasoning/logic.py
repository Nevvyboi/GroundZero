"""
Logic Reasoner
==============
Logical deduction and inference capabilities.
"""

import re
from typing import Tuple, List, Dict, Any, Optional


class LogicReasoner:
    """Handles logical reasoning and inference"""
    
    def __init__(self):
        # Logical connectives
        self.connectives = {
            'and': 'conjunction',
            'or': 'disjunction',
            'not': 'negation',
            'if': 'implication',
            'then': 'implication',
            'implies': 'implication',
            'therefore': 'conclusion',
            'because': 'reason',
            'since': 'reason'
        }
    
    def analyze(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Analyze a logical statement or question.
        Returns (conclusion, steps).
        """
        query_lower = query.lower()
        steps = []
        
        # Check for comparison reasoning
        if self._is_comparison_query(query_lower):
            return self._analyze_comparison(query_lower, steps)
        
        # Check for syllogism
        if self._is_syllogism(query_lower):
            return self._analyze_syllogism(query_lower, steps)
        
        # Check for conditional
        if 'if' in query_lower and 'then' in query_lower:
            return self._analyze_conditional(query_lower, steps)
        
        # Check for Boolean logic
        if any(c in query_lower for c in ['true', 'false', 'and', 'or', 'not']):
            return self._analyze_boolean(query_lower, steps)
        
        # Default analysis
        steps.append({
            'description': 'Analyzing logical structure',
            'operation': 'parse',
            'result': 'Statement parsed for logical components'
        })
        
        return "Unable to determine a logical conclusion from the given statement.", steps
    
    def _is_comparison_query(self, query: str) -> bool:
        """Check if query involves comparison reasoning"""
        patterns = [
            r'if\s+(\w+)\s*[><]=?\s*(\w+)\s+and\s+(\w+)\s*[><]=?\s*(\w+)',
            r'(\w+)\s+is\s+(greater|less|bigger|smaller|taller|shorter)',
            r'compare\s+(\w+)\s+and\s+(\w+)'
        ]
        return any(re.search(p, query) for p in patterns)
    
    def _analyze_comparison(self, query: str, steps: List) -> Tuple[str, List]:
        """Analyze transitive comparison reasoning"""
        steps.append({
            'description': 'Identifying comparison relationships',
            'operation': 'extract_relations',
            'result': 'Found comparison operators'
        })
        
        # Extract relationships like "A > B and B > C"
        pattern = r'(\w+)\s*([><]=?)\s*(\w+)'
        matches = re.findall(pattern, query)
        
        if len(matches) >= 2:
            relations = {}
            for left, op, right in matches:
                if left not in relations:
                    relations[left] = {}
                relations[left][right] = op
            
            steps.append({
                'description': 'Building relationship graph',
                'operation': 'build_graph',
                'result': f'Found {len(matches)} relationships'
            })
            
            # Look for transitive conclusions
            conclusions = self._find_transitive_conclusions(relations, matches)
            
            if conclusions:
                steps.append({
                    'description': 'Applying transitive property',
                    'operation': 'transitive_inference',
                    'result': conclusions[0]
                })
                return conclusions[0], steps
        
        return "Cannot determine a clear comparison conclusion.", steps
    
    def _find_transitive_conclusions(
        self,
        relations: Dict,
        matches: List
    ) -> List[str]:
        """Find transitive conclusions from comparison relations"""
        conclusions = []
        
        for left, op, mid in matches:
            if mid in relations:
                for right, op2 in relations[mid].items():
                    if op == op2 and right != left:
                        if '>' in op:
                            conclusions.append(
                                f"By transitivity: {left} > {right} "
                                f"(since {left} > {mid} and {mid} > {right})"
                            )
                        elif '<' in op:
                            conclusions.append(
                                f"By transitivity: {left} < {right} "
                                f"(since {left} < {mid} and {mid} < {right})"
                            )
        
        return conclusions
    
    def _is_syllogism(self, query: str) -> bool:
        """Check if query is a syllogism"""
        syllogism_patterns = [
            r'all\s+\w+\s+are\s+\w+',
            r'some\s+\w+\s+are\s+\w+',
            r'no\s+\w+\s+are\s+\w+',
            r'if\s+all\s+\w+'
        ]
        return any(re.search(p, query) for p in syllogism_patterns)
    
    def _analyze_syllogism(self, query: str, steps: List) -> Tuple[str, List]:
        """Analyze syllogistic reasoning"""
        steps.append({
            'description': 'Identifying premises',
            'operation': 'extract_premises',
            'result': 'Extracting logical premises'
        })
        
        # Extract "All X are Y" statements
        all_pattern = r'all\s+(\w+)\s+are\s+(\w+)'
        all_matches = re.findall(all_pattern, query)
        
        # Extract "X is a Y" statements
        is_pattern = r'(\w+)\s+is\s+a\s+(\w+)'
        is_matches = re.findall(is_pattern, query)
        
        if all_matches and is_matches:
            # Try to form a conclusion
            for subject, category in is_matches:
                for cat, property_name in all_matches:
                    if category.lower() == cat.lower():
                        conclusion = f"Therefore, {subject} is {property_name}."
                        steps.append({
                            'description': 'Applying syllogistic rule',
                            'operation': 'modus_ponens',
                            'result': f'{subject} belongs to {category}, which are all {property_name}'
                        })
                        return conclusion, steps
        
        return "Cannot form a syllogistic conclusion.", steps
    
    def _analyze_conditional(self, query: str, steps: List) -> Tuple[str, List]:
        """Analyze conditional (if-then) statements"""
        steps.append({
            'description': 'Parsing conditional statement',
            'operation': 'parse_conditional',
            'result': 'Identifying antecedent and consequent'
        })
        
        # Extract if-then structure
        pattern = r'if\s+(.+?)\s+then\s+(.+?)(?:\.|$)'
        match = re.search(pattern, query, re.IGNORECASE)
        
        if match:
            antecedent = match.group(1).strip()
            consequent = match.group(2).strip()
            
            steps.append({
                'description': 'Identified conditional structure',
                'operation': 'structure_analysis',
                'result': f'IF: {antecedent} → THEN: {consequent}'
            })
            
            return f"Given the condition '{antecedent}', we can conclude: {consequent}", steps
        
        return "Could not parse conditional statement.", steps
    
    def _analyze_boolean(self, query: str, steps: List) -> Tuple[str, List]:
        """Analyze Boolean logic expressions"""
        steps.append({
            'description': 'Analyzing Boolean expression',
            'operation': 'parse_boolean',
            'result': 'Extracting truth values and operators'
        })
        
        # Simple true/false evaluation
        query_clean = query.lower()
        
        # Handle basic Boolean operations
        if 'true and true' in query_clean:
            result = 'TRUE (both operands are true)'
        elif 'true and false' in query_clean or 'false and true' in query_clean:
            result = 'FALSE (AND requires both true)'
        elif 'false and false' in query_clean:
            result = 'FALSE (both operands are false)'
        elif 'true or' in query_clean or 'or true' in query_clean:
            result = 'TRUE (OR is true if any operand is true)'
        elif 'false or false' in query_clean:
            result = 'FALSE (both operands are false)'
        elif 'not true' in query_clean:
            result = 'FALSE (negation of true)'
        elif 'not false' in query_clean:
            result = 'TRUE (negation of false)'
        else:
            result = 'Unable to evaluate Boolean expression'
        
        steps.append({
            'description': 'Evaluated Boolean expression',
            'operation': 'evaluate',
            'result': result
        })
        
        return result, steps
