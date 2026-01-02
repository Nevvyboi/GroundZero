"""
Code Analyzer
=============
Code understanding and debugging capabilities.
"""

import re
from typing import Tuple, List, Dict, Any


class CodeAnalyzer:
    """Analyzes code for bugs, syntax issues, and improvements"""
    
    def analyze(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Analyze code in query. Returns (analysis, steps)."""
        steps = []
        
        # Extract code from query
        code = self._extract_code(query)
        steps.append({'description': 'Extracted code', 'operation': 'parse', 'result': code[:100] + '...' if len(code) > 100 else code})
        
        # Detect language
        language = self._detect_language(code)
        steps.append({'description': 'Detected language', 'operation': 'detect', 'result': language})
        
        # Find issues
        issues = self._find_issues(code, language)
        
        if issues:
            steps.append({'description': 'Found issues', 'operation': 'analyze', 'result': issues})
            return self._format_issues(issues, code), steps
        
        steps.append({'description': 'No issues found', 'operation': 'analyze', 'result': 'Code appears correct'})
        return "The code looks syntactically correct. No obvious issues found.", steps
    
    def _extract_code(self, query: str) -> str:
        """Extract code from query"""
        # Try to find code blocks
        block_match = re.search(r'```(?:\w+)?\s*(.*?)```', query, re.DOTALL)
        if block_match:
            return block_match.group(1).strip()
        
        # Look for code-like patterns
        code_patterns = [
            r'def\s+\w+\s*\(.*?\)',
            r'function\s+\w+\s*\(.*?\)',
            r'class\s+\w+',
            r'for\s+\w+\s+in',
            r'while\s+.*:',
            r'if\s+.*:'
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, query):
                # Extract from pattern onwards
                match = re.search(f'({pattern}.*)', query, re.DOTALL)
                if match:
                    return match.group(1).strip()
        
        # Return everything after common prefixes
        prefixes = ['debug:', 'fix:', 'analyze:', 'code:']
        for prefix in prefixes:
            if prefix in query.lower():
                idx = query.lower().find(prefix)
                return query[idx + len(prefix):].strip()
        
        return query
    
    def _detect_language(self, code: str) -> str:
        """Detect programming language"""
        patterns = {
            'python': [r'def\s+\w+\s*\(', r'import\s+\w+', r'from\s+\w+\s+import', r':\s*$', r'print\s*\('],
            'javascript': [r'function\s+\w+\s*\(', r'const\s+\w+', r'let\s+\w+', r'var\s+\w+', r'=>'],
            'java': [r'public\s+class', r'public\s+static\s+void', r'System\.out'],
            'sql': [r'SELECT\s+', r'FROM\s+', r'WHERE\s+', r'INSERT\s+INTO'],
            'html': [r'<html', r'<div', r'<p>', r'</\w+>']
        }
        
        code_lower = code.lower()
        scores = {}
        
        for lang, lang_patterns in patterns.items():
            scores[lang] = sum(1 for p in lang_patterns if re.search(p, code, re.IGNORECASE))
        
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return 'unknown'
    
    def _find_issues(self, code: str, language: str) -> List[Dict[str, str]]:
        """Find issues in code"""
        issues = []
        
        if language == 'python':
            issues.extend(self._check_python(code))
        elif language == 'javascript':
            issues.extend(self._check_javascript(code))
        
        # Generic checks
        issues.extend(self._check_generic(code))
        
        return issues
    
    def _check_python(self, code: str) -> List[Dict[str, str]]:
        """Check Python-specific issues"""
        issues = []
        
        # Missing colons
        patterns = [
            (r'def\s+\w+\s*\([^)]*\)\s*[^:]', 'Missing colon after function definition'),
            (r'if\s+.+[^:]\s*$', 'Missing colon after if statement'),
            (r'for\s+.+[^:]\s*$', 'Missing colon after for loop'),
            (r'while\s+.+[^:]\s*$', 'Missing colon after while loop'),
            (r'class\s+\w+[^:]*$', 'Missing colon after class definition'),
        ]
        
        for pattern, message in patterns:
            if re.search(pattern, code, re.MULTILINE):
                issues.append({'type': 'syntax', 'message': message, 'severity': 'error'})
        
        # Common Python mistakes
        if 'print ' in code and 'print(' not in code:
            issues.append({'type': 'syntax', 'message': 'Python 3 requires print()', 'severity': 'error'})
        
        # Undefined variables (simple check)
        if re.search(r'=\s*\w+\s*\+', code):
            var_match = re.search(r'(\w+)\s*=\s*(\w+)\s*\+', code)
            if var_match and var_match.group(1) == var_match.group(2):
                pass  # x = x + 1 is fine
        
        return issues
    
    def _check_javascript(self, code: str) -> List[Dict[str, str]]:
        """Check JavaScript-specific issues"""
        issues = []
        
        # Missing semicolons (basic check)
        lines = code.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if line and not line.endswith((';', '{', '}', ',', '(', ')')) and not line.startswith('//'):
                if any(kw in line for kw in ['var ', 'let ', 'const ', 'return ']):
                    issues.append({'type': 'style', 'message': f'Line {i+1}: Consider adding semicolon', 'severity': 'warning'})
        
        # Using var instead of let/const
        if 'var ' in code:
            issues.append({'type': 'style', 'message': 'Consider using let/const instead of var', 'severity': 'suggestion'})
        
        return issues
    
    def _check_generic(self, code: str) -> List[Dict[str, str]]:
        """Generic code checks"""
        issues = []
        
        # Unmatched brackets
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        for char in code:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack:
                    issues.append({'type': 'syntax', 'message': f'Unmatched closing bracket: {char}', 'severity': 'error'})
                else:
                    expected = brackets[stack.pop()]
                    if char != expected:
                        issues.append({'type': 'syntax', 'message': f'Mismatched brackets: expected {expected}, got {char}', 'severity': 'error'})
        
        if stack:
            issues.append({'type': 'syntax', 'message': f'Unclosed brackets: {stack}', 'severity': 'error'})
        
        return issues
    
    def _format_issues(self, issues: List[Dict], code: str) -> str:
        """Format issues into readable output"""
        if not issues:
            return "No issues found."
        
        output = ["**Code Analysis Results:**\n"]
        
        errors = [i for i in issues if i['severity'] == 'error']
        warnings = [i for i in issues if i['severity'] == 'warning']
        suggestions = [i for i in issues if i['severity'] == 'suggestion']
        
        if errors:
            output.append("❌ **Errors:**")
            for e in errors:
                output.append(f"  • {e['message']}")
        
        if warnings:
            output.append("\n⚠️ **Warnings:**")
            for w in warnings:
                output.append(f"  • {w['message']}")
        
        if suggestions:
            output.append("\n💡 **Suggestions:**")
            for s in suggestions:
                output.append(f"  • {s['message']}")
        
        return "\n".join(output)
