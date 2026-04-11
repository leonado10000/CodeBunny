import ast
import re
import io
import os
import tempfile
from typing import List, Dict
from pylint.lint import Run
from pylint.reporters.text import TextReporter

class PythonContextMapper(ast.NodeVisitor):
    """
    Combines AST structural mapping with deterministic Pylint checks.
    Maps diff changes directly into function context blocks for token efficiency.
    """
    def __init__(self, diff_mapping: Dict[int, List[str]]):
        self.diff_mapping = diff_mapping
        self.extracted_context = []
        self.functions_found = set()
        self.used_lines = set()

    def visit_FunctionDef(self, node):
        self._analyze_function(node, is_async=False)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self._analyze_function(node, is_async=True)
        self.generic_visit(node)

    def _analyze_function(self, node, is_async: bool):
        func_start = node.lineno
        func_end = getattr(node, 'end_lineno', func_start + 100)
        
        # 1. Map relevant diff lines (+/-) to this function's scope
        func_diff_lines = []
        for line_num, text_lines in self.diff_mapping.items():
            if func_start <= line_num <= func_end:
                func_diff_lines.extend(text_lines)
                self.used_lines.add(line_num)
        
        # Skip if no changes detected in this specific function
        if not func_diff_lines:
            return

        self.functions_found.add(node.name)
        
        # 2. Deterministic Pylint-style Checks (Hard Rules)
        linter_flags = []
        
        # Naming Convention (PEP8 snake_case)
        if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
            linter_flags.append(f"CRITICAL: Invalid name '{node.name}' (Use snake_case)")
        
        # Docstring Presence
        docstring = ast.get_docstring(node)
        if not docstring:
            linter_flags.append("WARNING: Missing docstring (C0116)")

        # Complexity/Size Check
        line_count = func_end - func_start
        size_flag = ""
        if line_count > 1000:
            size_flag = "⚠️ FLAG: LARGE_FUNCTION (>1000 lines)"
        elif line_count >= 100:
            size_flag = "⚠️ FLAG: LARGE_FUNCTION (>100 lines)"

        # 3. Metadata Extraction
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(f"@{dec.id}")
            elif isinstance(dec, ast.Call):
                func = dec.func
                name = getattr(func, 'id', getattr(getattr(func, 'value', None), 'id', 'decorator'))
                decorators.append(f"@{name}(...)")
        
        # Calculate if Minor or Major Fix based on modified lines count
        modified_lines_count = sum(1 for line in func_diff_lines if line.startswith(('+', '-')))
        change_label = "Minor Fix" if modified_lines_count <= 5 else "Major Refactor/Addition"

        # 4. Build Optimized Block for AI (Formatting matched to Grader)
        func_prefix = "async def" if is_async else "def"
        block = f"Function: `{func_prefix} {node.name}`\n"
        
        if decorators:
            block += f"Decorators: {', '.join(decorators)}\n"
        
        if docstring:
            # Clean docstring for token efficiency
            clean_doc = docstring[:80].replace('\n', ' ').strip()
            block += f"Docstring: \"\"\"{clean_doc}...\"\"\"\n"
        else:
            block += "Docstring: ⚠️ FLAG: MISSING_DOCSTRING\n"
            
        if size_flag:
            block += f"Size: {line_count} lines {size_flag}\n"
            
        if linter_flags:
            block += f"Pylint Flags: {'; '.join(linter_flags)}\n"
            
        block += f"Change Type: {change_label}\n"
        block += "--- CODE CHANGES (Small Fix Context) ---\n"
        for dl in func_diff_lines:
            block += f"  {dl}\n"

        self.extracted_context.append(block.strip())

def run_library_pylint(content: str) -> str:
    """
    Runs the official Pylint library on the provided content.
    Returns a concise summary including rating and top 5 issues.
    """
    pylint_output = io.StringIO()
    reporter = TextReporter(pylint_output)
    
    # Create a temporary file for Pylint to analyze
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Run Pylint
        # Fix: 'do_exit' is often replaced by 'exit' in newer Pylint versions
        results = Run([tmp_path, '--reports=n', '--score=y', '--disable=C0301,C0303'], reporter=reporter, exit=False)
        raw_output = pylint_output.getvalue()
        
        # Extract the score
        score = 0.0
        if hasattr(results.linter, 'stats'):
            stats = results.linter.stats
            # Pylint 3.x uses dictionary stats; older uses object attributes
            if isinstance(stats, dict):
                score = stats.get('global_note', 0)
            else:
                score = getattr(stats, 'global_note', 0)
        
        lines = raw_output.split('\n')
        important_issues = []
        needs_spacing_fix = False
        
        issue_pattern = re.compile(r':\d+:\s\[([A-Z]\d+)\]\s(.*)')
        
        for line in lines:
            match = issue_pattern.search(line)
            if match:
                code, msg = match.groups()
                if any(k in msg.lower() for k in ['indentation', 'whitespace', 'trailing-whitespace']):
                    needs_spacing_fix = True
                elif code.startswith(('E', 'F', 'W')):
                    important_issues.append(msg)
            
        report = f"OVERALL CODE RATING: {score:.2f}/10\n"
        report += "MAJOR ISSUES DETECTED (Max 5):\n"
        
        for issue in important_issues[:5]:
            report += f"- {issue}\n"
            
        if needs_spacing_fix:
            report += "- STYLE: General spacing or indentation issues detected (Fix suggested).\n"
            
        if not important_issues and not needs_spacing_fix:
            report += "- None (Great quality)\n"
            
        return report
    except Exception as e:
        return f"Pylint Analysis failed: {str(e)}"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def parse_diff_to_mapping(diff_text: str) -> Dict[int, List[str]]:
    """Strips git metadata and maps diff lines to line numbers."""
    diff_text = diff_text.replace('\r\n', '\n')
    mapping = {}
    current_line = 0
    for line in diff_text.split('\n'):
        if line.startswith('@@'):
            match = re.search(r'\+(\d+)', line)
            if match: current_line = int(match.group(1))
        elif line.startswith(('+++', '---')): continue
        elif line.startswith('+'):
            mapping.setdefault(current_line, []).append(line)
            current_line += 1
        elif line.startswith('-'): mapping.setdefault(current_line, []).append(line)
        elif not line.startswith('\\'): current_line += 1
    return mapping

def search_codebase_usages(func_name: str) -> str:
    """Mock cross-codebase usage scan."""
    return f"Usage Scan: `{func_name}` is likely referenced by downstream callers."

def optimize_payload_for_ai(filepath: str, full_file_content: str, raw_diff: str) -> str:
    """Main Orchestrator for token-efficient payloads."""
    diff_mapping = parse_diff_to_mapping(raw_diff)
    
    # Fallback for non-Python files
    if not full_file_content or not filepath.endswith('.py'):
        clean_diff = "\n".join([l for l in raw_diff.split('\n') if l.startswith(('+', '-'))])
        return f"File: {filepath}\n\n{clean_diff[:1500]}"

    # Run library-based Pylint
    pylint_report = run_library_pylint(full_file_content)

    try:
        tree = ast.parse(full_file_content)
        visitor = PythonContextMapper(diff_mapping)
        visitor.visit(tree)
    except SyntaxError:
        return f"File: {filepath}\n[SYNTAX ERROR]\n{raw_diff[:1000]}"

    prompt = f"--- ENGINEERING AUDIT: {filepath} ---\n\n"
    prompt += f"### LIBRARY QUALITY REPORT\n{pylint_report}\n\n"
    
    if visitor.functions_found:
        first_func = list(visitor.functions_found)[0]
        prompt += f"{search_codebase_usages(first_func)}\n\n"
    
    if visitor.extracted_context:
        prompt += "\n\n".join(visitor.extracted_context)
    else:
        prompt += "Global/Module changes (Imports or Variables) detected outside functions."
    
    # Add any diff lines that didn't fall into a function (e.g. imports)
    globals_diff = [line for ln, lines in diff_mapping.items() if ln not in visitor.used_lines for line in lines]
    if globals_diff:
        prompt += "\n\n[Global Scope Changes]\n--- CODE CHANGES (Small Fix Context) ---\n" + "\n".join([f"  {g}" for g in globals_diff])
        
    return prompt.strip()