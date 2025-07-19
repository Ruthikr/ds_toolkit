import ast
import sys
import io
import subprocess
import traceback
import contextlib
import uuid
import re
import os
from typing import Dict, Any, List, Tuple
from langchain.tools import BaseTool
from pydantic import Field, PrivateAttr

class JupyterCodeExecutor(BaseTool):
    name: str = Field("jupyter_code_executor", description="Tool name")

    description: str = Field(
        (
            "⚙️ Executes Python code in a persistent Jupyter-like REPL environment.\n\n"
            "📌 This tool maintains session context between runs (like variables, imports, models, etc).\n"
            "📦 Supports:\n"
            "  - Python code execution (multi-line, expressions, function/class definitions)\n"
            "  - Shell commands prefixed with `!` (e.g., `!ls`, `!pip install pandas`)\n"
            "  - Inline package installation via pip or conda\n"
            "  - Auto-imports common libraries like pandas (pd), numpy (np), matplotlib (plt), seaborn (sns)\n"
            "  - Reading/writing local files using standard Python (`open()`, `os.listdir()`, etc)\n\n"
            "✅ Input Format:\n"
            "  - Accepts raw string input of Python code (multi-line allowed)\n"
            "  - Markdown-style code blocks are cleaned automatically (```python ... ```)\n\n"
            "🧪 Examples:\n"
            "  - Python: `df = pd.read_csv('data.csv'); df.head()`\n"
            "  - Shell: `!pip install seaborn`\n"
            "  - Mixed:\n"
            "    ```python\n"
            "    !pip install pandas\n"
            "    import pandas as pd\n"
            "    df = pd.read_csv('file.csv')\n"
            "    df.describe()\n"
            "    ```\n\n"
            "🚫 Avoid:\n"
            "  - Dangerous shell commands (`rm`, `shutdown`, `kill`, `format`, etc are blocked)\n"
            "  - Unsupported languages (only Python is allowed)\n"
            "  - Improper indentation or syntax in code blocks\n\n"
            
            "🧠 Use for step-by-step execution, debugging, and interactive workflows like EDA, feature engineering, model training, plotting, and saving files. The tool behaves like an advanced code cell in Jupyter Notebook."
        ),
        description="Detailed tool description with usage guide"
    )
    
    # Private attributes for internal state
    _context: Dict[str, Any] = PrivateAttr()
    _history: List[str] = PrivateAttr()
    _auto_imports: List[Tuple[str, str]] = PrivateAttr()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._context = self._initialize_context()
        self._history = []
        self._auto_imports = [
            ('pd', 'pandas'),
            ('np', 'numpy'),
            ('plt', 'matplotlib.pyplot'),
            ('sns', 'seaborn'),
            ('sklearn', 'sklearn'),
            ('torch', 'torch'),
            ('tf', 'tensorflow')
        ]

    def _initialize_context(self) -> Dict[str, Any]:
        """Create execution context with essential imports"""
        context = {
            '__builtins__': __builtins__,
            'print': self._custom_print,
            'help': help
        }
        # Add common modules
        for name in ['os', 'sys', 'math', 'json', 'io', 'pathlib', 'subprocess']:
            context[name] = __import__(name)
        return context

    def _custom_print(self, *args, **kwargs):
        """Custom print function that captures output"""
        file = kwargs.get('file', sys.stdout)
        print(*args, **kwargs)
        if file == sys.stdout:
            sys.stdout.flush()
            
    def _clean_code(self, code: str) -> str:
        """Remove markdown code block indicators if present"""
        # Pattern to match code blocks with optional language specifier
        pattern = r'^\s*```(?:\w*)\s*\n(.*?)\n\s*```\s*$'
        match = re.search(pattern, code, re.DOTALL)
        if match:
            return match.group(1)
        return code

    def _run_shell(self, command: str) -> str:
        """Execute a single shell command with proper package handling"""
        # Normalize and tokenize command
        tokens = command.split()
        if not tokens:
            return ""

        # Handle package installation with current Python's pip
        if tokens[0] in ['pip', 'pip3'] and len(tokens) >= 2 and tokens[1] == 'install':
            # Construct safe installation command
            new_command = [sys.executable, "-m", "pip", "install"] + tokens[2:]
            try:
                result = subprocess.run(
                    new_command,
                    capture_output=True,
                    text=True,
                    check=True
                )
                return result.stdout + result.stderr
            except subprocess.CalledProcessError as e:
                return f"PIP INSTALL ERROR:\n{e.stdout}\n{e.stderr}\nReturn code: {e.returncode}"
        
        # Handle conda installation
        elif tokens[0] == 'conda' and len(tokens) >= 2 and tokens[1] == 'install':
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    check=True
                )
                return result.stdout + result.stderr
            except subprocess.CalledProcessError as e:
                return f"CONDA INSTALL ERROR:\n{e.stdout}\n{e.stderr}\nReturn code: {e.returncode}"
        
        # Block dangerous commands
        dangerous_commands = ['rm', 'shutdown', 'reboot', 'del', 'format', 'mv', 'dd', 'kill']
        if any(token in dangerous_commands for token in tokens):
            return "ERROR: Potentially dangerous command blocked for security reasons"
        
        # Execute regular shell commands
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout + result.stderr
        except subprocess.CalledProcessError as e:
            return f"SHELL COMMAND ERROR:\n{e.stdout}\n{e.stderr}\nReturn code: {e.returncode}"

    def _run_python(self, code: str) -> Tuple[str, str]:
        """Execute Python code with persistent context"""
        # Handle auto-imports for common modules
        for alias, module in self._auto_imports:
            if alias not in self._context:
                try:
                    self._context[alias] = __import__(module)
                except ImportError:
                    pass

        # Setup output capture
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        last_expr_value = None

        with contextlib.redirect_stdout(stdout_capture), \
             contextlib.redirect_stderr(stderr_capture):

            # Handle multi-line expressions for last expression capture
            try:
                parsed = ast.parse(code)
                
                # Only attempt to capture last expression if there's at least one statement
                if parsed.body and isinstance(parsed.body[-1], ast.Expr):
                    last_expr = parsed.body[-1]
                    temp_var = f"__temp_{uuid.uuid4().hex}"
                    
                    # Create assignment node
                    new_assign = ast.Assign(
                        targets=[ast.Name(id=temp_var, ctx=ast.Store())],
                        value=last_expr.value
                    )
                    
                    # Replace expression with assignment
                    parsed.body[-1] = new_assign
                    
                    # Fix missing locations in AST
                    ast.fix_missing_locations(parsed)
                    
                    # Compile and execute modified AST
                    code_obj = compile(parsed, filename="<ast>", mode="exec")
                    exec(code_obj, self._context)
                    
                    # Get result of last expression
                    last_expr_value = eval(temp_var, self._context)
                else:
                    # Execute code normally without modification
                    exec(code, self._context)
            except Exception:
                traceback.print_exc(file=sys.stderr)

        # Capture output
        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()

        # Add last expression to output if it exists
        if last_expr_value is not None:
            try:
                stdout += f"\n[Last expression result]: {repr(last_expr_value)}"
            except Exception:
                stdout += "\n[Last expression result] (unable to represent)"

        return stdout, stderr

    def _run(self, code: str) -> str:
        """Main execution method with robust shell handling"""
        try:
            # Clean code by removing markdown code blocks
            cleaned_code = self._clean_code(code)
            self._history.append(cleaned_code)
            
            # Separate shell commands and Python code
            shell_commands = []
            python_lines = []
            
            for line in cleaned_code.split('\n'):
                stripped_line = line.strip()
                if stripped_line.startswith('!'):
                    # Extract shell command (remove ! and leading whitespace)
                    cmd = line.replace('!', '', 1).strip()
                    if cmd:  # Only add non-empty commands
                        shell_commands.append(cmd)
                else:
                    python_lines.append(line)
            
            # Process shell commands first
            shell_outputs = []
            for command in shell_commands:
                shell_outputs.append(f"Executing: {command}")
                result = self._run_shell(command)
                shell_outputs.append(result)
            
            # Process Python code if any exists
            python_output = []
            if python_lines:
                python_code = '\n'.join(python_lines)
                stdout, stderr = self._run_python(python_code)
                if stdout:
                    python_output.append(f"PYTHON STDOUT:\n{stdout}")
                if stderr:
                    python_output.append(f"PYTHON STDERR:\n{stderr}")
            
            # Combine all outputs
            all_outputs = []
            if shell_outputs:
                all_outputs.append("SHELL OUTPUT:\n" + "\n".join(shell_outputs))
            if python_output:
                all_outputs.append("\n".join(python_output))
            
            return "\n\n".join(all_outputs) if all_outputs else "Code executed successfully with no output"
        
        except Exception as e:
            return f"EXECUTION ERROR: {str(e)}\n{traceback.format_exc()}"
    
    def reset_context(self) -> str:
        """Reset execution context while preserving core functionality"""
        self._context = self._initialize_context()
        return "Context reset successfully"