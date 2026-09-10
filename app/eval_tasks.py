from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .models import TaskCategory

@dataclass
class EvalTask:
    """Represents a single objective evaluation task."""
    task_id: str
    category: TaskCategory
    prompt: str
    checker: Callable[[str], bool]

def _check_code_output(model_output: str, test_code: str) -> bool:
    """
    Extracts Python code from the model output and runs it alongside the 
    provided test code to verify correctness.
    """
    match = re.search(r"```(?:python)?\s*\n(.*?)```", model_output, re.DOTALL)
    code = match.group(1) if match else model_output
    
    script_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(f"{code}\n\n{test_code}\n")
            script_path = Path(f.name)
            
        result = subprocess.run(
            [sys.executable, str(script_path)], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False
    finally:
        if script_path is not None:
            script_path.unlink(missing_ok=True)

def _check_numeric_answer(model_output: str, expected: float, tolerance: float = 0.01) -> bool:
    """Verifies if the expected numeric answer appears in the model's output."""
    numbers = re.findall(r"-?\d+\.?\d*", model_output.replace(",", ""))
    return any(abs(float(n) - expected) < tolerance for n in numbers)

def _check_label_match(model_output: str, expected_label: str) -> bool:
    """Verifies if a specific label exists within the output."""
    return expected_label.lower() in model_output.lower()

OBJECTIVE_TASKS = [
    EvalTask(
        task_id="code_reverse_string",
        category=TaskCategory.CODING,
        prompt="Write a Python function `reverse_string(s)` that returns the reverse of a string.",
        checker=lambda out: _check_code_output(
            out, 
            "assert reverse_string('hello') == 'olleh'\nassert reverse_string('') == ''"
        )
    ),
    EvalTask(
        task_id="code_fibonacci",
        category=TaskCategory.CODING,
        prompt="Write a Python function `fib(n)` that returns the nth Fibonacci number (fib(0)=0, fib(1)=1).",
        checker=lambda out: _check_code_output(
            out, 
            "assert fib(0) == 0\nassert fib(1) == 1\nassert fib(10) == 55"
        )
    ),
    EvalTask(
        task_id="code_is_palindrome",
        category=TaskCategory.CODING,
        prompt="Write a Python function `is_palindrome(s)` that returns True if s reads the same forwards and backwards.",
        checker=lambda out: _check_code_output(
            out, 
            "assert is_palindrome('racecar') is True\nassert is_palindrome('hello') is False"
        )
    ),
    EvalTask(
        task_id="math_percentage",
        category=TaskCategory.MATH_REASONING,
        prompt="A shirt costs $80. It's on sale for 25% off. What is the final price in dollars?",
        checker=lambda out: _check_numeric_answer(out, 60.0)
    ),
    EvalTask(
        task_id="math_word_problem",
        category=TaskCategory.MATH_REASONING,
        prompt="A train travels 240 miles in 4 hours. At the same speed, how many miles does it travel in 7 hours?",
        checker=lambda out: _check_numeric_answer(out, 420.0)
    ),
    EvalTask(
        task_id="extract_invoice_amount",
        category=TaskCategory.CLASSIFICATION_EXTRACTION,
        prompt="Extract just the dollar amount from: 'Invoice #4521 issued to John Carter for $1,250.00.' Reply with only the number.",
        checker=lambda out: _check_numeric_answer(out, 1250.0)
    ),
    EvalTask(
        task_id="classify_support_ticket",
        category=TaskCategory.CLASSIFICATION_EXTRACTION,
        prompt="Classify as billing, technical, account, or general: 'I was charged twice for my subscription this month.' Reply with only the category word.",
        checker=lambda out: _check_label_match(out, "billing")
    ),
]

def run_objective_eval(model, send_request_fn) -> dict[TaskCategory, float]:
    """
    Runs a suite of objective tasks against a model and calculates its score
    out of 10.0 for each evaluated category.
    """
    results = defaultdict(list)
    
    for task in OBJECTIVE_TASKS:
        try:
            response = send_request_fn(
                model, 
                task.prompt, 
                temperature=0.0, 
                max_tokens=300
            )
            passed = task.checker(response.text)
        except Exception:
            passed = False
            
        results[task.category].append(passed)
        
    scores = {}
    for category, passes in results.items():
        if passes:
            # Scale win rate (0.0 to 1.0) to a score out of 10.0
            win_rate = sum(passes) / len(passes)
            scores[category] = round(10.0 * win_rate, 2)
            
    return scores
