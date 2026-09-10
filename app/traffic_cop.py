from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .classifier import ClassificationResult, HeuristicClassifier
from .models import MODEL_REGISTRY, QualityTier, TaskCategory
from .providers import ProviderError, send_request


TRAFFIC_COP_MODEL_KEY = "groq-llama3-8b-instant"

_SYSTEM_PROMPT = """
You are a request router for an LLM gateway.

Read the user's prompt and classify it. Respond with ONLY a JSON object,
with exactly this shape:

{
  "task_category": "<one of: coding, math_reasoning, creative_writing,
  classification_extraction, summarization, general_qa, analysis>",
  "complexity": <integer 1-10>,
  "reasoning": "<one short sentence>"
}

Examples:

"What is 2+2?"
-> {"task_category": "math_reasoning", "complexity": 1,
    "reasoning": "trivial arithmetic"}

"Write a haiku about rain"
-> {"task_category": "creative_writing", "complexity": 2,
    "reasoning": "short, low-stakes creative task"}

"Implement a thread-safe LRU cache in Python with O(1) operations"
-> {"task_category": "coding", "complexity": 8,
    "reasoning": "requires concurrency and algorithmic precision"}
""".strip()


@dataclass
class TrafficCopResult:
    task_category: TaskCategory
    complexity: int
    reasoning: str
    used_llm: bool


def _parse_llm_json(raw_text: str) -> dict:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)

    if not match:
        raise ValueError(
            f"No JSON object found in traffic cop output: {raw_text!r}"
        )

    return json.loads(match.group(0))


def _heuristic_fallback(prompt: str) -> TrafficCopResult:
    result: ClassificationResult = (
        HeuristicClassifier().predict(prompt)
    )

    complexity_map = {
        QualityTier.LOW: 2,
        QualityTier.MEDIUM: 5,
        QualityTier.HIGH: 8,
    }

    return TrafficCopResult(
        task_category=TaskCategory.GENERAL_QA,
        complexity=complexity_map[result.tier],
        reasoning=(
            "[fallback, LLM traffic cop unavailable] "
            f"{result.reason}"
        ),
        used_llm=False,
    )


def classify_with_traffic_cop(prompt: str) -> TrafficCopResult:
    cop_model = MODEL_REGISTRY[TRAFFIC_COP_MODEL_KEY]

    full_prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f'Classify this prompt:\n"{prompt}"'
    )

    try:
        response = send_request(
            cop_model,
            full_prompt,
            temperature=0.0,
            max_tokens=150,
        )

        parsed = _parse_llm_json(response.text)

        task_category = TaskCategory(parsed["task_category"])
        complexity = int(parsed["complexity"])
        complexity = max(1, min(10, complexity))

        return TrafficCopResult(
            task_category=task_category,
            complexity=complexity,
            reasoning=parsed.get("reasoning", ""),
            used_llm=True,
        )

    except (
        ProviderError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ):
        return _heuristic_fallback(prompt)
