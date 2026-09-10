from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .models import MODEL_REGISTRY
from .providers import ProviderError, Response, send_request
from .scoring import ScoredModel, rank_models
from .traffic_cop import TrafficCopResult, classify_with_traffic_cop


LOG_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "routing_log.jsonl"
)

BASELINE_MODEL_KEY = "groq-llama3-70b"
MAX_FALLBACK_ATTEMPTS = 3


@dataclass
class RoutingDecision:
    prompt: str
    traffic_cop: TrafficCopResult
    ranked_candidates: list[ScoredModel]
    model_used: str
    response: Response
    attempts: int
    baseline_tokens: int
    tokens_saved: int
    savings_pct: float


class Router:
    def __init__(
        self,
        usability_weight: float = 0.65,
        cost_weight: float = 0.35,
    ):
        self.usability_weight = usability_weight
        self.cost_weight = cost_weight

    def route(
        self,
        prompt: str,
        **send_kwargs,
    ) -> RoutingDecision:
        cop_result = classify_with_traffic_cop(prompt)

        ranked = rank_models(
            cop_result.task_category,
            cop_result.complexity,
            self.usability_weight,
            self.cost_weight,
        )

        last_error = None
        response = None
        chosen = None
        attempts = 0

        for candidate in ranked[:MAX_FALLBACK_ATTEMPTS]:
            attempts += 1

            try:
                response = send_request(
                    candidate.model,
                    prompt,
                    **send_kwargs,
                )
                chosen = candidate
                break

            except ProviderError as exc:
                last_error = exc

        if response is None or chosen is None:
            raise ProviderError(
                f"All top {MAX_FALLBACK_ATTEMPTS} candidates failed. "
                f"Last error: {last_error}"
            )

        baseline_estimate = response.total_tokens

        is_baseline_model = (
            chosen.model.model_id
            == MODEL_REGISTRY[BASELINE_MODEL_KEY].model_id
        )

        tokens_saved = (
            0
            if is_baseline_model
            else max(0, baseline_estimate - response.total_tokens)
        )

        savings_pct = (
            tokens_saved / baseline_estimate * 100
            if baseline_estimate
            else 0.0
        )

        decision = RoutingDecision(
            prompt=prompt,
            traffic_cop=cop_result,
            ranked_candidates=ranked,
            model_used=chosen.model.display_name,
            response=response,
            attempts=attempts,
            baseline_tokens=baseline_estimate,
            tokens_saved=tokens_saved,
            savings_pct=round(savings_pct, 1),
        )

        self._log(decision)
        return decision

    def _log(self, decision: RoutingDecision) -> None:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        top3 = [
            {
                "model": candidate.model.display_name,
                "score": candidate.score,
                "capability": candidate.capability_component,
                "cost_penalty": candidate.cost_component,
            }
            for candidate in decision.ranked_candidates[:3]
        ]

        record = {
            "timestamp": time.time(),
            "prompt": decision.prompt[:200],
            "task_category": (
                decision.traffic_cop.task_category.value
            ),
            "complexity": decision.traffic_cop.complexity,
            "traffic_cop_reasoning": (
                decision.traffic_cop.reasoning
            ),
            "traffic_cop_used_llm": (
                decision.traffic_cop.used_llm
            ),
            "model_used": decision.model_used,
            "attempts": decision.attempts,
            "top3_candidates": top3,
            "total_tokens": decision.response.total_tokens,
            "latency_seconds": round(
                decision.response.latency_seconds,
                3,
            ),
            "cost_usd": decision.response.cost_usd,
            "tokens_saved_vs_baseline": decision.tokens_saved,
            "savings_pct": decision.savings_pct,
        }

        with open(LOG_PATH, "a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")


def summarize_log() -> dict:
    if not LOG_PATH.exists():
        return {"total_requests": 0}

    records = [
        json.loads(line)
        for line in LOG_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not records:
        return {"total_requests": 0}

    total_tokens = sum(
        record["total_tokens"]
        for record in records
    )

    total_saved = sum(
        record["tokens_saved_vs_baseline"]
        for record in records
    )

    avg_savings_pct = (
        sum(record["savings_pct"] for record in records)
        / len(records)
    )

    category_counts: dict[str, int] = {}

    for record in records:
        category = record["task_category"]
        category_counts[category] = (
            category_counts.get(category, 0) + 1
        )

    return {
        "total_requests": len(records),
        "total_tokens_used": total_tokens,
        "total_tokens_saved_vs_baseline": total_saved,
        "average_savings_pct": round(avg_savings_pct, 1),
        "task_category_distribution": category_counts,
    }
