from pathlib import Path
from unittest.mock import patch

import app.router as router_module
from app.models import MODEL_REGISTRY, TaskCategory
from app.providers import Response
from app.scoring import rank_models, score_model
from app.traffic_cop import TrafficCopResult


def test_qwen_beats_instant_model_on_hard_coding_task():
    qwen = MODEL_REGISTRY["groq-qwen2.5-coder-32b"]
    instant = MODEL_REGISTRY["groq-llama3-8b-instant"]

    qwen_score = score_model(
        qwen,
        TaskCategory.CODING,
        complexity=8,
    )

    instant_score = score_model(
        instant,
        TaskCategory.CODING,
        complexity=8,
    )

    assert qwen_score.score > instant_score.score


def test_instant_model_wins_on_trivial_classification_task():
    qwen = MODEL_REGISTRY["groq-qwen2.5-coder-32b"]
    instant = MODEL_REGISTRY["groq-llama3-8b-instant"]

    qwen_score = score_model(
        qwen,
        TaskCategory.CLASSIFICATION_EXTRACTION,
        complexity=1,
    )

    instant_score = score_model(
        instant,
        TaskCategory.CLASSIFICATION_EXTRACTION,
        complexity=1,
    )

    assert instant_score.score > qwen_score.score


def test_rank_models_returns_full_sorted_leaderboard():
    ranked = rank_models(
        TaskCategory.CODING,
        complexity=7,
    )

    assert len(ranked) == len(MODEL_REGISTRY)

    scores = [
        result.score
        for result in ranked
    ]

    assert scores == sorted(scores, reverse=True)


def _fake_response(model, prompt, *args, **kwargs):
    return Response(
        text=f"mock answer to: {prompt[:20]}",
        input_tokens=20,
        output_tokens=15,
        latency_seconds=0.05,
        model_key=model.model_id,
        provider=model.provider,
        cost_usd=0.0,
    )


def test_router_uses_traffic_cop_classification(tmp_path):
    router_module.LOG_PATH = tmp_path / "routing_log.jsonl"

    fake_cop_result = TrafficCopResult(
        task_category=TaskCategory.CODING,
        complexity=8,
        reasoning="mock: looks like a hard coding task",
        used_llm=True,
    )

    with (
        patch(
            "app.router.classify_with_traffic_cop",
            return_value=fake_cop_result,
        ),
        patch(
            "app.router.send_request",
            side_effect=_fake_response,
        ),
    ):
        router = router_module.Router()
        decision = router.route(
            "Implement a lock-free concurrent queue"
        )

    assert decision.traffic_cop.task_category == TaskCategory.CODING
    assert "Qwen" in decision.model_used
    assert router_module.LOG_PATH.exists()


def test_router_falls_back_to_next_best_scored_model_on_failure(
    tmp_path,
):
    router_module.LOG_PATH = tmp_path / "routing_log.jsonl"

    fake_cop_result = TrafficCopResult(
        task_category=TaskCategory.CODING,
        complexity=8,
        reasoning="mock",
        used_llm=True,
    )

    call_count = {"n": 0}

    def flaky(model, prompt, *args, **kwargs):
        call_count["n"] += 1

        if call_count["n"] == 1:
            from app.providers import ProviderError
            raise ProviderError(
                "simulated outage on top choice"
            )

        return _fake_response(model, prompt)

    with (
        patch(
            "app.router.classify_with_traffic_cop",
            return_value=fake_cop_result,
        ),
        patch(
            "app.router.send_request",
            side_effect=flaky,
        ),
    ):
        router = router_module.Router()
        decision = router.route(
            "Implement a lock-free concurrent queue"
        )

    assert decision.attempts == 2
