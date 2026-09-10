from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .providers import ProviderError
from .router import Router, summarize_log


app = FastAPI(title="LLM Cost Autopilot")
router = Router()


class RouteRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 512


class RouteResponse(BaseModel):
    output: str
    model_used: str
    task_category: str
    complexity: int
    traffic_cop_reasoning: str
    total_tokens: int
    latency_seconds: float
    cost_usd: float
    tokens_saved_vs_baseline: int
    savings_pct: float
    attempts: int
    top3_candidates: list[dict]


@app.post(
    "/v1/route",
    response_model=RouteResponse,
)
def route_prompt(req: RouteRequest):
    try:
        decision = router.route(
            req.prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )

    except ProviderError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    top3 = [
        {
            "model": candidate.model.display_name,
            "score": candidate.score,
            "capability": candidate.capability_component,
            "cost_penalty": candidate.cost_component,
        }
        for candidate in decision.ranked_candidates[:3]
    ]

    return RouteResponse(
        output=decision.response.text,
        model_used=decision.model_used,
        task_category=decision.traffic_cop.task_category.value,
        complexity=decision.traffic_cop.complexity,
        traffic_cop_reasoning=decision.traffic_cop.reasoning,
        total_tokens=decision.response.total_tokens,
        latency_seconds=round(
            decision.response.latency_seconds,
            3,
        ),
        cost_usd=decision.response.cost_usd,
        tokens_saved_vs_baseline=decision.tokens_saved,
        savings_pct=decision.savings_pct,
        attempts=decision.attempts,
        top3_candidates=top3,
    )


@app.get("/v1/stats")
def stats():
    return summarize_log()


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "LLM Cost Autopilot",
    }
