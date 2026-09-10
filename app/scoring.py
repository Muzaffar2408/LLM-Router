from __future__ import annotations

from dataclasses import dataclass

from .models import ModelConfig, TaskCategory, all_models


DEFAULT_USABILITY_WEIGHT = 0.50
DEFAULT_COST_WEIGHT = 0.30
DEFAULT_LATENCY_WEIGHT = 0.20


@dataclass
class ScoredModel:
    model_key: str
    model: ModelConfig
    capability_component: float
    cost_component: float
    latency_component: float
    score: float
    explanation: str


def _cost_component(model: ModelConfig, complexity: int) -> float:
    # Estimate tokens: assume a base of 500 input tokens.
    # Output tokens scale with complexity and the model's verbosity factor.
    estimated_input_tokens = 500
    estimated_output_tokens = complexity * 100 * model.verbosity_factor
    
    estimated_cost_usd = model.estimated_cost(estimated_input_tokens, estimated_output_tokens)
    
    # Normalize to a 0-10 penalty scale.
    # We use a reference max cost (e.g., $0.002) for the 10.0 penalty.
    MAX_COST_REFERENCE = 0.002
    
    if MAX_COST_REFERENCE > 0:
        cost_score = (estimated_cost_usd / MAX_COST_REFERENCE) * 10.0
    else:
        cost_score = 0.0
        
    return min(10.0, cost_score)

def _latency_component(model: ModelConfig) -> float:
    hint = model.typical_latency_hint.lower()
    if hint == "fast":
        return 0.0
    elif hint == "medium":
        return 5.0
    elif hint == "slow":
        return 10.0
    return 5.0

def score_model(
    model: ModelConfig,
    task_category: TaskCategory,
    complexity: int,
    usability_weight: float = DEFAULT_USABILITY_WEIGHT,
    cost_weight: float = DEFAULT_COST_WEIGHT,
    latency_weight: float = DEFAULT_LATENCY_WEIGHT,
) -> ScoredModel:
    raw_capability = model.capability_for(task_category)

    complexity_multiplier = 0.5 + (complexity / 10)
    
    # Dynamic Capability Scaling:
    # Penalize low-capability models more heavily on highly complex tasks.
    if raw_capability < 6.0 and complexity > 7:
        capability_component = min(10.0, raw_capability * complexity_multiplier * 0.5)
    else:
        capability_component = min(10.0, raw_capability * complexity_multiplier)

    cost_component = _cost_component(model, complexity)
    latency_component = _latency_component(model)

    score = (
        usability_weight * capability_component
        - cost_weight * cost_component
        - latency_weight * latency_component
    )

    explanation = (
        f"capability={raw_capability:.1f}/10 "
        f"for {task_category.value} "
        f"(x{complexity_multiplier:.2f} "
        f"for complexity={complexity}) "
        f"-> {capability_component:.1f}; "
        f"cost_penalty={cost_component:.1f}/10; "
        f"latency_penalty={latency_component:.1f}/10"
    )

    return ScoredModel(
        model_key=model.model_id,
        model=model,
        capability_component=round(capability_component, 2),
        cost_component=round(cost_component, 2),
        latency_component=round(latency_component, 2),
        score=round(score, 3),
        explanation=explanation,
    )


def rank_models(
    task_category: TaskCategory,
    complexity: int,
    usability_weight: float = DEFAULT_USABILITY_WEIGHT,
    cost_weight: float = DEFAULT_COST_WEIGHT,
    latency_weight: float = DEFAULT_LATENCY_WEIGHT,
) -> list[ScoredModel]:
    scored = [
        score_model(
            model,
            task_category,
            complexity,
            usability_weight,
            cost_weight,
            latency_weight,
        )
        for model in all_models()
    ]

    return sorted(
        scored,
        key=lambda scored_model: scored_model.score,
        reverse=True,
    )
