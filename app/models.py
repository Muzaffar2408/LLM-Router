from dataclasses import dataclass, field, replace, replace
from enum import Enum
import json
from pathlib import Path


class Provider(str, Enum):
    GROQ = "groq"
    HUGGINGFACE = "huggingface"


class QualityTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskCategory(str, Enum):
    CODING = "coding"
    MATH_REASONING = "math_reasoning"
    CREATIVE_WRITING = "creative_writing"
    CLASSIFICATION_EXTRACTION = "classification_extraction"
    SUMMARIZATION = "summarization"
    GENERAL_QA = "general_qa"
    ANALYSIS = "analysis"


@dataclass(frozen=True)
class ModelConfig:
    provider: Provider
    model_id: str
    display_name: str
    quality_tier: QualityTier
    context_window: int
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    typical_latency_hint: str = "unknown"
    capability_scores: dict[TaskCategory, float] = field(default_factory=dict)
    verbosity_factor: float = 1.0

    def estimated_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            (input_tokens / 1000) * self.cost_per_1k_input_tokens
            + (output_tokens / 1000) * self.cost_per_1k_output_tokens
        )

    def capability_for(self, task: TaskCategory) -> float:
        return self.capability_scores.get(task, 5.0)


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "groq-llama3-8b-instant": ModelConfig(
        provider=Provider.GROQ,
        model_id="openai/gpt-oss-20b",
        display_name="GPT OSS 20B (Groq)",
        quality_tier=QualityTier.LOW,
        context_window=131072,
        typical_latency_hint="fast",
        verbosity_factor=0.8,
        cost_per_1k_input_tokens=0.00005,
        cost_per_1k_output_tokens=0.00008,
        capability_scores={
            TaskCategory.CODING: 4.5,
            TaskCategory.MATH_REASONING: 4.0,
            TaskCategory.CREATIVE_WRITING: 5.5,
            TaskCategory.CLASSIFICATION_EXTRACTION: 7.0,
            TaskCategory.SUMMARIZATION: 6.0,
            TaskCategory.GENERAL_QA: 6.5,
            TaskCategory.ANALYSIS: 4.5,
        },
    ),
    "groq-llama3-70b": ModelConfig(
        provider=Provider.GROQ,
        model_id="openai/gpt-oss-120b",
        display_name="GPT OSS 120B (Groq)",
        quality_tier=QualityTier.HIGH,
        context_window=131072,
        typical_latency_hint="fast",
        verbosity_factor=1.2,
        cost_per_1k_input_tokens=0.00059,
        cost_per_1k_output_tokens=0.00079,
        capability_scores={
            TaskCategory.CODING: 7.0,
            TaskCategory.MATH_REASONING: 7.5,
            TaskCategory.CREATIVE_WRITING: 8.0,
            TaskCategory.CLASSIFICATION_EXTRACTION: 7.5,
            TaskCategory.SUMMARIZATION: 8.0,
            TaskCategory.GENERAL_QA: 8.0,
            TaskCategory.ANALYSIS: 8.0,
        },
    ),
    "groq-qwen2.5-coder-32b": ModelConfig(
        provider=Provider.GROQ,
        model_id="qwen/qwen3.6-27b",
        display_name="Qwen 3.6 27B (Groq)",
        quality_tier=QualityTier.HIGH,
        context_window=131072,
        typical_latency_hint="fast",
        verbosity_factor=1.4,
        cost_per_1k_input_tokens=0.00069,
        cost_per_1k_output_tokens=0.00069,
        capability_scores={
            TaskCategory.CODING: 9.2,
            TaskCategory.MATH_REASONING: 7.5,
            TaskCategory.CREATIVE_WRITING: 5.0,
            TaskCategory.CLASSIFICATION_EXTRACTION: 6.5,
            TaskCategory.SUMMARIZATION: 6.0,
            TaskCategory.GENERAL_QA: 6.0,
            TaskCategory.ANALYSIS: 7.0,
        },
    ),
    "groq-mixtral-8x7b": ModelConfig(
        provider=Provider.GROQ,
        model_id="groq/compound",
        display_name="Groq Compound (Groq)",
        quality_tier=QualityTier.MEDIUM,
        context_window=131072,
        typical_latency_hint="fast",
        verbosity_factor=1.0,
        cost_per_1k_input_tokens=0.00024,
        cost_per_1k_output_tokens=0.00024,
        capability_scores={
            TaskCategory.CODING: 6.0,
            TaskCategory.MATH_REASONING: 6.0,
            TaskCategory.CREATIVE_WRITING: 6.5,
            TaskCategory.CLASSIFICATION_EXTRACTION: 6.5,
            TaskCategory.SUMMARIZATION: 6.5,
            TaskCategory.GENERAL_QA: 6.5,
            TaskCategory.ANALYSIS: 6.5,
        },
    ),
    "hf-mistral-7b": ModelConfig(
        provider=Provider.HUGGINGFACE,
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        display_name="Mistral 7B Instruct (HF)",
        quality_tier=QualityTier.MEDIUM,
        context_window=32768,
        typical_latency_hint="slow",
        verbosity_factor=0.9,
        cost_per_1k_input_tokens=0.00020,
        cost_per_1k_output_tokens=0.00020,
        capability_scores={
            TaskCategory.CODING: 5.5,
            TaskCategory.MATH_REASONING: 5.0,
            TaskCategory.CREATIVE_WRITING: 6.5,
            TaskCategory.CLASSIFICATION_EXTRACTION: 6.0,
            TaskCategory.SUMMARIZATION: 6.5,
            TaskCategory.GENERAL_QA: 6.0,
            TaskCategory.ANALYSIS: 5.5,
        },
    ),
    "hf-phi-3-mini": ModelConfig(
        provider=Provider.HUGGINGFACE,
        model_id="microsoft/Phi-3-mini-4k-instruct",
        display_name="Phi-3 Mini (HF)",
        quality_tier=QualityTier.LOW,
        context_window=4096,
        typical_latency_hint="medium",
        verbosity_factor=0.7,
        cost_per_1k_input_tokens=0.00010,
        cost_per_1k_output_tokens=0.00010,
        capability_scores={
            TaskCategory.CODING: 5.0,
            TaskCategory.MATH_REASONING: 5.5,
            TaskCategory.CREATIVE_WRITING: 4.5,
            TaskCategory.CLASSIFICATION_EXTRACTION: 6.0,
            TaskCategory.SUMMARIZATION: 5.5,
            TaskCategory.GENERAL_QA: 5.5,
            TaskCategory.ANALYSIS: 4.5,
        },
    ),

}


def get_model(key: str) -> ModelConfig:
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model key '{key}'. Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[key]


def models_by_tier(tier: QualityTier) -> list[ModelConfig]:
    return [
        model for model in MODEL_REGISTRY.values()
        if model.quality_tier == tier
    ]


def all_models() -> list[ModelConfig]:
    return list(MODEL_REGISTRY.values())


# Phase 5: auto-onboarded model loading.
_CUSTOM_MODELS_PATH = Path(__file__).resolve().parent.parent / "data" / "models_config.yaml"
_MODEL_META_PATH = Path(__file__).resolve().parent.parent / "data" / "model_meta.json"
_CALIBRATED_SCORES_PATH = Path(__file__).resolve().parent.parent / "data" / "calibrated_scores.json"

def _load_custom_models() -> None:
    """
    Loads custom model configurations from YAML and adds them to the registry.
    """
    if not _CUSTOM_MODELS_PATH.exists():
        return
        
    try:
        import yaml
        entries = yaml.safe_load(_CUSTOM_MODELS_PATH.read_text(encoding="utf-8")) or []
    except Exception:
        return
        
    for entry in entries:
        key = entry.get("key")
        if not key or key in MODEL_REGISTRY:
            continue
            
        try:
            MODEL_REGISTRY[key] = ModelConfig(
                provider=Provider(entry["provider"]), 
                model_id=entry["model_id"], 
                display_name=entry.get("display_name", entry["model_id"]), 
                quality_tier=QualityTier.MEDIUM, 
                context_window=int(entry.get("context_window", 8192)), 
                cost_per_1k_input_tokens=float(entry.get("cost_per_1k_input_tokens", 0.0)),
                cost_per_1k_output_tokens=float(entry.get("cost_per_1k_output_tokens", 0.0)),
                verbosity_factor=1.0, 
                capability_scores={}
            )
        except (KeyError, ValueError, TypeError):
            continue

def _apply_model_meta() -> None:
    """
    Loads model metadata (e.g. tier overrides, measured verbosity) from JSON 
    and applies it to the corresponding models in the registry.
    """
    if not _MODEL_META_PATH.exists():
        return
        
    try:
        meta = json.loads(_MODEL_META_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
        
    for key, values in meta.items():
        if key not in MODEL_REGISTRY:
            continue
            
        updates = {}
        
        if "quality_tier" in values:
            try:
                updates["quality_tier"] = QualityTier(values["quality_tier"])
            except ValueError:
                pass
                
        if "verbosity_factor" in values:
            try:
                updates["verbosity_factor"] = float(values["verbosity_factor"])
            except (TypeError, ValueError):
                pass
                
        if "typical_latency_hint" in values:
            updates["typical_latency_hint"] = values["typical_latency_hint"]
            
        if updates:
            MODEL_REGISTRY[key] = replace(MODEL_REGISTRY[key], **updates)

def _apply_calibrated_scores() -> None:
    """
    Loads calibrated capability scores from JSON and applies them to 
    the models in the registry, replacing default baseline scores.
    """
    if not _CALIBRATED_SCORES_PATH.exists():
        return
        
    try:
        calibrated = json.loads(_CALIBRATED_SCORES_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
        
    for key, category_scores in calibrated.items():
        if key not in MODEL_REGISTRY:
            continue
            
        current = MODEL_REGISTRY[key]
        merged_scores = dict(current.capability_scores)
        
        for category_name, score in category_scores.items():
            try:
                merged_scores[TaskCategory(category_name)] = float(score)
            except (ValueError, TypeError):
                continue
                
        MODEL_REGISTRY[key] = replace(current, capability_scores=merged_scores)

_load_custom_models()
_apply_model_meta()
_apply_calibrated_scores()
