from __future__ import annotations

import json
import sys
from pathlib import Path
import yaml

from .calibrate import SUBJECTIVE_PROMPTS
from .eval_tasks import run_objective_eval
from .llm_judge import judge_pairwise
from .models import MODEL_REGISTRY, ModelConfig, Provider, QualityTier, TaskCategory
from .providers import ProviderError, send_request

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
CONFIG_PATH = DATA_DIR / 'models_config.yaml'
META_PATH = DATA_DIR / 'model_meta.json'
CALIBRATED_PATH = DATA_DIR / 'calibrated_scores.json'

_PROBE_BASELINE_TOKENS = 50

def _probe(provider: str, model_id: str, context_window: int) -> tuple[ModelConfig, float]:
    """
    Probes the model by sending a standard prompt to measure its natural 
    verbosity. Returns the model configuration and the estimated verbosity factor.
    """
    model = ModelConfig(
        provider=Provider(provider),
        model_id=model_id,
        display_name=model_id,
        quality_tier=QualityTier.MEDIUM,
        context_window=context_window
    )
    
    response = send_request(
        model, 
        "Explain what an LLM is in one short paragraph.", 
        max_tokens=200
    )
    
    verbosity = max(0.3, round(response.output_tokens / _PROBE_BASELINE_TOKENS, 2))
    return model, verbosity

def _run_tournament_against_existing(
    new_key: str, 
    new_model: ModelConfig
) -> dict[TaskCategory, float]:
    """
    Evaluates a new model by running a subjective pairwise tournament against 
    all existing models in the registry using the LLM-as-a-judge system.
    """
    existing = [
        (k, m) for k, m in MODEL_REGISTRY.items() 
        if k != new_key
    ]
    scores = {}
    
    for category, prompt in SUBJECTIVE_PROMPTS.items():
        print(f"  tournament: {category.value}...")
        
        try:
            new_response = send_request(
                new_model, 
                prompt, 
                temperature=0.7, 
                max_tokens=200
            ).text
        except ProviderError:
            continue
            
        wins = 0.0
        matches = 0
        
        for _, opponent in existing:
            try:
                opp_response = send_request(
                    opponent, 
                    prompt, 
                    temperature=0.7, 
                    max_tokens=200
                ).text
            except ProviderError:
                continue
                
            verdict = judge_pairwise(
                prompt, 
                new_response, 
                opp_response, 
                num_votes=2
            )
            matches += 1
            
            if verdict.winner == 'A':
                wins += 1.0
            elif verdict.winner == 'TIE':
                wins += 0.5
                
        if matches:
            scores[category] = round(10.0 * wins / matches, 2)
            
    return scores

def _derive_tier(scores: dict[TaskCategory, float]) -> QualityTier:
    """
    Determines the quality tier (LOW, MEDIUM, HIGH) based on the 
    average capability score across all tasks.
    """
    if scores:
        avg = sum(scores.values()) / len(scores)
    else:
        avg = 5.0
        
    if avg >= 7.5:
        return QualityTier.HIGH
    if avg >= 5.5:
        return QualityTier.MEDIUM
    return QualityTier.LOW

def onboard(
    key: str, 
    provider: str, 
    model_id: str, 
    display_name: str | None = None, 
    context_window: int = 8192
) -> None:
    """
    End-to-end process for onboarding a new model:
    1. Probes the model to determine if it is reachable and calculate verbosity.
    2. Runs objective evaluations.
    3. Runs subjective pairwise evaluations against existing models.
    4. Saves configuration and capability metadata.
    """
    print(f"Onboarding '{key}' ({provider}/{model_id})...\n")
    
    # Step 1: Probe the model
    print('Step 1/3: probing model...')
    model, verbosity = _probe(provider, model_id, context_window)
    print(f'  OK - reachable, verbosity_factor~{verbosity}\n')
    
    # Step 2: Objective evaluations
    print('Step 2/3: objective eval (coding, math, extraction)...')
    objective = run_objective_eval(model, send_request)
    for cat, score in objective.items():
        print(f'  {cat.value}: {score}/10')
        
    # Step 3: Subjective tournament
    print(f"\nStep 3/3: judged tournament vs. {len(MODEL_REGISTRY)} existing models...")
    subjective = _run_tournament_against_existing(key, model)
    for cat, score in subjective.items():
        print(f'  {cat.value}: {score}/10')
        
    # Compile scores and calculate averages
    scores = {**objective, **subjective}
    avg = sum(scores.values()) / len(scores) if scores else 5.0
    tier = _derive_tier(scores)
    
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model config to YAML
    configs = []
    if CONFIG_PATH.exists():
        configs = yaml.safe_load(CONFIG_PATH.read_text()) or []
    
    configs = [x for x in configs if x.get('key') != key]
    configs.append({
        'key': key,
        'provider': provider,
        'model_id': model_id,
        'display_name': display_name or model_id,
        'context_window': context_window
    })
    CONFIG_PATH.write_text(yaml.safe_dump(configs, sort_keys=False))
    
    # Save model metadata to JSON (quality tier, verbosity)
    meta = {}
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
        
    meta[key] = {
        'quality_tier': tier.value,
        'verbosity_factor': verbosity
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    
    # Save calibrated capability scores
    calibrated = {}
    if CALIBRATED_PATH.exists():
        calibrated = json.loads(CALIBRATED_PATH.read_text())
        
    calibrated[key] = {cat.value: score for cat, score in scores.items()}
    CALIBRATED_PATH.write_text(json.dumps(calibrated, indent=2))
    
    print(f"\nOnboarded '{key}': tier={tier.value}, avg capability={avg:.1f}/10")
    print('No manual scoring was entered; all routing metadata was measured.')

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: python -m app.onboard <key> <provider> <model_id> [display_name] [context_window]')
        sys.exit(1)
        
    key = sys.argv[1]
    provider = sys.argv[2]
    model_id = sys.argv[3]
    display_name = sys.argv[4] if len(sys.argv) > 4 else None
    context_window = int(sys.argv[5]) if len(sys.argv) > 5 else 8192
    
    onboard(key, provider, model_id, display_name, context_window)
