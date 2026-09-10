from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

from .eval_tasks import run_objective_eval
from .llm_judge import judge_pairwise, JUDGE_MODEL_KEY
from .models import MODEL_REGISTRY, TaskCategory
from .providers import send_request, ProviderError

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "calibrated_scores.json"

SUBJECTIVE_PROMPTS = {
    TaskCategory.CREATIVE_WRITING: [
        "Write a short, vivid opening paragraph for a mystery novel set in Victorian London.",
        "Compose a poem about a robot discovering emotion for the first time.",
        "Write a dialogue between two astronauts who just realized their ship is off course."
    ],
    TaskCategory.SUMMARIZATION: [
        "Summarize in 2 sentences: The committee reviewed twelve proposals over three days, ultimately selecting four for further funding based on feasibility, budget alignment, and projected community impact, while the remaining eight were deferred pending revisions.",
        "Summarize the plot of Romeo and Juliet in a single paragraph.",
        "Summarize the main principles of Agile software development in bullet points."
    ],
    TaskCategory.GENERAL_QA: [
        "What causes the seasons to change on Earth?",
        "How do black holes form?",
        "What is the difference between DNA and RNA?"
    ],
    TaskCategory.ANALYSIS: [
        "What are the main trade-offs between working remotely and working in an office?",
        "Analyze the economic impact of universal basic income based on current theories.",
        "Compare and contrast the advantages of functional programming versus object-oriented programming."
    ],
}

def run_objective_pass() -> dict[str, dict[str, float]]:
    """
    Runs the objective evaluation suite against all models in the registry.
    Returns a nested dictionary mapping model keys to category scores.
    """
    scores = {}
    
    for key, model in MODEL_REGISTRY.items():
        print(f"Objective eval: {model.display_name}...")
        try:
            eval_results = run_objective_eval(model, send_request)
            scores[key] = {
                cat.value: score 
                for cat, score in eval_results.items()
            }
        except Exception as exc:
            print(f"  skipped ({exc})")
            scores[key] = {}
            
    return scores

def run_subjective_tournament() -> dict[str, dict[str, float]]:
    """
    Runs a pairwise subjective tournament using an LLM-as-a-judge for all
    models in the registry (excluding the judge model itself).
    Returns Elo-based scores out of 10.0 for each category.
    """
    models = [
        (k, m) for k, m in MODEL_REGISTRY.items() 
        if k != JUDGE_MODEL_KEY
    ]
    
    # Initialize Elo ratings at 1200
    elo_ratings = {k: {cat.value: 1200.0 for cat in SUBJECTIVE_PROMPTS.keys()} for k, _ in models}
    
    # Higher K-Factor because we only have a few rounds (matches) per model
    K_FACTOR = 64
    
    for category, prompts in SUBJECTIVE_PROMPTS.items():
        ck = category.value
        print(f"Subjective tournament: {ck}...")
        
        for prompt in prompts:
            # Collect responses from all models for the current prompt
            responses = {}
            for key, model in models:
                try:
                    response_obj = send_request(
                        model, 
                        prompt, 
                        temperature=0.7, 
                        max_tokens=200
                    )
                    responses[key] = response_obj.text
                except ProviderError:
                    responses[key] = None
                    
            # Run pairwise combinations
            for (a, _), (b, _) in combinations(models, 2):
                if responses[a] is None or responses[b] is None:
                    continue
                    
                verdict = judge_pairwise(
                    prompt, 
                    responses[a], 
                    responses[b], 
                    num_votes=1
                )
                
                # Elo Calculation
                R_a = elo_ratings[a][ck]
                R_b = elo_ratings[b][ck]
                
                E_a = 1 / (1 + 10 ** ((R_b - R_a) / 400))
                E_b = 1 / (1 + 10 ** ((R_a - R_b) / 400))
                
                if verdict.winner == "A":
                    score_a, score_b = 1.0, 0.0
                elif verdict.winner == "B":
                    score_a, score_b = 0.0, 1.0
                else:
                    score_a, score_b = 0.5, 0.5
                    
                elo_ratings[a][ck] = R_a + K_FACTOR * (score_a - E_a)
                elo_ratings[b][ck] = R_b + K_FACTOR * (score_b - E_b)
                
    # Normalize final Elo ratings to a 0-10 scale
    # Baseline 1200 Elo = 5.0 score. 800 Elo = 0.0 score. 1600 Elo = 10.0 score.
    final_scores = {}
    for k, cats in elo_ratings.items():
        final_scores[k] = {}
        for ck, elo in cats.items():
            scaled = (elo - 800) / 80.0
            final_scores[k][ck] = round(max(0.0, min(10.0, scaled)), 2)
            
    return final_scores

def calibrate_all() -> dict[str, dict[str, float]]:
    """
    Runs both objective and subjective evaluations and saves the combined
    calibrated scores to the output JSON file.
    """
    print(
        f"NOTE: judge model is {JUDGE_MODEL_KEY}; "
        "excluded from the candidate tournament.\n"
    )
    
    objective = run_objective_pass()
    subjective = run_subjective_tournament()
    
    merged = {}
    for k in MODEL_REGISTRY:
        merged[k] = {
            **objective.get(k, {}), 
            **subjective.get(k, {})
        }
        
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"\nCalibrated scores written to {OUTPUT_PATH}")
    
    return merged

if __name__ == "__main__":
    calibrate_all()
