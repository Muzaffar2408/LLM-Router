from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from .models import MODEL_REGISTRY
from .providers import ProviderError, send_request

JUDGE_MODEL_KEY = "groq-llama3-70b"
JUDGE_PROMPT_TEMPLATE = """You are an impartial evaluator comparing two AI responses to the same prompt.
Judge accuracy, relevance, and substance only. Do NOT prefer an answer just because it is longer; penalize unnecessary padding.

Original prompt:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Reply with ONLY: A, B, or TIE."""

@dataclass
class JudgeVerdict:
    """Represents the verdict of an LLM-as-a-judge pairwise comparison."""
    winner: str
    votes: list[str]
    position_bias_detected: bool

def _single_judge_call(prompt: str, response_a: str, response_b: str) -> str:
    """
    Sends a single prompt with two responses to the judge LLM.
    Returns the judge's vote: "A", "B", or "TIE".
    """
    try:
        result = send_request(
            MODEL_REGISTRY[JUDGE_MODEL_KEY],
            JUDGE_PROMPT_TEMPLATE.format(
                prompt=prompt,
                response_a=response_a,
                response_b=response_b
            ),
            temperature=0.0,
            max_tokens=10,
        )
        vote = result.text.strip().upper()
        if vote in {"A", "B", "TIE"}:
            return vote
        return "TIE"
    except ProviderError:
        return "TIE"

def _majority(votes: list[str]) -> str:
    """
    Calculates the majority vote from a list of votes.
    Returns "TIE" if there is no clear majority.
    """
    if not votes:
        return "TIE"
        
    counts = Counter(votes)
    max_votes = max(counts.values())
    
    # Find all choices that received the maximum number of votes
    winners = [choice for choice, count in counts.items() if count == max_votes]
    
    if len(winners) == 1:
        return winners[0]
        
    return "TIE"

def judge_pairwise(
    prompt: str, 
    response_a: str, 
    response_b: str, 
    num_votes: int = 1
) -> JudgeVerdict:
    """
    Performs a pairwise comparison between two responses using an LLM judge.
    
    To counteract position bias (where a model might always prefer the first 
    option), it runs the comparison multiple times both in the original order 
    (A vs B) and in swapped order (B vs A).
    """
    # Cast votes in the original order (A vs B)
    votes = [
        _single_judge_call(prompt, response_a, response_b) 
        for _ in range(num_votes)
    ]
    
    # Cast votes in swapped order (B vs A) to check for position bias
    swapped_raw = [
        _single_judge_call(prompt, response_b, response_a) 
        for _ in range(num_votes)
    ]
    
    # Flip the swapped votes back to the original reference frame
    flip_map = {"A": "B", "B": "A", "TIE": "TIE"}
    swapped_votes = [flip_map[vote] for vote in swapped_raw]
    
    # Determine the majority for each direction
    orig_majority = _majority(votes)
    swapped_majority = _majority(swapped_votes)
    
    # Position bias is detected if the winner changes purely based on order
    bias = orig_majority != swapped_majority and "TIE" not in (orig_majority, swapped_majority)
    
    if bias:
        winner = "TIE"
    else:
        # Aggregate all votes to determine the final winner
        all_votes = votes + swapped_votes
        winner = _majority(all_votes)
        
    return JudgeVerdict(
        winner=winner, 
        votes=all_votes, 
        position_bias_detected=bias
    )

def spot_check_sample(judged_comparisons: list[dict], sample_size: int = 5) -> list[dict]:
    """Returns a random sample of judged comparisons for manual review."""
    return random.sample(
        judged_comparisons, 
        min(sample_size, len(judged_comparisons))
    )
