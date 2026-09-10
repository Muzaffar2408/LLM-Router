import sys
from typing import List, Dict

from app.models import MODEL_REGISTRY
from app.traffic_cop import classify_with_traffic_cop
from app.scoring import rank_models

# The expensive baseline model most devs default to
BASELINE_MODEL_KEY = "groq-llama3-70b"

PROMPTS = [
    "What is 2+2?",
    "Write a haiku about rain.",
    "Implement a thread-safe LRU cache in Python with O(1) operations.",
    "Extract the company names from this text: Microsoft and Google announced a partnership.",
    "Summarize the plot of the movie Inception in one sentence.",
    "What is the capital of France?",
    "Analyze the time complexity of the quicksort algorithm.",
    "Write a romantic poem about the ocean.",
    "Solve the differential equation dy/dx = x^2 + y.",
    "Categorize this support ticket: 'My screen is cracked and I need a replacement.'",
    "Explain quantum computing to a 5 year old.",
    "Generate a 5-day itinerary for a trip to Tokyo.",
    "What are the symptoms of the common cold?",
    "Write a SQL query to find the second highest salary from an Employee table.",
    "Translate 'Hello, how are you?' to Spanish.",
    "What is the meaning of life?",
    "Write a short story about a robot who learns to love.",
    "Calculate 15% of $120.",
    "Extract the dates from: The meeting is on 2023-10-25 and the deadline is 2023-11-01.",
    "Explain the difference between a list and a tuple in Python."
]

def estimate_latency_ms(model, input_tokens: float, output_tokens: float) -> float:
    """Estimates realistic latency using Time-To-First-Token and tokens/sec speed."""
    if model.provider == "groq":
        ttft = 200.0
        # 70B is slower than 8B even on Groq
        if "70b" in model.model_id:
            speed_tps = 300.0
        elif "32b" in model.model_id:
            speed_tps = 500.0
        else:
            speed_tps = 800.0
    elif model.provider == "huggingface":
        ttft = 1200.0
        speed_tps = 30.0
    else:
        ttft = 500.0
        speed_tps = 100.0
        
    generation_time_ms = (output_tokens / speed_tps) * 1000.0
    return ttft + generation_time_ms

def run_benchmark():
    print(f"Running benchmark on {len(PROMPTS)} prompts...\n")
    
    baseline_model = MODEL_REGISTRY[BASELINE_MODEL_KEY]
    
    total_baseline_cost = 0.0
    total_autopilot_cost = 0.0
    
    total_baseline_latency = 0.0
    total_autopilot_latency = 0.0
    
    total_baseline_capability = 0.0
    total_autopilot_capability = 0.0
    
    for i, prompt in enumerate(PROMPTS, 1):
        # 1. Traffic Cop classification
        cop_result = classify_with_traffic_cop(prompt)
        category = cop_result.task_category
        complexity = cop_result.complexity
        
        # 2. Get Autopilot's chosen model
        ranked = rank_models(category, complexity)
        autopilot_model = ranked[0].model
        
        # 3. Calculate metrics for Baseline
        est_input_tokens = len(prompt.split()) * 1.3
        
        baseline_out_tokens = complexity * 100 * baseline_model.verbosity_factor
        baseline_cost = baseline_model.estimated_cost(est_input_tokens, baseline_out_tokens)
        baseline_cap = baseline_model.capability_for(category)
        baseline_lat = estimate_latency_ms(baseline_model, est_input_tokens, baseline_out_tokens)
        
        # 4. Calculate metrics for Autopilot
        autopilot_out_tokens = complexity * 100 * autopilot_model.verbosity_factor
        autopilot_cost = autopilot_model.estimated_cost(est_input_tokens, autopilot_out_tokens)
        autopilot_cap = autopilot_model.capability_for(category)
        autopilot_lat = estimate_latency_ms(autopilot_model, est_input_tokens, autopilot_out_tokens)
        
        # 5. Accumulate
        total_baseline_cost += baseline_cost
        total_autopilot_cost += autopilot_cost
        
        total_baseline_latency += baseline_lat
        total_autopilot_latency += autopilot_lat
        
        total_baseline_capability += baseline_cap
        total_autopilot_capability += autopilot_cap
        
        print(f"[{i}/{len(PROMPTS)}] Prompt: {prompt[:30]}... | Cat: {category.value} | Cpx: {complexity}")
        print(f"  -> Autopilot selected: {autopilot_model.display_name} (Cost: ${autopilot_cost:.6f}, Cap: {autopilot_cap})")
        
    
    # Calculate averages and percentages
    cost_savings_pct = ((total_baseline_cost - total_autopilot_cost) / total_baseline_cost) * 100
    
    avg_baseline_latency = total_baseline_latency / len(PROMPTS)
    avg_autopilot_latency = total_autopilot_latency / len(PROMPTS)
    latency_reduction_pct = ((avg_baseline_latency - avg_autopilot_latency) / avg_baseline_latency) * 100
    
    avg_baseline_cap = total_baseline_capability / len(PROMPTS)
    avg_autopilot_cap = total_autopilot_capability / len(PROMPTS)
    capability_retained_pct = (avg_autopilot_cap / avg_baseline_cap) * 100
    
    print("\n" + "="*50)
    print("[BENCHMARK RESULTS (FOR CV)]")
    print("="*50)
    print(f"Compared Autopilot vs. Always using {baseline_model.display_name}\n")
    
    print(f"Total Cost Savings:        {cost_savings_pct:.2f}%")
    print(f"   (Cost reduced from ${total_baseline_cost:.4f} down to ${total_autopilot_cost:.4f})\n")
    
    print(f"Average Latency Reduction: {latency_reduction_pct:.2f}%")
    print(f"   (Latency reduced from {avg_baseline_latency:.0f}ms down to {avg_autopilot_latency:.0f}ms)\n")
    
    print(f"Capability Retained:       {capability_retained_pct:.2f}%")
    print(f"   (Maintained {avg_autopilot_cap:.2f}/10 average score vs {avg_baseline_cap:.2f}/10 baseline)")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()
