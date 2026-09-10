import json
from pathlib import Path

from .models import MODEL_REGISTRY
from .providers import ProviderError, send_request


TEST_PROMPTS = [
    "What is the capital of France?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Write a Python function that reverses a string.",
    "Explain the difference between TCP and UDP in simple terms.",
    "What is 17 * 24?",
    (
        "Classify this email as billing, technical, account, or general: "
        "'Hi, I was charged twice for my subscription this month, can you help?'"
    ),
    "Compare the pros and cons of remote work vs office work.",
    "Write a haiku about autumn.",
    "What year did the Berlin Wall fall?",
    (
        "Extract the key entities (names, dates, amounts) from: "
        "'Invoice #4521 issued to John Carter on March 3, 2024 for $1,250.00.'"
    ),
]

OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "baseline_results.json"
)


def run_baseline() -> list[dict]:
    results = []

    for model_key, model in MODEL_REGISTRY.items():
        print(
            f"\n=== Testing {model.display_name} ({model_key}) ==="
        )

        for prompt in TEST_PROMPTS:
            try:
                response = send_request(model, prompt)

                record = {
                    "model_key": model_key,
                    "provider": model.provider.value,
                    "prompt": prompt,
                    "output": response.text,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "total_tokens": response.total_tokens,
                    "latency_seconds": round(
                        response.latency_seconds,
                        3,
                    ),
                    "cost_usd": response.cost_usd,
                    "success": True,
                }

                print(
                    f" OK [{response.total_tokens:>4} tokens, "
                    f"{response.latency_seconds:.2f}s] "
                    f"{prompt[:50]}..."
                )

            except ProviderError as exc:
                record = {
                    "model_key": model_key,
                    "provider": model.provider.value,
                    "prompt": prompt,
                    "error": str(exc),
                    "success": False,
                }

                print(f" FAIL: {exc}")

            results.append(record)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\nSaved {len(results)} records to {OUTPUT_PATH}")
    return results


if __name__ == "__main__":
    run_baseline()
