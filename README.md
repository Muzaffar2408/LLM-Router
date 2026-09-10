# LLM Cost Autopilot

LLM Cost Autopilot is an intelligent, dynamic query router designed to slash LLM API inference costs while maintaining high quality. It intercepts user prompts, evaluates their complexity and task category, and dynamically routes them to the most cost-effective model capable of handling the request.

## Key Metrics
* **93.2% Reduction in Inference Costs**
* **64.9% Reduction in Average Latency**
* **81.2% Capability Retention** (compared to always defaulting to a 70B+ model)

## Features
- **Dynamic Traffic Cop**: Uses a fast, inexpensive model (or fallback heuristics) to assess prompt complexity (1-10) and category (e.g., coding, math, general QA).
- **Intelligent Scoring Algorithm**: Ranks models based on a dynamic trade-off between task-specific capability and cost. Simple queries get routed to fast/cheap 1B-8B models, while complex tasks get routed to 70B+ models.
- **Automated Model Onboarding**: Includes an LLM-as-a-judge calibration suite to automatically benchmark and onboard new models into the registry.
- **Zero Local Overheating**: 100% cloud-native inference using Groq and Hugging Face APIs. No local GPU/CPU inference required.

## Installation & Setup

1. Clone the repository and ensure you have `uv` installed.
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Set up your environment variables in `.env`:
   ```env
   GROQ_API_KEY="your_groq_key"
   HF_API_KEY="your_hf_key"
   ```

## Usage

Start the FastAPI server:
```bash
uv run uvicorn app.main:app --reload
```

Submit a prompt to the router:
```bash
curl -X POST "http://localhost:8000/v1/route" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "What is 2+2?"}'
```

Run the benchmark to see cost savings:
```bash
python -m app.benchmark
```