# Architecture

The LLM Cost Autopilot is built on a modular, asynchronous architecture designed for high throughput and extensible model management.

## Core Components

### 1. Traffic Cop (`app/traffic_cop.py`)
The entry point for all prompts. The Traffic Cop uses an ultra-fast, cheap LLM to analyze the prompt in milliseconds. 
- It outputs a JSON payload containing the **task category** (coding, math, creative, etc.) and a **complexity score (1-10)**.
- **Fallback Heuristics**: If the API fails or is too slow, it falls back to regex-based heuristics (e.g., detecting keywords like `def`, `class`, `calculate`) to guess the category and complexity.

### 2. The Router & Scorer (`app/router.py` & `app/scoring.py`)
Once the prompt is classified, the router evaluates all active models in the registry.
- **Cost Estimation**: Estimates the number of input and output tokens (incorporating a model's specific `verbosity_factor` and the prompt's `complexity`).
- **Scoring Formula**: Models are ranked using a dynamic capability penalty. The penalty is exponentially applied based on the gap between the model's capability for the specific task and the prompt's complexity. 
- The model with the lowest combined `cost_component + capability_penalty` is selected.

### 3. Provider Integrations (`app/providers.py`)
Abstracts away the specific API calls. Currently supports **Groq** and **Hugging Face** endpoints. It standardizes the responses, tracking exact tokens used and latency (Time-to-First-Token). Local Ollama models were removed to prevent machine overheating and ensure cloud scalability.

### 4. Calibration & Onboarding (`app/onboard.py`, `app/calibrate.py`, `app/llm_judge.py`)
A self-contained testing suite for adding new models.
- **Probing**: Sends basic prompts to test latency and verbosity.
- **LLM-as-a-Judge**: Runs the new model against a suite of benchmark prompts across all categories. A high-tier judge model evaluates the answers to automatically assign objective capability scores (0-10) and a `QualityTier` (High, Medium, Low).
- Updates are saved to `data/calibrated_scores.json` and `data/models_config.yaml`.
