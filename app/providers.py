from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

load_dotenv()

from .models import ModelConfig, Provider


@dataclass
class Response:
    text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    model_key: str
    provider: Provider
    cost_usd: float
    raw: dict | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ProviderError(RuntimeError):
    """Raised when a provider call fails."""


def _approx_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def _call_groq(
    model: ModelConfig,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> Response:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ProviderError("GROQ_API_KEY not set in environment")

    start = time.perf_counter()
    try:
        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=30.0,
        )
    except httpx.HTTPError as exc:
        raise ProviderError(f"Groq request failed: {exc}") from exc

    latency = time.perf_counter() - start

    if resp.status_code != 200:
        raise ProviderError(f"Groq error {resp.status_code}: {resp.text}")

    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    input_tokens = usage.get(
        "prompt_tokens",
        _approx_token_count(prompt),
    )
    output_tokens = usage.get(
        "completion_tokens",
        _approx_token_count(text),
    )

    return Response(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_seconds=latency,
        model_key=model.model_id,
        provider=Provider.GROQ,
        cost_usd=model.estimated_cost(input_tokens, output_tokens),
        raw=data,
    )


def _call_huggingface(
    model: ModelConfig,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> Response:
    api_key = os.environ.get("HF_API_TOKEN") or os.environ.get("HF_API_KEY")
    if not api_key:
        raise ProviderError("HF_API_TOKEN or HF_API_KEY not set in environment")

    start = time.perf_counter()
    try:
        resp = httpx.post(
            f"https://api-inference.huggingface.co/models/{model.model_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "inputs": prompt,
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "return_full_text": False,
                },
            },
            timeout=60.0,
        )
    except httpx.HTTPError as exc:
        raise ProviderError(f"Hugging Face request failed: {exc}") from exc

    latency = time.perf_counter() - start

    if resp.status_code != 200:
        raise ProviderError(
            f"Hugging Face error {resp.status_code}: {resp.text}"
        )

    data = resp.json()

    if (
        isinstance(data, list)
        and data
        and isinstance(data[0], dict)
        and "generated_text" in data[0]
    ):
        text = data[0]["generated_text"]
    else:
        text = str(data)

    input_tokens = _approx_token_count(prompt)
    output_tokens = _approx_token_count(text)

    return Response(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_seconds=latency,
        model_key=model.model_id,
        provider=Provider.HUGGINGFACE,
        cost_usd=model.estimated_cost(input_tokens, output_tokens),
        raw=data if isinstance(data, dict) else {"data": data},
    )


_DISPATCH = {
    Provider.GROQ: _call_groq,
    Provider.HUGGINGFACE: _call_huggingface,
}


def send_request(
    model: ModelConfig,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> Response:
    handler = _DISPATCH.get(model.provider)

    if handler is None:
        raise ProviderError(
            f"No handler registered for provider {model.provider}"
        )

    return handler(model, prompt, temperature, max_tokens)
