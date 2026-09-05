import json
import os

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BACKEND = "openai"
_clients: dict[str, OpenAI] = {}


def _get_client(backend: str) -> OpenAI:
    if backend not in {"openai", "openrouter"}:
        raise ValueError("LLM backend must be 'openai' or 'openrouter'")
    if backend not in _clients:
        load_dotenv()
        if backend == "openrouter":
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY is required for OpenRouter generation"
                )
            _clients[backend] = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=key,
                max_retries=5,
                default_headers={
                    "HTTP-Referer": "https://github.com/rivf",
                    "X-OpenRouter-Title": "RIVF GRAFT-v1",
                },
            )
        else:
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY is required for OpenAI generation")
            _clients[backend] = OpenAI(api_key=key, max_retries=5)
    return _clients[backend]


def generate_answer(
    prompt: str,
    model: str = DEFAULT_MODEL,
    *,
    backend: str = DEFAULT_BACKEND,
) -> str:
    return generate_answer_with_metadata(prompt, model=model, backend=backend)[
        "prediction"
    ]


def generate_answer_with_metadata(
    prompt: str,
    model: str = DEFAULT_MODEL,
    *,
    backend: str = DEFAULT_BACKEND,
) -> dict:
    """Generate one answer and retain API provenance/usage when available."""
    response = _get_client(backend).chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    usage = response.usage
    response_extra = getattr(response, "model_extra", None) or {}
    usage_extra = (getattr(usage, "model_extra", None) or {}) if usage else {}
    return {
        "prediction": response.choices[0].message.content.strip(),
        "api_backend": backend,
        "api_provider": response_extra.get("provider"),
        "api_response_id": getattr(response, "id", None),
        "resolved_llm_model": getattr(response, "model", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "api_prompt_tokens": usage.prompt_tokens if usage else None,
        "api_completion_tokens": usage.completion_tokens if usage else None,
        "api_total_tokens": usage.total_tokens if usage else None,
        "api_cost_usd": usage_extra.get("cost"),
    }


class StructuredGenerationError(RuntimeError):
    """Raised when a model does not return the exact JSON contract requested."""


def generate_json_with_metadata(
    prompt: str,
    model: str = DEFAULT_MODEL,
    *,
    backend: str = DEFAULT_BACKEND,
) -> dict:
    """Generate one strict JSON object and retain API provenance.

    WISE-v3 treats the reasoning plan as part of the method, not as a hint. An
    invalid planner response is therefore an explicit error; it is never
    repaired heuristically or replaced with a different planning policy.
    """
    response = _get_client(backend).chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw_content = response.choices[0].message.content
    if not isinstance(raw_content, str) or not raw_content.strip():
        raise StructuredGenerationError("planner returned empty JSON content")
    content = raw_content.strip()
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise StructuredGenerationError(
            "planner returned invalid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise StructuredGenerationError("planner JSON must be an object")

    usage = response.usage
    response_extra = getattr(response, "model_extra", None) or {}
    usage_extra = (getattr(usage, "model_extra", None) or {}) if usage else {}
    return {
        "payload": payload,
        "api_backend": backend,
        "api_provider": response_extra.get("provider"),
        "api_response_id": getattr(response, "id", None),
        "resolved_llm_model": getattr(response, "model", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "api_prompt_tokens": usage.prompt_tokens if usage else None,
        "api_completion_tokens": usage.completion_tokens if usage else None,
        "api_total_tokens": usage.total_tokens if usage else None,
        "api_cost_usd": usage_extra.get("cost"),
    }
