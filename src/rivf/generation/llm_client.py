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
