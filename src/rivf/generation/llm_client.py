import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        # A ~200-800 call experiment run is long enough to hit a transient
        # DNS/network blip; the SDK's built-in retry (exponential backoff)
        # covers that without a bespoke retry loop.
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=5)
    return _client


def generate_answer(prompt: str, model: str = DEFAULT_MODEL) -> str:
    return generate_answer_with_metadata(prompt, model=model)["prediction"]


def generate_answer_with_metadata(prompt: str, model: str = DEFAULT_MODEL) -> dict:
    """Generate one answer and retain API provenance/usage when available."""
    response = _get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    usage = response.usage
    return {
        "prediction": response.choices[0].message.content.strip(),
        "api_response_id": getattr(response, "id", None),
        "resolved_llm_model": getattr(response, "model", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "api_prompt_tokens": usage.prompt_tokens if usage else None,
        "api_completion_tokens": usage.completion_tokens if usage else None,
        "api_total_tokens": usage.total_tokens if usage else None,
    }
