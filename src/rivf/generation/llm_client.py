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
    response = _get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()
