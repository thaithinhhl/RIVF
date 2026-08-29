import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


LOCAL_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
OPENROUTER_MODEL_NAME = "qwen/qwen3-reranker-8b"
MODEL_NAME = LOCAL_MODEL_NAME  # Backward-compatible public constant.

_backend = os.environ.get("RIVF_RERANKER_BACKEND", "local")
_model_name = os.environ.get("RIVF_RERANKER_MODEL", LOCAL_MODEL_NAME)
_model = None
_score_cache: dict[tuple[str, str, str, str], float] = {}
_usage = {"calls": 0, "search_units": 0, "total_tokens": 0}


def configure(backend: str = "local", model: str | None = None) -> None:
    global _backend, _model_name, _model
    if backend not in {"local", "openrouter"}:
        raise ValueError("reranker backend must be 'local' or 'openrouter'")
    resolved_model = model or (
        OPENROUTER_MODEL_NAME if backend == "openrouter" else LOCAL_MODEL_NAME
    )
    if (_backend, _model_name) != (backend, resolved_model):
        _model = None
    _backend = backend
    _model_name = resolved_model


def backend_name() -> str:
    return _backend


def model_name() -> str:
    return _model_name


def usage() -> dict[str, int]:
    return dict(_usage)


def _get_local_model():
    global _model
    if _model is None:
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "local reranker backend requires the 'local-models' extra"
            ) from exc
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        _model = CrossEncoder(_model_name, device=device)
    return _model


def _openrouter_scores(query: str, documents: list[str]) -> list[float]:
    load_dotenv()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter reranking")
    body = json.dumps(
        {
            "model": _model_name,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        }
    ).encode("utf-8")
    request = Request(
        "https://openrouter.ai/api/v1/rerank",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/rivf",
            "X-OpenRouter-Title": "RIVF GRAFT-v1",
        },
        method="POST",
    )
    payload = None
    for attempt in range(7):
        try:
            with urlopen(request, timeout=120) as response:
                payload = json.load(response)
            break
        except HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt == 6:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                raise RuntimeError(
                    f"OpenRouter rerank failed with HTTP {exc.code}: {detail}"
                ) from exc
        except (URLError, TimeoutError) as exc:
            if attempt == 6:
                raise RuntimeError("OpenRouter rerank request failed") from exc
        time.sleep(min(2 ** (attempt + 1), 30))
    if payload is None:
        raise RuntimeError("OpenRouter rerank returned no payload")

    scores: list[float | None] = [None] * len(documents)
    for result in payload.get("results", []):
        index = int(result["index"])
        if not 0 <= index < len(documents):
            raise RuntimeError(f"OpenRouter rerank returned invalid index {index}")
        scores[index] = float(result["relevance_score"])
    if any(score is None for score in scores):
        raise RuntimeError(
            f"OpenRouter rerank returned {len(payload.get('results', []))} scores "
            f"for {len(documents)} documents"
        )
    response_usage = payload.get("usage") or {}
    _usage["calls"] += 1
    _usage["search_units"] += int(response_usage.get("search_units") or 0)
    _usage["total_tokens"] += int(response_usage.get("total_tokens") or 0)
    return [float(score) for score in scores]


def rerank_scores(question: str, texts: list[str]) -> list[float]:
    """Cross-encoder scores aligned with the original document order."""
    if not texts:
        return []
    missing = list(
        dict.fromkeys(
            text
            for text in texts
            if (_backend, _model_name, question, text) not in _score_cache
        )
    )
    if missing:
        if _backend == "local":
            pairs = [[question, text] for text in missing]
            raw_scores = _get_local_model().predict(
                pairs, batch_size=min(16, len(pairs))
            )
            scores = [float(score) for score in raw_scores]
        else:
            scores = _openrouter_scores(question, missing)
        _score_cache.update(
            {
                (_backend, _model_name, question, text): score
                for text, score in zip(missing, scores)
            }
        )
    return [
        _score_cache[(_backend, _model_name, question, text)] for text in texts
    ]
