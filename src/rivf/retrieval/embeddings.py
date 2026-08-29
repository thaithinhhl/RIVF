import atexit
import hashlib
import os
import pickle
from pathlib import Path

import numpy as np
from dotenv import load_dotenv


LOCAL_MODEL_NAME = "BAAI/bge-m3"
OPENROUTER_MODEL_NAME = "baai/bge-m3"
MODEL_NAME = LOCAL_MODEL_NAME  # Backward-compatible public constant.
_CACHE_PATH = Path("data/processed/embedding_cache.pkl")

_backend = os.environ.get("RIVF_EMBEDDING_BACKEND", "local")
_model_name = os.environ.get("RIVF_EMBEDDING_MODEL", LOCAL_MODEL_NAME)
_model = None
_client = None
_cache: dict[str, np.ndarray] = {}
_cache_dirty = False
_usage = {"calls": 0, "prompt_tokens": 0, "total_tokens": 0}


def configure(backend: str = "local", model: str | None = None) -> None:
    """Select the embedding backend before the first experiment row runs."""
    global _backend, _model_name, _model, _client
    if backend not in {"local", "openrouter"}:
        raise ValueError("embedding backend must be 'local' or 'openrouter'")
    resolved_model = model or (
        OPENROUTER_MODEL_NAME if backend == "openrouter" else LOCAL_MODEL_NAME
    )
    if (_backend, _model_name) != (backend, resolved_model):
        _model = None
        _client = None
    _backend = backend
    _model_name = resolved_model


def backend_name() -> str:
    return _backend


def model_name() -> str:
    return _model_name


def usage() -> dict[str, int]:
    return dict(_usage)


def _cache_key(text: str) -> str:
    # Backend/model are part of the identity: vectors from different services
    # must never silently share the historical text-only cache entries.
    payload = f"v2\0{_backend}\0{_model_name}\0{text}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_disk_cache() -> None:
    if _CACHE_PATH.exists():
        with _CACHE_PATH.open("rb") as f:
            _cache.update(pickle.load(f))


def _save_disk_cache() -> None:
    if not _cache_dirty:
        return
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_PATH.open("wb") as f:
        pickle.dump(_cache, f)


_load_disk_cache()
atexit.register(_save_disk_cache)


def _get_local_model():
    global _model
    if _model is None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "local embedding backend requires the 'local-models' extra"
            ) from exc
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        _model = SentenceTransformer(_model_name, device=device)
    return _model


def _get_openrouter_client():
    global _client
    if _client is None:
        load_dotenv()
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter embeddings")
        from openai import OpenAI

        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
            max_retries=5,
        )
    return _client


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _encode(texts: list[str]) -> list[np.ndarray]:
    if _backend == "local":
        vectors = _get_local_model().encode(
            texts, normalize_embeddings=True, batch_size=32
        )
        return [np.asarray(vector, dtype=np.float32) for vector in vectors]

    response = _get_openrouter_client().embeddings.create(
        model=_model_name,
        input=texts,
        encoding_format="float",
    )
    ordered = sorted(response.data, key=lambda item: item.index)
    response_usage = getattr(response, "usage", None)
    _usage["calls"] += 1
    if response_usage is not None:
        _usage["prompt_tokens"] += int(
            getattr(response_usage, "prompt_tokens", 0) or 0
        )
        _usage["total_tokens"] += int(
            getattr(response_usage, "total_tokens", 0) or 0
        )
    return [
        _normalize(np.asarray(item.embedding, dtype=np.float32)) for item in ordered
    ]


def embed_batch(texts: list[str]) -> list[np.ndarray]:
    """Embed texts, caching vectors by backend, model and normalized input."""
    global _cache_dirty
    keys = [_cache_key(text) for text in texts]
    unique_missing = {
        text: key for text, key in zip(texts, keys) if key not in _cache
    }
    if unique_missing:
        missing_texts = list(unique_missing)
        vectors = _encode(missing_texts)
        if len(vectors) != len(missing_texts):
            raise RuntimeError(
                f"embedding backend returned {len(vectors)} vectors for "
                f"{len(missing_texts)} inputs"
            )
        for text, vector in zip(missing_texts, vectors):
            _cache[unique_missing[text]] = vector
        _cache_dirty = True
    return [_cache[key] for key in keys]


def embed(text: str) -> np.ndarray:
    return embed_batch([text])[0]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
