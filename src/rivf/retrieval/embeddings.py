import atexit
import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "BAAI/bge-m3"
_CACHE_PATH = Path("data/processed/embedding_cache.pkl")

_model: SentenceTransformer | None = None
_cache: dict[str, np.ndarray] = {}
_cache_dirty = False


def _cache_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


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


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _model = SentenceTransformer(_MODEL_NAME, device=device)
    return _model


def embed_batch(texts: list[str]) -> list[np.ndarray]:
    """Embeddings for a list of texts, in order. Cached both in-process and
    on disk (keyed by text hash) so the same sentence -- reused as a
    distractor across many questions, or re-encountered in a later
    experiment run entirely -- is only ever embedded once."""
    global _cache_dirty
    keys = [_cache_key(t) for t in texts]
    to_encode = [t for t, k in dict(zip(texts, keys)).items() if k not in _cache]
    if to_encode:
        vecs = _get_model().encode(to_encode, normalize_embeddings=True, batch_size=32)
        for t, v in zip(to_encode, vecs):
            _cache[_cache_key(t)] = v
        _cache_dirty = True
    return [_cache[k] for k in keys]


def embed(text: str) -> np.ndarray:
    return embed_batch([text])[0]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
