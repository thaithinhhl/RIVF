from types import SimpleNamespace

import numpy as np
import yaml

from rivf.generation import llm_client
from rivf.retrieval import embeddings, reranker


def test_openrouter_embeddings_are_aligned_normalized_and_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(embeddings, "_backend", embeddings._backend)
    monkeypatch.setattr(embeddings, "_model_name", embeddings._model_name)
    monkeypatch.setattr(embeddings, "_model", embeddings._model)
    monkeypatch.setattr(embeddings, "_client", embeddings._client)

    class FakeEmbeddings:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                # Deliberately reverse response order to exercise index alignment.
                data=[
                    SimpleNamespace(index=1, embedding=[0.0, 2.0]),
                    SimpleNamespace(index=0, embedding=[3.0, 0.0]),
                ],
                usage=SimpleNamespace(prompt_tokens=4, total_tokens=4),
            )

    monkeypatch.setattr(
        embeddings,
        "_get_openrouter_client",
        lambda: SimpleNamespace(embeddings=FakeEmbeddings()),
    )
    monkeypatch.setattr(embeddings, "_cache", {})
    embeddings.configure("openrouter", "baai/bge-m3-test")

    first = embeddings.embed_batch(["alpha", "beta"])
    second = embeddings.embed_batch(["alpha", "beta"])

    assert np.allclose(first[0], [1.0, 0.0])
    assert np.allclose(first[1], [0.0, 1.0])
    assert all(np.allclose(a, b) for a, b in zip(first, second))
    assert len(calls) == 1


def test_openrouter_reranker_preserves_input_order_and_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(reranker, "_backend", reranker._backend)
    monkeypatch.setattr(reranker, "_model_name", reranker._model_name)
    monkeypatch.setattr(reranker, "_model", reranker._model)

    def fake_scores(query, documents):
        calls.append((query, documents))
        return [0.2, 0.9]

    monkeypatch.setattr(reranker, "_openrouter_scores", fake_scores)
    monkeypatch.setattr(reranker, "_score_cache", {})
    reranker.configure("openrouter", "qwen/qwen3-reranker-8b-test")

    assert reranker.rerank_scores("q", ["a", "b"]) == [0.2, 0.9]
    assert reranker.rerank_scores("q", ["b", "a"]) == [0.9, 0.2]
    assert len(calls) == 1


def test_openrouter_generation_records_backend_and_usage(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=2,
            total_tokens=12,
            model_extra={"cost": 0.001},
        ),
        id="gen-1",
        model="openai/gpt-4o-mini-2024-07-18",
        system_fingerprint="fp",
        model_extra={"provider": "OpenAI"},
    )
    completions = SimpleNamespace(create=lambda **kwargs: response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(llm_client, "_get_client", lambda backend: client)

    result = llm_client.generate_answer_with_metadata(
        "prompt",
        model="openai/gpt-4o-mini-2024-07-18",
        backend="openrouter",
    )

    assert result["prediction"] == "answer"
    assert result["api_backend"] == "openrouter"
    assert result["api_provider"] == "OpenAI"
    assert result["api_total_tokens"] == 12
    assert result["api_cost_usd"] == 0.001


def test_hotpot_openrouter_rq1_config_has_five_paired_rows():
    config = yaml.safe_load(
        open("configs/graft_v1_hotpot_rq1_openrouter_n1000.yaml")
    )
    assert config["n_questions"] == 1000
    assert config["backend"] == "openrouter"
    assert config["embedding_model"] == "baai/bge-m3"
    assert config["reranker_model"] == "qwen/qwen3-reranker-8b"
    assert config["llm_model"] == "openai/gpt-4o-mini-2024-07-18"
    assert sum(
        len(method["budget_tokens"])
        if isinstance(method["budget_tokens"], list)
        else 1
        for method in config["methods"]
    ) == 5
