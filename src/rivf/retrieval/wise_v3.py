"""WISE-v3: minimum-cost claim-complete witness graph construction.

The implementation deliberately has one inference policy. It induces one
reasoning plan, verifies claim assignments and dependency paths, and solves a
minimum serialized-cost constrained search problem. Invalid plans, uncovered
claims, unverified dependencies, infeasible budgets, and exhausted searches
raise :class:`WiseV3Error`; none of them switches to another retriever or
selection rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice, product
import json
import math
import re
from typing import Callable

import networkx as nx

from rivf.data.schema import QAItem, Sentence
from rivf.eval.efficiency import count_tokens
from rivf.generation.llm_client import (
    StructuredGenerationError,
    generate_json_with_metadata,
)
from rivf.generation.prompts import build_context_compact_selection_order
from rivf.graph.build import bfs_hop_distances_with_parents
from rivf.graph.entities import (
    canonical_entity_key,
    entity_alias_match,
    extract_entities,
    mentions_title,
    normalize_text,
)
from rivf.methods.base import Candidate
from rivf.retrieval.embeddings import cosine_sim, embed_batch
from rivf.retrieval.reranker import rerank_scores


EDGE_SAME_PASSAGE_NEXT = "same_passage_next"
EDGE_SAME_PASSAGE_PREVIOUS = "same_passage_previous"
EDGE_ENTITY_COREFERENCE = "entity_coreference"
EDGE_TITLE_MENTION = "mentions_title"


class WiseV3Error(RuntimeError):
    """A typed, serializable WISE-v3 inference failure."""

    def __init__(self, code: str, message: str, **details):
        self.code = code
        self.details = details
        super().__init__(f"WISE-v3 {code}: {message}")

    def to_dict(self) -> dict:
        return {"code": self.code, "message": str(self), "details": self.details}


@dataclass(frozen=True)
class Claim:
    claim_id: str
    subject: str
    relation: str
    object_value: str
    query: str
    source_title: str | None

    def to_dict(self) -> dict:
        return {
            "id": self.claim_id,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object_value,
            "query": self.query,
            "source_title": self.source_title,
        }


@dataclass(frozen=True)
class Dependency:
    source_claim: str
    target_claim: str
    binding: str

    def to_dict(self) -> dict:
        return {
            "from": self.source_claim,
            "to": self.target_claim,
            "binding": self.binding,
        }


@dataclass(frozen=True)
class ReasoningPlan:
    claims: tuple[Claim, ...]
    dependencies: tuple[Dependency, ...]
    operation: dict

    @property
    def by_id(self) -> dict[str, Claim]:
        return {claim.claim_id: claim for claim in self.claims}

    def topological_claim_ids(self) -> tuple[str, ...]:
        graph = nx.DiGraph()
        graph.add_nodes_from(claim.claim_id for claim in self.claims)
        graph.add_edges_from(
            (dependency.source_claim, dependency.target_claim)
            for dependency in self.dependencies
        )
        return tuple(nx.lexicographical_topological_sort(graph, key=str))

    def to_dict(self) -> dict:
        return {
            "claims": [claim.to_dict() for claim in self.claims],
            "dependencies": [dependency.to_dict() for dependency in self.dependencies],
            "operation": self.operation,
        }


@dataclass(frozen=True)
class ClaimAssignment:
    claim_id: str
    sentence_key: tuple[str, int]
    raw_score: float
    verifier_confidence: float
    confidence: float
    bindings: tuple[tuple[str, str], ...]
    evidence_quote: str

    @property
    def binding_map(self) -> dict[str, str]:
        return dict(self.bindings)


@dataclass(frozen=True)
class VerifiedDependency:
    dependency: Dependency
    source_key: tuple[str, int]
    target_key: tuple[str, int]
    path_keys: tuple[tuple[str, int], ...]
    raw_score: float
    verifier_confidence: float
    confidence: float
    binding_value: str
    path_rank: int


@dataclass(frozen=True)
class Witness:
    plan: ReasoningPlan
    candidates: tuple[Candidate, ...]
    selected: tuple[Sentence, ...]
    assignments: tuple[ClaimAssignment, ...]
    dependencies: tuple[VerifiedDependency, ...]
    serialized_tokens: int
    bottleneck_confidence: float
    sufficiency_threshold: float
    search_states: int
    planner_metadata: dict

    def diagnostics(self) -> dict:
        return {
            "wise_plan": self.plan.to_dict(),
            "wise_claim_assignments": [
                {
                    "claim_id": assignment.claim_id,
                    "sentence_key": list(assignment.sentence_key),
                    "raw_score": assignment.raw_score,
                    "verifier_confidence": assignment.verifier_confidence,
                    "confidence": assignment.confidence,
                    "bindings": dict(assignment.bindings),
                    "evidence_quote": assignment.evidence_quote,
                }
                for assignment in self.assignments
            ],
            "wise_verified_dependencies": [
                {
                    **dependency.dependency.to_dict(),
                    "source_key": list(dependency.source_key),
                    "target_key": list(dependency.target_key),
                    "path_keys": [list(key) for key in dependency.path_keys],
                    "binding_value": dependency.binding_value,
                    "path_rank": dependency.path_rank,
                    "raw_score": dependency.raw_score,
                    "verifier_confidence": dependency.verifier_confidence,
                    "confidence": dependency.confidence,
                }
                for dependency in self.dependencies
            ],
            "wise_serialized_tokens": self.serialized_tokens,
            "wise_evidence_representation": "verified_atomic_quotes",
            "wise_selected_evidence": [
                {
                    "sentence_key": list(sentence.key),
                    "text": sentence.text,
                }
                for sentence in self.selected
            ],
            "wise_bottleneck_confidence": self.bottleneck_confidence,
            "wise_sufficiency_threshold": self.sufficiency_threshold,
            "wise_search_states": self.search_states,
            "wise_operation_complete": True,
            "wise_certified": True,
            "wise_irreducible": True,
            "wise_planner": self.planner_metadata,
        }


def planner_prompt(question: str, passage_titles: list[str]) -> str:
    """Strict claim/operation contract used by WISE-v3's label-free planner."""
    titles = json.dumps(passage_titles, ensure_ascii=False)
    explicit_titles = [
        title for title in passage_titles if mentions_title(question, title)
    ]
    grounded = json.dumps(explicit_titles, ensure_ascii=False)
    return f"""Decompose the question into atomic evidence claims and one final operation.
Return exactly one JSON object with these keys:
{{
  "claims": [
    {{
      "id": "c1",
      "subject": "an entity name or ?variable",
      "relation": "a concise normalized relation",
      "object": "an entity, value, or ?variable",
      "query": "a self-contained evidence retrieval statement"
    }}
  ],
  "operation": {{
    "type": "lookup|compare|boolean|intersection",
    "operator": "...",
    "inputs": ["claim ids consumed by the final operation"],
    "output": "?answer"
  }}
}}

Rules:
- Produce atomic evidence claims only; the final comparison/boolean operation is not a claim.
- Use one claim for each independently required fact.
- For a comparison, operation.inputs must be terminal claims that retrieve the
  actual comparable attribute requested by the question, such as birth date
  for younger/older. Never compare intermediate entity identities.
- For film questions asking `released first`, `released earlier`, or release
  order, retrieve each film's release year (not a full release date) unless the
  question explicitly requires day-level dates. A Wikipedia lead of the form
  `Film is a YEAR ... film` directly supplies that release year.
- If a comparable attribute belongs to an entity reached through another
  relation, create both claims and reuse the exact variable. Example: film ->
  ?director, then ?director -> ?birth_date; compare the birth-date claims.
- Decompose kinship composition in the same way. `paternal grandfather of A`
  is A ->father-> ?father, then ?father ->father-> ?answer; `maternal
  grandfather` uses mother then father. Never create a direct grandfather
  claim when the supplied passages expose the two parent relations.
- Decompose in-law and step relations rather than treating them as atomic:
  `father-in-law of A` is A ->spouse-> ?spouse, then
  ?spouse ->father-> ?answer; `mother-in-law` analogously ends with mother;
  `stepchild of A` is A ->spouse-> ?spouse, then
  ?spouse ->child-> ?answer.
- Variables begin with "?" and must be reused exactly from a source claim's
  object as the dependent claim's subject. WISE deterministically induces the
  dependency edge from this equality; do not output a dependencies field.
- Every ?variable subject must exactly reuse one earlier claim's object.
- Keep parallel chains paired. For A and B use c1: A -> ?entity1,
  c2: B -> ?entity2, c3: ?entity1 -> ?value1, and
  c4: ?entity2 -> ?value2. Never cross the numbered variables.
- Preserve the left-to-right order of alternatives as written in the question
  when constructing parallel chains and operation.inputs.
- operation.inputs must list every and only terminal (sink) claim in the DAG.
- Every terminal claim consumed by operation.inputs must retrieve its value
  into a ?variable; never place a guessed date, country, person, or answer
  constant in a terminal claim's object.
- lookup has exactly one input; compare and intersection have at least two.
- For non-lookup operations, operation.output must be exactly `?answer`.
- For lookup, operation.inputs must contain only the final claim and
  operation.output must exactly equal that final claim's object variable.
- Use the exact explicitly grounded title below as a named claim subject when
  available. WISE deterministically grounds named subjects to passage titles;
  do not output a source_title field.
- Do not use dataset labels, answers, or facts not present in the question.
- Do not output markdown or explanatory text.

Available passage titles:
{titles}

Titles explicitly grounded by the question:
{grounded}

Question:
{question}"""


def parse_reasoning_plan(payload: dict, passage_titles: set[str]) -> ReasoningPlan:
    """Validate claims/operation and deterministically induce grounding/DAG."""
    if not isinstance(payload, dict):
        raise WiseV3Error("plan_invalid", "planner output must be a JSON object")
    if set(payload) != {"claims", "operation"}:
        raise WiseV3Error(
            "plan_invalid",
            "planner output must contain exactly claims and operation",
            keys=sorted(payload),
        )
    raw_claims = payload["claims"]
    operation = payload["operation"]
    if not isinstance(raw_claims, list) or not raw_claims:
        raise WiseV3Error("plan_invalid", "claims must be a non-empty list")
    if not isinstance(operation, dict) or set(operation) != {
        "type",
        "operator",
        "inputs",
        "output",
    }:
        raise WiseV3Error(
            "plan_invalid", "operation must follow the exact operation schema"
        )
    if operation["type"] not in {"lookup", "compare", "boolean", "intersection"}:
        raise WiseV3Error(
            "plan_invalid", "operation type is not supported", operation=operation
        )
    if not isinstance(operation["operator"], str) or not operation["operator"].strip():
        raise WiseV3Error(
            "plan_invalid", "operation operator must be a non-empty string"
        )
    if (
        not isinstance(operation["inputs"], list)
        or not operation["inputs"]
        or any(not isinstance(value, str) for value in operation["inputs"])
        or len(operation["inputs"]) != len(set(operation["inputs"]))
    ):
        raise WiseV3Error(
            "plan_invalid", "operation inputs must be a non-empty unique claim-id list"
        )
    if not isinstance(operation["output"], str) or not operation["output"].startswith("?"):
        raise WiseV3Error("plan_invalid", "operation output must be a ?variable")

    claims = []
    for index, raw in enumerate(raw_claims):
        required = {"id", "subject", "relation", "object", "query"}
        if not isinstance(raw, dict) or set(raw) != required:
            raise WiseV3Error(
                "plan_invalid",
                "each claim must follow the exact claim schema",
                claim_index=index,
            )
        string_fields = ("id", "subject", "relation", "object", "query")
        if any(
            not isinstance(raw[field], str) or not raw[field].strip()
            for field in string_fields
        ):
            raise WiseV3Error(
                "plan_invalid", "claim string fields must be non-empty", claim_index=index
            )
        subject = raw["subject"].strip()
        source_title = None
        if not subject.startswith("?"):
            exact_titles = sorted(
                title
                for title in passage_titles
                if normalize_text(title) == normalize_text(subject)
            )
            alias_titles = sorted(
                title
                for title in passage_titles
                if entity_alias_match(title, subject)
            )
            if len(exact_titles) == 1:
                source_title = exact_titles[0]
            elif not exact_titles and len(alias_titles) == 1:
                source_title = alias_titles[0]
        claims.append(
            Claim(
                claim_id=raw["id"].strip(),
                subject=subject,
                relation=raw["relation"].strip(),
                object_value=raw["object"].strip(),
                query=raw["query"].strip(),
                source_title=source_title,
            )
        )
    claim_ids = [claim.claim_id for claim in claims]
    if len(claim_ids) != len(set(claim_ids)):
        raise WiseV3Error("plan_invalid", "claim ids must be unique")
    by_id = {claim.claim_id: claim for claim in claims}

    dependencies = []
    for target_index, target in enumerate(claims):
        if not target.subject.startswith("?"):
            continue
        sources = [
            source
            for source in claims[:target_index]
            if source.object_value == target.subject
        ]
        if len(sources) != 1:
            raise WiseV3Error(
                "plan_invalid",
                "a variable-subject claim must reuse exactly one earlier object variable",
                claim_id=target.claim_id,
                subject=target.subject,
                matching_sources=[source.claim_id for source in sources],
            )
        dependencies.append(
            Dependency(sources[0].claim_id, target.claim_id, target.subject)
        )

    canonical_operation = dict(operation)
    if canonical_operation["type"] != "lookup":
        # The comparison/boolean result is not an evidence binding. Its name
        # carries no information, so do not let an arbitrary branch variable
        # bias the answer stage.
        canonical_operation["output"] = "?answer"
    plan = ReasoningPlan(tuple(claims), tuple(dependencies), canonical_operation)
    graph = nx.DiGraph()
    graph.add_nodes_from(claim_ids)
    graph.add_edges_from(
        (dependency.source_claim, dependency.target_claim)
        for dependency in dependencies
    )
    if not nx.is_directed_acyclic_graph(graph):
        raise WiseV3Error("plan_invalid", "claim dependencies must form a DAG")
    sink_claims = {claim_id for claim_id in claim_ids if graph.out_degree(claim_id) == 0}
    operation_inputs = set(operation["inputs"])
    if operation_inputs != sink_claims:
        raise WiseV3Error(
            "plan_invalid",
            "operation inputs must equal all DAG sink claims",
            expected=sorted(sink_claims),
            actual=sorted(operation_inputs),
        )
    if any(not by_id[claim_id].object_value.startswith("?") for claim_id in operation_inputs):
        raise WiseV3Error(
            "plan_invalid",
            "every operation input claim must retrieve a variable object",
        )
    operation_type = operation["type"]
    if operation_type == "lookup" and len(operation_inputs) != 1:
        raise WiseV3Error("plan_invalid", "lookup operation requires exactly one input")
    if operation_type == "lookup":
        terminal_claim = by_id[operation["inputs"][0]]
        if terminal_claim.object_value != operation["output"]:
            raise WiseV3Error(
                "plan_invalid",
                "lookup output must equal its terminal claim object variable",
            )
    if operation_type in {"compare", "intersection"} and len(operation_inputs) < 2:
        raise WiseV3Error(
            "plan_invalid", f"{operation_type} operation requires at least two inputs"
        )
    return plan


def induce_reasoning_plan(
    item: QAItem,
    *,
    model: str,
    backend: str,
    generator: Callable[..., dict] = generate_json_with_metadata,
) -> tuple[ReasoningPlan, dict]:
    titles = [passage.title for passage in item.passages]
    try:
        generated = generator(
            planner_prompt(item.question, titles), model=model, backend=backend
        )
    except StructuredGenerationError as exc:
        raise WiseV3Error("plan_invalid", str(exc)) from exc
    if "payload" not in generated:
        raise WiseV3Error("plan_invalid", "planner generator returned no payload")
    try:
        plan = parse_reasoning_plan(generated["payload"], set(titles))
    except WiseV3Error as exc:
        raise WiseV3Error(
            exc.code,
            "planner output violated the strict reasoning-plan contract",
            validation_error=str(exc),
            validation_details=exc.details,
            planner_output=generated["payload"],
        ) from exc
    metadata = {key: value for key, value in generated.items() if key != "payload"}
    metadata["requested_model"] = model
    return plan, metadata


def _add_typed_edge(
    graph: nx.DiGraph,
    source: tuple[str, int],
    target: tuple[str, int],
    kind: str,
) -> None:
    previous = graph.get_edge_data(source, target, {}).get("kinds", ())
    graph.add_edge(source, target, kinds=tuple(sorted({*previous, kind})))


def build_typed_evidence_graph(item: QAItem) -> nx.DiGraph:
    """Build WISE-v3's directed, typed sentence graph."""
    graph = nx.DiGraph()
    sentences = item.all_sentences()
    by_title = {passage.title: passage.sentences for passage in item.passages}
    for sentence in sentences:
        graph.add_node(sentence.key, sentence=sentence)

    for passage in item.passages:
        for left, right in zip(passage.sentences, passage.sentences[1:]):
            _add_typed_edge(graph, left.key, right.key, EDGE_SAME_PASSAGE_NEXT)
            _add_typed_edge(graph, right.key, left.key, EDGE_SAME_PASSAGE_PREVIOUS)

    entities = {sentence.key: extract_entities(sentence.text) for sentence in sentences}
    for index, left in enumerate(sentences):
        for right in sentences[index + 1 :]:
            if left.passage_title == right.passage_title:
                continue
            if entities[left.key] & entities[right.key]:
                _add_typed_edge(graph, left.key, right.key, EDGE_ENTITY_COREFERENCE)
                _add_typed_edge(graph, right.key, left.key, EDGE_ENTITY_COREFERENCE)

    for sentence in sentences:
        for title, target_sentences in by_title.items():
            if title == sentence.passage_title or not mentions_title(sentence.text, title):
                continue
            for target in target_sentences:
                _add_typed_edge(graph, sentence.key, target.key, EDGE_TITLE_MENTION)
    return graph


def propose_candidates(
    item: QAItem,
    plan: ReasoningPlan,
    graph: nx.DiGraph,
    *,
    seed_n: int,
    max_hops: int,
) -> list[Candidate]:
    """Claim-aware semantic seeds followed by one bounded graph expansion."""
    if seed_n <= 0 or max_hops < 0:
        raise WiseV3Error(
            "configuration_invalid", "seed_n must be positive and max_hops non-negative"
        )
    sentences = item.all_sentences()
    if not sentences:
        raise WiseV3Error("candidate_empty", "question contains no candidate sentences")
    vectors = embed_batch([sentence.embedding_text for sentence in sentences])
    queries = [item.question, *(claim.query for claim in plan.claims)]
    query_vectors = embed_batch(queries)
    seed_keys: set[tuple[str, int]] = set()
    seed_scores: dict[tuple[str, int], float] = {}
    for query_vector in query_vectors:
        scored = sorted(
            (
                (cosine_sim(query_vector, vector), sentence.key)
                for sentence, vector in zip(sentences, vectors)
            ),
            reverse=True,
        )
        for score, key in scored[:seed_n]:
            seed_keys.add(key)
            seed_scores[key] = max(seed_scores.get(key, float("-inf")), score)

    grounded_titles = {
        claim.source_title for claim in plan.claims if claim.source_title is not None
    }
    seed_keys.update(
        sentence.key
        for sentence in sentences
        if sentence.passage_title in grounded_titles
    )
    distances, parents = bfs_hop_distances_with_parents(
        graph, sorted(seed_keys)
    )
    included = {
        key: distance for key, distance in distances.items() if distance <= max_hops
    }
    by_key = {sentence.key: sentence for sentence in sentences}
    vector_by_key = {
        sentence.key: vector for sentence, vector in zip(sentences, vectors)
    }

    def path_to(key: tuple[str, int]) -> tuple[tuple[str, int], ...]:
        path = []
        current: tuple[str, int] | None = key
        while current is not None:
            path.append(current)
            current = parents.get(current)
        return tuple(reversed(path))

    candidates = [
        Candidate(
            sentence=by_key[key],
            semantic_score=seed_scores.get(key),
            graph_score=1.0 / (1 + distance),
            hop_distance=distance,
            embedding=vector_by_key[key],
            path_keys=path_to(key),
            neighbor_keys=tuple(
                neighbor for neighbor in graph.successors(key) if neighbor in included
            ),
        )
        for key, distance in sorted(included.items())
    ]
    if not candidates:
        raise WiseV3Error("candidate_empty", "graph proposal produced no candidates")
    return candidates


def _calibrate_confidence(confidence: float, temperature: float) -> float:
    if not 0.0 <= confidence <= 1.0 or temperature <= 0.0:
        raise WiseV3Error(
            "verification_invalid", "verifier confidence or temperature is invalid"
        )
    clipped = min(max(confidence, 1e-6), 1.0 - 1e-6)
    logit = math.log(clipped / (1.0 - clipped)) / temperature
    return 1.0 / (1.0 + math.exp(-logit))


def _structured_stage(
    prompt: str,
    *,
    model: str,
    backend: str,
    error_code: str,
    generator: Callable[..., dict],
) -> tuple[dict, dict]:
    try:
        generated = generator(prompt, model=model, backend=backend)
    except StructuredGenerationError as exc:
        raise WiseV3Error(error_code, str(exc)) from exc
    if not isinstance(generated, dict) or not isinstance(generated.get("payload"), dict):
        raise WiseV3Error(error_code, "structured verifier returned no JSON object")
    metadata = {key: value for key, value in generated.items() if key != "payload"}
    metadata["requested_model"] = model
    return generated["payload"], metadata


def _claim_variables(claim: Claim) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            value
            for value in (claim.subject, claim.object_value)
            if value.startswith("?")
        )
    )


def _project_exact_quote(
    source_text: str,
    proposed_quote: str,
    required_values: list[str],
) -> str | None:
    """Project a verifier quote onto one exact contiguous source span.

    The verifier may omit parenthetical material while copying a clause. We
    align every proposed token and every required value token monotonically to
    the original sentence, then retain the source substring spanning them all.
    A paraphrase with tokens absent or out of order is rejected.
    """
    source_matches = list(re.finditer(r"\w+", source_text, flags=re.UNICODE))
    source_tokens = [
        canonical_entity_key(match.group(0)) for match in source_matches
    ]
    if not proposed_quote.strip() or not source_tokens:
        return None

    def align(value: str) -> list[int] | None:
        tokens = [
            canonical_entity_key(match.group(0))
            for match in re.finditer(r"\w+", value, flags=re.UNICODE)
        ]
        tokens = [token for token in tokens if token]
        if not tokens:
            return None
        indices = []
        cursor = 0
        for token in tokens:
            try:
                index = source_tokens.index(token, cursor)
            except ValueError:
                return None
            indices.append(index)
            cursor = index + 1
        return indices

    aligned_groups = [align(proposed_quote), *(align(value) for value in required_values)]
    if any(group is None for group in aligned_groups):
        return None
    indices = [index for group in aligned_groups for index in group or ()]
    start = source_matches[min(indices)].start()
    end = source_matches[max(indices)].end()
    return source_text[start:end]


def _claim_verifier_prompt(tasks: list[dict]) -> str:
    payload = json.dumps(tasks, ensure_ascii=False)
    return f"""Judge whether each evidence sentence entails its atomic claim.
Return exactly one JSON object:
{{"judgments": [{{
  "task_id": "t0",
  "label": "entailed|not_entailed",
  "confidence": 0.0,
  "bindings": {{"?variable": "exact entity or value stated by the evidence"}},
  "evidence_quote": "shortest exact contiguous quote that entails the claim"
}}]}}

Rules:
- Return every entailed task exactly once. Negative tasks may be returned as
  not_entailed or omitted. Never return an unknown or duplicate task_id.
- Entailed means the sentence itself, together with its passage title, asserts
  the requested subject-relation-object fact; topical relevance is insufficient.
- Interpret standard Wikipedia lead conventions as explicit assertions:
  `Person (birth date, birth place – death date, death place)` states the birth
  date/place and death date/place; `Person (birth date – death date)` states
  both dates; `Film is a YEAR ... film` states its release year; and a demonym
  or phrase such as `American group` or `based in South Korea` states the
  country/national origin. In particular, `British film`, `Canadian film`,
  `Bollywood filmmaker`, and `active in Indian cinema` entail the corresponding
  origin/country relation. A parenthetical pair such as
  `(February 24, 1896 – May 1, 1991)` entails both birth and death dates.
- For an entailed task, bindings must contain exactly every required variable.
- Copy every supplied binding_constraint exactly into bindings. The instantiated
  claim replaces constrained variables with their concrete entity values.
- Every variable in link_binding_variables transports one entity to another
  claim. Resolve it to exactly one entity surface, never a coordinated list.
  Its alias must match one supplied linkable_title (parenthetical
  disambiguators, accents, punctuation, and spacing may differ).
- Resolve entity variables transported to another claim to an exact entity
  surface explicitly present in the evidence sentence or its passage title.
  A terminal attribute may instead use its canonical typed value when the
  evidence expresses it by a standard morphological form (for example
  `Canadian` -> `Canada`). Never emit unknown, N/A, or unsupported knowledge.
- For entailed, evidence_quote must be a non-empty exact contiguous substring
  copied from evidence.text. Use the shortest self-contained clause that, with
  the supplied title, asserts the requested relation and its resolved object.
  Do not paraphrase, normalize, or join non-contiguous spans.
- Example: subject=?person, relation=birth date, object=?date, constraint
  ?person=Alice, evidence "Alice (born 1 January 1900)" requires bindings
  {{"?person":"Alice", "?date":"1 January 1900"}} and an exact quote such as
  "Alice (born 1 January 1900)". Omitting ?date is invalid.
- For not_entailed, bindings must be an empty object; evidence_quote may be
  omitted or must be "". For entailed, evidence_quote is mandatory.
- Confidence is the probability that the entailment label is correct.
- Do not use outside knowledge and do not output markdown.

Tasks:
{payload}"""


def verify_claim_assignments(
    plan: ReasoningPlan,
    candidates: list[Candidate],
    *,
    threshold: float,
    top_k: int,
    prefilter_k: int,
    model: str,
    backend: str,
    confidence_temperature: float,
    generator: Callable[..., dict] = generate_json_with_metadata,
) -> tuple[dict[str, list[ClaimAssignment]], dict]:
    """CE proposes claim pairs; a structured verifier decides entailment."""
    if (
        not 0.0 <= threshold <= 1.0
        or top_k <= 0
        or prefilter_k < top_k
        or confidence_temperature <= 0.0
    ):
        raise WiseV3Error(
            "configuration_invalid",
            "claim verifier thresholds, sizes, or temperature are invalid",
        )
    all_raw_by_key: dict[tuple[str, int], list[float]] = {
        candidate.sentence.key: [] for candidate in candidates
    }
    assignments: dict[str, list[ClaimAssignment]] = {
        claim.claim_id: [] for claim in plan.claims
    }
    incoming: dict[str, list[Dependency]] = {
        claim.claim_id: [] for claim in plan.claims
    }
    for dependency in plan.dependencies:
        incoming[dependency.target_claim].append(dependency)

    grounding_rejections = 0
    stage_metadata: list[dict] = []
    claims = plan.by_id
    dependency_bindings = {
        dependency.binding for dependency in plan.dependencies
    }
    terminal_variables = {
        claims[claim_id].object_value for claim_id in plan.operation["inputs"]
    }
    forbidden_values = {
        "unknown",
        "n/a",
        "na",
        "not available",
        "not provided",
        "not specified",
        "not stated",
        "none",
    }

    # Verify in DAG order. A dependent claim may only inspect passages whose
    # title is a concrete binding already established by every incoming claim.
    # This prevents a later hop from independently attaching to a distractor.
    for claim_id in plan.topological_claim_ids():
        claim = claims[claim_id]
        dependencies = incoming[claim_id]
        allowed_subjects: set[str] | None = None
        for dependency in dependencies:
            resolved = {
                value.strip()
                for assignment in assignments[dependency.source_claim]
                if (value := assignment.binding_map.get(dependency.binding))
            }
            allowed_subjects = (
                resolved
                if allowed_subjects is None
                else {
                    left
                    for left in allowed_subjects
                    if any(entity_alias_match(left, right) for right in resolved)
                }
            )

        eligible = [
            candidate
            for candidate in candidates
            if (
                claim.source_title is None
                or candidate.sentence.passage_title == claim.source_title
            )
            and (
                allowed_subjects is None
                or any(
                    entity_alias_match(
                        candidate.sentence.passage_title, resolved_subject
                    )
                    for resolved_subject in allowed_subjects
                )
            )
        ]
        if not eligible:
            raise WiseV3Error(
                "claim_uncovered",
                "no proposed sentence satisfies the claim's grounded subject",
                claim_id=claim.claim_id,
                source_title=claim.source_title,
                resolved_subjects=sorted(allowed_subjects or ()),
            )

        raw_scores = rerank_scores(
            claim.query, [candidate.sentence.embedding_text for candidate in eligible]
        )
        scored = list(zip(eligible, (float(score) for score in raw_scores)))
        ranked = sorted(
            scored,
            key=lambda item: (item[1], item[0].sentence.key),
            reverse=True,
        )[:prefilter_k]
        for candidate, raw_score in scored:
            all_raw_by_key[candidate.sentence.key].append(raw_score)

        task_lookup = {}
        tasks = []
        variables = _claim_variables(claim)
        for candidate, raw_score in ranked:
            task_id = f"t{len(tasks)}"
            link_binding_variables = sorted(
                variable
                for variable in variables
                if variable in dependency_bindings
            )
            linkable_titles = sorted({
                linked.sentence.passage_title
                for linked in candidates
                if mentions_title(
                    candidate.sentence.embedding_text,
                    linked.sentence.passage_title,
                )
            })
            binding_constraints = (
                {claim.subject: candidate.sentence.passage_title}
                if claim.subject.startswith("?")
                else {}
            )
            instantiated_claim = claim.to_dict()
            if claim.subject in binding_constraints:
                instantiated_claim["subject"] = binding_constraints[claim.subject]
            task_lookup[task_id] = (
                candidate,
                raw_score,
                binding_constraints,
                link_binding_variables,
                linkable_titles,
            )
            tasks.append(
                {
                    "task_id": task_id,
                    "claim": instantiated_claim,
                    "required_variables": list(variables),
                    "binding_constraints": binding_constraints,
                    "link_binding_variables": link_binding_variables,
                    "linkable_titles": linkable_titles,
                    "evidence": {
                        "title": candidate.sentence.passage_title,
                        "sent_id": candidate.sentence.sent_id,
                        "text": candidate.sentence.text,
                    },
                }
            )

        payload, call_metadata = _structured_stage(
            _claim_verifier_prompt(tasks),
            model=model,
            backend=backend,
            error_code="claim_verification_invalid",
            generator=generator,
        )
        stage_metadata.append({"claim_id": claim_id, **call_metadata})
        if set(payload) != {"judgments"} or not isinstance(
            payload["judgments"], list
        ):
            raise WiseV3Error(
                "claim_verification_invalid",
                "claim verifier schema is invalid",
                claim_id=claim_id,
                verifier_output=payload,
            )
        judgments = payload["judgments"]
        ids = [
            item.get("task_id") for item in judgments if isinstance(item, dict)
        ]
        if (
            len(judgments) > len(tasks)
            or not set(ids) <= set(task_lookup)
            or len(ids) != len(set(ids))
        ):
            raise WiseV3Error(
                "claim_verification_invalid",
                "claim verifier returned an unknown or duplicate task",
                claim_id=claim_id,
                expected_task_ids=sorted(task_lookup),
                verifier_output=payload,
            )

        retained: list[ClaimAssignment] = []
        for judgment in judgments:
            base_fields = {
                "task_id",
                "label",
                "confidence",
                "bindings",
            }
            if (
                not isinstance(judgment, dict)
                or frozenset(judgment) not in {frozenset(base_fields), frozenset({
                    *base_fields, "evidence_quote"
                })}
            ):
                raise WiseV3Error(
                    "claim_verification_invalid",
                    "claim judgment schema is invalid",
                    claim_id=claim_id,
                    expected_fields=[
                        "bindings",
                        "confidence",
                        "evidence_quote",
                        "label",
                        "task_id",
                    ],
                    judgment=judgment,
                )
            (
                candidate,
                raw_score,
                binding_constraints,
                link_binding_variables,
                linkable_titles,
            ) = task_lookup[judgment["task_id"]]
            label = judgment["label"]
            confidence = judgment["confidence"]
            bindings = judgment["bindings"]
            evidence_quote = judgment.get("evidence_quote", "")
            if (
                label not in {"entailed", "not_entailed"}
                or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0.0 <= float(confidence) <= 1.0
                or not isinstance(bindings, dict)
                or not isinstance(evidence_quote, str)
            ):
                raise WiseV3Error(
                    "claim_verification_invalid", "claim judgment values are invalid"
                )
            if label == "not_entailed":
                if bindings or evidence_quote:
                    raise WiseV3Error(
                        "claim_verification_invalid",
                        "a negative entailment judgment cannot contain bindings or a quote",
                    )
                continue
            if set(bindings) != set(variables) or any(
                not isinstance(value, str) or not value.strip()
                for value in bindings.values()
            ):
                grounding_rejections += 1
                continue
            if any(
                normalize_text(bindings[variable]) != normalize_text(expected)
                for variable, expected in binding_constraints.items()
            ):
                grounding_rejections += 1
                continue
            if any(
                not any(
                    entity_alias_match(bindings[variable], title)
                    for title in linkable_titles
                )
                for variable in link_binding_variables
            ):
                grounding_rejections += 1
                continue
            if any(
                normalize_text(value) in forbidden_values
                for value in bindings.values()
            ):
                grounding_rejections += 1
                continue
            evidence_surface = normalize_text(candidate.sentence.embedding_text)
            lexical_variables = {
                variable
                for variable in variables
                if variable not in terminal_variables
            }
            if any(
                normalize_text(bindings[variable]) not in evidence_surface
                and not any(
                    entity_alias_match(bindings[variable], title)
                    for title in linkable_titles
                )
                for variable in lexical_variables
                if variable not in binding_constraints
            ):
                grounding_rejections += 1
                continue
            unconstrained_values = [
                value
                for variable, value in bindings.items()
                if variable not in binding_constraints
                and variable not in terminal_variables
            ]
            quote = _project_exact_quote(
                candidate.sentence.text,
                evidence_quote,
                unconstrained_values,
            )
            if quote is None:
                grounding_rejections += 1
                continue
            calibrated = _calibrate_confidence(
                float(confidence), confidence_temperature
            )
            if calibrated < threshold:
                continue
            retained.append(
                ClaimAssignment(
                    claim_id=claim.claim_id,
                    sentence_key=candidate.sentence.key,
                    raw_score=raw_score,
                    verifier_confidence=float(confidence),
                    confidence=calibrated,
                    bindings=tuple(
                        sorted((key, value.strip()) for key, value in bindings.items())
                    ),
                    evidence_quote=quote,
                )
            )

        retained = sorted(
            retained,
            key=lambda assignment: (
                assignment.confidence,
                assignment.raw_score,
                assignment.sentence_key,
            ),
            reverse=True,
        )[:top_k]
        if not retained:
            raise WiseV3Error(
                "claim_uncovered",
                "no sentence is entailed above the calibrated claim threshold",
                claim_id=claim.claim_id,
                threshold=threshold,
                extractive_binding_rejections=grounding_rejections,
                claim=claim.to_dict(),
                verifier_tasks=tasks,
                verifier_output=payload,
            )
        assignments[claim.claim_id] = retained

    for candidate in candidates:
        values = all_raw_by_key[candidate.sentence.key]
        candidate.semantic_score = max(values) if values else None
    metadata = {
        "requested_model": model,
        "calls": stage_metadata,
        "extractive_binding_rejections": grounding_rejections,
    }
    for key in (
        "api_prompt_tokens",
        "api_completion_tokens",
        "api_total_tokens",
        "api_cost_usd",
    ):
        values = [
            call[key]
            for call in stage_metadata
            if isinstance(call.get(key), (int, float))
        ]
        metadata[key] = sum(values) if values else None
    return assignments, metadata


def _path_text(path: tuple[tuple[str, int], ...], by_key: dict) -> str:
    return "\n".join(by_key[key].sentence.embedding_text for key in path)


def _ground_binding_values(
    path: tuple[tuple[str, int], ...],
    graph: nx.DiGraph,
    by_key: dict[tuple[str, int], Candidate],
) -> tuple[str, ...]:
    """Return concrete entity surfaces transported by a dependency path."""
    values: set[str] = set()
    if len(path) == 1:
        values.update(extract_entities(by_key[path[0]].sentence.text))
    for left, right in zip(path, path[1:]):
        kinds = set(graph[left][right]["kinds"])
        left_sentence = by_key[left].sentence
        right_sentence = by_key[right].sentence
        if EDGE_TITLE_MENTION in kinds:
            values.add(right_sentence.passage_title)
        if EDGE_ENTITY_COREFERENCE in kinds:
            values.update(
                extract_entities(left_sentence.text)
                & extract_entities(right_sentence.text)
            )
    return tuple(sorted(value for value in values if value))


def _enumerate_typed_paths(
    graph: nx.DiGraph,
    source: tuple[str, int],
    target: tuple[str, int],
    *,
    max_path_hops: int,
    path_top_k: int,
) -> list[tuple[tuple[str, int], ...]]:
    try:
        generated = nx.shortest_simple_paths(graph, source, target)
        paths = []
        for path in islice(generated, path_top_k):
            if len(path) - 1 > max_path_hops:
                break
            paths.append(tuple(path))
        return paths
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def _dependency_verifier_prompt(tasks: list[dict]) -> str:
    payload = json.dumps(tasks, ensure_ascii=False)
    return f"""Verify each proposed multi-hop evidence transition.
Return exactly one JSON object:
{{"judgments": [{{
  "task_id": "p0",
  "label": "valid|invalid",
  "confidence": 0.0
}}]}}

Rules:
- Produce exactly one judgment for every task_id and no others.
- A valid path must jointly support both atomic claims in the stated order.
- The explicit binding value must instantiate the source claim object and the
  target claim subject. Wikipedia-title aliases may differ only by a trailing
  parenthetical disambiguator, accents, punctuation, or spacing; mere entity
  co-occurrence or topical relevance is invalid.
- Every bridge sentence must contribute to that transition.
- Confidence is the probability that the validity label is correct.
- Use only the supplied evidence and do not output markdown.

Tasks:
{payload}"""


def verify_dependencies(
    plan: ReasoningPlan,
    candidates: list[Candidate],
    graph: nx.DiGraph,
    assignments: dict[str, list[ClaimAssignment]],
    *,
    threshold: float,
    max_path_hops: int,
    path_top_k: int,
    model: str,
    backend: str,
    confidence_temperature: float,
    generator: Callable[..., dict] = generate_json_with_metadata,
) -> tuple[
    dict[
        tuple[str, str, tuple[str, int], tuple[str, int]],
        tuple[VerifiedDependency, ...],
    ],
    dict,
]:
    if (
        not 0.0 <= threshold <= 1.0
        or max_path_hops < 0
        or path_top_k <= 0
        or confidence_temperature <= 0.0
    ):
        raise WiseV3Error(
            "configuration_invalid",
            "dependency verifier thresholds, paths, or temperature are invalid",
        )
    if not plan.dependencies:
        return {}, {"skipped": True, "reason": "plan_has_no_dependencies"}
    by_key = {candidate.sentence.key: candidate for candidate in candidates}
    candidate_graph = graph.subgraph(by_key).copy()
    verified: dict[tuple, list[VerifiedDependency]] = {}
    claims = plan.by_id
    tasks = []
    task_lookup = {}
    possible_by_dependency = {dependency: 0 for dependency in plan.dependencies}
    for dependency in plan.dependencies:
        source_assignments = assignments[dependency.source_claim]
        target_assignments = assignments[dependency.target_claim]
        query = (
            f"Question: {claims[dependency.source_claim].query}\n"
            f"Required next claim: {claims[dependency.target_claim].query}\n"
            f"Shared binding: {dependency.binding}"
        )
        for source in source_assignments:
            for target in target_assignments:
                source_value = source.binding_map.get(dependency.binding)
                target_value = target.binding_map.get(dependency.binding)
                if (
                    source_value is None
                    or target_value is None
                    or not entity_alias_match(source_value, target_value)
                ):
                    continue
                binding_value = source_value.strip()
                paths = _enumerate_typed_paths(
                    candidate_graph,
                    source.sentence_key,
                    target.sentence_key,
                    max_path_hops=max_path_hops,
                    path_top_k=path_top_k,
                )
                grounded_paths = [
                    path
                    for path in paths
                    if any(
                        entity_alias_match(binding_value, grounded)
                        for grounded in _ground_binding_values(
                            path, candidate_graph, by_key
                        )
                    )
                ]
                if not grounded_paths:
                    continue
                raw_scores = rerank_scores(
                    query,
                    [_path_text(path, by_key) for path in grounded_paths],
                )
                for path_rank, (path, raw_score) in enumerate(
                    zip(grounded_paths, raw_scores), start=1
                ):
                    task_id = f"p{len(tasks)}"
                    key = (
                        dependency.source_claim,
                        dependency.target_claim,
                        source.sentence_key,
                        target.sentence_key,
                    )
                    task_lookup[task_id] = (
                        dependency,
                        source,
                        target,
                        path,
                        float(raw_score),
                        binding_value,
                        path_rank,
                        key,
                    )
                    possible_by_dependency[dependency] += 1
                    tasks.append(
                        {
                            "task_id": task_id,
                            "source_claim": claims[dependency.source_claim].to_dict(),
                            "target_claim": claims[dependency.target_claim].to_dict(),
                            "binding_variable": dependency.binding,
                            "binding_value": binding_value,
                            "typed_path": [
                                {
                                    "title": by_key[node].sentence.passage_title,
                                    "sent_id": by_key[node].sentence.sent_id,
                                    "text": by_key[node].sentence.text,
                                    "outgoing_edge_types": list(
                                        candidate_graph[node][path[index + 1]]["kinds"]
                                    )
                                    if index + 1 < len(path)
                                    else [],
                                }
                                for index, node in enumerate(path)
                            ],
                        }
                    )
    missing = [
        dependency.to_dict()
        for dependency, count in possible_by_dependency.items()
        if count == 0
    ]
    if missing:
        raise WiseV3Error(
            "dependency_unverified",
            "no explicitly bound typed path exists for a dependency",
            dependencies=missing,
        )

    payload, metadata = _structured_stage(
        _dependency_verifier_prompt(tasks),
        model=model,
        backend=backend,
        error_code="dependency_verification_invalid",
        generator=generator,
    )
    if set(payload) != {"judgments"} or not isinstance(payload["judgments"], list):
        raise WiseV3Error(
            "dependency_verification_invalid", "dependency verifier schema is invalid"
        )
    judgments = payload["judgments"]
    ids = [item.get("task_id") for item in judgments if isinstance(item, dict)]
    if len(judgments) != len(tasks) or set(ids) != set(task_lookup) or len(ids) != len(set(ids)):
        raise WiseV3Error(
            "dependency_verification_invalid",
            "dependency verifier must judge every requested path exactly once",
        )
    for judgment in judgments:
        if not isinstance(judgment, dict) or set(judgment) != {
            "task_id",
            "label",
            "confidence",
        }:
            raise WiseV3Error(
                "dependency_verification_invalid", "dependency judgment schema is invalid"
            )
        label = judgment["label"]
        confidence = judgment["confidence"]
        if (
            label not in {"valid", "invalid"}
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise WiseV3Error(
                "dependency_verification_invalid", "dependency judgment values are invalid"
            )
        if label == "invalid":
            continue
        calibrated = _calibrate_confidence(
            float(confidence), confidence_temperature
        )
        if calibrated < threshold:
            continue
        (
            dependency,
            source,
            target,
            path,
            raw_score,
            binding_value,
            path_rank,
            key,
        ) = task_lookup[judgment["task_id"]]
        verified.setdefault(key, []).append(
            VerifiedDependency(
                dependency=dependency,
                source_key=source.sentence_key,
                target_key=target.sentence_key,
                path_keys=path,
                raw_score=raw_score,
                verifier_confidence=float(confidence),
                confidence=calibrated,
                binding_value=binding_value,
                path_rank=path_rank,
            )
        )
    for dependency in plan.dependencies:
        if not any(
            options and options[0].dependency == dependency
            for options in verified.values()
        ):
            raise WiseV3Error(
                "dependency_unverified",
                "no explicitly bound path is entailed above the calibrated threshold",
                dependency=dependency.to_dict(),
                threshold=threshold,
            )
    return {
        key: tuple(
            sorted(
                options,
                key=lambda item: (
                    item.path_rank,
                    -item.confidence,
                    item.path_keys,
                ),
            )
        )
        for key, options in verified.items()
    }, metadata


def _ordered_union(
    current: tuple[tuple[str, int], ...],
    additions: tuple[tuple[str, int], ...],
) -> tuple[tuple[str, int], ...]:
    seen = set(current)
    output = list(current)
    for key in additions:
        if key not in seen:
            output.append(key)
            seen.add(key)
    return tuple(output)


def _serialized_cost(
    keys: tuple[tuple[str, int], ...], by_key: dict[tuple[str, int], Candidate]
) -> int:
    sentences = [by_key[key].sentence for key in keys]
    return count_tokens(build_context_compact_selection_order(sentences))


def _render_certificate_sentences(
    keys: tuple[tuple[str, int], ...],
    mapping: dict[str, ClaimAssignment],
    by_key: dict[tuple[str, int], Candidate],
) -> tuple[Sentence, ...]:
    """Serialize verified atomic quotes; retain raw text only for path bridges."""
    quotes_by_key: dict[tuple[str, int], list[str]] = {}
    for assignment in mapping.values():
        if not assignment.evidence_quote:
            raise WiseV3Error(
                "certificate_invalid",
                "a claim assignment is missing its verified evidence quote",
                claim_id=assignment.claim_id,
                sentence_key=list(assignment.sentence_key),
            )
        quotes = quotes_by_key.setdefault(assignment.sentence_key, [])
        if assignment.evidence_quote not in quotes:
            quotes.append(assignment.evidence_quote)

    rendered = []
    for key in keys:
        source = by_key[key].sentence
        quotes = quotes_by_key.get(key)
        rendered.append(
            Sentence(
                passage_title=source.passage_title,
                sent_id=source.sent_id,
                text="\n".join(quotes) if quotes else source.text,
            )
        )
    return tuple(rendered)


def _certificate_serialized_cost(
    keys: tuple[tuple[str, int], ...],
    mapping: dict[str, ClaimAssignment],
    by_key: dict[tuple[str, int], Candidate],
) -> int:
    return count_tokens(
        build_context_compact_selection_order(
            list(_render_certificate_sentences(keys, mapping, by_key))
        )
    )


def solve_minimum_cost_witness(
    plan: ReasoningPlan,
    candidates: list[Candidate],
    assignments: dict[str, list[ClaimAssignment]],
    verified_dependencies: dict[
        tuple[str, str, tuple[str, int], tuple[str, int]],
        tuple[VerifiedDependency, ...],
    ],
    *,
    budget_tokens: int,
    max_search_states: int,
    sufficiency_threshold: float,
) -> Witness:
    """Minimize serialized cost subject to a hard calibrated sufficiency bound."""
    if (
        budget_tokens <= 0
        or max_search_states <= 0
        or not 0.0 <= sufficiency_threshold <= 1.0
    ):
        raise WiseV3Error(
            "configuration_invalid",
            "budget, search states, or sufficiency threshold is invalid",
        )
    order = plan.topological_claim_ids()
    operation_inputs = plan.operation.get("inputs")
    sink_claims = set(order) - {
        dependency.source_claim for dependency in plan.dependencies
    }
    if (
        not isinstance(operation_inputs, list)
        or not operation_inputs
        or set(operation_inputs) != sink_claims
    ):
        raise WiseV3Error(
            "certificate_invalid", "operation does not consume every DAG sink claim"
        )
    by_key = {candidate.sentence.key: candidate for candidate in candidates}
    if len(by_key) != len(candidates):
        raise WiseV3Error("certificate_invalid", "candidate sentence keys must be unique")
    if set(assignments) != set(order) or any(
        not assignments.get(claim_id) for claim_id in order
    ):
        raise WiseV3Error(
            "claim_uncovered", "every planned claim requires retained assignments"
        )
    for claim_id, claim_assignments in assignments.items():
        if any(
            assignment.claim_id != claim_id or assignment.sentence_key not in by_key
            for assignment in claim_assignments
        ):
            raise WiseV3Error(
                "certificate_invalid", "claim assignment references invalid evidence"
            )
    incoming: dict[str, list[Dependency]] = {claim_id: [] for claim_id in order}
    for dependency in plan.dependencies:
        incoming[dependency.target_claim].append(dependency)

    best = None
    search_states = 0

    def search(
        depth: int,
        mapping: dict[str, ClaimAssignment],
        chosen_dependencies: tuple[VerifiedDependency, ...],
        selected_keys: tuple[tuple[str, int], ...],
    ) -> None:
        nonlocal best, search_states
        search_states += 1
        if search_states > max_search_states:
            raise WiseV3Error(
                "search_space_exhausted",
                "exact witness search exceeded max_search_states",
                max_search_states=max_search_states,
            )
        cost = (
            _certificate_serialized_cost(selected_keys, mapping, by_key)
            if selected_keys
            else 0
        )
        if cost > budget_tokens or (best is not None and cost > best[0]):
            return
        if depth == len(order):
            confidences = [assignment.confidence for assignment in mapping.values()]
            confidences.extend(
                dependency.confidence for dependency in chosen_dependencies
            )
            bottleneck = min(confidences) if confidences else 0.0
            objective = (cost, -bottleneck, selected_keys)
            if best is None or objective < best[:3]:
                best = (
                    cost,
                    -bottleneck,
                    selected_keys,
                    tuple(mapping[claim_id] for claim_id in order),
                    chosen_dependencies,
                )
            return

        claim_id = order[depth]
        for assignment in assignments[claim_id]:
            if assignment.confidence < sufficiency_threshold:
                continue
            dependency_option_groups = []
            feasible = True
            for dependency in incoming[claim_id]:
                source_assignment = mapping[dependency.source_claim]
                key = (
                    dependency.source_claim,
                    dependency.target_claim,
                    source_assignment.sentence_key,
                    assignment.sentence_key,
                )
                source_binding = source_assignment.binding_map.get(dependency.binding)
                target_binding = assignment.binding_map.get(dependency.binding)
                options = tuple(
                    item
                    for item in verified_dependencies.get(key, ())
                    if item.confidence >= sufficiency_threshold
                    and source_binding is not None
                    and target_binding is not None
                    and entity_alias_match(source_binding, target_binding)
                    and entity_alias_match(source_binding, item.binding_value)
                )
                if not options:
                    feasible = False
                    break
                dependency_option_groups.append(options)
            if not feasible:
                continue
            combinations = product(*dependency_option_groups) if dependency_option_groups else [()]
            for dependency_choices in combinations:
                additions: tuple[tuple[str, int], ...] = ()
                if dependency_choices:
                    for dependency in dependency_choices:
                        additions = _ordered_union(additions, dependency.path_keys)
                else:
                    additions = (assignment.sentence_key,)
                next_keys = _ordered_union(selected_keys, additions)
                next_mapping = {**mapping, claim_id: assignment}
                search(
                    depth + 1,
                    next_mapping,
                    (*chosen_dependencies, *dependency_choices),
                    next_keys,
                )

    search(0, {}, (), ())
    if best is None:
        raise WiseV3Error(
            "no_feasible_witness",
            "no claim-complete witness satisfies all constraints under the budget",
            budget_tokens=budget_tokens,
            sufficiency_threshold=sufficiency_threshold,
            search_states=search_states,
            assignments={
                claim_id: [
                    {
                        "sentence_key": list(assignment.sentence_key),
                        "confidence": assignment.confidence,
                        "bindings": assignment.binding_map,
                        "serialized_tokens": _certificate_serialized_cost(
                            (assignment.sentence_key,),
                            {claim_id: assignment},
                            by_key,
                        ),
                    }
                    for assignment in claim_assignments
                ]
                for claim_id, claim_assignments in assignments.items()
            },
            verified_dependencies=[
                {
                    "from": dependency.dependency.source_claim,
                    "to": dependency.dependency.target_claim,
                    "source_key": list(dependency.source_key),
                    "target_key": list(dependency.target_key),
                    "path_keys": [list(key) for key in dependency.path_keys],
                    "confidence": dependency.confidence,
                    "binding_value": dependency.binding_value,
                    "serialized_tokens": _serialized_cost(
                        dependency.path_keys, by_key
                    ),
                }
                for options in verified_dependencies.values()
                for dependency in options
            ],
        )
    cost, negative_bottleneck, keys, chosen_assignments, chosen_dependencies = best
    selected = _render_certificate_sentences(
        keys,
        {assignment.claim_id: assignment for assignment in chosen_assignments},
        by_key,
    )

    required_keys = {assignment.sentence_key for assignment in chosen_assignments}
    for dependency in chosen_dependencies:
        required_keys.update(dependency.path_keys)
    if required_keys != set(keys):
        raise WiseV3Error(
            "certificate_invalid",
            "selected witness contains a sentence not required by its certificate",
        )

    return Witness(
        plan=plan,
        candidates=tuple(candidates),
        selected=selected,
        assignments=chosen_assignments,
        dependencies=chosen_dependencies,
        serialized_tokens=cost,
        bottleneck_confidence=-negative_bottleneck,
        sufficiency_threshold=sufficiency_threshold,
        search_states=search_states,
        planner_metadata={},
    )


def construct_witness(
    item: QAItem,
    *,
    planner_model: str,
    planner_backend: str,
    budget_tokens: int,
    seed_n: int = 5,
    max_hops: int = 2,
    claim_support_threshold: float = 0.7,
    claim_top_k: int = 5,
    claim_prefilter_k: int = 8,
    dependency_threshold: float = 0.7,
    max_path_hops: int = 3,
    dependency_path_top_k: int = 3,
    confidence_temperature: float = 1.0,
    witness_sufficiency_threshold: float = 0.8,
    max_search_states: int = 100_000,
    verifier_model: str | None = None,
    verifier_backend: str | None = None,
    generator: Callable[..., dict] = generate_json_with_metadata,
) -> Witness:
    """Execute the complete WISE-v3 pipeline with no alternative policy."""
    plan, planner_metadata = induce_reasoning_plan(
        item, model=planner_model, backend=planner_backend, generator=generator
    )
    graph = build_typed_evidence_graph(item)
    candidates = propose_candidates(
        item, plan, graph, seed_n=seed_n, max_hops=max_hops
    )
    resolved_verifier_model = verifier_model or planner_model
    resolved_verifier_backend = verifier_backend or planner_backend
    assignments, claim_metadata = verify_claim_assignments(
        plan,
        candidates,
        threshold=claim_support_threshold,
        top_k=claim_top_k,
        prefilter_k=claim_prefilter_k,
        model=resolved_verifier_model,
        backend=resolved_verifier_backend,
        confidence_temperature=confidence_temperature,
        generator=generator,
    )
    dependencies, dependency_metadata = verify_dependencies(
        plan,
        candidates,
        graph,
        assignments,
        threshold=dependency_threshold,
        max_path_hops=max_path_hops,
        path_top_k=dependency_path_top_k,
        model=resolved_verifier_model,
        backend=resolved_verifier_backend,
        confidence_temperature=confidence_temperature,
        generator=generator,
    )
    witness = solve_minimum_cost_witness(
        plan,
        candidates,
        assignments,
        dependencies,
        budget_tokens=budget_tokens,
        max_search_states=max_search_states,
        sufficiency_threshold=witness_sufficiency_threshold,
    )
    return Witness(
        plan=witness.plan,
        candidates=witness.candidates,
        selected=witness.selected,
        assignments=witness.assignments,
        dependencies=witness.dependencies,
        serialized_tokens=witness.serialized_tokens,
        bottleneck_confidence=witness.bottleneck_confidence,
        sufficiency_threshold=witness.sufficiency_threshold,
        search_states=witness.search_states,
        planner_metadata={
            "planner": planner_metadata,
            "claim_verifier": claim_metadata,
            "dependency_verifier": dependency_metadata,
        },
    )
