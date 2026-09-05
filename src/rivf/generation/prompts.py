import json

from rivf.data.schema import QAItem, Sentence
from rivf.retrieval.question_router import (
    BRIDGE_COMPARISON,
    CHAINED_REASONING,
    DIRECT_COMPARISON,
    route_question,
)


def build_context(item: QAItem, selected: list[Sentence]) -> str:
    """Re-groups selected sentences under their original passage titles, in
    original sentence order -- regardless of the score order pruning picked
    them in."""
    selected_keys = {s.key for s in selected}
    lines = []
    for passage in item.passages:
        kept = [s for s in passage.sentences if s.key in selected_keys]
        if not kept:
            continue
        lines.append(f"[{passage.title}]")
        lines.extend(s.text for s in kept)
    return "\n".join(lines)


def build_context_in_selection_order(selected: list[Sentence]) -> str:
    """Serialize evidence in the chain/pair order produced by pruning."""
    lines = []
    for index, sentence in enumerate(selected, start=1):
        lines.append(f"[Evidence {index} | {sentence.passage_title}]")
        lines.append(sentence.text)
    return "\n".join(lines)


def build_context_compact_selection_order(selected: list[Sentence]) -> str:
    """Preserve witness order while paying for each consecutive title once."""
    lines = []
    previous_title = None
    for sentence in selected:
        if sentence.passage_title != previous_title:
            lines.append(f"[{sentence.passage_title}]")
            previous_title = sentence.passage_title
        lines.append(sentence.text)
    return "\n".join(lines)


def build_prompt(
    item: QAItem,
    selected: list[Sentence],
    chain_aware: bool = False,
    preserve_selection_order: bool = False,
    compact_selection_order: bool = False,
    reasoning_contract: dict | None = None,
) -> str:
    if compact_selection_order:
        context = build_context_compact_selection_order(selected)
    elif chain_aware or preserve_selection_order:
        context = build_context_in_selection_order(selected)
    else:
        context = build_context(item, selected)
    # Not branched on item.type: HotpotQA "comparison" questions include both
    # yes/no questions AND "which one" questions with an entity as the gold
    # answer, so the instruction has to let the model judge from the question
    # itself rather than assuming every comparison question is yes/no.
    reasoning_hint = ""
    if chain_aware:
        route = route_question(item.question)
        if route == BRIDGE_COMPARISON:
            reasoning_hint = (
                "Internally identify the intermediate entity and requested attribute for each "
                "option, compare the two attributes, then return only the winning option.\n"
            )
        elif route == DIRECT_COMPARISON:
            reasoning_hint = (
                "Internally identify the requested attribute for each option, compare them, "
                "then return only the answer.\n"
            )
        elif route == CHAINED_REASONING:
            reasoning_hint = (
                "Internally follow the relation chain across the ordered evidence before answering.\n"
            )
    contract_text = ""
    if reasoning_contract is not None:
        contract_text = (
            "Execute the verified reasoning certificate below exactly. Trace each final "
            "operation input through its dependency branch and use its resolved bindings.\n"
            "For a comparison phrased as 'Which/Who/A or B', return the winning option "
            "from the question, never yes/no. Respect the operator direction: younger/later "
            "means the later date; older/first/earlier means the earlier date.\n"
            "Render only the final answer in the form requested by the question. Preserve "
            "an explicitly supported nationality or demonym for a nationality question; "
            "use a country noun only when the question unambiguously requests the named "
            "country. Do not expand or convert a supported surface merely to make it more "
            "specific. For a birth/death place, return the conventional "
            "place entity: retain a region when it is part of the place name, but omit a "
            "descriptive geographic hierarchy beyond that entity.\n"
            "Verified certificate:\n"
            f"{json.dumps(reasoning_contract, ensure_ascii=False)}\n\n"
        )
    answer_style = (
        "Otherwise, return only the requested entity, attribute, locality, or option "
        "required by the verified certificate -- not a full sentence.\n\n"
        if reasoning_contract is not None
        else "Otherwise, answer as briefly as possible with the exact name or phrase "
        "from the context -- not a full sentence.\n\n"
    )
    return (
        "Answer the question using only the context below.\n"
        'If the question is a yes-or-no question, answer with exactly one word: "yes" or "no".\n'
        f"{answer_style}"
        f"{reasoning_hint}"
        f"{contract_text}"
        f"Context:\n{context}\n\n"
        f"Question: {item.question}\n"
        "Answer:"
    )
