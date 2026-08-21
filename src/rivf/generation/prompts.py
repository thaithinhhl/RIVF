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


def build_prompt(
    item: QAItem,
    selected: list[Sentence],
    chain_aware: bool = False,
) -> str:
    context = (
        build_context_in_selection_order(selected)
        if chain_aware
        else build_context(item, selected)
    )
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
    return (
        "Answer the question using only the context below.\n"
        'If the question is a yes-or-no question, answer with exactly one word: "yes" or "no".\n'
        "Otherwise, answer as briefly as possible with the exact name or phrase "
        "from the context -- not a full sentence.\n\n"
        f"{reasoning_hint}"
        f"Context:\n{context}\n\n"
        f"Question: {item.question}\n"
        "Answer:"
    )
