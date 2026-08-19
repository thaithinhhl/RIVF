from rivf.data.schema import QAItem, Sentence


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


def build_prompt(item: QAItem, selected: list[Sentence]) -> str:
    context = build_context(item, selected)
    # Not branched on item.type: HotpotQA "comparison" questions include both
    # yes/no questions AND "which one" questions with an entity as the gold
    # answer, so the instruction has to let the model judge from the question
    # itself rather than assuming every comparison question is yes/no.
    return (
        "Answer the question using only the context below.\n"
        'If the question is a yes-or-no question, answer with exactly one word: "yes" or "no".\n'
        "Otherwise, answer as briefly as possible with the exact name or phrase "
        "from the context -- not a full sentence.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {item.question}\n"
        "Answer:"
    )
