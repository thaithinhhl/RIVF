from rivf.data.schema import Passage, QAItem, Sentence
from rivf.generation.prompts import build_prompt


def test_chain_aware_prompt_preserves_selection_order_and_comparison_hint():
    film_a = Sentence("Film A", 0, "Film A was directed by Person A.")
    person_a = Sentence("Person A", 0, "Person A was born in 1900.")
    film_b = Sentence("Film B", 0, "Film B was directed by Person B.")
    person_b = Sentence("Person B", 0, "Person B was born in 1910.")
    item = QAItem(
        qid="q",
        question="Which film has the director born first, Film A or Film B?",
        answer="Film A",
        type="",
        level="",
        passages=[
            Passage(sentence.passage_title, [sentence])
            for sentence in [person_b, film_b, person_a, film_a]
        ],
        supporting_facts=[],
    )
    prompt = build_prompt(
        item,
        [film_a, person_a, film_b, person_b],
        chain_aware=True,
    )
    assert prompt.index("Evidence 1 | Film A") < prompt.index("Evidence 2 | Person A")
    assert prompt.index("Evidence 2 | Person A") < prompt.index("Evidence 3 | Film B")
    assert "compare the two attributes" in prompt
