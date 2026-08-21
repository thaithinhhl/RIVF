from rivf.retrieval.question_router import (
    BRIDGE_COMPARISON,
    CHAINED_REASONING,
    DIRECT_COMPARISON,
    route_question,
)


def test_routes_direct_comparison_to_semantic_policy():
    assert (
        route_question("Which film came out first, Blind Shaft or The Mask Of Fu Manchu?")
        == DIRECT_COMPARISON
    )
    assert (
        route_question("Are Fire In Hell and The Tiger both from the same country?")
        == DIRECT_COMPARISON
    )


def test_routes_bridge_comparison_to_conditional_policy():
    assert (
        route_question(
            "Which film has the director who was born later, El Extraño Viaje or Love In Pawn?"
        )
        == BRIDGE_COMPARISON
    )


def test_routes_compositional_and_inference_to_conditional_policy():
    assert (
        route_question("Who is the mother of the director of film Polish-Russian War?")
        == CHAINED_REASONING
    )
    assert (
        route_question("Who is Charles Bretagne Marie De La Trémoille's paternal grandfather?")
        == CHAINED_REASONING
    )
