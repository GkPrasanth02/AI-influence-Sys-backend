def aggregate_similarities(similarities: list[float]) -> float:
    """
    Aggregate similarity scores from multiple LLMs.
    """
    if not similarities:
        return 0.0

    return sum(similarities) / len(similarities)
