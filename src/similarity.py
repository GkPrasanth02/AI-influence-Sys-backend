import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cosine_sim(vec1, vec2) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Parameters:
    vec1, vec2: SBERT embedding vectors (same dimension)

    Returns:
    float: similarity score in the range 0–1
    """

    v1 = np.asarray(vec1).reshape(1, -1)
    v2 = np.asarray(vec2).reshape(1, -1)

    score = cosine_similarity(v1, v2)[0][0]

    # Numerical safety
    return float(max(0.0, min(1.0, score)))
