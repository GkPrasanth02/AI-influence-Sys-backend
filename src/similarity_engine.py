from src.preprocess import preprocess_text
from src.embedding import get_embedding
from src.similarity import cosine_sim
from src.aggregate import aggregate_similarities


def compute_similarity_score(student_text: str, llm_texts: list[str]) -> float:
    """
    Compute Similarity Influence Score (S%)
    using SBERT embeddings + cosine similarity.
    """

    # Preprocess student text
    student_text = preprocess_text(student_text)
    student_vec = get_embedding(student_text)

    similarities = []

    # Compare student answer with each LLM answer
    for text in llm_texts:
        clean_text = preprocess_text(text)
        llm_vec = get_embedding(clean_text)

        sim = cosine_sim(student_vec, llm_vec)
        similarities.append(sim)

    # Aggregate similarities and convert to percentage
    final_score = aggregate_similarities(similarities) * 100
    return final_score
