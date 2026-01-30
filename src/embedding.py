from sentence_transformers import SentenceTransformer
import numpy as np

# Load model once at startup
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str) -> np.ndarray:
    """
    Generate SBERT embedding for input text.
    """
    return model.encode(text)
