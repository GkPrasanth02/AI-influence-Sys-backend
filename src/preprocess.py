import re


def preprocess_text(text: str) -> str:
    """
    Preprocess input text for similarity computation.

    Steps:
    1. Lowercase conversion
    2. Remove extra whitespace
    3. Remove non-alphanumeric characters (basic noise)

    Parameters:
    text (str): Raw input text

    Returns:
    str: Cleaned text
    """

    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove unwanted characters (keep letters, numbers, punctuation)
    text = re.sub(r"[^a-z0-9.,?! ]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()
