import os
import json
import re
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

DEFAULT_SEARCH_LIMIT = 5
SCORE_PRECISION = 3
MODEL_NAME = "all-MiniLM-L6-v2"


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def semantic_chunk(text: str, size: int, overlap: int) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", stripped)

    if len(sentences) == 1 and not text.endswith((".", "!", "?")):
        sentences = [text]

    chunks = []
    i = 0
    while i < len(sentences) - overlap:
        chunk_sentences = sentences[i : i + size]

        cleaned_sentences = []
        for sentence in chunk_sentences:
            stripped = sentence.strip()
            if stripped:
                cleaned_sentences.append(stripped)

        chunk = " ".join(cleaned_sentences)
        chunks.append(chunk)

        i += size - overlap

    return chunks


def chunk(text: str, size: int, overlap: int) -> list[str]:
    if not overlap:
        overlap = size // 5

    words = text.split()

    chunks = []
    i = 0
    while i < len(words) - overlap:
        chunks.append(" ".join(words[i : i + size]))
        i += size - overlap

    return chunks
