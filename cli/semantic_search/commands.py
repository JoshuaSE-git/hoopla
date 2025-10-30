import re

from .semantic_search import SemanticSearch
from .helpers import load_movies


def semantic_chunk(text: str, size: int, overlap: int):
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    i = 0
    while i < len(sentences) - overlap:
        chunks.append(" ".join(sentences[i : i + size]))
        i += size - overlap

    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk}")


def chunk(text: str, size: int, overlap: int):
    if not overlap:
        overlap = size // 5

    words = text.split()

    chunks = []
    i = 0
    while i < len(words) - overlap:
        chunks.append(" ".join(words[i : i + size]))
        i += size - overlap

    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk}")


def verify_model():
    ss = SemanticSearch()

    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def embed_text(text: str):
    ss = SemanticSearch()

    embedding = ss.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    ss = SemanticSearch()

    documents = load_movies()

    embeddings = ss.load_or_create_embeddings(documents)

    print(f"Nubmer of docs: {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str):
    ss = SemanticSearch()

    embedding = ss.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def search(query: str, limit: int):
    ss = SemanticSearch()
    documents = load_movies()
    ss.load_or_create_embeddings(documents)

    results = ss.search(query, limit)

    for i, r in enumerate(results, start=1):
        print(f"{i}. {r['title']} (score: {r['score']:.4f})")
        print(f"   {r['description'][:100]} ...")
        print()
