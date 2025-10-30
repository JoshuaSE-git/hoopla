from .semantic_search import SemanticSearch
from .helpers import load_movies


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
