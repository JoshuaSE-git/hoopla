from .semantic_search import SemanticSearch
from .chunked_semantic_search import ChunkedSemanticSearch
from .helpers import load_movies, MODEL_NAME, semantic_chunk, chunk


def handler_search_chunks(query: str, limit: int):
    css = ChunkedSemanticSearch(MODEL_NAME)
    documents = load_movies()
    css.load_or_create_chunk_embeddings(documents)

    results = css.search_chunks(query, limit)

    print(f"Query: {query}")
    print("Results:")
    for i, result in enumerate(results, start=1):
        print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")


def handler_embed_chunks():
    css = ChunkedSemanticSearch(MODEL_NAME)
    documents = load_movies()
    embeddings = css.load_or_create_chunk_embeddings(documents)

    print(f"Generated {len(embeddings)} chunked embeddings")


def handler_semantic_chunk(text: str, size: int, overlap: int):
    chunks = semantic_chunk(text, size, overlap)

    print(f"Semantically chunking {len(text)} characters")
    for i, c in enumerate(chunks, start=1):
        print(f"{i}. {c}")


def handler_chunk(text: str, size: int, overlap: int):
    chunks = chunk(text, size, overlap)

    print(f"Chunking {len(text)} characters")
    for i, c in enumerate(chunks, start=1):
        print(f"{i}. {c}")


def handler_verify_model():
    ss = SemanticSearch(MODEL_NAME)

    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def handler_embed_text(text: str):
    ss = SemanticSearch(MODEL_NAME)

    embedding = ss.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def handler_verify_embeddings():
    ss = SemanticSearch(MODEL_NAME)

    documents = load_movies()

    embeddings = ss.load_or_create_embeddings(documents)

    print(f"Nubmer of docs: {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def handler_embed_query(query: str):
    ss = SemanticSearch(MODEL_NAME)

    embedding = ss.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def handler_search(query: str, limit: int):
    ss = SemanticSearch(MODEL_NAME)
    documents = load_movies()
    ss.load_or_create_embeddings(documents)

    results = ss.search(query, limit)

    for i, r in enumerate(results, start=1):
        print(f"{i}. {r['title']} (score: {r['score']:.4f})")
        print(f"   {r['description'][:100]} ...")
        print()
