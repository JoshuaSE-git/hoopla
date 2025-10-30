import math

from .index import InvertedIndex
from .helpers import BM25_K1, BM25_B, DEFAULT_SEARCH_LIMIT, tokenize_text


def build_command() -> None:
    print("Building inverted index...")

    idx = InvertedIndex()
    idx.build()
    idx.save()

    print("Inverted index built successfully.")


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    print("Searching for: " + query)

    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for query_token in query_tokens:
        matching_doc_ids = idx.get_documents(query_token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            if not doc:
                continue
            results.append(doc)
            if len(results) >= limit:
                _print_results(results)
                return

    _print_results(results)


def tf_command(doc_id: int, term: str):
    idx = InvertedIndex()
    idx.load()

    print(idx.get_tf(doc_id, term))


def idf_command(term: str):
    idx = InvertedIndex()
    idx.load()

    term_tokens = tokenize_text(term)
    if len(term_tokens) != 1:
        raise Exception("invalid token")

    ids = idx.get_documents(term_tokens[0])

    idf = math.log((len(idx.docmap) + 1) / (len(ids) + 1))

    print(f"Inverse document frequency of '{term}': {idf:.2f}")


def tfidf_command(doc_id: int, term: str):
    idx = InvertedIndex()
    idx.load()

    tf = idx.get_tf(doc_id, term)

    term_tokens = tokenize_text(term)
    if len(term_tokens) != 1:
        raise Exception("invalid token")

    ids = idx.get_documents(term_tokens[0])

    idf = math.log((len(idx.docmap) + 1) / (len(ids) + 1))

    tfidf = tf * idf

    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tfidf:.2f}")


def bmf25idf_command(term: str):
    idx = InvertedIndex()
    idx.load()

    score = idx.get_bm25_idf(term)

    print(f"BM25 IDF score of '{term}': {score:.2f}")


def bm25tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B):
    idx = InvertedIndex()
    idx.load()

    bm25_tf = idx.get_bm25_tf(doc_id, term, k1, b)

    print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25_tf:.2f}")


def bm25_command(query: str, limit: int, k1: float, b: float):
    idx = InvertedIndex()
    idx.load()

    results = idx.bm25_search(query, limit, k1, b)

    for i, res in enumerate(results, 1):
        print(f"{i}. ({res[0]['id']}) {res[0]['title']} - Score: {res[1]:.2f}")


def _print_results(results: list[dict]) -> None:
    for i, res in enumerate(results, 1):
        print(f"{i}. ({res['id']}) {res['title']}")
