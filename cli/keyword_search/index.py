import os
import pickle
import math
from collections import defaultdict, Counter

from .helpers import CACHE_DIR, load_movies, tokenize_text


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}

        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        counter = self.term_frequencies.get(doc_id, None)
        if not counter:
            raise Exception("doc_id doesn't exist")

        term_tokens = tokenize_text(term)
        if len(term_tokens) != 1:
            raise Exception("invalid token")

        return counter[term_tokens[0]]

    def get_bm25_idf(self, term: str) -> float:
        term_tokens = tokenize_text(term)
        if len(term_tokens) != 1:
            raise Exception("invalid term")

        df = len(self.index[term_tokens[0]])
        n = len(self.docmap)

        return math.log(((n - df + 0.5) / (df + 0.5) + 1))

    def get_bm25_tf(self, doc_id: int, term: str, k1: float, b: float) -> float:
        if doc_id not in self.doc_lengths:
            raise Exception("doc_id doesn't exist")

        tf = self.get_tf(doc_id, term)

        length_norm = (
            1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        )
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)

        return bm25_tf

    def bm25_search(self, query: str, limit: int, k1: float, b: float) -> list[tuple]:
        query_tokens = tokenize_text(query)
        scores: dict[int, float] = defaultdict(float)
        for tk in query_tokens:
            idf = self.get_bm25_idf(tk)
            for doc_id in self.docmap:
                tf = self.get_bm25_tf(doc_id, tk, k1, b)
                scores[doc_id] += tf * idf

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]

        return [(self.docmap[scores[0]], scores[1]) for scores in sorted_scores]

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0

        avg = sum(self.doc_lengths.values()) / len(self.doc_lengths)

        return avg

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        self.term_frequencies[doc_id] = Counter(tokens)
        self.doc_lengths[doc_id] = len(tokens)
        for token in set(tokens):
            self.index[token].add(doc_id)
