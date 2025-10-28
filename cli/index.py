import os
import pickle
import math
from collections import defaultdict, Counter

from helpers import (
    CACHE_DIR,
    load_movies,
    tokenize_text,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")

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

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

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

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        self.term_frequencies[doc_id] = Counter(tokens)
        for token in set(tokens):
            self.index[token].add(doc_id)
