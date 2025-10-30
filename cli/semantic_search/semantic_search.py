import numpy as np
import os

from sentence_transformers import SentenceTransformer
from .helpers import CACHE_DIR, cosine_similarity


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = None
        self.embeddings = None
        self.document_map = {}

        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarities.append(
                (self.documents[i], cosine_similarity(query_embedding, doc_embedding))
            )

        sorted_scores = sorted(similarities, key=lambda x: x[1], reverse=True)[:limit]

        results = []
        for doc, score in sorted_scores:
            result = {
                "score": score,
                "title": doc["title"],
                "description": doc["description"],
            }
            results.append(result)

        return results

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents

        docs = []
        for d in documents:
            self.document_map[d["id"]] = d
            docs.append(f"{d['title']}: {d['description']}")

        self.embeddings = self.model.encode(docs, show_progress_bar=True)

        np.save(self.embeddings_path, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {d["id"]: d for d in documents}

        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def generate_embedding(self, text: str):
        if not text or text.isspace():
            raise ValueError("text must be nonempty")

        embedding = self.model.encode([text])[0]

        return embedding
