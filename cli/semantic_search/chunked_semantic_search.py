import os
import json
import numpy as np

from .semantic_search import SemanticSearch
from .helpers import cosine_similarity, semantic_chunk, CACHE_DIR, SCORE_PRECISION


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

        self.chunk_embeddings_path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
        self.chunk_metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")

    def search_chunks(self, query: str, limit: int):
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError("run load_or_create_chunk_embeddings first")

        if not self.documents or not self.document_map:
            raise ValueError("run load_or_create_chunk_embeddings first")

        query_embedding = self.generate_embedding(query)
        chunk_scores = []
        for i, chunk in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk)
            chunk_scores.append(
                {
                    "chunk_idx": self.chunk_metadata[i]["chunk_idx"],
                    "movie_idx": self.chunk_metadata[i]["movie_idx"],
                    "score": score,
                }
            )

        idx_scores = {}
        for chunk in chunk_scores:
            idx_scores[chunk["movie_idx"]] = max(
                idx_scores.get(chunk["movie_idx"], 0), chunk["score"]
            )

        sorted_idx_scores = sorted(
            idx_scores.items(), key=lambda x: x[1], reverse=True
        )[:limit]

        results = []
        for i, score in sorted_idx_scores:
            doc = self.documents[i]
            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "document": doc["description"][:100],
                    "score": round(score, SCORE_PRECISION),
                }
            )

        return results

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {d["id"]: d for d in documents}

        chunks = []
        metadata = []
        for i, doc in enumerate(documents):
            if not doc["description"]:
                continue

            chunked = semantic_chunk(doc["description"], 4, 1)
            chunk_count = len(chunked)
            for j, c in enumerate(chunked):
                chunks.append(c)
                data = {"movie_idx": i, "chunk_idx": j, "total_chunks": chunk_count}
                metadata.append(data)

        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = metadata

        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w") as f:
            json.dump(
                {
                    "chunks": self.chunk_metadata,
                    "total_chunks": len(self.chunk_embeddings),
                },
                f,
                indent=2,
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {d["id"]: d for d in documents}

        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(
            self.chunk_metadata_path
        ):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]

            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)
