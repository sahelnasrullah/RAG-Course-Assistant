"""Retrieve top-k chunks: dense (embeddings), BM25 (keywords) or hybrid (RRF)."""
from __future__ import annotations

import json
import re

from . import config


STOPWORDS = set(
    "a an and are as at be by can do does for from has have how i in is it many much "
    "of on or the this that to was what when where which who why will with you your "
    "course".split()  # "course" appears in every chunk, so it carries no signal
)


def _stem(word: str) -> str:
    """Very light plural stripping so 'assignments' matches 'assignment'."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokenize(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS]


class Retriever:
    def __init__(self, index_dir=config.INDEX_DIR):
        import chromadb
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        chunks_file = index_dir / "chunks.json"
        if not chunks_file.exists():
            raise SystemExit("No index found. Run: rag ingest")

        self.chunks = {c["id"]: c for c in json.loads(chunks_file.read_text())}
        self.ids = list(self.chunks)
        self.bm25 = BM25Okapi([_tokenize(self.chunks[i]["text"]) for i in self.ids])
        self.col = chromadb.PersistentClient(path=str(index_dir)).get_collection(
            config.COLLECTION
        )
        self.model = SentenceTransformer(config.EMBED_MODEL)

    def dense(self, query: str, k: int) -> list[str]:
        q = self.model.encode(config.QUERY_PREFIX + query, normalize_embeddings=True)
        res = self.col.query(query_embeddings=[q.tolist()], n_results=min(k, len(self.ids)))
        return res["ids"][0]

    def keyword(self, query: str, k: int) -> list[str]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(self.ids)), key=lambda i: scores[i], reverse=True)
        return [self.ids[i] for i in ranked[:k]]

    def search(self, query: str, k: int = config.TOP_K, mode: str = config.RETRIEVAL_MODE):
        if mode == "dense":
            ids = self.dense(query, k)
        elif mode == "bm25":
            ids = self.keyword(query, k)
        elif mode == "hybrid":
            # Reciprocal Rank Fusion of both rankings.
            n = max(k * 3, 10)
            fused: dict[str, float] = {}
            for ranking in (self.dense(query, n), self.keyword(query, n)):
                for rank, cid in enumerate(ranking):
                    fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)
            ids = sorted(fused, key=fused.get, reverse=True)[:k]
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        return [self.chunks[i] for i in ids]
