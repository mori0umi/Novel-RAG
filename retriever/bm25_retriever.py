import os
import pickle
import jieba
from typing import List, Optional
from rank_bm25 import BM25Okapi
from abc import ABC, abstractmethod
from retriever.base import BaseRetriever
from sentence_transformers import CrossEncoder

class BM25Retriever(BaseRetriever):
    def __init__(
        self,
        index_path: str = "bm25_index.pkl",
        reranker_model_name: Optional[str] = None,  # e.g., "BAAI/bge-reranker-base"
        use_rerank: bool = False
    ):
        self.index_path = index_path
        self.chunks: List[str] = []
        self.bm25 = None
        self.use_rerank = use_rerank
        self.reranker = None

        if use_rerank:
            if CrossEncoder is None:
                raise ImportError("Please install sentence-transformers to use re-ranking: pip install sentence-transformers")
            self.reranker = CrossEncoder(reranker_model_name or "BAAI/bge-reranker-base")

    def _tokenize(self, text: str) -> List[str]:
        # 中文分词
        return list(jieba.cut_for_search(text))

    def build_index(self, chunks: List[str]):
        self.chunks = chunks
        tokenized_chunks = [self._tokenize(chunk) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        with open(self.index_path, "wb") as f:
            pickle.dump((chunks, tokenized_chunks), f)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if self.bm25 is None:
            raise ValueError("Index not built or loaded. Call build_index() or load_index() first.")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        # 先取稍多一点的候选（比如 top_k * 2），用于重排
        candidate_k = min(len(scores), top_k * 5 if self.use_rerank else top_k)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:candidate_k]
        candidates = [self.chunks[i] for i in top_indices]

        if self.use_rerank and self.reranker is not None:
            # 使用 cross-encoder 重排
            pairs = [[query, doc] for doc in candidates]
            rerank_scores = self.reranker.predict(pairs)
            # 按 rerank 分数降序排序
            reranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
            results = [doc for doc, _ in reranked[:top_k]]
        else:
            results = candidates[:top_k]

        return results

    def exists(self) -> bool:
        return os.path.exists(self.index_path)

    def load_index(self):
        with open(self.index_path, "rb") as f:
            self.chunks, _ = pickle.load(f)
        tokenized = [self._tokenize(c) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)