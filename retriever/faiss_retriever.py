import os
import faiss
import numpy as np
from typing import List, Optional
from models.embedding import EmbeddingModel
from retriever.base import BaseRetriever
from config import EMBEDDING_MODEL_NAME
from sentence_transformers import CrossEncoder


class FaissRetriever(BaseRetriever):
    def __init__(
        self,
        index_path: str = "novel_index.faiss",
        reranker_model_name: Optional[str] = None,
        use_rerank: bool = False
    ):
        self.index_path = index_path
        self.embedder = EmbeddingModel(EMBEDDING_MODEL_NAME)
        self.chunks: List[str] = []
        self.index = None
        self.use_rerank = use_rerank
        self.reranker = None

        if use_rerank:
            if CrossEncoder is None:
                raise ImportError(
                    "Please install sentence-transformers to enable re-ranking: "
                    "pip install sentence-transformers"
                )
            # 默认使用支持中英双语的 reranker
            model = reranker_model_name or "BAAI/bge-reranker-base"
            self.reranker = CrossEncoder(model)

    def build_index(self, chunks: List[str]):
        self.chunks = chunks
        embeddings = self.embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
        embeddings = embeddings.astype(np.float32)  # FAISS 要求 float32
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        faiss.write_index(self.index, self.index_path)
        self.save_chunks(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if self.index is None:
            raise RuntimeError("FAISS 索引未加载！请先调用 build_index() 或 load_index()")

        # 第一阶段：FAISS 向量召回
        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        candidate_k = min(self.index.ntotal, top_k * 5 if self.use_rerank else top_k)
        _, I = self.index.search(q_emb, candidate_k)
        indices = I[0]
        candidates = [self.chunks[i] for i in indices if i < len(self.chunks)]

        # 第二阶段：Cross-Encoder 重排（可选）
        if self.use_rerank and self.reranker is not None:
            pairs = [[query, doc] for doc in candidates]
            rerank_scores = self.reranker.predict(pairs)
            reranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
            results = [doc for doc, _ in reranked[:top_k]]
        else:
            results = candidates[:top_k]

        return results

    def exists(self) -> bool:
        return os.path.exists(self.index_path) and os.path.exists("chunks_cache.txt")

    def save_chunks(self, chunks: List[str]):
        with open("chunks_cache.txt", "w", encoding="utf-8") as f:
            f.write("\n===CHUNK===\n".join(chunks))

    def load_chunks(self) -> List[str]:
        with open("chunks_cache.txt", "r", encoding="utf-8") as f:
            text = f.read()
            self.chunks = [chunk for chunk in text.split("===CHUNK===\n") if chunk.strip()]
            return self.chunks

    def load_index(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS 索引文件不存在: {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        self.load_chunks()