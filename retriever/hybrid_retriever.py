from typing import List, Optional
from retriever.base import BaseRetriever
from retriever.faiss_retriever import FaissRetriever
from retriever.bm25_retriever import BM25Retriever
from collections import defaultdict


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        faiss_path: str = "novel_index.faiss",
        bm25_path: str = "bm25_index.pkl",
        use_rerank: bool = False,
        reranker_model_name: Optional[str] = None
    ):
        # 初始化子 retriever，传递 rerank 配置
        self.faiss = FaissRetriever(
            index_path=faiss_path,
            use_rerank=False,  # 子检索器不重排，由 Hybrid 统一重排
        )
        self.bm25 = BM25Retriever(
            index_path=bm25_path,
            use_rerank=False,  # 同上
        )
        self.use_rerank = use_rerank
        self.reranker = None

        if use_rerank:
            try:
                from sentence_transformers import CrossEncoder
                model = reranker_model_name or "BAAI/bge-reranker-base"
                self.reranker = CrossEncoder(model)
            except ImportError:
                raise ImportError(
                    "Please install sentence-transformers for re-ranking: "
                    "pip install sentence-transformers"
                )

    def build_index(self, chunks: List[str]):
        # 构建两个索引
        self.faiss.build_index(chunks)
        self.bm25.build_index(chunks)


    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        # 第一阶段：分别召回（取稍多一点，用于融合）
        recall_k = max(top_k * 2, 10)  # 至少取10个，避免漏掉

        faiss_results = self.faiss.retrieve(query, top_k=recall_k)
        bm25_results = self.bm25.retrieve(query, top_k=recall_k)

        # 过滤空字符串
        faiss_results = [doc.strip() for doc in faiss_results if doc.strip()]
        bm25_results = [doc.strip() for doc in bm25_results if doc.strip()]

        # === RRF 融合 ===
        k_rrf = 60  # RRF 平滑常数，通常设为60
        rrf_scores = defaultdict(float)

        # 给 FAISS 结果打分（rank 从 1 开始）
        for rank, doc in enumerate(faiss_results, start=1):
            rrf_scores[doc] += 1.0 / (k_rrf + rank)

        # 给 BM25 结果打分
        for rank, doc in enumerate(bm25_results, start=1):
            rrf_scores[doc] += 1.0 / (k_rrf + rank)

        # 按 RRF 分数降序排序
        rrf_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [doc for doc, _ in rrf_ranked]

        # 如果不需要 rerank，直接返回 top_k
        if not self.use_rerank or self.reranker is None:
            return "\n".join(candidates[:top_k])

        # === 第二阶段：Cross-Encoder 重排（可选）===
        rerank_candidates = candidates[:top_k * 2]  # 可选：只重排前 N 个以节省计算
        pairs = [[query, doc] for doc in rerank_candidates]
        scores = self.reranker.predict(pairs)
        reranked = sorted(zip(rerank_candidates, scores), key=lambda x: x[1], reverse=True)
        final_results = [doc for doc, _ in reranked[:top_k]]

        return final_results

    def exists(self) -> bool:
        return self.faiss.exists() and self.bm25.exists()

    def load_index(self):
        # 分别加载索引（无需传 chunks）
        self.faiss.load_index()
        self.bm25.load_index()