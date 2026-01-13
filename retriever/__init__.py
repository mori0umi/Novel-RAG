from config import RETRIEVER_TYPE, USE_RERANK, RERANK_MODEL_NAME
from .faiss_retriever import FaissRetriever
from .bm25_retriever import BM25Retriever
from .hybrid_retriever import HybridRetriever

def get_retriever(index_path: str = "index"):
    if RETRIEVER_TYPE == "faiss":
        return FaissRetriever(index_path + ".faiss", use_rerank=USE_RERANK, reranker_model_name=RERANK_MODEL_NAME)
    elif RETRIEVER_TYPE == "bm25":
        return BM25Retriever(index_path + ".pkl", use_rerank=USE_RERANK, reranker_model_name=RERANK_MODEL_NAME)
    elif RETRIEVER_TYPE == "hybrid":
        return HybridRetriever(
            faiss_path=index_path + ".faiss",
            bm25_path=index_path + ".pkl",
            use_rerank=USE_RERANK,
            reranker_model_name=RERANK_MODEL_NAME
        )
    else:
        raise ValueError(f"Unknown retriever type: {RETRIEVER_TYPE}")
    
    