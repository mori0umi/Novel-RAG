import os

# ========== LLM 模式选择 ==========
USE_LOCAL_LLM = True

# ========== 基础配置 ==========
DATA_DIR = "data"
NOVEL_PATH = os.path.join(DATA_DIR, "novel.txt")
QUESTIONS_FILE = os.path.join(DATA_DIR, "TBP161.json")
INDEX_PATH = "novel_index.faiss"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 5
MAX_NEW_TOKENS = 500
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

RETRIEVER_TYPE = "faiss"  # 可选：faiss, bm25，hybrid

USE_RERANK = True
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"

# ========== 本地 配置（当 USE_LOCAL_LLM=True 时生效）==========

LOCAL_LLM_MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"

# ========== API 配置（当 USE_LOCAL_LLM=False 时生效）==========

LLM_API_KEY = os.getenv("LLM_API_KEY", "") # 请在环境变量中设置您的 API Key
LLM_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"   
API_LLM_MODEL_NAME = "doubao-seed-1-6-251015"          

