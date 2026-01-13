from config import (
    API_LLM_MODEL_NAME, LOCAL_LLM_MODEL_NAME, MAX_NEW_TOKENS, TOP_K,
    USE_LOCAL_LLM, LLM_API_KEY, LLM_BASE_URL
)
import re

# 动态导入 LLM
if USE_LOCAL_LLM:
    LLM_MODEL_NAME = LOCAL_LLM_MODEL_NAME
    from models.llm_local import QuantizedLLM  # 保留原本地模型为 llm_local.py
    LLMClass = QuantizedLLM
else:
    LLM_MODEL_NAME = API_LLM_MODEL_NAME
    from models.llm_api import APILLM
    LLMClass = lambda _: APILLM(LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME)

class RAGEngine:
    def __init__(self, novel_text: str, index_path: str = "novel_index"):
        from retriever import get_retriever
        from retriever.chunking import split_text
        from config import CHUNK_SIZE, CHUNK_OVERLAP

        self.retriever = get_retriever(index_path)
        self.chunks = split_text(novel_text, CHUNK_SIZE, CHUNK_OVERLAP, show_progress=True)

        if self.retriever.exists():
            print(f"加载已有的索引...")
            self.retriever.load_index()
        else:
            print(f"构建索引...")
            self.retriever.build_index(self.chunks)
        
        # 动态初始化 LLM
        self.llm = LLMClass(LLM_MODEL_NAME)

        print(f"共切分为 {len(self.chunks)} 个 chunk")
        print(f"最长 chunk 长度: {max(len(c) for c in self.chunks)}")

    def answer(self, question: str) -> str:
        context = self.retriever.retrieve(question, top_k=TOP_K)
        context_text = "\n".join(context)
        prompt = (
            "你是一个专注于分析小说内容的智能助手。\n"
            "请严格参考以下提供的小说片段回答问题，不要编造信息。\n"
            "【相关小说片段】\n"
            f"{context_text}\n\n"
            f"【问题】\n{question}\n\n"
            "【回答】\n"
        )

        raw_answer = self.llm.generate(prompt, max_new_tokens=MAX_NEW_TOKENS)
        clean_answer = re.split(r'\n\n|\n【问题】|\n用户：', raw_answer)[0].strip()
        return clean_answer, context