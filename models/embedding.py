from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name: str):
        # 强制使用 CPU 避免与 LLM 抢显存
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts, **kwargs):
        return self.model.encode(texts, **kwargs)