from abc import ABC, abstractmethod
from typing import List

class BaseRetriever(ABC):
    @abstractmethod
    def build_index(self, chunks: List[str]):
        """构建索引（离线）"""
        pass

    @abstractmethod
    def load_index(self):
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """检索并返回拼接后的上下文"""
        pass

    @abstractmethod
    def exists(self) -> bool:
        """判断索引是否已存在"""
        pass