import os
import requests
import json
from typing import Dict, Any, List, Union

class APILLM:
    def __init__(self, api_key: str, base_url: str, model_name: str):
        """
        支持 OpenAI 兼容的多模态 API（包括火山方舟 vision 模型）
        """
        self.api_key = api_key or os.getenv("ARK_API_KEY")
        self.base_url = base_url or "https://ark.cn-beijing.volces.com/api/v3"
        self.model_name = model_name
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate(
        self,
        messages: Union[str, List[Dict[str, Any]]],
        max_new_tokens: int = 150,
        temperature: float = 0.0
    ) -> str:
        """
        支持两种输入：
        - 纯文本：传入字符串 prompt
        - 多模态：传入 OpenAI 格式的 messages 列表（含 image_url）
        """
        if isinstance(messages, str):
            payload_messages = [{"role": "user", "content": messages}]
        else:
            payload_messages = messages

        payload = {
            "model": self.model_name,
            "messages": payload_messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": False
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            error_msg = f"API 调用失败: {e}"
            print(error_msg)
            if hasattr(e, 'response') and e.response is not None:
                print("响应内容:", e.response.text)
            return "抱歉，模型调用出错，请检查网络或 API 配置。"
