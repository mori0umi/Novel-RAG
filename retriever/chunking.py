import re
from typing import List
from tqdm import tqdm

def split_text(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 60,
    show_progress: bool = False
) -> List[str]:
    if not text or not text.strip():
        return []

    text = re.sub(r'\s+', ' ', text).strip()

    if len(text) <= chunk_size:
        return [text]

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    # 1. 按句子分割（基于 . ! ? 等结束标点）
    # 使用正向先行断言保留分隔符，并过滤空句
    
    sentences = [s.strip() for s in re.split(r'(?<=[。！？.!?])\s*', text) if s.strip()]

    if not sentences:
        return [text[:chunk_size]]  # fallback

    chunks = []
    current_chunk = ""
    last_chunk = ""

    i = 0
    total = len(sentences)
    sentence_iter = tqdm(enumerate(sentences), total=total, disable=not show_progress, desc="Splitting text")
    while i < total:
        sentence = sentences[i]
        # 如果当前块为空，直接加入
        if not current_chunk:
            current_chunk = sentence
            i += 1
        else:
            # 尝试加入下一句
            test_chunk = current_chunk + " " + sentence
            if len(test_chunk) <= chunk_size:
                current_chunk = test_chunk
                i += 1
            else:
                # 无法加入，保存当前块
                chunks.append(current_chunk)
                last_chunk = current_chunk

                # 计算重叠部分：从上一块末尾回退，找能放入 overlap 长度的句子子序列
                overlap_text = ""
                if chunk_overlap > 0 and chunks:
                    # 从上一个 chunk 的末尾开始，取最多 chunk_overlap 字符的内容
                    tail = last_chunk[-chunk_overlap:]
                    overlap_sentences = []
                    temp_len = 0
                    for sent in reversed(sentences[:i]):
                        if temp_len + len(sent) + 1 > chunk_overlap:
                            break
                        overlap_sentences.insert(0, sent)
                        temp_len += len(sent) + 1  # +1 for space
                    overlap_text = " ".join(overlap_sentences)

                # 下一块以重叠部分开头（如果有的话）
                if overlap_text:
                    current_chunk = overlap_text
                    if not any(sentences[j] not in overlap_text for j in range(i, min(i+1, total))):
                        pass
                else:
                    current_chunk = ""

    # 添加最后一个块
    if current_chunk and (not chunks or current_chunk != chunks[-1]):
        chunks.append(current_chunk)

    # 边界情况：如果结果为空（极短句子等），兜底返回
    if not chunks:
        chunks = [text[:chunk_size]]

    return chunks