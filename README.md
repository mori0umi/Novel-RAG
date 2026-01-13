# RAG 小说问答系统

一个基于检索增强生成（RAG）的智能小说问答系统，支持本地大语言模型和API调用，提供终端和Web界面。

## 功能特性

- **多模态检索**：支持FAISS、BM25和混合检索器
- **灵活的LLM集成**：支持本地模型（如Qwen）和云端API（如豆包）
- **Web界面**：基于Flask的现代化Web应用
- **终端界面**：命令行交互式问答
- **评估工具**：内置评估脚本，支持选择题准确率测试
- **可配置**：高度可配置的参数，包括分块大小、检索数量等

## 安装

### 环境要求

- Python 3.8+
- Windows/Linux/macOS

### 步骤

1. 克隆或下载项目到本地：

   ```bash
   git clone <repository-url>
   cd RAG-git
   ```

2. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

3. 准备小说数据：

   将小说文本文件保存为 `data/novel.txt`（UTF-8编码）。

4. （可选）配置LLM：

   - 本地模型：确保 `config.py` 中 `USE_LOCAL_LLM = True`，并下载相应模型。
   - API模式：设置环境变量 `LLM_API_KEY`，并调整 `config.py` 中的API配置。

## 使用

### 终端模式

运行终端应用：

```bash
python app_terminal.py
```

输入问题进行问答，输入 `quit` 或 `exit` 退出。

### Web模式

运行Web应用：

```bash
python app_web.py
```

打开浏览器访问 `http://127.0.0.1:5000`，在界面中输入问题。

### 评估模式

运行评估脚本：

```bash
python evaluate_rag.py
```

该脚本会使用 `data/TBP161.json` 中的问题测试系统准确率。

## 配置

主要配置位于 `config.py`：

- `USE_LOCAL_LLM`: 是否使用本地LLM（True）或API（False）
- `CHUNK_SIZE`: 文本分块大小
- `TOP_K`: 检索的文档数量
- `RETRIEVER_TYPE`: 检索器类型（faiss, bm25, hybrid）
- `USE_RERANK`: 是否启用重排序
- 其他模型和路径配置

## 项目结构

```
├── app_terminal.py          # 终端应用
├── app_web.py              # Web应用
├── config.py               # 配置
├── evaluate_rag.py         # 评估脚本
├── requirements.txt        # 依赖
├── core/
│   └── rag_engine.py       # RAG引擎核心
├── data/
│   ├── novel.txt           # 小说文本
│   ├── TBP161.json         # 评估问题
│   └── TBP30.json          # 其他数据
├── models/
│   ├── embedding.py        # 嵌入模型
│   ├── llm_api.py          # API LLM
│   └── llm_local.py        # 本地LLM
├── retriever/
│   ├── base.py             # 基础检索器
│   ├── bm25_retriever.py   # BM25检索器
│   ├── faiss_retriever.py  # FAISS检索器
│   ├── hybrid_retriever.py # 混合检索器
│   └── chunking.py         # 分块工具
└── templates/
    └── index.html          # Web模板
```

## 依赖

主要依赖包见 `requirements.txt`：

- torch: 深度学习框架
- sentence-transformers: 句子嵌入
- faiss-cpu: 向量检索
- transformers: 预训练模型
- flask: Web框架
- tqdm: 进度条

## 注意事项

- 首次运行时会自动构建索引，可能需要一些时间。
- 本地LLM模式需要足够的GPU内存或CPU资源。
- 确保小说文件编码为UTF-8。

## 说明

本项目为北航计算机学院2025-2026年研究生课程《信息检索原理》课程大作业
