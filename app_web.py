import os
from flask import Flask, render_template, request, jsonify
from config import NOVEL_PATH, DATA_DIR
from core.rag_engine import RAGEngine

# 初始化 Flask 应用
app = Flask(__name__)

# 全局 RAG 引擎（只加载一次）
rag_engine = None

def init_rag_engine():
    global rag_engine
    if not os.path.exists(NOVEL_PATH):
        os.makedirs(DATA_DIR, exist_ok=True)
        raise FileNotFoundError(f"小说文件未找到！请将小说保存为：{NOVEL_PATH}")
    
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        novel_text = f.read()
    
    rag_engine = RAGEngine(novel_text)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    if rag_engine is None:
        return jsonify({"error": "RAG引擎未初始化，请检查小说文件。"}), 500

    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "问题不能为空。"}), 400

    if question.lower() in {"quit", "exit"}:
        return jsonify({"answer": "👋 再见！", "context": ""})

    try:
        answer, context = rag_engine.answer(question)
        return jsonify({"answer": answer, "context": context})
    except Exception as e:
        return jsonify({"error": f"处理问题时出错：{str(e)}"}), 500

if __name__ == "__main__":
    try:
        init_rag_engine()
        print("✅ RAG 引擎已加载，启动 Flask 应用...")
        app.run(host="127.0.0.1", port=5000, debug=True)
    except Exception as e:
        print(f"❌ 启动失败：{e}")