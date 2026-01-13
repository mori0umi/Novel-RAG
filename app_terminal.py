import os
from config import NOVEL_PATH, DATA_DIR
from core.rag_engine import RAGEngine

def main():
    if not os.path.exists(NOVEL_PATH):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"❌ 小说文件未找到！请将小说保存为：{NOVEL_PATH}")
        return

    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        novel_text = f.read()

    engine = RAGEngine(novel_text)

    print("\n✅ 小说问答系统已启动！输入 'quit' 或 'exit' 退出。\n")
    while True:
        question = input("❓ 你的问题： ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("👋 再见！")
            break
        try:
            answer, context = engine.answer(question)
            print(f"💡 回答：{answer}\n")
            # print(f"💡 相关上下文：{context}\n")
        except Exception as e:
            print(f"⚠️ 出错：{e}\n")

if __name__ == "__main__":
    main()