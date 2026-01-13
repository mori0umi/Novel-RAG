import os
import json
import re

from tqdm import tqdm
from config import NOVEL_PATH, QUESTIONS_FILE
from core.rag_engine import RAGEngine


def extract_option_letter(answer_text):
    """
    从模型回答中提取选项字母（如 'B'），支持多种格式：
    - “答案是 B”
    - “选 B”
    - “B. 借助外来力量...”
    - “B”
    """
    answer_text = answer_text.strip().upper()
    # 匹配单独的 A/B/C/D 或带标点的
    match = re.search(r'\b([ABCD])\b', answer_text)
    if match:
        return match.group(1)
    # 如果直接以 A. 开头
    match2 = re.match(r'^([ABCD])\.', answer_text)
    if match2:
        return match2.group(1)
    return None

def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    # 检查小说文件
    if not os.path.exists(NOVEL_PATH):
        print(f"❌ 小说文件未找到！请将《三体》全文保存为：{NOVEL_PATH}")
        return

    # 加载小说
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        novel_text = f.read()

    # 初始化引擎
    print("🔧 正在加载 RAG 引擎...")
    engine = RAGEngine(novel_text)
    print("✅ RAG 引擎加载完成。\n")

    # 加载问题
    questions = load_questions()
    total = len(questions)
    correct = 0
    results = []

    print(f"📊 开始评估 {total} 道选择题...\n")

    for i, q in enumerate(tqdm(questions, desc="处理题目", unit="题"), 1):
        question = q["question"]
        options = "\n".join(q["options"])
        full_prompt = f"{question}\n{options}\n请直接回答选项字母（A/B/C/D）。"

        model_answer, contexts = engine.answer(full_prompt)
        pred = extract_option_letter(model_answer)
        gold = q["answer"].strip().upper()

        is_correct = (pred == gold)
        if is_correct:
            correct += 1

        results.append({
            "id": i,
            "question": question,
            "options": " ".join(q["options"]),
            "gold": gold,
            "predicted": pred,
            "model_output": model_answer,
            "correct": is_correct,
            "contexts": contexts
        })

        status = "✅" if is_correct else "❌"
        tqdm.write(f"{status} 第 {i} 题 | 预测: {pred} | 真实: {gold}")

    print(f"\n准确率: {correct}/{total} ({100 * correct / total:.2f}%)")

    # 输出总结
    accuracy = correct / total * 100
    print("\n" + "="*60)
    print(f"🎯 总体准确率: {correct}/{total} = {accuracy:.2f}%")
    print("="*60)

    # 可选：保存详细结果
    output_file = os.path.join(os.path.dirname(__file__), "evaluation_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"📝 详细结果已保存至: {output_file}")

if __name__ == "__main__":
    main()