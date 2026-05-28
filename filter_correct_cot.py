#你已经有了包含思维链的文件 predict_dev_with_cot.json，现在需要根据评测结果，把做对的题目单独提取出来，保存成一个新的文件。
import json
import os

# ================= 配置区域 =================
# 1. 包含思维链的预测文件 (源文件)
INPUT_COT_FILE = "predict_dev_with_cot.json"

# 2. 评测结果文件 (由 evaluation.py 生成, 包含 res=1/0)
EVAL_RESULT_FILE = "evaluation_results.json"

# 3. 输出文件 (只包含正确题目的思维链)
OUTPUT_FILE = "correct_predict_dev_with_cot.json"
# ===========================================

def main():
    print("🚀 开始筛选正确的思维链样本...")

    # 1. 检查文件
    if not os.path.exists(INPUT_COT_FILE):
        print(f"❌ 错误：找不到输入文件 {INPUT_COT_FILE}")
        return
    if not os.path.exists(EVAL_RESULT_FILE):
        print(f"❌ 错误：找不到评测结果 {EVAL_RESULT_FILE}")
        print("💡 提示：请先运行修改后的 evaluation.py 来生成此文件。")
        return

    # 2. 读取评测结果，获取正确题目的 ID
    print(f"正在读取 {EVAL_RESULT_FILE} ...")
    correct_ids = set()
    with open(EVAL_RESULT_FILE, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
        for item in eval_data:
            # item 结构: {'sql_idx': 0, 'res': 1}
            # res=1 代表正确
            if item.get('res') == 1:
                # 转为字符串以匹配 JSON key
                correct_ids.add(str(item['sql_idx']))
    
    print(f"📊 评测结果中共有 {len(correct_ids)} 道做对的题目。")

    # 3. 读取思维链文件并筛选
    print(f"正在读取 {INPUT_COT_FILE} ...")
    with open(INPUT_COT_FILE, 'r', encoding='utf-8') as f:
        cot_data = json.load(f) # 这是一个字典 { "0": {...}, "1": {...} }

    filtered_data = []
    
    # 遍历 cot_data
    # 注意：cot_data 是字典，key 是 question_id
    for q_id, content in cot_data.items():
        if str(q_id) in correct_ids:
            # 将其加入结果列表
            # 我们可以保留原来的结构，或者变成列表
            filtered_data.append(content)

    # 按 question_id 排序 (为了美观)
    filtered_data.sort(key=lambda x: int(x['question_id']) if str(x['question_id']).isdigit() else x['question_id'])

    # 4. 保存结果
    print(f"正在保存筛选后的数据到 {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)

    print("-" * 30)
    print(f"✅ 筛选完成！")
    print(f"输入文件条数: {len(cot_data)}")
    print(f"正确样本条数: {len(filtered_data)}")
    print(f"结果已保存至: {OUTPUT_FILE}")
    print("-" * 30)

if __name__ == "__main__":
    main()