import json
import os

# ================= 配置区域 =================
# 1. 原始的 prompt 文件 (你想过滤的文件)
INPUT_PROMPT_FILE = "dev_prompt.jsonl"

# 2. 评测结果文件 (修改后的 evaluation.py 生成的)
EVAL_RESULT_FILE = "evaluation_results.json"

# 3. 输出的新文件 (只包含做对的 prompt)
OUTPUT_PROMPT_FILE = "correct_dev_prompt.jsonl"
# ===========================================

def main():
    print("🚀 开始根据评测结果筛选 dev_prompt.jsonl ...")

    # 1. 检查文件是否存在
    if not os.path.exists(INPUT_PROMPT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_PROMPT_FILE}")
        return
    if not os.path.exists(EVAL_RESULT_FILE):
        print(f"❌ 找不到评测结果: {EVAL_RESULT_FILE} (请先运行修改后的 evaluation.py)")
        return

    # 2. 获取所有做对的题目索引 (sql_idx)
    print("正在读取评测结果...")
    correct_indices = set()
    with open(EVAL_RESULT_FILE, 'r', encoding='utf-8') as f:
        results = json.load(f)
        for item in results:
            # item 示例: {"sql_idx": 0, "res": 1}
            if item['res'] == 1:
                correct_indices.add(item['sql_idx'])
    
    print(f"共发现 {len(correct_indices)} 道做对的题目。")

    # 3. 逐行读取 jsonl 并筛选
    print(f"正在读取 {INPUT_PROMPT_FILE} 并写入 {OUTPUT_PROMPT_FILE} ...")
    
    kept_count = 0
    total_count = 0
    
    with open(INPUT_PROMPT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_PROMPT_FILE, 'w', encoding='utf-8') as f_out:
        
        for idx, line in enumerate(f_in):
            # 这里的 idx 就是行号，对应 evaluation_results 里的 sql_idx
            if idx in correct_indices:
                f_out.write(line) # 直接写入原内容，保持格式不变
                kept_count += 1
            
            total_count += 1

    # 4. 结果摘要
    print("-" * 30)
    print(f"✅ 筛选完成！")
    print(f"原始文件行数: {total_count}")
    print(f"正确题目行数: {kept_count}")
    print(f"生成文件: {OUTPUT_PROMPT_FILE}")
    
    if kept_count != len(correct_indices):
        print("⚠️ 警告: 原始文件行数少于评测结果中的最大索引，可能文件不匹配！")
    else:
        print("数据一致性检查通过。")

if __name__ == "__main__":
    main()