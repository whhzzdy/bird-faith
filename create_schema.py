import json
import re
import os

# ================= 配置区域 =================
INPUT_FILE = "correct_predict_dev_with_cot.json"
OUTPUT_FILE = "cleaned_predict_dev_with_cot.json"
# ===========================================

def extract_schema_only(prompt_text):
    """
    仅从原始 prompt 中提取 schema，取消 question 提取逻辑
    """
    # 1. 提取 Schema
    # 匹配 "Database Schema:" 之后，直到 "This schema describes" 或 "Question:" 之前的内容
    schema_pattern = r"Database Schema:\n(.*?)\n(?:This schema describes|Question:)"
    schema_match = re.search(schema_pattern, prompt_text, re.DOTALL)
    schema_text = schema_match.group(1).strip() if schema_match else ""

    # 取消返回 question，只返回 schema
    return schema_text

def main():
    print(f"🚀 开始处理 {INPUT_FILE} ...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到文件: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 兼容处理：判断最外层是 list 还是 dict
    is_dict = isinstance(data, dict)
    items = data.values() if is_dict else data

    cleaned_data =[]
    
    for item in items:
        prompt_text = item.get("prompt", "")
        
        # 仅提取干净的 schema（取消 question 提取）
        schema_text = extract_schema_only(prompt_text)
        
        # 重新组装 JSON 结构 (让重要的字段排在前面)
        new_item = {
            "question_id": item.get("question_id"),
            # 保留 schema 字段
            "schema": schema_text,
            "thought_process": item.get("thought_process", ""),
            "generated_sql": item.get("generated_sql", ""),
            "full_raw_output": item.get("full_raw_output", "")
        }
        
        # 把其他可能存在的字段 (比如 db_id, evidence) 也顺便挪过来
        for k, v in item.items():
            if k not in["prompt", "question_id", "schema", "question", "thought_process", "generated_sql", "full_raw_output"]:
                new_item[k] = v

        cleaned_data.append(new_item)

    # 如果原来的 JSON 是字典结构，这里恢复回去
    if is_dict:
        cleaned_data = {str(item["question_id"]): item for item in cleaned_data}

    # 保存新的 JSON 文件
    print(f"正在保存到 {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=4, ensure_ascii=False)
        
    print(f"✅ 处理完成！已成功提取 schema 字段，未修改 question 字段。")

if __name__ == "__main__":
    main()