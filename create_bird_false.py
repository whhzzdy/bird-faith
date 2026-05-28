import json
import os

# ================= 配置区域 =================
INPUT_FILE = "bird_cot_quality_evaluation_cn_3.json"
OUTPUT_FILE = "filtered_bird_cot_false_cases.json"
# ===========================================

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

    filtered_data = []
    
    for item in items:
        # 取出 eval_result 用于判断
        eval_result = item.get("eval_result", {})
        is_reasonable = eval_result.get("is_reasonable", True)
        
        # 只保留 is_reasonable 为 false 的案例
        if not is_reasonable:
            # 复制原数据，删除 eval_result 字段
            new_item = {k: v for k, v in item.items() if k != "eval_result"}
            filtered_data.append(new_item)

    # 如果原来的 JSON 是字典结构，这里恢复回去
    if is_dict:
        filtered_data = {str(item["question_id"]): item for item in filtered_data}

    # 保存新的 JSON 文件
    print(f"正在保存到 {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)
        
    print(f"✅ 处理完成！共筛选出 {len(filtered_data)} 条 is_reasonable=false 的案例，已移除 eval_result 字段。")

if __name__ == "__main__":
    main()