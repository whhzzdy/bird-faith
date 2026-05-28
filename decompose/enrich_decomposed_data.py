import json
import os

# ================= 配置区域 =================
# 1. 你现在不完整的分解文件
INPUT_DECOMPOSED_FILE = "decompose/decomposed_cot.json"

# 2. 包含 question 和 evidence 的官方源文件
# (根据你之前发的信息，可能是 ./train/train_small.json 或 train.json)
GT_FILE = "./train/train.json" 

# 3. 融合后的终极完美文件
OUTPUT_FILE = "decompose/decomposed_cot_final.json"
# ===========================================

def main():
    print("🚀 开始数据融合与清洗...")

    if not os.path.exists(INPUT_DECOMPOSED_FILE):
        print(f"❌ 找不到输入文件: {INPUT_DECOMPOSED_FILE}")
        return
    if not os.path.exists(GT_FILE):
        print(f"❌ 找不到官方源文件: {GT_FILE}")
        return

    # 1. 加载官方源数据，建立 ID 映射字典
    print(f"正在读取官方题库 {GT_FILE} ...")
    with open(GT_FILE, 'r', encoding='utf-8') as f:
        gt_list = json.load(f)
    
    # 建立字典，方便通过 question_id 快速查找
    gt_dict = {}
    for idx, item in enumerate(gt_list):
        # 兼容处理：有的叫 question_id，有的叫 id，如果没有就用索引
        q_id = str(item.get('question_id', item.get('id', idx)))
        gt_dict[q_id] = item

    # 2. 加载待处理的分解文件
    print(f"正在读取待处理文件 {INPUT_DECOMPOSED_FILE} ...")
    with open(INPUT_DECOMPOSED_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 兼容处理字典或列表结构
    items = list(data.values()) if isinstance(data, dict) else data

    # 3. 遍历清洗与富化
    success_count = 0
    for item in items:
        q_id = str(item.get("question_id"))
        
        # 找到对应的官方题目信息
        gt_item = gt_dict.get(q_id)
        if gt_item:
            # ✨ 加入欠缺的字段
            item["question"] = gt_item.get("question", "")
            item["evidence"] = gt_item.get("evidence", "")
            success_count += 1
        else:
            print(f"⚠️ 警告: ID {q_id} 在官方题库中未找到，将填入空值。")
            item["question"] = ""
            item["evidence"] = ""

        # 🧹 删除冗余的臃肿字段
        if "full_raw_output" in item:
            del item["full_raw_output"]
            
        # 如果你也不想看原来的整体 thought_process（因为已经切片了），也可以取消下面这行的注释删掉它
        # if "thought_process" in item:
        #     del item["thought_process"]

    # 4. 排序并保存
    # 让重要的字段排在最前面，方便肉眼查看
    ordered_items =[]
    for item in items:
        ordered_item = {
            "question_id": item.get("question_id"),
            "question": item.get("question"),
            "evidence": item.get("evidence"),
            "schema": item.get("schema"),
            "generated_sql": item.get("generated_sql"),
            "decomposed_steps": item.get("decomposed_steps",[])
        }
        # 把其他未明确列出的原字段补上
        for k, v in item.items():
            if k not in ordered_item:
                ordered_item[k] = v
        ordered_items.append(ordered_item)

    ordered_items.sort(key=lambda x: int(x["question_id"]) if str(x["question_id"]).isdigit() else str(x["question_id"]))

    print(f"正在保存最终文件到 {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(ordered_items, f, ensure_ascii=False, indent=4)

    print(f"\n✅ 任务完成！成功融合 {success_count} 条数据。")
    print("👉 现在你可以放心地去修改 evaluate_decomposed_cot.py，把里面连接数据库和读取 GT_DATA 的所有代码全都删掉了！")

if __name__ == "__main__":
    main()