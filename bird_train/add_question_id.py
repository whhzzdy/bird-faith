import json
import os

# ================= 配置区域 =================
# 你的原始 train.json 路径
INPUT_FILE = "./train/train.json"
# 处理后生成的新文件路径（会保留原文件，不覆盖）
OUTPUT_FILE = "./train/train_with_id.json"
# ===========================================

def add_question_id_to_train_json():
    # 1. 检查原文件是否存在
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：找不到输入文件 {INPUT_FILE}")
        return

    # 2. 读取原始 JSON 数据
    print(f"正在读取 {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 3. 批量添加 question_id
    print(f"正在为 {len(data)} 条数据添加 question_id...")
    for idx, item in enumerate(data):
        # 给每条数据加上 question_id，从 0 开始递增
        item["question_id"] = idx

    # 4. 保存新文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 处理完成！")
    print(f"   新文件已保存至: {OUTPUT_FILE}")
    print(f"   你可以直接把逻辑审计代码里的 GT_FILE 改成这个新文件路径使用！")

if __name__ == "__main__":
    add_question_id_to_train_json()