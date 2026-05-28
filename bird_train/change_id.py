#改变生成predict_train_with_cot.json的序号
import json

# -------------------------- 请修改这里的文件路径 --------------------------
input_json_path = "exp_result/train_output_2/predict_train_with_cot.json"    # 输入：原始json文件
output_json_path = "bird_train/修改后_question_id从500开始.json"  # 输出：新保存文件
start_id = 500  # 从500开始计数
# ------------------------------------------------------------------------

# 1. 读取原始JSON文件
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. 重新排序并重新编号（保证顺序和原来一致）
old_keys = list(data.keys())  # 取出原来0,1,2...的顺序
new_data = {}

for new_idx, old_key in enumerate(old_keys):
    new_question_id = start_id + new_idx  # 500,501,502...
    item = data[old_key]
    item["question_id"] = new_question_id  # 修改内部question_id
    new_data[str(new_question_id)] = item # 修改外层字典键为字符串500,501...

# 3. 保存到新文件（格式化缩进，和你原图格式一致）
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

print("修改完成！")
print(f"原始条目数量：{len(old_keys)}")
print(f"新编号范围：{start_id} ~ {start_id+len(old_keys)-1}")