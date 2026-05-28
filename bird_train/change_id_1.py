#改变predict_train.json的序号
import json

# ===================== 请修改这里的文件路径 =====================
input_file = "exp_result/train_output_2/predict_train.json"    # 你的原始json路径
output_file = "bird_train/predict_train.json"  # 修改后保存的新文件
start_num = 500                 # 从500开始计数
# =================================================================

# 读取原始JSON
with open(input_file, "r", encoding="utf-8") as f:
    original_data = json.load(f)

# 按原来顺序重新编号
old_keys = list(original_data.keys())
new_data = {}

for idx, old_key in enumerate(old_keys):
    new_key = str(start_num + idx)  # 500,501,502...转字符串键
    new_data[new_key] = original_data[old_key]

# 格式化保存（和你截图缩进排版一致，不压缩）
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

print("修改完成！")
print(f"原始共 {len(old_keys)} 条数据")
print(f"新编号范围：{start_num} ~ {start_num + len(old_keys) - 1}")