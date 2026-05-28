#挑选出思维链不忠实的案例
import json

INPUT_FILE = "decompose/step_level_evaluation_results_1.json"
OUTPUT_FILE = "decompose/error_samples_only_1.json"

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 筛选 overall_valid = false 的样本
error_samples = [item for item in data if not item.get("overall_valid", False)]

print(f"总样本数: {len(data)}")
print(f"错误样本数: {len(error_samples)}")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(error_samples, f, ensure_ascii=False, indent=4)

print(f"已保存至: {OUTPUT_FILE}")