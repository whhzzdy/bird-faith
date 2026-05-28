"""从三条流水线结果中抽取样本供人工标注。

Category A: PRM 判定通过（overall_valid=true）— 检查 false negative
Category B: PRM 判定错误 + 纠正成功 — 检查纠正是否真正合理
Category C: PRM 判定错误 + 纠正失败 — 分析不可修复的根因
"""

import json
import random
import os

# ================= 配置 =================
EVAL_FILE = "decompose/simplify_prompt_result/step_level_evaluation_results_1.json"
ERROR_FILE = "decompose/simplify_prompt_result/error_samples_only_1.json"
CORRECTION_FILE = "decompose/simplify_prompt_result/self_correction_results_blind_1.json"
OUTPUT_FILE = "decompose/human_review_samples.json"

SEED = 42
SAMPLES_PER_CATEGORY = 15
# ========================================

random.seed(SEED)

# 1. 加载评估全量数据
print("Loading evaluation results...")
with open(EVAL_FILE, 'r', encoding='utf-8') as f:
    all_eval = json.load(f)

# 2. 加载纠错结果
print("Loading correction results...")
with open(CORRECTION_FILE, 'r', encoding='utf-8') as f:
    all_corrections = json.load(f)

# 3. 加载错误样本（含完整 CoT 和 step evaluations）
print("Loading error samples...")
with open(ERROR_FILE, 'r', encoding='utf-8') as f:
    error_samples = json.load(f)

# ---- 构建索引 ----
error_qids = {str(s['question_id']) for s in error_samples}
correction_map = {str(c['question_id']): c for c in all_corrections}
eval_map_by_qid = {str(e['question_id']): e for e in all_eval}

# ---- Category A: PRM 判定通过的样本 ----
passed_samples = [
    e for e in all_eval
    if str(e['question_id']) not in error_qids and e.get('overall_valid')
]
print(f"Category A pool: {len(passed_samples)} passed samples")

# 随机抽
selected_a = random.sample(passed_samples, min(SAMPLES_PER_CATEGORY, len(passed_samples)))

# 构建 category A 输出
cat_a_output = []
for s in selected_a:
    cat_a_output.append({
        "category": "A_PRM_passed",
        "question_id": s['question_id'],
        "question": s['question'],
        "evidence": s['evidence'],
        "schema": s['schema'][:3000] if s.get('schema') else "",
        "generated_sql": s['generated_sql'],
        "step_evaluations": s.get('step_evaluations', []),
        "prm_overall_valid": s.get('overall_valid'),
    })

# ---- Category B & C: 从错误样本 + 纠错结果中抽取 ----
fixed_qids = []
unfixed_qids = []
for corr in all_corrections:
    qid = str(corr['question_id'])
    if corr.get('overall_valid_after'):
        fixed_qids.append(qid)
    else:
        unfixed_qids.append(qid)

print(f"Category B pool (fixed): {len(fixed_qids)}")
print(f"Category C pool (unfixed): {len(unfixed_qids)}")

# 随机抽
selected_b_qids = set(random.sample(fixed_qids, min(SAMPLES_PER_CATEGORY, len(fixed_qids))))
selected_c_qids = set(random.sample(unfixed_qids, min(SAMPLES_PER_CATEGORY, len(unfixed_qids))))

def build_category_bc_output(qid, category_name):
    """为 category B 或 C 构建人工审查用数据"""
    err = next((e for e in error_samples if str(e['question_id']) == qid), None)
    corr = correction_map.get(qid)

    if not err:
        return None

    # 提取纠错前后的对比
    corrections_detail = []
    if corr:
        for cm in corr.get('corrections_made', []):
            corrections_detail.append({
                "step_id": cm.get('step_id'),
                "original_content": cm.get('original_content'),
                "critique": cm.get('critique'),
                "corrected_content": cm.get('corrected_content'),
            })

    return {
        "category": category_name,
        "question_id": err['question_id'],
        "question": err['question'],
        "evidence": err['evidence'],
        "schema": err['schema'][:3000] if err.get('schema') else "",
        "generated_sql": err['generated_sql'],
        "step_evaluations_before": err.get('step_evaluations', []),
        "overall_valid_before": err.get('overall_valid'),
        "first_error_step_before": err.get('first_error_step'),
        "corrections": corrections_detail,
        "overall_valid_after": corr.get('overall_valid_after') if corr else None,
        "re_evaluation": corr.get('re_evaluation') if corr else None,
    }

cat_b_output = []
for qid in selected_b_qids:
    entry = build_category_bc_output(qid, "B_fixed_successfully")
    if entry:
        cat_b_output.append(entry)

cat_c_output = []
for qid in selected_c_qids:
    entry = build_category_bc_output(qid, "C_failed_to_fix")
    if entry:
        cat_c_output.append(entry)

# ---- 合并输出 ----
all_samples = cat_a_output + cat_b_output + cat_c_output
random.shuffle(all_samples)  # 打乱顺序避免标注偏差

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(all_samples, f, ensure_ascii=False, indent=2)

print(f"\nDone! {len(all_samples)} samples saved to {OUTPUT_FILE}")
print(f"  Category A (PRM passed): {len(cat_a_output)}")
print(f"  Category B (fixed):       {len(cat_b_output)}")
print(f"  Category C (unfixed):     {len(cat_c_output)}")
