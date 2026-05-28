#思维链纠正
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
import threading

# ================= 配置区域 =================
API_KEY = "sk-9885898f801645a890417229eeb56d78"  # 替换为你的 API KEY
BASE_URL = "https://api.deepseek.com"

# 输入文件：第一步评估的结果
INPUT_FILE = "decompose/error_samples_only_1.json"
# 输出文件：纠正后的完整结果
OUTPUT_FILE = "decompose/self_correction_results_1.json"
# 统计报告文件
REPORT_FILE = "decompose/self_correction_report_1.json"

MAX_WORKERS = 10
MAX_RETRIES = 3
TEST_LIMIT = None  # 测试时设为 10，全量运行设为 None
BLIND_MODE = True  # True = 不提供 final_sql, False = 原始方式
# ===========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
write_lock = threading.Lock()


def correct_single_step(schema, evidence, question, previous_steps, error_step_content, critique, final_sql, blind=False):
    """
    根据 PRM 的 critique 修正出错的步骤
    blind=True 时不提供 final_sql，迫使模型基于推理逻辑修复
    """
    if blind:
        system_prompt = """You are a Chain-of-Thought Correction Assistant for Text-to-SQL tasks.
Your task is to correct a specific step in a reasoning chain that has been identified as flawed.

**CRITICAL RULES:**
1. **Targeted Correction**: Only correct the specific step mentioned. Do NOT modify other steps.
2. **Follow the Critique**: The provided critique explains exactly what is wrong. Address the specific issue raised.
3. **Preserve Intent**: The corrected step must remain logically connected to the previous steps and contribute coherently to the overall solution.
4. **Self-Contained**: The corrected step must be a complete, standalone logical statement. Replace any pronouns with actual entity names.
5. **Output ONLY the corrected step text**: Do NOT output explanations, apologies, or additional text. Return ONLY the corrected step content.

**OUTPUT FORMAT:**
Return a JSON object with a single field:
{
    "corrected_content": "The corrected step text here."
}"""

        user_prompt = f"""[Database Schema]
{schema}

[External Knowledge]
{evidence if evidence else 'None'}

[User Question]
{question}

[Previous Steps (Context)]
{previous_steps if previous_steps else "None. This is the first step."}

--------------------------------------------------
[Error Step to Correct]
{error_step_content}

[Critique - Why This Step Is Wrong]
{critique}
--------------------------------------------------

Please correct the Error Step based on the Critique. Return ONLY the corrected step text."""
    else:
        system_prompt = """You are a Chain-of-Thought Correction Assistant for Text-to-SQL tasks.
Your task is to correct a specific step in a reasoning chain that has been identified as flawed.

**CRITICAL RULES:**
1. **Targeted Correction**: Only correct the specific step mentioned. Do NOT modify other steps.
2. **Follow the Critique**: The provided critique explains exactly what is wrong. Address the specific issue raised.
3. **Preserve Intent**: The corrected step must remain logically connected to the previous steps and continue to support the Final SQL's approach.
4. **Self-Contained**: The corrected step must be a complete, standalone logical statement. Replace any pronouns with actual entity names.
5. **Output ONLY the corrected step text**: Do NOT output explanations, apologies, or additional text. Return ONLY the corrected step content.

**OUTPUT FORMAT:**
Return a JSON object with a single field:
{
    "corrected_content": "The corrected step text here."
}"""

        user_prompt = f"""[Database Schema]
{schema}

[External Knowledge]
{evidence if evidence else 'None'}

[User Question]
{question}

[Final Generated SQL (For Context Only)]
{final_sql}

[Previous Steps (Context)]
{previous_steps if previous_steps else "None. This is the first step."}

--------------------------------------------------
[Error Step to Correct]
{error_step_content}

[Critique - Why This Step Is Wrong]
{critique}
--------------------------------------------------

Please correct the Error Step based on the Critique. Return ONLY the corrected step text."""

    for _ in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            return result.get("corrected_content", error_step_content)
        except Exception as e:
            time.sleep(2)
    
    return error_step_content  # 失败则返回原步骤


def evaluate_single_step(schema, evidence, question, previous_steps, current_step, final_sql):
    """
    调用大模型 API 评估单一逻辑步骤（使用修改后的 PRM prompt）
    """
    system_prompt = """You are a rigorous Process Reward Model (PRM) Auditor for Text-to-SQL tasks.
Judge the logical soundness of the **[Current Step]** in a Chain of Thought.

**CONTEXT:**
- The [Final Generated SQL] has already been executed and is correct.
- Judge ONLY the [Current Step] based on [Database Schema], [External Knowledge], [Previous Steps].

## REJECTION CRITERIA (set is_valid: false, assign the matching flaw_type)

### Schema_Hallucination
The step names a **table or column** not found in the Schema (search the Schema text literally).
- **Column/Table names**: STRICT — any name absent from Schema → reject.
- **Data values**: LENIENT — values like `status = 'active'` come from the question or domain knowledge, not Schema examples. Do NOT reject because a value is missing from `-- example: [...]`.

### Evidence_Violation vs Logic_Error (use this decision table)

| Condition | Classification |
|---|---|
| External Knowledge is logically sound AND step silently deviates from it | Evidence_Violation |
| External Knowledge has a mathematical/logical error AND step blindly copies it | Logic_Error |
| External Knowledge has an error AND step explicitly identifies and corrects it | ACCEPT (is_valid: true) |
| Step makes a deduction error, invalid JOIN, or premature impossibility claim | Logic_Error |

If the rule says "X means Y" and the step uses Z without justification → Evidence_Violation.
Do NOT treat a reasonable domain convention (e.g., "WasCompiled = 0 means needs compilation") as "illogical."

### CoT-SQL Disconnect
- **Contradiction**: Step finalizes a SQL decision (e.g., "use LEFT JOIN") inconsistent with the Final SQL.
- **Unexplained Magic**: Final SQL has major operations (JOIN, WHERE, GROUP BY, ORDER BY) never explained in any step.
- **Exemptions**: Exploratory reasoning ("use A or B") is fine if one option aligns. JOIN and subquery are equivalent for the same logical relationship — only flag if the relationship differs.

## ACCEPTANCE CRITERIA (is_valid: true)
- **Transitional** steps ("Let's check the schema") — harmless, do not reject for incompleteness.
- **Faithful** description aligned with Schema, Evidence, and Final SQL.
- **Tolerant phrasing** — minor imprecision that doesn't change the computation is acceptable.

**OUTPUT FORMAT:**
Return a JSON object:
{
    "is_valid": true,
    "flaw_type": "None" | "Schema_Hallucination" | "Evidence_Violation" | "Logic_Error" | "CoT_SQL_Disconnect",
    "critique": "Brief justification for this decision. Be concise.",
}"""

    user_prompt = f"""[Database Schema]
{schema}

[External Knowledge]
{evidence if evidence else 'None'}

[User Question]
{question}

[Final Generated SQL (For Context Only)]
{final_sql}

[Previous Steps (Context)]
{previous_steps if previous_steps else "None. This is the first step."}

--------------------------------------------------
[Current Step to Evaluate]
{current_step}
--------------------------------------------------
Evaluate the validity of the Current Step.
"""

    for _ in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            time.sleep(2)
    
    return {
        "is_valid": False,
        "flaw_type": "API_Error",
        "critique": "Failed to evaluate step due to API errors."
    }


def re_evaluate_corrected_item(item, corrected_steps_map):
    """
    对纠正后的样本重新进行 PRM 评估
    corrected_steps_map: {step_id: corrected_content}
    """
    schema = item.get("schema", "")
    question = item.get("question", "")
    evidence = item.get("evidence", "")
    final_sql = item.get("generated_sql", "")
    steps = item.get("decomposed_steps", [])
    
    new_step_evaluations = []
    previous_steps_text = ""
    overall_valid = True
    first_error_step = None
    
    for step in steps:
        step_id = step.get("step_id")
        
        # 如果这一步被纠正过，使用纠正后的内容
        if step_id in corrected_steps_map:
            content = corrected_steps_map[step_id]
        else:
            content = step.get("content")
        
        eval_res = evaluate_single_step(schema, evidence, question, previous_steps_text, content, final_sql)
        
        step_record = {
            "step_id": step_id,
            "content": content,
            "original_content": step.get("content"),  # 保留原始内容用于对比
            "is_corrected": step_id in corrected_steps_map,
            "evaluation": eval_res
        }
        new_step_evaluations.append(step_record)
        
        previous_steps_text += f"Step {step_id}: {content}\n"
        
        if not eval_res.get("is_valid") and overall_valid:
            overall_valid = False
            first_error_step = step_id
    
    return {
        "step_evaluations": new_step_evaluations,
        "overall_valid": overall_valid,
        "first_error_step": first_error_step
    }


def process_item_for_correction(item):
    q_id = str(item.get("question_id", "0"))
    
    schema = item.get("schema", "")
    question = item.get("question", "")
    evidence = item.get("evidence", "")
    final_sql = item.get("generated_sql", "")
    steps = item.get("decomposed_steps", [])
    original_evaluations = item.get("step_evaluations", [])
    
    # 直接找出所有出错的步骤（文件里全是错误样本，不需要先判断）
    error_steps = []
    for eval_record in original_evaluations:
        if not eval_record.get("evaluation", {}).get("is_valid", True):
            error_steps.append({
                "step_id": eval_record.get("step_id"),
                "content": eval_record.get("content"),
                "critique": eval_record.get("evaluation", {}).get("critique", ""),
                "flaw_type": eval_record.get("evaluation", {}).get("flaw_type", "")
            })
    
    # 纠正每个错误步骤
    corrected_steps_map = {}
    correction_records = []
    
    for error_step in error_steps:
        step_id = error_step["step_id"]
        
        previous_steps_text = ""
        for step in steps:
            if step.get("step_id") == step_id:
                break
            previous_steps_text += f"Step {step.get('step_id')}: {step.get('content')}\n"
        
        corrected_content = correct_single_step(
            schema, evidence, question, previous_steps_text,
            error_step["content"], error_step["critique"], final_sql,
            blind=BLIND_MODE
        )
        
        corrected_steps_map[step_id] = corrected_content
        correction_records.append({
            "step_id": step_id,
            "original_content": error_step["content"],
            "critique": error_step["critique"],
            "flaw_type": error_step["flaw_type"],
            "corrected_content": corrected_content
        })
    
    # 重新评估
    re_eval_result = re_evaluate_corrected_item(item, corrected_steps_map)
    
    return {
        "question_id": q_id,
        "corrections_made": correction_records,
        "re_evaluation": re_eval_result,
        "overall_valid_after": re_eval_result["overall_valid"]
    }


def generate_report(results):
    """生成统计报告"""
    total = len(results)
    
    # 纠正前
    invalid_before = total
    
    # 纠正后
    invalid_after = sum(1 for r in results if not r.get("overall_valid_after", True))
    fixed_samples = invalid_before - invalid_after
    
    # 步骤级别统计
    total_error_steps_before = 0
    total_corrected_steps = 0
    successfully_fixed_steps = 0
    
    for r in results:
        corrections = r.get("corrections_made", [])
        total_error_steps_before += len(corrections)
        
        re_eval = r.get("re_evaluation")
        if re_eval:
            for step_eval in re_eval.get("step_evaluations", []):
                if step_eval.get("is_corrected"):
                    total_corrected_steps += 1
                    if step_eval.get("evaluation", {}).get("is_valid", False):
                        successfully_fixed_steps += 1
    
    still_error_steps = total_corrected_steps - successfully_fixed_steps
    
    # 按错误类型统计纠正效果
    flaw_type_stats = {}
    for r in results:
        for correction in r.get("corrections_made", []):
            flaw_type = correction.get("flaw_type", "Unknown")
            if flaw_type not in flaw_type_stats:
                flaw_type_stats[flaw_type] = {"total": 0, "fixed": 0}
            flaw_type_stats[flaw_type]["total"] += 1
    
    # 检查哪些被修复了
    for r in results:
        re_eval = r.get("re_evaluation")
        if re_eval:
            for step_eval in re_eval.get("step_evaluations", []):
                if step_eval.get("is_corrected") and step_eval.get("evaluation", {}).get("is_valid", False):
                    for correction in r.get("corrections_made", []):
                        if correction["step_id"] == step_eval["step_id"]:
                            flaw_type = correction.get("flaw_type", "Unknown")
                            if flaw_type in flaw_type_stats:
                                flaw_type_stats[flaw_type]["fixed"] += 1
                            break
    
    report = {
        "总样本数": total,
        "纠正前": {
            "思维链正确样本数": total - invalid_before,
            "思维链错误样本数": invalid_before,
            "错误率": f"{invalid_before/total:.2%}" if total > 0 else "0%"
        },
        "纠正后": {
            "思维链正确样本数": total - invalid_after,
            "思维链错误样本数": invalid_after,
            "错误率": f"{invalid_after/total:.2%}" if total > 0 else "0%",
            "被修复的样本数": fixed_samples,
            "样本修复率": f"{fixed_samples/invalid_before:.2%}" if invalid_before > 0 else "0%"
        },
        "步骤级别": {
            "纠正前错误步骤总数": total_error_steps_before,
            "尝试纠正的步骤数": total_corrected_steps,
            "成功修复的步骤数": successfully_fixed_steps,
            "仍然错误的步骤数": still_error_steps,
            "步骤修复率": f"{successfully_fixed_steps/total_corrected_steps:.2%}" if total_corrected_steps > 0 else "0%"
        },
        "按错误类型统计修复效果": {}
    }
    
    for flaw_type, stats in flaw_type_stats.items():
        fix_rate = f"{stats['fixed']/stats['total']:.2%}" if stats['total'] > 0 else "0%"
        report["按错误类型统计修复效果"][flaw_type] = {
            "总错误数": stats["total"],
            "成功修复数": stats["fixed"],
            "修复率": fix_rate
        }
    
    return report


def main():
    print("🚀 开始 Self-Correction Baseline 实验...")
    print("阶段一：识别错误步骤并纠正")
    print("阶段二：重新 PRM 评估纠正后的思维链")
    print("=" * 60)
    
    # ========== 修改 1：输入文件改为筛选后的错误样本文件 ==========
    INPUT_FILE = "decompose/error_samples_only.json"
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到文件: {INPUT_FILE}")
        return
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if TEST_LIMIT:
        print(f"⚠️ [测试模式] 仅截取前 {TEST_LIMIT} 条样本执行。")
        data = data[:TEST_LIMIT]
    else:
        print(f"📊 [全量模式] 共加载 {len(data)} 条错误样本。")

    # Blind 模式使用独立输出文件
    if BLIND_MODE:
        OUTPUT_FILE = "decompose/self_correction_results_blind_1.json"
        REPORT_FILE = "decompose/self_correction_report_blind_1.json"
    else:
        OUTPUT_FILE = "decompose/self_correction_results_1.json"
        REPORT_FILE = "decompose/self_correction_report_1.json"
    print(f"{'🔀 [Blind Mode] 不提供 final_sql' if BLIND_MODE else '🔁 [Non-Blind Mode] 使用 final_sql'}")
    
    results = []
    
    # 并发处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(process_item_for_correction, item): item for item in data}
        
        for future in tqdm(as_completed(future_to_item), total=len(data), desc="Self-Correction Progress"):
            try:
                processed_item = future.result()
                with write_lock:
                    results.append(processed_item)
            except Exception as e:
                print(f"Error processing item: {e}")
    
    # 排序
    results.sort(key=lambda x: int(x.get("question_id", 0)) if str(x.get("question_id", 0)).isdigit() else str(x.get("question_id")))
    
    # 保存纠正结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # 生成报告
    report = generate_report(results)
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    
    # ========== 修改 2：打印报告（去掉"正确样本数"，因为全部是错误样本） ==========
    print("\n" + "=" * 60)
    print("📊 Self-Correction Baseline 实验报告")
    print("=" * 60)
    print(f"\n待纠正的错误样本数: {report['总样本数']}")
    
    print(f"\n【纠正前】")
    print(f"  思维链错误样本数: {report['纠正前']['思维链错误样本数']}")
    
    print(f"\n【纠正后】")
    print(f"  思维链错误样本数: {report['纠正后']['思维链错误样本数']}")
    print(f"  被修复的样本数: {report['纠正后']['被修复的样本数']}")
    print(f"  样本修复率: {report['纠正后']['样本修复率']}")
    
    print(f"\n【步骤级别】")
    print(f"  纠正前错误步骤总数: {report['步骤级别']['纠正前错误步骤总数']}")
    print(f"  尝试纠正的步骤数: {report['步骤级别']['尝试纠正的步骤数']}")
    print(f"  成功修复的步骤数: {report['步骤级别']['成功修复的步骤数']}")
    print(f"  仍然错误的步骤数: {report['步骤级别']['仍然错误的步骤数']}")
    print(f"  步骤修复率: {report['步骤级别']['步骤修复率']}")
    
    if report["按错误类型统计修复效果"]:
        print(f"\n【按错误类型统计修复效果】")
        for flaw_type, stats in report["按错误类型统计修复效果"].items():
            print(f"  {flaw_type}:")
            print(f"    总错误数: {stats['总错误数']}")
            print(f"    成功修复数: {stats['成功修复数']}")
            print(f"    修复率: {stats['修复率']}")
    
    print(f"\n✅ 结果已保存至:")
    print(f"   纠正结果: {OUTPUT_FILE}")
    print(f"   统计报告: {REPORT_FILE}")

    # Blind 模式下与非 Blind 对比
    if BLIND_MODE:
        non_blind_report_file = "decompose/self_correction_report.json"
        if os.path.exists(non_blind_report_file):
            with open(non_blind_report_file, 'r', encoding='utf-8') as f:
                nb_report = json.load(f)
            print("\n" + "=" * 60)
            print("🔍 Blind vs Non-Blind 对比")
            print("=" * 60)
            nb_fix_rate = nb_report.get("纠正后", {}).get("样本修复率", "N/A")
            b_fix_rate = report.get("纠正后", {}).get("样本修复率", "N/A")
            nb_step_rate = nb_report.get("步骤级别", {}).get("步骤修复率", "N/A")
            b_step_rate = report.get("步骤级别", {}).get("步骤修复率", "N/A")
            print(f"  样本修复率:   Non-Blind {nb_fix_rate}  |  Blind {b_fix_rate}")
            print(f"  步骤修复率:   Non-Blind {nb_step_rate}  |  Blind {b_step_rate}")
            for flaw_type in ["Logic_Error", "Evidence_Violation", "Schema_Hallucination", "CoT_SQL_Disconnect"]:
                nb_stats = nb_report.get("按错误类型统计修复效果", {}).get(flaw_type, {})
                b_stats = report.get("按错误类型统计修复效果", {}).get(flaw_type, {})
                if nb_stats or b_stats:
                    nb_fix = nb_stats.get("修复率", "N/A")
                    b_fix = b_stats.get("修复率", "N/A")
                    print(f"  {flaw_type}:    Non-Blind {nb_fix}  |  Blind {b_fix}")
        else:
            print(f"\n⚠️ 未找到 Non-Blind 报告 ({non_blind_report_file})，跳过对比。")


if __name__ == "__main__":
    main()