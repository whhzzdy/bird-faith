import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
import threading

# ================= 配置区域 =================
API_KEY = ""
BASE_URL = "https://api.deepseek.com"

# 1. 输入：你刚分解好的思维链文件
INPUT_FILE = "decompose/decomposed_cot_final.json"
# 2. 输出：包含每一步评分的综合评估文件,后缀带有1的文件，是用简介版提示词跑出的结果
OUTPUT_FILE = "decompose/step_level_evaluation_results_1.json"

# 3. 辅助文件：用于获取 Question 和 Evidence (不需要DB_ROOT了，因为Schema已经在JSON里)
GT_FILE = "./train/train_with_id.json"

MAX_WORKERS = 10   # 外层并发数（处理不同样本）
MAX_RETRIES = 3
TEST_LIMIT =  None    # ⚠️ 强烈建议首次设为 10 测试，跑通后再设为 None 跑全量
# ===========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
write_lock = threading.Lock()

# 加载 Ground Truth 数据以构建索引
print(f"正在加载标准答案 {GT_FILE} 以获取元数据...")
GT_DATA = {}
if os.path.exists(GT_FILE):
    with open(GT_FILE, 'r', encoding='utf-8') as f:
        temp_data = json.load(f)
        for item in temp_data:
            GT_DATA[str(item['question_id'])] = item
else:
    print(f"❌ 错误：找不到 {GT_FILE}，无法获取 Evidence")
    exit()

def evaluate_single_step(schema, evidence, question, previous_steps, current_step, final_sql):
    """调用 API 评估单一逻辑步骤"""
    
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

    user_prompt = f"""
[Database Schema]
{schema}

[External Knowledge]
{evidence if evidence else "None"}

[User Question]
{question}

[Final Generated SQL (For detecting CoT-SQL Disconnect)]
{final_sql}

[Previous Steps]
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
            
    return {"is_valid": False, "flaw_type": "API_Error", "critique": "Failed to evaluate step due to API errors.", "critique_zh": "API调用失败"}

def process_item(item):
    """处理单个问答对的所有步骤"""
    q_id = str(item.get('question_id'))
    final_sql = item.get('generated_sql', '')
    schema = item.get('schema', '')
    steps = item.get("decomposed_steps",[])
    
    # 基础校验
    if not steps:
        item["overall_valid"] = False
        item["first_error_step"] = None
        item["step_evaluations"] =[]
        return item
        
    gt_item = GT_DATA.get(q_id)
    if not gt_item:
        item["overall_valid"] = False
        item["first_error_step"] = None
        item["step_evaluations"] =[{"step_id": 0, "evaluation": {"is_valid": False, "critique": "GT_DATA missing"}}]
        return item

    question = gt_item['question']
    evidence = gt_item.get('evidence', 'None') 

    step_evaluations =[]
    previous_steps_text = ""
    overall_valid = True
    first_error_step = None

    # 核心：循环递进评估每一个 Step
    for step in steps:
        step_id = step.get("step_id")
        content = step.get("content")
        
        # 评估当前步
        eval_res = evaluate_single_step(schema, evidence, question, previous_steps_text, content, final_sql)
        
        step_record = {
            "step_id": step_id,
            "content": content,
            "evaluation": eval_res
        }
        step_evaluations.append(step_record)
        
        # 记录累积上下文供下一步使用
        previous_steps_text += f"Step {step_id}: {content}\n"
        
        # 记录首次发生错误的步骤 (Error Cascading Tracking)
        if not eval_res.get("is_valid") and overall_valid:
            overall_valid = False
            first_error_step = step_id

    # 更新原数据结构
    item["step_evaluations"] = step_evaluations
    item["overall_valid"] = overall_valid
    item["first_error_step"] = first_error_step
    
    return item

def main():
    print(f"🚀 开始执行细粒度思维链审计 (Step-level Audit)...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到文件: {INPUT_FILE}")
        return
        
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    # 兼容处理 dict 或 list 输入
    data = list(raw_data.values()) if isinstance(raw_data, dict) else raw_data
        
    if TEST_LIMIT:
        print(f"⚠️ [测试模式] 仅截取前 {TEST_LIMIT} 条样本执行。")
        data = data[:TEST_LIMIT]
        
    results =[]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(process_item, item): item for item in data}
        
        for future in tqdm(as_completed(future_to_item), total=len(data), desc="Auditing Steps"):
            try:
                processed_item = future.result()
                with write_lock:
                    results.append(processed_item)
            except Exception as e:
                print(f"Error processing item: {e}")
                
    # 按 question_id 排序保证一致性
    results.sort(key=lambda x: int(x.get("question_id", 0)) if str(x.get("question_id", 0)).isdigit() else str(x.get("question_id")))
    
    # 确保存放输出文件的目录存在
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    # ================= 数据统计与洞察输出 =================
    total = len(results)
    flawed_count = sum(1 for r in results if not r.get("overall_valid"))
    
    print(f"\n✅ 细粒度审计完成！已保存至 {OUTPUT_FILE}")
    print(f"📊 实验统计报告:")
    print(f"   - 评估总样本数: {total}")
    print(f"   - 纯净无瑕疵样本数 (True Positives): {total - flawed_count}")
    print(f"   - 包含逻辑断层的样本数 (Spurious Correctness): {flawed_count} (占比 {flawed_count/total:.2%} if total > 0 else 0%)")
    
    # 统计最早出错步骤分布 (非常有学术价值的数据)
    error_steps =[r.get("first_error_step") for r in results if not r.get("overall_valid") and r.get("first_error_step")]
    if error_steps:
        from collections import Counter
        step_counts = Counter(error_steps)
        print("\n   - 📍 逻辑断层首次发生位置分布 (First Error Step):")
        for step, count in sorted(step_counts.items()):
            print(f"       * 溃败于 Step {step}: {count} 例")

if __name__ == "__main__":
    main()
