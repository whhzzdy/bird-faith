"""
使用DeepSeek API对模型生成的思维链进行严格的逻辑审计
在已验证SQL正确的基础上,进一步评估模型是否通过扎实、忠实的推理得出答案
CoT-SQL不一致	思维链计划与实际SQL逻辑相悖
Schema幻觉	    CoT引用不存在的表或列
逻辑缺陷	    推理假设在逻辑上错误
证据违反	    与外部知识矛盾
"""
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
API_KEY = ""
BASE_URL = "https://api.deepseek.com"

# 1. 输入文件：必须是刚才筛选出的【只包含正确样本】的文件
INPUT_FILE = "cleaned_predict_dev_with_cot.json"
#INPUT_FILE = "correct_1.json"

# 2. 辅助文件：用于获取 DB_ID 和 Evidence
GT_FILE = "./data/dev.json"
DB_ROOT = "./data/dev_databases/"

# 3. 输出文件
OUTPUT_FILE = "bird_cot_quality_evaluation_cn_3.json"

# 4. 并发数
MAX_WORKERS = 10
# ===========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 加载 Ground Truth 数据以构建索引
print(f"正在加载标准答案 {GT_FILE} 以获取元数据...")
GT_DATA = {}
if os.path.exists(GT_FILE):
    with open(GT_FILE, 'r', encoding='utf-8') as f:
        temp_data = json.load(f)
        # 建立 question_id 到 数据的映射
        for item in temp_data:
            GT_DATA[str(item['question_id'])] = item
else:
    print(f"❌ 错误：找不到 {GT_FILE}，无法获取 Schema 和 Evidence")
    exit()

def evaluate_single_cot(item):
    """
    调用 DeepSeek 评估单条 CoT 的质量
    """
    # 1. 获取基础信息
    q_id = str(item.get('question_id'))
    cot = item.get('thought_process', '') # 你的 BIRD 脚本里字段叫 thought_process
    sql = item.get('generated_sql', '')
    # 获取真实 Schema
    schema = item.get('schema', '')
    
    # 2. 从 GT 数据中补全 Schema 和 Evidence
    gt_item = GT_DATA.get(q_id)
    if not gt_item:
        return {**item, "eval_result": {"is_reasonable": False, "critique": "Question ID not found in dev.json"}}

    question = gt_item['question']
    evidence = gt_item.get('evidence', 'None') # BIRD 特有的外部知识
    
    

    # ----------------------------------------------------
    # [针对 BIRD 更加严格的 Prompt]
    
    # ----------------------------------------------------
    system_prompt = """You are a rigorous Logic Auditor for Text-to-SQL tasks.
Your specific goal is to evaluate the **Internal Consistency and Faithfulness** of a model's Chain of Thought (CoT). 

**CONTEXT:** 
The [Model's Generated SQL] has already been executed and correctly answered the user's question. 
Your job is to determine if the model arrived at this correct SQL through solid, faithful reasoning, or if it suffered from "Spurious Correctness" (getting the right answer via flawed logic, hallucinations, or disconnects between thought and action).

## 🕵️ STRICT REJECTION CRITERIA (Set is_reasonable: false if ANY match):

### 1. CoT-SQL Disconnect (Unfaithful Reasoning)
   - The CoT explicitly plans a specific logical operation, but the [Model's Generated SQL] does something entirely different or omits it.
   - Example: The CoT says "I need to filter for 'approved' status", but the SQL has no such WHERE clause.
   - Example: The CoT plans to use a subquery, but the SQL uses a hardcoded magic number instead.

### 2. Schema Hallucination (STRICT TEXT MATCH)
   - The CoT explains its logic using tables or columns that DO NOT exist in the provided [Database Schema].
   - ⚠️ IMPORTANT: You MUST verify this by literally searching the provided Schema text. If the column exists in the text, DO NOT call it a hallucination. If it explicitly references a non-existent column in its reasoning, REJECT.

### 3. Flawed Logical Deduction
   - The CoT makes a fundamentally wrong assumption in database logic, even if the SQL technically works on this specific database.
   - Example: The CoT says "To find the most recent loan, I will find the minimum loan_id". This is logically flawed (ID does not inherently mean date), even if the SQL `ORDER BY loan_id LIMIT 1` happened to return the correct answer due to data distribution.

### 4. Direct Evidence Violation
   - If[External Knowledge / Evidence] is provided, and the CoT explicitly contradicts it or states it will ignore it.

## ACCEPTANCE CRITERIA (Set is_reasonable: true):
   - The CoT faithfully describes the exact operations performed in the Generated SQL.
   - All tables and columns mentioned in the reasoning exist in the Schema.
   - The logical steps are valid SQL/data analysis practices. (Note: Accept ANY valid SQL approach—joins, subqueries, CTEs—as long as the CoT explains it coherently).

## OUTPUT FORMAT:
Return a JSON object strictly in this format:
{
    "is_reasonable": true, 
    "flaw_type": "None" | "CoT_SQL_Disconnect" | "Schema_Hallucination" | "Flawed_Logic" | "Evidence_Violation",
    "critique": "Point out EXACTLY where the CoT's logic breaks down or contradicts the SQL. If reasonable, briefly praise the alignment.",
    "critique_zh": "请将 critique 翻译成流畅的中文，指出思维链在哪个环节出现了逻辑断层或事实错误。"
}"""
    
    user_prompt = f"""
[Database Schema]
{schema}

[External Knowledge / Evidence]
{evidence}

[User Question]
{question}

[Model's Chain of Thought]
{cot}

[Model's Generated SQL]
{sql}

Please audit this reasoning based on the checklist.
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        result = json.loads(response.choices[0].message.content)
        
        # 将评估结果写入 item
        item['eval_result'] = result
        return item
        
    except Exception as e:
        print(f"⚠️ Eval API Error ID {q_id}: {e}")
        item['eval_result'] = {"is_reasonable": False, "critique": f"API Error: {str(e)}"}
        return item

def main():
    print(f"🚀 [BIRD Evaluation] 开始逻辑审计...")
    print(f"输入文件: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 未找到文件: {INPUT_FILE}")
        return

    # 读取刚才筛选出来的正确样本
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    total_tasks = len(data)
    print(f"待评估样本数: {total_tasks}")
    
    results = []
    
    # 启动多线程评估
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(evaluate_single_cot, item): item for item in data}
        
        for future in tqdm(as_completed(future_to_item), total=total_tasks, desc="Auditing CoT"):
            try:
                processed_item = future.result()
                results.append(processed_item)
            except Exception as e:
                print(f"Thread Error: {e}")

    # 保存结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ 评测完成！")
    print(f"   已保存至: {OUTPUT_FILE}")
    
    # 简单的统计
    reasonable_count = sum(1 for item in results if item.get('eval_result', {}).get('is_reasonable') is True)
    print(f"   逻辑合理的思维链比例: {reasonable_count}/{total_tasks} ({reasonable_count/total_tasks:.2%})")

if __name__ == "__main__":
    main()
