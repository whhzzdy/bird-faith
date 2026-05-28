import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
API_KEY = ""
BASE_URL = "https://api.deepseek.com"

# 1. 输入文件：必须是 Step 2 生成的【只包含正确样本】的文件
#INPUT_FILE = "output/deepseek/spider_correct_execution.json"
INPUT_FILE = "output/deepseek/error_cot.json"

# 2. 输出文件：包含评估结果
#OUTPUT_FILE = "output/deepseek/spider_evaluated.json"
OUTPUT_FILE = "output/deepseek/error_cot_test_2.json"

# 3. 并发数：评估通常比较快，可以开大点
MAX_WORKERS = 15 
# ===========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def evaluate_single_cot(item):
    """
    调用 DeepSeek 评估单条 CoT 的质量
    """
    # 提取 Step 1 生成的信息
    schema = item.get('schema', '')
    question = item.get('question', '')
    cot = item.get('deepseek_reason', 'No CoT provided')
    sql = item.get('deepseek_pred', '')

    # ----------------------------------------------------
    # [V9 高可靠平衡版 Prompt]
    # 目标：精准打击"逻辑漏洞"和"幻觉"，但放行清晰的"直接逻辑映射"。
    # ----------------------------------------------------
    system_prompt = """You are a SQL Reasoning Chain Auditor. Your task is to rigorously evaluate whether a Chain of Thought (CoT) contains logical errors or hallucinations, even if it leads to the correct SQL.

## CRITICAL EVALUATION RULES:

### ❌ **REJECT if ANY of these issues exist:**

**1. SCHEMA HALLUCINATION (Highest Priority)**
   - CoT mentions ANY table, column, or schema element NOT present in the provided schema
   - This includes: 
     * Guessing/assuming non-existent columns (e.g., "there might be a Date column")
     * Misremembering schema details (e.g., "I recall the table has X column")
     * Hypothetical schema elements (e.g., "in some databases there would be...")
   - **Exception**: Question values (e.g., 'Volvo', '2014') are NOT hallucinations

**2. LOGICAL MISMATCH / MISINTERPRETATION**
   - CoT explains SQL operations with WRONG reasoning
   - Example: Says "INTERSECT avoids duplicate records" when it actually finds common elements
   - Example: Misinterprets JOIN logic or filter conditions

**3. UNEXPLAINED OPERATIONS**
   - SQL contains ORDER BY/LIMIT/GROUP BY/DISTINCT but CoT doesn't explain WHY
   - SQL has specific WHERE conditions but CoT doesn't justify them

**4. CONTRADICTION OR OMISSION**
   - CoT mentions a filter/condition that SQL omits
   - SQL includes something CoT never planned for

### ✅ **ACCEPT ONLY IF:**
1. **Schema Faithful**: All referenced tables/columns exist in schema
2. **Logically Sound**: Every SQL operation is correctly explained
3. **Complete Reasoning**: No unexplained critical operations
4. **Consistent**: CoT and SQL align perfectly in intent and logic

Output strictly in JSON format:
{
    "is_reasonable": true, 
    "critique": "Detailed analysis. If false, specify EXACT error type and evidence."
}"""
    
    user_prompt = f"""
[Database Schema]
{schema}

[User Question]
{question}

[Model's Chain of Thought]
{cot}

[Model's Generated SQL]
{sql}

Please audit this reasoning.
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat", # 评估可以用 chat，也可以用 coder
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0 # 评估必须客观，温度设为0
        )
        result = json.loads(response.choices[0].message.content)
        
        # 将评估结果写入 item
        item['eval_result'] = result
        return item
        
    except Exception as e:
        print(f"⚠️ Eval API Error: {e}")
        # 如果 API 挂了，默认标记为 False 并记录原因
        item['eval_result'] = {"is_reasonable": False, "critique": f"API Error: {str(e)}"}
        return item

def main():
    print(f"🚀 [Step 3] 开始逻辑审计 (Agent Review)...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 未找到 Step 2 的输出文件: {INPUT_FILE}")
        print("请先运行 2_filter_correct.py")
        return

    data = json.load(open(INPUT_FILE, 'r', encoding='utf-8'))
    total_tasks = len(data)
    
    if total_tasks == 0:
        print("⚠️ 输入文件为空，没有正确执行的样本需要评估。")
        return
        
    print(f"   待评估样本数 (执行正确的): {total_tasks}")
    
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
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ Step 3 完成！")
    print(f"   已保存至: {OUTPUT_FILE}")
    print("   (包含字段 'eval_result': {'is_reasonable': ..., 'critique': ...})")

if __name__ == "__main__":
    main()
