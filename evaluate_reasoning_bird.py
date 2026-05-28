import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
API_KEY = "sk-9885898f801645a890417229eeb56d78"
BASE_URL = "https://api.deepseek.com"

# 1. 输入文件：必须是刚才筛选出的【只包含正确样本】的文件
INPUT_FILE = "correct_predict_dev_with_cot.json"
#INPUT_FILE = "correct_1.json"

# 2. 辅助文件：用于获取 DB_ID 和 Evidence
GT_FILE = "./data/dev.json"
DB_ROOT = "./data/dev_databases/"

# 3. 输出文件
OUTPUT_FILE = "bird_cot_quality_evaluation_cn_2.json"

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

def get_db_schema(db_id):
    """从 SQLite 文件读取 Schema，作为评测的依据"""
    db_path = os.path.join(DB_ROOT, db_id, f"{db_id}.sqlite")
    if not os.path.exists(db_path):
        return f"Schema file not found for {db_id}"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        return "\n".join([t[0] for t in tables if t[0] is not None])
    except Exception as e:
        return f"Error reading schema: {e}"

def evaluate_single_cot(item):
    """
    调用 DeepSeek 评估单条 CoT 的质量
    """
    # 1. 获取基础信息
    q_id = str(item.get('question_id'))
    cot = item.get('thought_process', '') # 你的 BIRD 脚本里字段叫 thought_process
    sql = item.get('generated_sql', '')
    
    # 2. 从 GT 数据中补全 Schema 和 Evidence
    gt_item = GT_DATA.get(q_id)
    if not gt_item:
        return {**item, "eval_result": {"is_reasonable": False, "critique": "Question ID not found in dev.json"}}

    db_id = gt_item['db_id']
    question = gt_item['question']
    evidence = gt_item.get('evidence', 'None') # BIRD 特有的外部知识
    
    # 获取真实 Schema
    schema = get_db_schema(db_id)

    # ----------------------------------------------------
    # [针对 BIRD 优化的 Prompt]
    # 增加了对 External Knowledge (Evidence) 的检查
    # ----------------------------------------------------
    system_prompt = """You are a SQL Reasoning Chain Auditor for the BIRD benchmark. 
Your task is to strictly evaluate whether the model's Chain of Thought (CoT) is logically sound and consistent with the Schema and External Knowledge.

## EVALUATION CHECKLIST:

### 1. Schema & Knowledge Consistency
   - Does the CoT use tables/columns that actually exist in the [Database Schema]?
   - Does the CoT correctly apply the [External Knowledge]? (e.g., if Knowledge says "M means Male", does CoT use it?)
   - **Reject** if the CoT hallucinates columns not in Schema.

### 2. Logical Alignment
   - Does the reasoning step-by-step lead to the final SQL?
   - If the SQL uses `JOIN`, `WHERE`, `GROUP BY`, does the CoT explain WHY?
   - **Reject** if the CoT logic contradicts the SQL logic (e.g., CoT says "find max" but SQL uses "min").

### 3. Completeness
   - Does the CoT address the user's question completely?

Output strictly in JSON format:
{
    "is_reasonable": true, 
    "critique": "Short explanation of the quality. If false, point out the specific error."
    "critique_zh": "请将 critique 翻译成流畅的中文，使用专业的数据库术语。"
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

Please audit this reasoning.
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