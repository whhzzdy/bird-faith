#判断逻辑模型
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
API_KEY = ""
BASE_URL = "https://api.deepseek.com"

# 1. 输入文件：上一轮评估生成的 JSON (包含 eval_result)
INPUT_FILE = "bird_cot_quality_evaluation_cn.json"

# 2. 辅助文件：用于重新获取 Schema 和 Evidence (保证上下文完整)
GT_FILE = "./data/dev.json"
DB_ROOT = "./data/dev_databases/"

# 3. 输出文件：最终确认的“真·逻辑错误”样本
OUTPUT_FILE = "final_verified_logic_errors_cn.json"

# 4. 并发数
MAX_WORKERS = 10
# ===========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 加载 Ground Truth 数据
print(f"正在加载标准答案 {GT_FILE} ...")
GT_DATA = {}
if os.path.exists(GT_FILE):
    with open(GT_FILE, 'r', encoding='utf-8') as f:
        temp_data = json.load(f)
        for item in temp_data:
            GT_DATA[str(item['question_id'])] = item

def get_db_schema(db_id):
    """从 SQLite 文件读取 Schema"""
    db_path = os.path.join(DB_ROOT, db_id, f"{db_id}.sqlite")
    if not os.path.exists(db_path): return ""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
        return "\n".join([t[0] for t in cursor.fetchall() if t[0]])
    except: return ""

def verify_single_case(item):
    """
    二轮复核：判断初审的 Critique 是否合理
    """
    q_id = str(item.get('question_id'))
    
    # 1. 获取上一轮的“起诉书”
    prev_eval = item.get('eval_result', {})
    prev_critique = prev_eval.get('critique', '')
    prev_flaw = prev_eval.get('flaw_type', 'Unknown')
    
    # 2. 补全上下文
    gt_item = GT_DATA.get(q_id)
    if not gt_item: return None
    
    db_id = gt_item['db_id']
    schema = get_db_schema(db_id)
    evidence = gt_item.get('evidence', 'None')
    question = gt_item['question']
    
    cot = item.get('thought_process', '')
    sql = item.get('generated_sql', '')

    # ----------------------------------------------------
    # [V3.0 法官提示词：仲裁模式]
    # ----------------------------------------------------
    system_prompt = """You are a Senior Data Science Judge. 
You are reviewing a disputed evaluation of a Text-to-SQL task.
A junior auditor has flagged a model's Chain of Thought (CoT) as having a "Logical Error", even though the final SQL executes correctly.

**Your Goal:** Determine if the junior auditor's critique is valid (True Positive) or if they are being too pedantic/wrong (False Positive).

**CRITERIA FOR A "REAL" ERROR:**
1. **Hallucination**: The CoT explicitly invents columns/tables that don't exist.
2. **Ignored Evidence**: The CoT clearly violates the provided External Knowledge (Evidence).
3. **Logic Contradiction**: The CoT says "Select Max" but the SQL does "Select Min".

**CRITERIA FOR "ACCEPTABLE" (False Alarm):**
1. **Implicit Reasoning**: The CoT didn't explicitly quote the Evidence, but the logic clearly implies they understood it.
2. **Different Phrasing**: The CoT used different words but meant the same logical operation.
3. **Minor Omission**: The CoT skipped a trivial step (like "I will output the result") but the core logic is sound.

**INPUTS:**
- Context: Schema, Evidence, Question
- Student Work: CoT, SQL
- Junior Auditor's Critique: [Why they failed it]

**OUTPUT:**
Return a JSON object:
{
    "confirm_error": true,  // Set True if the error is REAL and SIGNIFICANT. Set False if the CoT is actually acceptable.
    "final_verdict": "Detailed explanation of why you agree or disagree with the auditor."
}
"""

    user_prompt = f"""
[Context]
Schema: {schema}
Evidence: {evidence}
Question: {question}

[Student's CoT]
{cot}

[Student's SQL]
{sql}

[Junior Auditor's Accusation]
Flaw Type: {prev_flaw}
Critique: {prev_critique}

---------------------------
Judge, is this a real logical error?
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
        
        # 将复核结果写入 item
        item['judge_result'] = result
        return item
        
    except Exception as e:
        print(f"⚠️ Judge API Error ID {q_id}: {e}")
        return None

def main():
    print("🚀 [Step 4] 开始二轮复核 (Judge Review)...")
    
    # 1. 读取上一轮结果
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到上一轮文件: {INPUT_FILE}")
        return
        
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. 筛选出被判错误的样本 (Is_Reasonable = False)
    # 我们只复核那些被认为是错的，看看是不是冤枉了
    suspects = [item for item in data if item.get('eval_result', {}).get('is_reasonable') is False]
    
    print(f"总样本数: {len(data)}")
    print(f"初审判错样本数 (待复核): {len(suspects)}")
    
    if not suspects:
        print("🎉 没有需要复核的错误样本！")
        return

    verified_errors = []
    
    # 3. 并发复核
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(verify_single_case, item): item for item in suspects}
        
        for future in tqdm(as_completed(future_to_item), total=len(suspects), desc="Judging"):
            try:
                res = future.result()
                if res and res['judge_result']['confirm_error']:
                    # 只有法官确认是真的错，才加入最终列表
                    verified_errors.append(res)
            except Exception as e:
                print(f"Error: {e}")

    # 4. 保存最终结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(verified_errors, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ 复核完成！")
    print(f"   初审认为错误: {len(suspects)}")
    print(f"   法官确认错误: {len(verified_errors)}")
    print(f"   冤假错案平反: {len(suspects) - len(verified_errors)}")
    print(f"   最终实锤的错误样本已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
