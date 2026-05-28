import json
import time
import re
import os
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区域 =================
API_KEY = "" 
BASE_URL = "https://api.deepseek.com"
INPUT_FILE = "dev_prompt.jsonl"
OUTPUT_FILE = "predict_dev_1.json"
MODEL_NAME = "deepseek-chat"
MAX_WORKERS = 20  
# 为了挽救 ID 丢失的问题，我们需要读取标准答案来对齐 ID
GROUND_TRUTH_FILE = "data/dev.json" 
# ===========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def clean_sql_output(text):
    pattern = r"```sql\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.replace("```sql", "").replace("```", "").strip()

def process_single_item(item, correct_id):
    """
    item: 输入的单条数据
    correct_id: 从 dev.json 强制匹配的正确 ID
    """
    # 尝试各种可能的字段名获取 prompt
    prompt_content = item.get('prompt') or item.get('input') or item.get('prompt_content')
    
    # 优先使用 correct_id，如果没传则尝试自己找
    q_id = correct_id if correct_id is not None else (item.get('question_id') or item.get('id'))

    if not prompt_content:
        return q_id, None, "No prompt found"

    # 增强 System Prompt，减少格式错误
    messages = [
        {"role": "system", "content": "You are a professional data analyst. Output ONLY the SQL query directly. No markdown, no explanations."},
        {"role": "user", "content": prompt_content}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
            stream=False,
            timeout=60
        )
        
        raw_content = response.choices[0].message.content
        sql = clean_sql_output(raw_content)
        sql_oneline = " ".join(sql.split())
        
        return q_id, sql_oneline, None

    except Exception as e:
        return q_id, "SELECT 'error'", str(e)

def run_inference_robust():
    print(f"📂 正在读取 {INPUT_FILE}...")
    
    # 1. 读取 Prompts
    prompts = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    # 2. 读取标准答案以修正 ID (关键步骤)
    print(f"🔧 正在读取 {GROUND_TRUTH_FILE} 以修正 ID 顺序...")
    dev_ids = []
    try:
        with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
            dev_data = json.load(f)
            # 提取所有正确的 question_id
            dev_ids = [item['question_id'] for item in dev_data]
            
        if len(dev_ids) != len(prompts):
            print(f"⚠️ 警告: Prompt数量 ({len(prompts)}) 与 标准答案数量 ({len(dev_ids)}) 不一致！")
            print("程序将尝试尽力按顺序匹配...")
    except FileNotFoundError:
        print(f"❌ 错误: 找不到 {GROUND_TRUTH_FILE}。请确保该文件存在，否则 ID 可能再次出错。")
        return

    print(f"🚀 开始并发推理 | 数据量: {len(prompts)} | 并发数: {MAX_WORKERS}")

    results = {}
    errors = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for idx, item in enumerate(prompts):
            # 如果索引在范围内，强制使用 dev.json 里的正确 ID
            correct_id = dev_ids[idx] if idx < len(dev_ids) else None
            futures.append(executor.submit(process_single_item, item, correct_id))
        
        for future in tqdm(as_completed(futures), total=len(prompts)):
            q_id, sql, error = future.result()
            
            if q_id is None:
                q_id = "unknown_" + str(time.time()) # 兜底防止报错

            if error:
                # errors.append(q_id) # 这里就不打印了，直接保存 error SQL
                results[str(q_id)] = "SELECT 'error'"
            else:
                results[str(q_id)] = sql

    # 3. 保存结果
    print(f"\n💾 保存结果到 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        json.dump(results, out_f, indent=4, ensure_ascii=False)
    
    print("✅ 推理完成。")

if __name__ == "__main__":
    run_inference_robust()
