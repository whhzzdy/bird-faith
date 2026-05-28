# 通过提前构建schema相关的提示词，运行bird训练集,将输出格式从数组对象改为键值对嵌套对象（以 question_id 为键）
import json
import time
import re
import os
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ================= 配置区域 =================
API_KEY = "sk-9885898f801645a890417229eeb56d78" 
BASE_URL = "https://api.deepseek.com"

# 1. 你的新输入文件（之前构建好的包含完整 Prompt 的 jsonl）
INPUT_FILE = "train_prompt_dataset.jsonl" 

# 2. 输出文件 (自动存入 train_output 文件夹)
OUTPUT_DIR = "./exp_result/train_output_all"
OUTPUT_FILE_EVAL = f"{OUTPUT_DIR}/predict_train.json"           # 供 evaluation.py 直接使用的格式
OUTPUT_FILE_COT = f"{OUTPUT_DIR}/predict_train_with_cot.json"   # 分析版 (包含 CoT 和 Gold SQL)

LOG_FILE = f"deepseek_train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# ⚠️ 强烈建议：先用 10 跑通，查看生成文件没问题后，再改为 None 跑 9000 多条的全量！
TEST_LIMIT = None
MODEL_NAME = "deepseek-chat"
MAX_TOKENS = 2048
TEMPERATURE = 0.0  

MAX_WORKERS = 10
REQUEST_DELAY = 0.2  
MAX_RETRIES = 3  
# ===========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
log_lock = threading.Lock()
result_lock = threading.Lock()

def parse_model_output_robust(text):
    """鲁棒的解析函数：提取 CoT 和 SQL"""
    sql = ""
    cot = ""
    
    start_marker = "@@@SQL_START@@@"
    end_marker = "@@@SQL_END@@@"
    
    if start_marker in text:
        parts = text.split(start_marker)
        cot = parts[0].strip() 
        
        sql_part = parts[1]
        if end_marker in sql_part:
            sql = sql_part.split(end_marker)[0].strip()
        else:
            sql = sql_part.strip()
    else:
        pattern = r"```sql\s*(.*?)\s*```"
        matches = list(re.finditer(pattern, text, re.DOTALL | re.IGNORECASE))
        
        if matches:
            last_match = matches[-1]
            sql = last_match.group(1).strip()
            cot = text[:last_match.start()].strip()
        else:
            match = re.search(r"\b(SELECT|WITH)\b", text, re.IGNORECASE)
            if match:
                sql = text.strip() 
                cot = "Parse Warning: No separators found."
            else:
                sql = text.strip()
                cot = "Parse Error"

    sql = sql.replace("```sql", "").replace("```", "").strip()
    sql = sql.rstrip(';')
    sql_clean = " ".join(sql.split())
    
    return sql_clean, cot

def write_log(content):
    with log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            log_line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {content}\n"
            f.write(log_line)

def call_deepseek_api(prompt_content, q_id, idx):
    time.sleep(REQUEST_DELAY)
    
    messages =[
        {"role": "system", "content": """You are a SQLite expert. 
Please reason step-by-step about the database schema and the user question.

Output format requirements:
1. First, write your "Chain of Thought" (analysis).
2. Then, wrap your FINAL SQL query inside these exact separators:

@@@SQL_START@@@
SELECT ...
@@@SQL_END@@@

Do not output any text after the SQL end marker.
"""},
        {"role": "user", "content": prompt_content}
    ]
    
    for retry in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False
            )
            raw_output = response.choices[0].message.content
            clean_sql, cot = parse_model_output_robust(raw_output)
            
            if not clean_sql:
                raise ValueError("Parsed SQL is empty")

            write_log(f"✅ ID {q_id} 成功. SQL: {clean_sql[:50]}...")
            
            result_data = {
                "sql": clean_sql,
                "cot": cot,
                "raw": raw_output
            }
            return q_id, result_data, "success"
        
        except Exception as e:
            error_msg = f"ID {q_id} 重试{retry+1}失败: {str(e)}"
            write_log(f"❌ {error_msg}")
            time.sleep(1 + retry)
    
    return q_id, {"sql": "SELECT 'Error';", "cot": "Failed", "raw": ""}, "failed"

def process_single_item(item, idx):
    # 提取预构建好的 prompt
    prompt_content = item.get('prompt') 
    q_id = item.get('question_id', idx)
    db_id = item.get('db_id', 'unknown') # 关键：提取 db_id 供评测脚本格式化使用
    
    if not prompt_content:
        return q_id, db_id, {}, "skipped"
            
    q_id, result_data, status = call_deepseek_api(prompt_content, q_id, idx)
    return q_id, db_id, result_data, status

def main():
    write_log(f"任务开始 - 模型: {MODEL_NAME} - 鲁棒 CoT 模式 (Train Set)")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    data_list =[]
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data_list.append(json.loads(line))
    except FileNotFoundError:
        print(f"❌ 文件 {INPUT_FILE} 未找到！请确保你已经运行了构建 prompt 的脚本。")
        return

    if TEST_LIMIT: 
        print(f"⚠️ 测试模式：只运行前 {TEST_LIMIT} 条")
        data_list = data_list[:TEST_LIMIT]
    else:
        print(f"🚀 全量模式：即将处理 {len(data_list)} 条训练集数据！")
    
    results_for_eval = {}   
    results_for_analysis = {} 

    print(f"开始处理数据 (并发数: {MAX_WORKERS})...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {
            executor.submit(process_single_item, item, idx): (item, idx)
            for idx, item in enumerate(data_list)
        }

        pbar = tqdm(total=len(future_to_item))
        for future in as_completed(future_to_item):
            try:
                # 接收返回的 db_id
                q_id, db_id, res_data, status = future.result()
                q_id_str = str(q_id)  # 统一转为字符串键，匹配图二格式
                
                with result_lock:
                    if status == "success":
                        # 1. 核心修改：直接拼接成 evaluation.py 认识的格式 (SQL \t----- bird -----\t db_id)
                        results_for_eval[q_id_str] = f"{res_data['sql']}\t----- bird -----\t{db_id}"
                        
                        # 2. 给分析用：改为图二的「键值对嵌套」格式，以question_id为键
                        item, _ = future_to_item[future]
                        results_for_analysis[q_id_str] = {
                            "question_id": q_id,
                            "prompt": item.get('prompt'),
                            "thought_process": res_data['cot'],
                            "generated_sql": res_data['sql'],
                            "full_raw_output": res_data['raw']  # 匹配图二的字段名
                        }
                    elif status == "failed":
                        results_for_eval[q_id_str] = f"SELECT 'Error';\t----- bird -----\t{db_id}"
                        results_for_analysis[q_id_str] = {
                            "question_id": q_id,
                            "prompt": item.get('prompt'),
                            "thought_process": "Failed",
                            "generated_sql": "SELECT 'Error';",
                            "full_raw_output": ""
                        }
            except Exception as e:
                write_log(f"严重错误: {e}")
            finally:
                pbar.update(1)
        pbar.close()

    # 保存文件
    print(f"正在保存评测文件: {OUTPUT_FILE_EVAL}")
    # 按question_id升序排序，保证输出顺序一致
    sorted_eval = {k: v for k, v in sorted(results_for_eval.items(), key=lambda x: int(x[0]))}
    with open(OUTPUT_FILE_EVAL, 'w', encoding='utf-8') as f:
        json.dump(sorted_eval, f, indent=4, ensure_ascii=False)

    print(f"正在保存分析文件: {OUTPUT_FILE_COT}")
    # 按question_id升序排序，保证输出顺序一致
    sorted_analysis = {k: v for k, v in sorted(results_for_analysis.items(), key=lambda x: int(x[0]))}
    with open(OUTPUT_FILE_COT, 'w', encoding='utf-8') as f:
        json.dump(sorted_analysis, f, indent=4, ensure_ascii=False)

    print("\n✅ 任务完成！")
    print(f"👉 如果你要跑分，请直接运行: python evaluation.py --predicted_sql_path {OUTPUT_DIR}/ --data_mode train ...")

if __name__ == "__main__":
    main()