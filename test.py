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

INPUT_FILE = "dev_prompt.jsonl" 

# 输出两个文件
OUTPUT_FILE_EVAL = "predict_dev_3.json"           # 纯净版 SQL (给 evaluation.py 用)
OUTPUT_FILE_COT = "predict_dev_with_cot_1.json"   # 包含 CoT 和 Prompt (供分析)

LOG_FILE = f"deepseek_sql_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

TEST_LIMIT = None  
MODEL_NAME = "deepseek-chat"
MAX_TOKENS = 2048  # CoT 可能需要更多 Token
TEMPERATURE = 0.0  

MAX_WORKERS = 8  
REQUEST_DELAY = 0.2  
MAX_RETRIES = 3  

PRINT_DETAIL_LOG = True if TEST_LIMIT else False
# ===========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
log_lock = threading.Lock()
result_lock = threading.Lock()

def parse_model_output_robust(text):
    """
    鲁棒解析函数：优先特殊分隔符，再 Markdown，最后关键字。
    返回 (clean_sql, cot)
    """
    sql = ""
    cot = ""
    
    # 策略 A: 强分隔符 (在 System Prompt 中强制要求)
    start_marker = "@@@SQL_START@@@"
    end_marker = "@@@SQL_END@@@"
    
    if start_marker in text and end_marker in text:
        parts = text.split(start_marker, 1)
        cot = parts[0].strip()
        sql_part = parts[1]
        
        sql_end_index = sql_part.find(end_marker)
        if sql_end_index != -1:
            sql = sql_part[:sql_end_index].strip()
        else:
            sql = sql_part.strip() # 如果模型忘了写结束符
            cot += "\n[Parse Warning: End marker missing]"

    # 策略 B: Markdown 代码块 (兜底)
    else:
        pattern = r"```sql\s*(.*?)\s*```"
        matches = list(re.finditer(pattern, text, re.DOTALL | re.IGNORECASE))
        
        if matches:
            last_match = matches[-1]
            sql = last_match.group(1).strip()
            cot = text[:last_match.start()].strip() # 代码块之前的内容视为 CoT
            if not cot: # 如果代码块在最前面
                cot = "No explicit thought process before SQL block."
        else:
            # 策略 C: 纯文本，难以区分 CoT 和 SQL，直接整个作为 SQL (可能会失败)
            # 这里的逻辑可以根据实际模型输出再微调
            # 暂时假设纯文本就是 SQL，CoT为空
            sql = text.strip() 
            cot = "Parse Warning: No clear separators or markdown found. Treating entire output as potential SQL."

    # 清理 SQL
    sql = sql.replace("```sql", "").replace("```", "").strip()
    sql = sql.rstrip(';')
    sql_clean = " ".join(sql.split()) # 压缩成单行
    
    # 如果 CoT 没提取出来，给个默认值
    if not cot:
        cot = "No thought process extracted."
        
    return sql_clean, cot

def write_log(content):
    with log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            log_line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {content}\n"
            f.write(log_line)
            f.write("-" * 80 + "\n")

def call_deepseek_api(prompt_content, q_id, idx):
    time.sleep(REQUEST_DELAY)
    
    # === 关键修改：System Prompt 强制要求 CoT 和特殊分隔符 ===
    messages = [
        {"role": "system", "content": """You are a SQLite expert. 
Please reason step-by-step about the database schema and the user question.
1. Analyze the database schema, question, and any external knowledge.
2. Break down the problem into smaller logical steps.
3. Identify the tables, columns, joins, filters, and aggregations needed.
4. Finally, present your Chain of Thought (reasoning process) first,
   and then wrap your FINAL SQL query inside these exact markers:

@@@SQL_START@@@
SELECT ...
@@@SQL_END@@@

Ensure there is NO other text outside the reasoning process and the SQL markers.
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
            
            # 使用鲁棒的解析器
            clean_sql, cot = parse_model_output_robust(raw_output)
            
            if not clean_sql or "Error" in clean_sql: # 检查 SQL 是否有效
                raise ValueError(f"Parsed SQL is invalid or empty: '{clean_sql}'")

            write_log(f"✅ ID {q_id} 成功. SQL: {clean_sql[:50]}...")
            write_log(f"【CoT (ID:{q_id})】\n{cot}")
            write_log(f"【原始输出 (ID:{q_id})】\n{raw_output}")

            result_data = {
                "sql": clean_sql,
                "cot": cot,
                "raw": raw_output
            }
            return q_id, result_data, "success"
        
        except Exception as e:
            error_msg = f"ID {q_id} 重试{retry+1}/{MAX_RETRIES}失败: {str(e)}"
            write_log(f"❌ {error_msg}")
            if PRINT_DETAIL_LOG:
                print(f"\n{error_msg}")
            time.sleep(1 + retry)
    
    # 所有重试失败
    write_log(f"❌ ID {q_id} 所有重试失败，记录错误SQL和CoT")
    return q_id, {"sql": "SELECT 'Error';", "cot": "API Failed after retries.", "raw": ""}, "failed"

def process_single_item(item, idx):
    prompt_content = item.get('prompt') or item.get('prompt_content')
    q_id = item.get('question_id', idx)
    
    if not prompt_content:
        instruction = item.get('instruction', '')
        db_schema = item.get('input', '')
        if instruction and db_schema:
            prompt_content = f"{instruction}\n\n{db_schema}"
        else:
            # 没有Prompt，跳过
            write_log(f"跳过 ID {q_id}: 无有效Prompt")
            return q_id, {}, "skipped"
            
    return call_deepseek_api(prompt_content, q_id, idx)

def main():
    write_log(f"===== DeepSeek SQL + CoT 生成任务开始 =====")
    write_log(f"模型: {MODEL_NAME} | CoT 模式: 开启")
    
    data_list = []
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data_list.append(json.loads(line))
    except FileNotFoundError:
        print(f"❌ 错误: 找不到输入文件 {INPUT_FILE}")
        return

    if TEST_LIMIT: 
        print(f"⚠️ 测试模式：只运行前 {TEST_LIMIT} 条")
        data_list = data_list[:TEST_LIMIT]
    else:
        print(f"🚀 全量模式：共 {len(data_list)} 条数据")
    
    # 两个输出文件
    results_for_eval = {}   # 仅SQL，给 evaluation.py
    results_for_analysis = {} # SQL + CoT + Prompt，供分析

    print(f"开始多线程处理 (线程数: {MAX_WORKERS})...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {
            executor.submit(process_single_item, item, idx): (item, idx)
            for idx, item in enumerate(data_list)
        }

        pbar = tqdm(total=len(future_to_item), desc="处理进度")
        for future in as_completed(future_to_item):
            try:
                q_id, res_data, status = future.result()
                
                with result_lock:
                    if status == "success":
                        # 1. 存入供评测用的纯净 SQL
                        results_for_eval[str(q_id)] = res_data['sql']
                        
                        # 2. 存入供分析用的完整数据 (含 CoT)
                        item, _ = future_to_item[future] # 获取原始数据
                        results_for_analysis[str(q_id)] = {
                            "question_id": q_id,
                            "prompt": item.get('prompt') or item.get('input'), # 原始Prompt
                            "thought_process": res_data['cot'], # 模型生成的思维链
                            "generated_sql": res_data['sql'], # 模型生成的SQL
                            "full_raw_output": res_data['raw'] # 模型完整原始输出
                        }
                    else: # failed or skipped
                        results_for_eval[str(q_id)] = "SELECT 'Error';"
                        # 即使失败，也尝试记录一下（如果需要）
                        results_for_analysis[str(q_id)] = {
                            "question_id": q_id,
                            "prompt": item.get('prompt') or item.get('input'),
                            "thought_process": res_data.get('cot', 'API Failed or Skipped'),
                            "generated_sql": res_data.get('sql', 'SELECT \'Error:\''),
                            "full_raw_output": res_data.get('raw', '')
                        }
            except Exception as e:
                item, idx = future_to_item[future]
                q_id = item.get('question_id', idx)
                error_msg = f"ID {q_id} 线程异常: {str(e)}"
                write_log(f"❌ {error_msg}")
                if PRINT_DETAIL_LOG:
                    print(f"\n{error_msg}")
                with result_lock:
                    results_for_eval[str(q_id)] = "SELECT 'Error';"
            finally:
                pbar.update(1)
        pbar.close()

    # 保存两个结果文件
    print(f"\n正在保存评测文件 (纯净 SQL) 到: {OUTPUT_FILE_EVAL}")
    # 排序后保存
    sorted_eval = dict(sorted(results_for_eval.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else x[0]))
    with open(OUTPUT_FILE_EVAL, 'w', encoding='utf-8') as f:
        json.dump(sorted_eval, f, indent=4, ensure_ascii=False)

    print(f"正在保存分析文件 (含 CoT) 到: {OUTPUT_FILE_COT}")
    sorted_analysis = dict(sorted(results_for_analysis.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else item[0]))
    with open(OUTPUT_FILE_COT, 'w', encoding='utf-8') as f:
        json.dump(sorted_analysis, f, indent=4, ensure_ascii=False)

    # 总结
    success_count = sum(1 for s in results_for_eval.values() if not s.startswith("SELECT 'Error'"))
    total_processed = len(results_for_eval)
    finish_log = f"""
✅ 任务完成！
- 总处理条数: {len(data_list)}
- 成功生成 SQL: {success_count} / {total_processed}
- 结果文件 (供评测): {OUTPUT_FILE_EVAL}
- 分析文件 (含 CoT): {OUTPUT_FILE_COT}
- 日志文件: {LOG_FILE}
"""
    print(finish_log)
    write_log(finish_log)

if __name__ == "__main__":
    main()