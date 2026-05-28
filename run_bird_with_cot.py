"""
BIRD数据集SQL生成脚本,利用DeepSeek API通过Chain-of-Thought(CoT)推理生成数据库查询语句。
从 dev_prompt.jsonl 读取提示词数据，每行包含数据库架构和自然语言问题
"""
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
API_KEY = "" 
BASE_URL = "https://api.deepseek.com"

INPUT_FILE = "dev_prompt.jsonl" 

# 输出文件
OUTPUT_FILE_EVAL = "predict_dev_2.json"           # 纯净版 (给评测用)
OUTPUT_FILE_COT = "predict_dev_with_cot.json"   # 分析版 (包含CoT)

LOG_FILE = f"deepseek_sql_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

TEST_LIMIT = None  # None 跑全量
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
    """
    鲁棒的解析函数：
    1. 优先寻找特殊分隔符 @@@SQL_START@@@
    2. 其次寻找 Markdown ```sql
    3. 最后尝试寻找 SELECT/WITH 关键字
    """
    sql = ""
    cot = ""
    
    # 策略 A: 强分隔符 (对应 System Prompt)
    start_marker = "@@@SQL_START@@@"
    end_marker = "@@@SQL_END@@@"
    
    if start_marker in text:
        parts = text.split(start_marker)
        cot = parts[0].strip() # 分隔符前面的都是思考过程
        
        # 处理 SQL 部分
        sql_part = parts[1]
        if end_marker in sql_part:
            sql = sql_part.split(end_marker)[0].strip()
        else:
            # 如果模型忘了写结束符，就取剩下的所有
            sql = sql_part.strip()
            
    # 策略 B: Markdown 代码块 (兜底)
    else:
        pattern = r"```sql\s*(.*?)\s*```"
        matches = list(re.finditer(pattern, text, re.DOTALL | re.IGNORECASE))
        
        if matches:
            # 取最后一个代码块 (通常模型会在最后给出最终答案)
            last_match = matches[-1]
            sql = last_match.group(1).strip()
            cot = text[:last_match.start()].strip()
        else:
            # 策略 C: 暴力寻找 SQL 关键字 (最后的手段)
            # 寻找最后一个 SELECT 或 WITH 出现的位置（假设 CoT 在前）
            match = re.search(r"\b(SELECT|WITH)\b", text, re.IGNORECASE)
            if match:
                # 这里很难精确区分，只能假设从这里开始是 SQL
                # 但为了安全，如果没找到明确的分隔符，我们倾向于报错或者记录原始文本
                # 既然是 BIRD，我们尝试提取整个字符串
                sql = text.strip() 
                cot = "Parse Warning: No separators found."
            else:
                sql = text.strip()
                cot = "Parse Error"

    # 清理 SQL: 去掉可能存在的 markdown 符号 (防止策略A中混入 markdown)
    sql = sql.replace("```sql", "").replace("```", "").strip()
    
    # 移除末尾的分号 (Evaluation 脚本有时不喜欢分号)
    sql = sql.rstrip(';')
    
    # 压缩成单行
    sql_clean = " ".join(sql.split())
    
    return sql_clean, cot

def write_log(content):
    with log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            log_line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {content}\n"
            f.write(log_line)
            f.write("-" * 80 + "\n")

def call_deepseek_api(prompt_content, q_id, idx):
    time.sleep(REQUEST_DELAY)
    
    # === 关键修改：System Prompt 强制使用特殊分隔符 ===
    messages = [
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
            
            # 使用鲁棒解析器
            clean_sql, cot = parse_model_output_robust(raw_output)
            
            # 简单验证 SQL 是否为空
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
    prompt_content = item.get('prompt') or item.get('prompt_content')
    q_id = item.get('question_id', idx)
    
    if not prompt_content:
        # 尝试兼容其他格式
        instruction = item.get('instruction', '')
        db_schema = item.get('input', '')
        if instruction and db_schema:
            prompt_content = f"{instruction}\n\n{db_schema}"
        else:
            return q_id, {}, "skipped"
            
    return call_deepseek_api(prompt_content, q_id, idx)

def main():
    write_log(f"任务开始 - 模型: {MODEL_NAME} - 鲁棒 CoT 模式")
    
    data_list = []
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data_list.append(json.loads(line))
    except FileNotFoundError:
        print("❌ 文件未找到")
        return

    if TEST_LIMIT: 
        print(f"⚠️ 测试模式：只运行前 {TEST_LIMIT} 条")
        data_list = data_list[:TEST_LIMIT]
    
    results_for_eval = {}   
    results_for_analysis = {} 

    print(f"开始处理 {len(data_list)} 条数据 (并发数: {MAX_WORKERS})...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {
            executor.submit(process_single_item, item, idx): (item, idx)
            for idx, item in enumerate(data_list)
        }

        pbar = tqdm(total=len(future_to_item))
        for future in as_completed(future_to_item):
            try:
                q_id, res_data, status = future.result()
                
                if status == "success":
                    with result_lock:
                        # 1. 给评测用
                        results_for_eval[str(q_id)] = res_data['sql']
                        
                        # 2. 给分析用
                        item, _ = future_to_item[future]
                        results_for_analysis[str(q_id)] = {
                            "question_id": q_id,
                            "prompt": item.get('prompt') or item.get('input'),
                            "thought_process": res_data['cot'],
                            "generated_sql": res_data['sql'],
                            "full_response": res_data['raw']
                        }
                elif status == "failed":
                    with result_lock:
                        results_for_eval[str(q_id)] = "SELECT 'Error';"
            except Exception as e:
                write_log(f"严重错误: {e}")
            finally:
                pbar.update(1)
        pbar.close()

    # 保存
    print(f"正在保存评测文件: {OUTPUT_FILE_EVAL}")
    # 按数字ID排序
    sorted_eval = {k: v for k, v in sorted(results_for_eval.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else x[0])}
    with open(OUTPUT_FILE_EVAL, 'w', encoding='utf-8') as f:
        json.dump(sorted_eval, f, indent=4, ensure_ascii=False)

    print(f"正在保存分析文件: {OUTPUT_FILE_COT}")
    sorted_analysis = dict(sorted(results_for_analysis.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else item[0]))
    with open(OUTPUT_FILE_COT, 'w', encoding='utf-8') as f:
        json.dump(sorted_analysis, f, indent=4, ensure_ascii=False)

    print("\n✅ 任务完成！SQL 提取已增强。")

if __name__ == "__main__":
    main()
