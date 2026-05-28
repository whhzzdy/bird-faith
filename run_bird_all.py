#分为8个进程用deepseek-v3.2按问题顺序运行
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
# 1. 填入你的 DeepSeek API Key
API_KEY = "" 
BASE_URL = "https://api.deepseek.com"

# 2. 文件路径
INPUT_FILE = "dev_prompt.jsonl" 
OUTPUT_FILE = "predict_dev.json"
LOG_FILE = f"deepseek_sql_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# 3. 测试数量限制（跑全量设为 None）
TEST_LIMIT = None  # 改为 None 跑全量，测试时可改回 5

# 4. 模型配置
MODEL_NAME = "deepseek-chat"
MAX_TOKENS = 1024  # SQL生成最大长度
TEMPERATURE = 0.0  # 0=确定性输出，适合SQL生成

# 5. 多线程配置（核心！根据API限流调整）
MAX_WORKERS = 8  # 并发线程数（建议5-10，避免触发API限流）
REQUEST_DELAY = 0.2  # 每个线程的请求间隔（秒），防止请求过快
MAX_RETRIES = 3  # 单条请求失败重试次数

# 6. 日志配置（全量模式关闭控制台详细日志）
PRINT_DETAIL_LOG = True if TEST_LIMIT else False
# ===========================================

# 初始化OpenAI客户端（线程安全）
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 线程锁：保证日志写入和结果保存的线程安全
log_lock = threading.Lock()
result_lock = threading.Lock()

def extract_sql(text):
    """从模型输出中提取纯SQL（保持原有逻辑）"""
    pattern = r"```sql\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        sql = match.group(1)
    else:
        sql = text.replace("```sql", "").replace("```", "")
    return " ".join(sql.split())

def write_log(content):
    """线程安全的日志写入"""
    with log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            log_line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {content}\n"
            f.write(log_line)
            f.write("-" * 80 + "\n")

def call_deepseek_api(prompt_content, q_id, idx):
    """单条请求的API调用逻辑（带重试）"""
    # 每个请求前短暂延迟，避免限流
    time.sleep(REQUEST_DELAY)
    
    messages = [{"role": "user", "content": prompt_content}]
    raw_output = ""
    clean_sql = ""
    
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
            clean_sql = extract_sql(raw_output)
            
            # 记录成功日志
            write_log(f"✅ ID {q_id} (索引{idx}) 调用成功")
            write_log(f"【模型原始输出 (ID:{q_id})】\n{raw_output}")
            write_log(f"【提取后SQL (ID:{q_id})】\n{clean_sql}")
            return q_id, clean_sql, "success"
        
        except Exception as e:
            error_msg = f"ID {q_id} (索引{idx}) 重试{retry+1}/{MAX_RETRIES}失败: {str(e)}"
            write_log(f"❌ {error_msg}")
            if PRINT_DETAIL_LOG:
                print(f"\n{error_msg}")
            # 重试间隔递增（指数退避）
            time.sleep(1 * (retry + 1))
    
    # 所有重试失败，返回默认SQL
    clean_sql = "SELECT 'Error';"
    write_log(f"❌ ID {q_id} (索引{idx}) 所有重试失败，填充默认SQL")
    return q_id, clean_sql, "failed"

def process_single_item(item, idx):
    """处理单条数据（解析Prompt + 调用API）"""
    # 解析数据
    prompt_content = item.get('prompt') or item.get('prompt_content')
    q_id = item.get('question_id', idx)  # ID从0开始
    
    # 日志记录处理开始
    process_log = f"开始处理 ID {q_id} (索引{idx})"
    write_log(process_log)
    if PRINT_DETAIL_LOG:
        print(f"\n{process_log}")
    
    # 检查Prompt是否存在
    if not prompt_content:
        instruction = item.get('instruction', '')
        db_schema = item.get('input', '')
        if instruction and db_schema:
            prompt_content = f"{instruction}\n\n{db_schema}"
        else:
            skip_msg = f"跳过 ID {q_id} (索引{idx}): 无有效Prompt"
            write_log(skip_msg)
            if PRINT_DETAIL_LOG:
                print(skip_msg)
            return q_id, "", "skipped"
    
    # 记录Prompt
    write_log(f"【模型输入-Prompt (ID:{q_id})】\n{prompt_content[:1000]}..." if len(prompt_content) > 1000 else prompt_content)
    
    # 调用API
    return call_deepseek_api(prompt_content, q_id, idx)

def main():
    # 初始化日志
    write_log("===== DeepSeek SQL 全量生成任务开始 =====")
    write_log(f"并发线程数: {MAX_WORKERS} | 重试次数: {MAX_RETRIES} | 请求间隔: {REQUEST_DELAY}s")
    write_log(f"测试模式: {TEST_LIMIT is not None} (数量限制: {TEST_LIMIT if TEST_LIMIT else '全量'})")
    write_log(f"模型名称: {MODEL_NAME} | 输入文件: {INPUT_FILE} | 输出文件: {OUTPUT_FILE}")

    # 1. 读取数据
    print(f"正在读取 {INPUT_FILE}...")
    write_log(f"开始读取输入文件: {INPUT_FILE}")
    
    data_list = []
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data_list.append(json.loads(line))
    except FileNotFoundError:
        error_msg = f"❌ 错误：找不到文件 {INPUT_FILE}"
        print(error_msg)
        write_log(error_msg)
        return

    # 应用数量限制
    if TEST_LIMIT is not None:
        print(f"⚠️ 测试模式：只处理前 {TEST_LIMIT} 条数据")
        data_list = data_list[:TEST_LIMIT]
    else:
        print(f"🚀 全量模式：共 {len(data_list)} 条数据待处理")
    
    write_log(f"实际处理数据条数: {len(data_list)}")

    # 2. 多线程批量处理
    results = {}  # 存储最终结果 {q_id: sql}
    success_count = 0
    failed_count = 0
    skipped_count = 0

    print(f"\n开始多线程调用 DeepSeek API（{MAX_WORKERS} 线程）...")
    write_log(f"开始多线程推理，线程数: {MAX_WORKERS}")

    # 创建线程池
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_item = {
            executor.submit(process_single_item, item, idx): (item, idx)
            for idx, item in enumerate(data_list)
        }

        # 进度条展示
        pbar = tqdm(total=len(future_to_item), desc="处理进度")
        for future in as_completed(future_to_item):
            try:
                q_id, clean_sql, status = future.result()
                # 线程安全保存结果
                with result_lock:
                    results[str(q_id)] = clean_sql
                
                # 统计状态
                if status == "success":
                    success_count += 1
                elif status == "failed":
                    failed_count += 1
                elif status == "skipped":
                    skipped_count += 1
            
            except Exception as e:
                item, idx = future_to_item[future]
                q_id = item.get('question_id', idx)
                error_msg = f"ID {q_id} (索引{idx}) 线程执行异常: {str(e)}"
                write_log(f"❌ {error_msg}")
                if PRINT_DETAIL_LOG:
                    print(f"\n{error_msg}")
                # 保存异常结果
                with result_lock:
                    results[str(q_id)] = "SELECT 'Error';"
                failed_count += 1
            
            finally:
                pbar.update(1)
        pbar.close()

    # 3. 保存结果
    print(f"\n正在保存结果到 {OUTPUT_FILE}...")
    write_log(f"开始保存结果，成功: {success_count} | 失败: {failed_count} | 跳过: {skipped_count}")
    
    # 按ID排序后保存（保证顺序）
    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda x: int(x[0]))}
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(sorted_results, f, indent=4, ensure_ascii=False)

    # 4. 任务总结
    finish_log = f"""
✅ 任务完成！
- 总处理条数: {len(data_list)}
- 成功条数: {success_count}
- 失败条数: {failed_count}
- 跳过条数: {skipped_count}
- 结果文件: {OUTPUT_FILE}
- 日志文件: {LOG_FILE}
"""
    print(finish_log)
    write_log(finish_log)
    
    # 验证ID起始值
    if results:
        first_id = sorted(results.keys())[0]
        write_log(f"结果中最小Question ID: {first_id} (符合从0开始的要求)")
        print(f"🔍 结果中最小Question ID: {first_id} (符合从0开始的要求)")

    write_log("===== DeepSeek SQL 全量生成任务结束 =====")

if __name__ == "__main__":
    main()
