#添加了日志，跑前5个
import json
import time
import re
import os
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 填入你的 DeepSeek API Key
API_KEY = "sk-9885898f801645a890417229eeb56d78" 
BASE_URL = "https://api.deepseek.com"

# 2. 文件路径
# 确保你已经下载了截图里的 dev_prompt.jsonl
INPUT_FILE = "dev_prompt.jsonl" 
OUTPUT_FILE = "predict_dev.json"
# 日志文件路径（按时间命名，避免覆盖）
LOG_FILE = f"deepseek_sql_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# 3. 测试数量限制
# 设置为 None 则跑全量数据 (1534条)
# 设置为 5 则只跑前 5 条 (用于测试)
TEST_LIMIT = 5  

# 4. 模型名称
MODEL_NAME = "deepseek-chat"

# 5. 日志配置
# 是否在控制台打印详细日志（测试模式建议开启，全量模式建议关闭）
PRINT_DETAIL_LOG = True if TEST_LIMIT else False
# ===========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def extract_sql(text):
    """
    从模型输出中提取 SQL。
    如果模型输出了 ```sql ... ```，则提取中间内容；
    否则移除多余的 markdown 符号。
    """
    # 匹配 markdown 代码块
    pattern = r"```sql\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        sql = match.group(1)
    else:
        # 如果没有代码块，尝试移除可能存在的 ``` 符号
        sql = text.replace("```sql", "").replace("```", "")
    
    # 压缩为单行，去除多余换行符（BIRD评测脚本通常喜欢单行SQL）
    return " ".join(sql.split())

def write_log(content):
    """写入日志文件（追加模式）"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        # 增加时间戳
        log_line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {content}\n"
        f.write(log_line)
        # 分隔线，方便阅读
        f.write("-" * 80 + "\n")

def main():
    # 初始化日志文件
    write_log("===== DeepSeek SQL 生成任务开始 =====")
    write_log(f"测试模式: {TEST_LIMIT is not None} (数量限制: {TEST_LIMIT if TEST_LIMIT else '全量'})")
    write_log(f"模型名称: {MODEL_NAME}")
    write_log(f"输入文件: {INPUT_FILE}")
    write_log(f"输出文件: {OUTPUT_FILE}")
    write_log(f"Question ID 起始值: 0（兼容文件中显式ID）")

    # 1. 读取验证集文件
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
        print("请确保你已经下载了 dev_prompt.jsonl 并放在同级目录下。")
        return

    # 应用数量限制
    if TEST_LIMIT is not None:
        print(f"⚠️ 注意：当前处于测试模式，只运行前 {TEST_LIMIT} 条数据。")
        data_list = data_list[:TEST_LIMIT]
    else:
        print(f"🚀 准备运行全量验证集，共 {len(data_list)} 条数据。")
    
    write_log(f"实际处理数据条数: {len(data_list)}")

    results = {}  # 格式: {question_id: sql}
    
    # 2. 开始推理
    print("开始调用 DeepSeek API...")
    write_log("开始调用 DeepSeek API 进行推理")
    
    for idx, item in enumerate(tqdm(data_list)):  # idx 从 0 开始（原生enumerate默认）
        # 解析输入数据结构
        prompt_content = item.get('prompt') or item.get('prompt_content')
        # 核心修改：优先用文件中的question_id，无则用从0开始的idx
        q_id = item.get('question_id', idx)  
        
        # 日志：当前处理的ID（明确显示从0开始）
        process_log = f"处理第 {idx+1}/{len(data_list)} 条 | Question ID: {q_id} (原始索引: {idx})"
        write_log(process_log)
        if PRINT_DETAIL_LOG:
            print(f"\n{process_log}")

        # 如果文件中没有显式的 prompt 字段，可能需要根据 schema 和 question 拼接
        if not prompt_content:
            # 备用方案：尝试拼接 instruction + input (如果有)
            instruction = item.get('instruction', '')
            db_schema = item.get('input', '')
            if instruction and db_schema:
                prompt_content = f"{instruction}\n\n{db_schema}"
            else:
                skip_msg = f"跳过 ID {q_id} (索引{idx}): 无法找到 prompt 内容"
                print(skip_msg)
                write_log(skip_msg)
                results[str(q_id)] = ""  # 用q_id作为key，保证从0开始
                continue

        # 日志：模型输入（Prompt）
        input_log = f"【模型输入-Prompt (ID:{q_id})】\n{prompt_content}"
        write_log(input_log)
        if PRINT_DETAIL_LOG:
            print(f"\n📥 模型输入 (ID: {q_id}):")
            print(prompt_content[:500] + "..." if len(prompt_content) > 500 else prompt_content)  # 截断长Prompt

        # 构造对话消息
        messages = [
            {"role": "user", "content": prompt_content}
        ]

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0, # SQL生成不需要随机性
                max_tokens=1024,
                stream=False
            )
            
            raw_output = response.choices[0].message.content
            clean_sql = extract_sql(raw_output)
            
            # 日志：模型原始输出
            raw_output_log = f"【模型原始输出 (ID:{q_id})】\n{raw_output}"
            write_log(raw_output_log)
            
            # 日志：提取后的SQL
            sql_log = f"【提取后SQL (ID:{q_id})】\n{clean_sql}"
            write_log(sql_log)

            # 控制台打印（测试模式）
            if PRINT_DETAIL_LOG:
                print(f"\n📤 模型原始输出 (ID: {q_id}):")
                print(raw_output[:500] + "..." if len(raw_output) > 500 else raw_output)
                print(f"\n✅ 提取后SQL (ID: {q_id}):")
                print(clean_sql)

            # 保存结果：key为q_id（保证从0开始）
            results[str(q_id)] = clean_sql
            write_log(f"ID {q_id} 处理成功")

        except Exception as e:
            error_msg = f"ID {q_id} (索引{idx}) 请求失败: {str(e)}"
            print(f"\n❌ {error_msg}")
            write_log(f"【错误】{error_msg}")
            
            # 失败时填空字符串，保证文件完整性
            results[str(q_id)] = "SELECT 'Error';" 
            write_log(f"ID {q_id} 错误处理：填充默认SQL -> SELECT 'Error';")
            
            time.sleep(1)

    # 3. 保存结果
    print(f"\n正在保存结果到 {OUTPUT_FILE}...")
    write_log(f"开始保存结果到输出文件: {OUTPUT_FILE}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    # 日志：任务完成
    finish_log = f"✅ 完成！生成了 {len(results)} 条 SQL，结果已保存到 {OUTPUT_FILE}"
    print(finish_log)
    write_log(finish_log)
    
    # 验证ID起始值
    if results:
        first_id = sorted(results.keys())[0]
        write_log(f"结果中最小Question ID: {first_id} (符合从0开始的要求)")
        print(f"🔍 结果中最小Question ID: {first_id} (符合从0开始的要求)")
    
    if TEST_LIMIT is not None:
        tip_msg = "💡 提示：确认输出格式没问题后，请将代码中的 TEST_LIMIT 改为 None 以跑完所有数据。"
        print(tip_msg)
        write_log(tip_msg)
    
    write_log("===== DeepSeek SQL 生成任务结束 =====")

if __name__ == "__main__":
    main()