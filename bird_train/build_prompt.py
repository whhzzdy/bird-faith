# 构建 BIRD Train 集的 Prompt
import json
import sqlite3
import os
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 原始数据输入
INPUT_JSON = "./train/train.json"            # 原始问题集 
DB_ROOT = "./train/train_databases/"         # 数据库物理文件目录

# 2. 输出文件
OUTPUT_JSONL = "train_prompt_dataset.jsonl"  # 生成的缓存文件
# ===========================================

def get_column_examples(cursor, table_name, column_name):
    """
    从数据库中查询指定列的 2 个非空不重复的样本数据，并带截断防抖
    """
    try:
        # 使用反引号防止关键字冲突
        cursor.execute(f"SELECT DISTINCT `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL LIMIT 2")
        rows = cursor.fetchall()
        
        examples = []
        for row in rows:
            val = row[0]
            # 对字符串加引号，数字不加
            if isinstance(val, str):
                # 简单处理掉内部的单引号，防止格式乱掉
                val = val.replace("'", "")
                # 防干扰截断：如果字符串太长，直接斩断，防止冲散大模型注意力
                if len(val) > 30:
                    val = val[:27] + "..."
                examples.append(f"'{val}'")
            else:
                examples.append(str(val))
                
        if examples:
            return f"[{', '.join(examples)}]"
        return "[]"
    except Exception as e:
        return "[]"

def build_bird_schema(db_id):
    """
    仿照 BIRD 官方格式，重建带有 example 的 CREATE TABLE 语句
    """
    db_path = os.path.join(DB_ROOT, db_id, f"{db_id}.sqlite")
    if not os.path.exists(db_path):
        return ""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 获取所有表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall() if row[0] != 'sqlite_sequence']

    schema_str = ""

    for table in tables:
        schema_str += f"CREATE TABLE {table} (\n"
        
        # 1. 获取列信息
        cursor.execute(f"PRAGMA table_info(`{table}`);")
        columns = cursor.fetchall()
        
        col_defs = []
        pk_cols = []
        
        for col in columns:
            col_id, name, type_, notnull, dflt_value, pk = col
            # 兼容处理没有类型的情况
            type_ = type_ if type_ else "text"
            
            if pk > 0:
                pk_cols.append(name)
                
            # 核心：获取真实数据示例
            examples_str = get_column_examples(cursor, table, name)
            
            # 格式化列定义
            formatted_name = f"`{name}`" if " " in name or "-" in name or name.lower() in ['order', 'group', 'by', 'date'] else name
            col_defs.append(f"    {formatted_name} {type_}, -- example: {examples_str}")

        # 2. 获取外键信息
        cursor.execute(f"PRAGMA foreign_key_list(`{table}`);")
        fks = cursor.fetchall()
        fk_defs = []
        for fk in fks:
            id_, seq, table_ref, from_col, to_col, on_upd, on_del, match = fk
            
            formatted_from = f"`{from_col}`" if " " in from_col else from_col
            formatted_ref = f"`{table_ref}`" if " " in table_ref or "-" in table_ref else table_ref
            
            # 【核心修复】：判断 to_col 是否为 None，解决 SQLite 隐式外键报错问题
            if to_col:
                formatted_to = f"`{to_col}`" if " " in to_col else to_col
                fk_defs.append(f"    CONSTRAINT fk_{table}_{from_col} FOREIGN KEY ({formatted_from}) REFERENCES {formatted_ref} ({formatted_to})")
            else:
                # 如果数据库中没写关联到哪一列（默认关联主键），我们就省略括号
                fk_defs.append(f"    CONSTRAINT fk_{table}_{from_col} FOREIGN KEY ({formatted_from}) REFERENCES {formatted_ref}")

        # 3. 组装表结构
        all_defs = col_defs
        if pk_cols:
            formatted_pks = ", ".join([f"`{c}`" if " " in c else c for c in pk_cols])
            all_defs.append(f"    PRIMARY KEY ({formatted_pks})")
        all_defs.extend(fk_defs)

        schema_str += ",\n".join(all_defs)
        schema_str += "\n);\n\n"

    conn.close()
    return schema_str.strip()

def build_full_prompt(schema, question, evidence):
    """
    组装完整的提示词模板
    """
    evidence_text = f"{evidence}\n" if evidence else ""
    full_question = evidence_text + question
    
    prompt = f"""Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{schema}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{full_question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```sql
-- Your SQL query
Take a deep breath and think step by step to find the correct SQL query.
"""
    return prompt, full_question

def main():
    print(f"🚀 开始读取 {INPUT_JSON} 并构建 Prompt 数据集...")
    if not os.path.exists(INPUT_JSON):
        print(f"❌ 找不到文件: {INPUT_JSON}，请检查路径。")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 缓存 Schema，防止同一个库被反复查询（训练集中同一个 db 会出现很多次）
    db_schema_cache = {}

    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f_out:
        for item in tqdm(data, desc="Processing"):
            q_id = item.get('question_id', '')
            db_id = item['db_id']
            question = item['question']
            evidence = item.get('evidence', '')
            gold_sql = item.get('SQL', '')
            difficulty = item.get('difficulty', 'simple')

            # 如果这个库的 schema 还没构建过，就查一次数据库并缓存
            if db_id not in db_schema_cache:
                db_schema_cache[db_id] = build_bird_schema(db_id)
        
            schema = db_schema_cache[db_id]
        
            if not schema:
                print(f"⚠️ 警告: 找不到数据库 {db_id}")
                continue

            prompt, full_question = build_full_prompt(schema, question, evidence)

            final_item = {
                "question_id": q_id,
                "db_id": db_id,
                "question": question,
                "evidence": evidence,
                "full_question": full_question,
                "SQL": gold_sql,
                "difficulty": difficulty,
                "schema": schema,
                "prompt": prompt,
            }

            f_out.write(json.dumps(final_item, ensure_ascii=False) + '\n')

    print(f"\n✅ 构建完成！已保存为: {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()