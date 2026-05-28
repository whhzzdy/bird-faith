"""ReAct (Reasoning + Acting) architecture for Text-to-SQL on BIRD dev set.
v2: Use standard prompt as Turn 1 base, always 2+ turns with reflection.
"""

import json
import os
import re
import sqlite3
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
import threading

# ================= 配置 =================
API_KEY = ""
BASE_URL = "https://api.deepseek.com"

# v2: 直接复用标准 prompt（dev_prompt.jsonl），保证 prompt 质量与 baseline 一致
PROMPT_FILE = "../dev_prompt.jsonl"
DATABASES_DIR = "../data/dev_databases"
OUTPUT_FILE = "predict_dev_react.json"
OUTPUT_FILE_EVAL = "predict_dev_react_eval.json"
LOG_FILE = f"react_sql_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

MODEL_NAME = "deepseek-chat"
MAX_TURNS = 3
MAX_TOKENS = 2048
TEMPERATURE = 0.0
MAX_WORKERS = 8
MAX_RETRIES = 3
TEST_LIMIT = None
# =========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
log_lock = threading.Lock()
result_lock = threading.Lock()


def write_log(content):
    with log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {content}\n")
            f.write("-" * 80 + "\n")


def extract_sql(text):
    """Extract SQL from model output"""
    pattern = r"```sql\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return " ".join(matches[-1].split())
    return text.strip()


def execute_sql(sql, db_path):
    """Execute SQL and return results or error"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        col_names = [d[0] for d in cursor.description] if cursor.description else []
        conn.close()

        if not rows:
            return "(empty result set)", col_names
        preview = rows[:20]
        result_str = " | ".join(col_names) + "\n" if col_names else ""
        result_str += "\n".join(" | ".join(str(v) for v in row) for row in preview)
        if len(rows) > 20:
            result_str += f"\n... ({len(rows)} rows total, showing first 20)"
        return result_str, col_names
    except Exception as e:
        return f"ERROR: {str(e)}", []


def build_reflection_prompt(original_prompt, previous_sql, execution_result, turn):
    """Turn 2+: ask model to reflect on execution results and refine"""
    return f"""{original_prompt}

---

[Previous Attempt]
You previously wrote:
```sql
{previous_sql}
```

[Execution Result]
{execution_result}

---

This is refinement round {turn}. Carefully check the execution result:

1. Does the result actually answer the question correctly?
   - If YES: output the SAME SQL again (no changes).
   - If the result is EMPTY: reconsider your WHERE conditions or JOIN logic.
   - If the result looks WRONG (wrong values, wrong columns): identify the error in your logic and fix it.
   - If there was an ERROR: fix the syntax, column name, or table reference.

2. Common pitfalls to check:
   - Wrong column in SELECT or WHERE
   - Missing or incorrect JOIN
   - Filter conditions too strict or too loose
   - Column name escaping (use backticks for names with spaces)
   - Did you use the External Knowledge correctly?

Output your refined SQL in a code block:
```sql
<your refined SQL query here>
```"""


def run_react_loop(item, db_path, qid):
    """ReAct multi-turn loop. item has: prompt, db_id, question_id, SQL (gold)"""
    standard_prompt = item.get("prompt", "")
    if not standard_prompt:
        return "SELECT 'Error' AS reason;", "no_prompt", 0

    db_id = item["db_id"]
    trace = ""
    final_sql = ""
    last_result = ""
    status = "unknown"
    turns_used = 0

    for turn in range(1, MAX_TURNS + 1):
        turns_used = turn

        if turn == 1:
            prompt = standard_prompt
        else:
            prompt = build_reflection_prompt(
                standard_prompt, final_sql, last_result, turn
            )

        for retry in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False
                )
                raw_output = response.choices[0].message.content
                break
            except Exception as e:
                if retry == MAX_RETRIES - 1:
                    write_log(f"QID {qid} turn {turn} API call failed: {e}")
                    return final_sql or "SELECT 'Error' AS reason;", "api_error", turn
                time.sleep(2 * (retry + 1))

        sql = extract_sql(raw_output)
        trace += f"\n--- Turn {turn} ---\n{raw_output[:600]}\n"

        if not sql or not sql.upper().startswith("SELECT"):
            last_result = "No valid SQL found. Please output a SELECT query in a ```sql code block."
            continue

        # Execute SQL
        exec_result, columns = execute_sql(sql, db_path)
        last_result = exec_result

        if final_sql and sql.strip() == final_sql.strip():
            # Model confirmed the previous SQL — we're done
            status = "confirmed"
            break

        final_sql = sql

        if exec_result.startswith("ERROR:"):
            write_log(f"QID {qid} turn {turn}: SQL error - {exec_result[:100]}")
            continue
        elif exec_result == "(empty result set)":
            write_log(f"QID {qid} turn {turn}: empty result - refining")
            continue
        else:
            # Got results — but still do at least one more turn to verify
            if turn >= 2:
                status = "success"
                break
            # turn=1 with results: let Turn 2 do a confirmation
            last_result = exec_result
            continue
    else:
        if final_sql:
            status = "max_turns_reached"
        else:
            status = "no_valid_sql"
            final_sql = "SELECT 'Error' AS reason;"

    write_log(f"QID {qid}: status={status}, turns={turns_used}, SQL={final_sql[:200]}")
    return final_sql, status, turns_used


def check_execution_match(pred_sql, gold_sql, db_path):
    """Check EX metric: do pred and gold produce the same result set?"""
    if not pred_sql or pred_sql.startswith("SELECT 'Error'"):
        return False
    pred_result, _ = execute_sql(pred_sql, db_path)
    gold_result, _ = execute_sql(gold_sql, db_path)
    if pred_result.startswith("ERROR:") or gold_result.startswith("ERROR:"):
        return False
    return pred_result == gold_result


def main():
    write_log("===== ReAct v2 Text-to-SQL on BIRD Dev =====")
    write_log(f"Model: {MODEL_NAME} | Max turns: {MAX_TURNS} | Workers: {MAX_WORKERS}")

    # 1. Load data from dev_prompt.jsonl (standard prompt, same as baseline)
    print("Loading prompt data...")
    items = []
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    if TEST_LIMIT:
        print(f"TEST mode: {TEST_LIMIT} samples")
        items = items[:TEST_LIMIT]
    else:
        print(f"Full mode: {len(items)} samples")

    write_log(f"Total samples: {len(items)}")

    # 2. Multi-threaded processing
    results = {}
    status_counts = {"success": 0, "confirmed": 0, "max_turns_reached": 0,
                     "no_valid_sql": 0, "no_prompt": 0, "api_error": 0}
    total_turns = 0
    ex_correct = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {}
        for item in items:
            qid = item["question_id"]
            db_id = item["db_id"]
            db_path = os.path.join(DATABASES_DIR, db_id, f"{db_id}.sqlite")
            future = executor.submit(run_react_loop, item, db_path, qid)
            future_to_item[future] = item

        pbar = tqdm(total=len(future_to_item), desc="ReAct v2")
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                sql, status, turns = future.result()
                db_id = item["db_id"]
                with result_lock:
                    results[str(item["question_id"])] = {
                        "question_id": item["question_id"],
                        "db_id": db_id,
                        "pred_sql": sql,
                        "status": status,
                        "turns": turns,
                    }
                    if status in status_counts:
                        status_counts[status] += 1
                    total_turns += turns

                db_path = os.path.join(DATABASES_DIR, db_id, f"{db_id}.sqlite")
                if sql and os.path.exists(db_path):
                    if check_execution_match(sql, item["SQL"], db_path):
                        ex_correct += 1
                        with result_lock:
                            results[str(item["question_id"])]["ex_match"] = True
                    else:
                        with result_lock:
                            results[str(item["question_id"])]["ex_match"] = False

            except Exception as e:
                write_log(f"EXCEPTION QID {item['question_id']}: {e}")
                with result_lock:
                    results[str(item["question_id"])] = {
                        "question_id": item["question_id"],
                        "db_id": item.get("db_id", "unknown"),
                        "pred_sql": "SELECT 'Error' AS reason;",
                        "status": "exception",
                        "turns": 0,
                        "ex_match": False,
                    }
            finally:
                pbar.update(1)
        pbar.close()

    # 3. Save detailed results
    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda x: int(x[0]))}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted_results, f, indent=2, ensure_ascii=False)

    # 4. Save BIRD evaluation-compatible format
    eval_format = {}
    for qid_str, info in sorted_results.items():
        sql = info.get("pred_sql", "SELECT 'Error' AS reason;")
        db_id = info.get("db_id", "unknown")
        eval_format[qid_str] = f"{sql}\t----- bird -----\t{db_id}"
    with open(OUTPUT_FILE_EVAL, "w", encoding="utf-8") as f:
        json.dump(eval_format, f, indent=4, ensure_ascii=False)

    # 5. Report
    ex_acc = ex_correct / len(items) * 100 if items else 0
    avg_turns = total_turns / len(items) if items else 0

    report = f"""
===== ReAct v2 Results =====
Samples: {len(items)}
Status distribution: {json.dumps(status_counts, indent=2)}
Execution Accuracy (EX): {ex_correct}/{len(items)} = {ex_acc:.2f}%
Average turns: {avg_turns:.2f}
Detailed output: {OUTPUT_FILE}
Eval format output: {OUTPUT_FILE_EVAL}

Compare:
  Baseline (standard prompt, 1 turn): ~62.13%
  ReAct v1 (custom prompt, early stop): 51.24%
  ReAct v2 (standard prompt + reflection): ?
"""
    print(report)
    write_log(report)


if __name__ == "__main__":
    main()
