import json
import sqlite3
import os

# ================= 配置 =================
PREDICT_FILE = "./exp_result/turbo_output/predict_dev.json"
DB_ROOT = "./data/dev_databases/"
# =======================================

def check_errors():
    print("正在诊断错误原因...")
    
    if not os.path.exists(PREDICT_FILE):
        print(f"❌ 找不到预测文件: {PREDICT_FILE}")
        return

    with open(PREDICT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    error_count = 0
    success_count = 0
    
    # 遍历前 20 条数据进行测试
    for q_id, val in list(data.items())[:20]:
        try:
            # 1. 解析 evaluation.py 要求的奇怪格式
            if "\t----- bird -----\t" in val:
                sql, db_name = val.split('\t----- bird -----\t')
            else:
                print(f"⚠️ 格式警告 ID {q_id}: 格式不正确，缺少分隔符")
                print(f"内容: {val}")
                continue

            # 2. 找到数据库路径
            db_path = os.path.join(DB_ROOT, db_name, f"{db_name}.sqlite")
            
            if not os.path.exists(db_path):
                print(f"❌ 严重错误: 找不到数据库文件 {db_path}")
                print("请检查 data/dev_databases 文件夹结构是否正确！")
                break

            # 3. 尝试执行 SQL
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            res = cursor.fetchall()
            conn.close()
            
            success_count += 1
            # print(f"✅ ID {q_id}: 执行成功")

        except sqlite3.Error as e:
            error_count += 1
            print(f"\n❌ [执行报错] ID: {q_id} (DB: {db_name})")
            print(f"SQL内容:  >>>{sql}<<<")
            print(f"报错信息: {e}")
            print("-" * 40)
        except Exception as e:
            print(f"系统错误: {e}")

    print(f"\n诊断结束: 测试了 20 条，成功 {success_count} 条，失败 {error_count} 条。")
    print("请根据上面的报错信息，查看下方的【解决方案】。")

if __name__ == "__main__":
    check_errors()