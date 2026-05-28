#转换测评格式,负责把 predict_dev.json 转换成 evaluation.py 能认的格式
import os
import json
import shutil

# ================= 配置 =================
# 你现有的文件名
MY_PREDICTION = "predict_dev_3.json"
MY_GROUND_TRUTH = "dev.json"
MY_DB_FOLDER = "dev_databases"

# 目标路径 (对应 run_evaluation.sh 的变量)
PATH_DATA = "./data/"
PATH_OUTPUT = "./exp_result/turbo_output/"
PATH_DB = "./data/dev_databases/"
# =======================================

def main():
    print("正在根据 run_evaluation.sh 的要求构建目录结构...")

    # 1. 创建文件夹
    os.makedirs(PATH_DATA, exist_ok=True)
    os.makedirs(PATH_OUTPUT, exist_ok=True)

    # 2. 移动/检查数据库
    if os.path.exists(PATH_DB):
        print("✅ 数据库目录 ./data/dev_databases/ 已存在")
    elif os.path.exists(MY_DB_FOLDER):
        print(f"正在移动 {MY_DB_FOLDER} 到 {PATH_DB} ...")
        shutil.move(MY_DB_FOLDER, PATH_DATA)
    else:
        print("❌ 警告：找不到 dev_databases 文件夹！请确保你解压了数据库并放在当前目录。")

    # 3. 复制 dev.json
    if os.path.exists(MY_GROUND_TRUTH):
        shutil.copy(MY_GROUND_TRUTH, os.path.join(PATH_DATA, "dev.json"))
    else:
        print(f"❌ 错误：找不到 {MY_GROUND_TRUTH}")
        return

    # 4. 生成 dev_gold.sql (标准答案 SQL \t db_id)
    # evaluation.py 的 mode='gt' 需要这个文件
    print("正在生成 data/dev_gold.sql ...")
    with open(MY_GROUND_TRUTH, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    
    with open(os.path.join(PATH_DATA, "dev_gold.sql"), 'w', encoding='utf-8') as f_gold:
        for item in dev_data:
            # 格式: SQL \t db_id
            f_gold.write(f"{item['SQL'].strip()}\t{item['db_id']}\n")

    # 5. 转换并保存预测结果 predict_dev.json
    # evaluation.py 要求 value 格式为: "SQL \t----- bird -----\t db_id"
    print(f"正在转换预测结果到 {PATH_OUTPUT}predict_dev.json ...")
    
    if os.path.exists(MY_PREDICTION):
        with open(MY_PREDICTION, 'r', encoding='utf-8') as f:
            my_pred = json.load(f)
        
        # 建立 id 到 db_id 的映射
        id_to_db = {str(item['question_id']): item['db_id'] for item in dev_data}
        
        formatted_pred = {}
        for q_id, sql in my_pred.items():
            db_id = id_to_db.get(str(q_id), "financial") # 默认值防报错
            # 清理 SQL 换行
            clean_sql = " ".join(sql.replace("\n", " ").split())
            # 构造特殊格式
            formatted_pred[str(q_id)] = f"{clean_sql}\t----- bird -----\t{db_id}"
            
        with open(os.path.join(PATH_OUTPUT, "predict_dev.json"), 'w', encoding='utf-8') as f_out:
            json.dump(formatted_pred, f_out, indent=4, ensure_ascii=False)
    else:
        print(f"❌ 错误：找不到你的预测文件 {MY_PREDICTION}")

    print("\n✅ 环境准备完成！")
    print("现在可以直接运行下面的 Python 命令进行评测了。")

if __name__ == "__main__":
    main()