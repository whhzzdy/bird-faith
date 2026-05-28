import sqlite3
# 连接到你的数据库
conn = sqlite3.connect('./data/dev_databases/card_games/card_games.sqlite')
cursor = conn.cursor()
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='cards';")
print(cursor.fetchone()[0])
conn.close()