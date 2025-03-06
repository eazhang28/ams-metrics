import psycopg2
import pandas as pd

DB_CONFIG = {
    "dbname": "analytics_team",
    "user": "postgres",
    "password": "pasted-sprang-siesta-shortwave",
    "host": "192.168.1.24",
    "port": "5432"
}

def connect_and_query(q):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute(q)
    wr = pd.DataFrame(cursor.fetchall())
    cursor.close()
    conn.close()
    return wr    

table = "survey_data_DGData10"
q_tgt = "from public.\"" + table + "\""
q_op = "select"
q_cols = "\"Degree_Department\""
q_cond = ""
q = [q_op, q_cols, q_tgt]
if len(q_cond) > 0:
    q.append(q_cond)
q = " ".join(q)
print(q)

col = connect_and_query(q)