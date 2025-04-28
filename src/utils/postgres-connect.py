import psycopg2
import pandas as pd
import yaml

with open("misc\config.yaml", "r") as f:
    DB_CONFIG = yaml.safe_load(f)

print(DB_CONFIG["c_andreas"])

def connect_and_query(q):
    conn = psycopg2.connect(**DB_CONFIG["c_andreas"])
    cursor = conn.cursor()
    cursor.execute(q)
    wr = pd.DataFrame(cursor.fetchall())
    cursor.close()
    conn.close()
    return wr    

def query_build():
    table = "survey_data_DGData10"
    q_tgt = "from public.\"" + table + "\""
    q_op = "select"
    q_cols = "\"Degree_Department\""
    q_cond = ""
    q = [q_op, q_cols, q_tgt]
    if len(q_cond) > 0:
        q.append(q_cond)
    q = " ".join(q)
    print(f"selected query:{q}")
    return q

q = query_build()
print(f"Query Selected: \"{q}\"")
col = connect_and_query(q)
print(col)