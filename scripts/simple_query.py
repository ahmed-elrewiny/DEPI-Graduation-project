# scripts/simple_query.py
import sqlite3
import pandas as pd

db = "data/db/tech_stocks.db"
conn = sqlite3.connect(db)
query = """
SELECT Ticker, MIN(Date) as start_date, MAX(Date) as end_date, COUNT(*) as rows
FROM tech_stocks
GROUP BY Ticker
ORDER BY rows DESC
LIMIT 50;
"""
df = pd.read_sql(query, conn)
print(df.to_string(index=False))
conn.close()

