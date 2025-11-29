from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine

# -----------------------
# Configuration
# -----------------------
DATA_DIR = "/opt/airflow/data"
os.makedirs(DATA_DIR, exist_ok=True)

POSTGRES_CONN_STR = "postgresql+psycopg2://airflow:airflow@postgres/airflow"
TABLE_NAME = "stocks_clean"

SYMBOLS = ["AAPL", "TSLA", "MSFT", "NVDA"]

# -----------------------
# 1) Extract
# -----------------------
def extract():
    all_data = []

    for symbol in SYMBOLS:
        try:
            df = yf.download(symbol, period="5d")  # 5 days to ensure data
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            continue

        if df.empty:
            print(f"No data retrieved for {symbol}")
            continue
        
        df['symbol'] = symbol  # add ticker column
        all_data.append(df.reset_index())
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        raw_path = os.path.join(DATA_DIR, "raw.csv")
        final_df.to_csv(raw_path, index=False)
        print(f"Saved raw CSV to: {raw_path}")
    else:
        raise ValueError("No data retrieved for any symbol! Failing task to prevent empty downstream data.")

# -----------------------
# 2) Clean
# -----------------------
def clean():
    raw_path = os.path.join(DATA_DIR, "raw.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError("Raw CSV not found. Skipping clean step.")
    
    df = pd.read_csv(raw_path)
    df = df.dropna()
    
    clean_path = os.path.join(DATA_DIR, "clean.csv")
    df.to_csv(clean_path, index=False)
    print(f"Saved clean CSV to: {clean_path}")

# -----------------------
# 3) Load to Postgres
# -----------------------
def load_to_postgres():
    clean_path = os.path.join(DATA_DIR, "clean.csv")
    if not os.path.exists(clean_path):
        raise FileNotFoundError("Clean CSV not found. Skipping load step.")
    
    df = pd.read_csv(clean_path)
    engine = create_engine(POSTGRES_CONN_STR)
    df.to_sql(TABLE_NAME, engine, if_exists="append", index=False)
    print(f"Data loaded to Postgres table '{TABLE_NAME}'")

# -----------------------
# DAG Definition
# -----------------------
default_args = {
    "owner": "george",
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

with DAG(
    "stock_pipeline_full",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
    tags=["finance", "stocks"]
) as dag:

    t1 = PythonOperator(task_id="extract", python_callable=extract)
    t2 = PythonOperator(task_id="clean", python_callable=clean)
    t3 = PythonOperator(task_id="load_to_postgres", python_callable=load_to_postgres)

    t1 >> t2 >> t3
