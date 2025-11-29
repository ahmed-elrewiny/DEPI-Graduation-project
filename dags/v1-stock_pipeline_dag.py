from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os

DATA_DIR = "/opt/airflow/data"

def extract():
    import yfinance as yf
    symbols = ["AAPL","TSLA","MSFT","NVDA"]
    df = yf.download(symbols, period="1d", group_by='ticker', threads=False)
    out = os.path.join(DATA_DIR, "raw.csv")
    df.to_csv(out)
    print("Saved raw to", out)

def clean():
    import pandas as pd, os
    path = os.path.join(DATA_DIR, "raw.csv")
    df = pd.read_csv(path)
    # example cleaning
    df = df.dropna()
    out = os.path.join(DATA_DIR, "clean.csv")
    df.to_csv(out, index=False)
    print("Saved clean to", out)

default_args = {
    "owner": "george",
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

with DAG("stock_pipeline_simple", default_args=default_args, schedule_interval="@daily", catchup=False) as dag:
    t1 = PythonOperator(task_id="extract", python_callable=extract)
    t2 = PythonOperator(task_id="clean", python_callable=clean)

    t1 >> t2
