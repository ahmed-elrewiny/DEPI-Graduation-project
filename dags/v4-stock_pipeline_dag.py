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
TABLE_NAME = "tech_stocks_clean"

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX",
    "AMD", "INTC", "ADBE", "CRM", "PYPL", "ORCL", "CSCO", "UBER",
    "SHOP", "SQ", "BABA", "IBM", "QCOM", "SPOT", "ASML", "AVGO"
]

START_DATE = "2010-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# -----------------------
# 1) Extract + Compute Change & Change %
# -----------------------
def extract():
    all_data = []

    for ticker in TICKERS:
        print(f"Downloading {ticker}...")

        try:
            data = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=False)

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]

            data["Ticker"] = ticker
            data["Change"] = data["Close"].diff()
            data["Change %"] = (data["Change"] / data["Close"].shift(1)) * 100

            cleaned = data.reset_index()[["Date", "Ticker", "Close", "Change", "Change %"]]
            all_data.append(cleaned)

        except Exception as e:
            print(f"Error downloading {ticker}: {e}")

    if not all_data:
        raise ValueError("No data downloaded for any ticker!")

    final_df = pd.concat(all_data, ignore_index=True)

    raw_path = os.path.join(DATA_DIR, "raw.csv")
    final_df.to_csv(raw_path, index=False)

    print(f"Saved raw data to: {raw_path}")
    print(final_df.head(10))

# -----------------------
# 2) Clean (Drop NaN)
# -----------------------
def clean():
    raw_path = os.path.join(DATA_DIR, "raw.csv")

    if not os.path.exists(raw_path):
        raise FileNotFoundError("raw.csv not found!")

    df = pd.read_csv(raw_path)

    print("Before cleaning:")
    print(df.isna().sum())

    df_cleaned = df.dropna()

    print("After cleaning:")
    print(df_cleaned.isna().sum())

    clean_path = os.path.join(DATA_DIR, "clean.csv")
    df_cleaned.to_csv(clean_path, index=False)

    print(f"Saved cleaned data to: {clean_path}")
    print(df_cleaned.head(10))

# -----------------------
# 3) Load to Postgres
# -----------------------
def load_to_postgres():
    clean_path = os.path.join(DATA_DIR, "clean.csv")

    if not os.path.exists(clean_path):
        raise FileNotFoundError("clean.csv not found!")

    df = pd.read_csv(clean_path)
    engine = create_engine(POSTGRES_CONN_STR)

    df.to_sql(TABLE_NAME, engine, if_exists="append", index=False)

    print(f"Data loaded to PostgreSQL table: {TABLE_NAME}")

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
    "tech_stock_pipeline_2010_present",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
    tags=["stocks", "yfinance", "tech"]
) as dag:

    t1 = PythonOperator(task_id="extract", python_callable=extract)
    t2 = PythonOperator(task_id="clean", python_callable=clean)
    t3 = PythonOperator(task_id="load_to_postgres", python_callable=load_to_postgres)

    t1 >> t2 >> t3
