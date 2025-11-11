# scripts/etl_pipeline.py
import pandas as pd
import numpy as np
import sqlite3
import os

RAW = "D:\YAS\رواد مصر الرقمية\ETL\stock_data_etl_project/data/raw/tech_stocks_final.csv"
CLEAN_DIR = "data/clean"
DB_PATH = "data/db/tech_stocks.db"

def extract():
    if not os.path.exists(RAW):
        raise FileNotFoundError(f"Raw file not found: {RAW}")
    df = pd.read_csv(RAW)
    print("extract -> rows:", len(df))
    return df

def transform(df):
    if 'Abbreviation' in df.columns:
        df = df.rename(columns={'Abbreviation': 'Ticker'})
    cols = df.columns.tolist()
    keep = [c for c in ['Date','Ticker','Close','Change','Change %','Volume'] if c in cols]
    df = df[keep].copy()

    df = df.dropna(subset=['Date','Ticker','Close']).copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()

    df = df.sort_values(['Ticker','Date'])
    df = df.drop_duplicates(subset=['Ticker','Date'])

    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])

    df['Daily_Return'] = df.groupby('Ticker')['Close'].pct_change()
    df['MA_7'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df['MA_30'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    df['Volatility_30'] = df.groupby('Ticker')['Daily_Return'].transform(lambda x: x.rolling(window=30, min_periods=1).std())

    print("transform -> rows after transform:", len(df))
    return df

def load(df):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("tech_stocks", conn, if_exists="replace", index=False)
    conn.close()
    print("Loaded into DB:", DB_PATH)

def save_csv(df):
    os.makedirs(CLEAN_DIR, exist_ok=True)
    out = os.path.join(CLEAN_DIR, "clean_tech_stocks.csv")
    df.to_csv(out, index=False)
    print("Saved clean CSV:", out)

def run():
    df = extract()
    df_clean = transform(df)
    save_csv(df_clean)
    load(df_clean)
    print("ETL done. rows after transform:", len(df_clean))

if __name__ == "__main__":
    run()

