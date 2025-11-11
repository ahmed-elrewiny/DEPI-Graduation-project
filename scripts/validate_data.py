# scripts/validate_data.py
import pandas as pd
import os, sys

def validate():
    path = "data/raw/tech_stocks_final.csv"
    if not os.path.exists(path):
        print("ERROR: file not found:", path)
        sys.exit(2)

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} cols")
    required = ["Date", "Abbreviation", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("ERROR: missing columns:", missing)
        sys.exit(3)
    null_dates = df['Date'].isnull().sum() if 'Date' in df.columns else 0
    if null_dates > 0:
        print(f"WARNING: {null_dates} rows have null Date values")
    print("Validation OK")

if __name__ == "__main__":
    validate()

