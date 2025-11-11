from datetime import datetime, timedelta
import os
import pandas as pd
import sqlite3

from airflow import DAG
from airflow.operators.python import PythonOperator

# -------------------
#  مسارات البيانات
# -------------------
RAW = os.path.join("data", "raw", "tech_stocks_final.csv")
CLEAN_DIR = os.path.join("data", "clean")
DB_PATH = os.path.join("data", "db", "tech_stocks.db")

# إنشاء المجلدات لو مش موجودة
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/clean", exist_ok=True)
os.makedirs("data/db", exist_ok=True)

# -------------------
#  إعداد الـ DAG
# -------------------
default_args = {
    'owner': 'waleed',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'tech_stock_etl',
    default_args=default_args,
    description='ETL pipeline for stock data',
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
)

# -------------------
#  تعريف الدوال
# -------------------
def extract():
    df = pd.read_csv(RAW)
    print(f"✅ Extracted {len(df)} rows")
    return df.to_dict(orient='records')

def transform(**kwargs):
    ti = kwargs['ti']
    df_dict = ti.xcom_pull(task_ids='extract_task')
    df = pd.DataFrame(df_dict)
    df.dropna(inplace=True)
    df.to_csv(f"{CLEAN_DIR}/clean_from_airflow.csv", index=False)
    print(f"✅ Cleaned & saved {len(df)} rows")
    return df.to_dict(orient='records')

def load(**kwargs):
    ti = kwargs['ti']
    df_dict = ti.xcom_pull(task_ids='transform_task')
    df = pd.DataFrame(df_dict)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql('stocks', conn, if_exists='replace', index=False)
    conn.close()
    print("✅ Data loaded to SQLite DB")

# -------------------
#  تعريف التاسكات
# -------------------
extract_task = PythonOperator(
    task_id='extract_task',
    python_callable=extract,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_task',
    python_callable=transform,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_task',
    python_callable=load,
    dag=dag,
)

# ترتيب التاسكات
extract_task >> transform_task >> load_task
