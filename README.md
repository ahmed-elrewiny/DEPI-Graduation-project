# stock_data_etl_project

Local ETL project (EXTRACT → TRANSFORM → LOAD) using SQLite.

## Structure
```
stock_data_etl_project/
  ┣ data/
  ┃  ┣ raw/          # place your raw CSV here: tech_stocks_final.csv
  ┃  ┣ clean/        # cleaned CSV will be written here
  ┃  ┗ db/           # sqlite DB will be created here
  ┣ scripts/
  ┃  ┣ validate_data.py
  ┃  ┣ etl_pipeline.py
  ┃  ┗ simple_query.py
  ┗ README.md
```

## Quick start (local, no Docker)

1. Copy your CSV (`tech_stocks_final.csv`) into `data/raw/`.
   - Current path expected: `data/raw/tech_stocks_final.csv`
2. Create a Python virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate    # Windows (PowerShell)
   ```
3. Install requirements:
   ```bash
   pip install pandas numpy
   ```
4. Validate the raw file:
   ```bash
   python scripts/validate_data.py
   ```
5. Run the ETL pipeline:
   ```bash
   python scripts/etl_pipeline.py
   ```
   - Output: `data/clean/clean_tech_stocks.csv` and `data/db/tech_stocks.db`
6. Quick check of DB:
   ```bash
   python scripts/simple_query.py
   ```

## Notes
- This setup uses **SQLite** for simplicity and portability. If you prefer PostgreSQL, update `etl_pipeline.py` to use SQLAlchemy connection string.
- After the team finishes scraping, they should replace the raw CSV in `data/raw/` and re-run the ETL to update results.
- If you want, I can also prepare a `docker-compose.yaml` + Airflow DAG later to schedule this ETL automatically.

## Support
If anything fails when you run the scripts, copy the terminal output and share it so I can help debug step-by-step.

