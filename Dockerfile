FROM apache/airflow:2.7.0

USER airflow

# Install dependencies
COPY requirements.txt /requirements.txt
RUN pip install --user --no-cache-dir -r /requirements.txt
