FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY artifacts/ artifacts/
COPY data/ data/

CMD ["python", "src/pipeline_aws.py"]