FROM python:3.12-slim
 
WORKDIR /app
 
# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 gcc libpq-dev \
    tesseract-ocr tesseract-ocr-rus tesseract-ocr-kir \
    && rm -rf /var/lib/apt/lists/*
 
COPY requirements.txt .
 
# Увеличиваем таймаут и добавляем retry для нестабильного соединения
RUN pip install --no-cache-dir \
    --timeout=120 \
    --retries=5 \
    -r requirements.txt
 
COPY . .
 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]