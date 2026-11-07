FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd \
      tesseract-ocr-hin tesseract-ocr-urd tesseract-ocr-ara tesseract-ocr-fra tesseract-ocr-deu tesseract-ocr-spa \
      curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# High-accuracy English model (tessdata_best). Other languages fall back to the system models
# unless you add them here, e.g. `download_models.sh eng hin`.
RUN ./scripts/download_models.sh eng

EXPOSE 8000
CMD ["uvicorn", "ocrscanner.server:app", "--host", "0.0.0.0", "--port", "8000"]
