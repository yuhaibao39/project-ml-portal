FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN chown -R app:app /app
USER app

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.routes.health:router --host 0.0.0.0 --port ${PORT:-8000}"]
