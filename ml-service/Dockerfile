FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8001

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . ./

RUN chown -R app:app /app
USER app

EXPOSE 8001
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8001} model:app"]
