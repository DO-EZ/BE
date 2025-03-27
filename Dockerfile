FROM python:3.9-alpine

# Set the working directory
WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir uv==0.6.3 && \
    uv sync

COPY src/ .

CMD ["uv", "run", "uvicorn", "--host", "0.0.0.0", "--port", "8000", "main:app"]

