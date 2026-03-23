FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY . /app

RUN uv pip install --system --no-cache-dir -e .

RUN mkdir -p /root/.deepagents

EXPOSE 8001

CMD ["uvicorn", "admin_app.main:app", "--host", "0.0.0.0", "--port", "8001"]
