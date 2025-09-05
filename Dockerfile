FROM python:3.12
LABEL authors="Hau"

WORKDIR /app

# Copy và cài dependencies
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy code và model
COPY . .

EXPOSE 8888


CMD ["sh", "-c", "uvicorn src.server:app --host 0.0.0.0 --port $PORT"]


