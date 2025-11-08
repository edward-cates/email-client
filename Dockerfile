FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY tests/ ./tests/
COPY pyproject.toml .

# Expose the application port
EXPOSE 9000

# Run the application
CMD ["uvicorn", "src.web.main:app", "--host", "0.0.0.0", "--port", "9000", "--reload"]

