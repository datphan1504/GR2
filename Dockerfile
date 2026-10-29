# Base Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt .

# Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and configurations
COPY configs/ ./configs/
COPY src/ ./src/
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY artifacts/ ./artifacts/

# Expose FastAPI service port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Command to run API service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
