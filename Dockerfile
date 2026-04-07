FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock requirements.txt ./
COPY src/ ./src/
COPY openenv.yaml ./

# Install dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Install the package in editable mode
RUN pip install -e .

# Expose port 8000
EXPOSE 8000

# Run the FastAPI application
CMD ["python", "-m", "uvicorn", "dlq_triage.main:app", "--host", "0.0.0.0", "--port", "8000"]
