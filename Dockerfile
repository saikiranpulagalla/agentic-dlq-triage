FROM python:3.11-slim

WORKDIR /app

# Install uv for dependency management
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml requirements.txt ./
COPY src/ ./src/
COPY openenv.yaml ./

# Install dependencies using uv
RUN uv pip install --system -r requirements.txt

# Expose port 8000
EXPOSE 8000

# Run the FastAPI application
CMD ["python", "-m", "uvicorn", "dlq_triage.main:app", "--host", "0.0.0.0", "--port", "8000"]
