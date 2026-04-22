

FROM python:3.11-slim

# Set working dir before any COPY
WORKDIR /app

# Install system packages and then install Python dependencies (keep pip-install in same layer
# after apt install so pip can build any packages that require headers)
COPY requirements.txt .

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl \
       gcc \
       g++ \
       unixodbc-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code into the image (per rules: copy code/ contents into /app)
COPY code/ .

# Create a non-root user and drop privileges
RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup --no-create-home appuser \
    && chown -R appuser:appgroup /app

USER appuser

# Runtime environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AGENT_NAME=""

# Expose port expected by healthcheck and runtime contract
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command — FastAPI app is available as `app` in agent.py
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8000"]

