# GridWise LLM-Assisted Energy Optimizer — production Dockerfile
# Multi-stage build: tiny final image, no model files baked in.
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build deps for PuLP wheels (CBC is bundled in PuLP for manylinux).
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy source
COPY app /app/app


FROM python:3.11-slim

WORKDIR /app

# Copy installed packages and source from the builder.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/app /app/app
COPY --from=builder /app/requirements.txt /app/requirements.txt

# Use a non-root user for the runtime.
RUN useradd --create-home --uid 1000 gridwise
USER gridwise

EXPOSE 8000

# Bind to 0.0.0.0 so the port is reachable externally.
ENV PORT=8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
