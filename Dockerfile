# HuggingFace Spaces — ToolOrchestratorEnv
# Builds a FastAPI server that exposes the OpenEnv endpoints.
#
# Required secret (set in Space Settings → Variables and secrets):
#   CERAMIC_API_KEY=cer_sk_live_...
#
# Optional:
#   TOGETHER_API_KEY=...  (enables the llm_reason tool)
#   HF_TOKEN=...          (enables gated datasets like GPQA)

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker layer-caches them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# HuggingFace Spaces runs as a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# The Space README sets base_path: /web so the demo UI loads on open
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
