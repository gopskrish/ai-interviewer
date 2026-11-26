FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/huggingface

WORKDIR /app

# Dependencies first, in their own layer, so code changes rebuild in seconds.
COPY pyproject.toml ./
RUN mkdir -p interviewer && touch interviewer/__init__.py \
    && pip install . \
    && pip uninstall -y ai-interviewer \
    && rm -rf interviewer build

COPY interviewer ./interviewer
RUN pip install --no-deps . && rm -rf build
COPY kits ./kits
COPY scripts ./scripts

# Runs as the usual first WSL user (uid 1000), so files in data/ and models/ stay yours.
RUN useradd --create-home --uid 1000 app && mkdir -p /data /models && chown app:app /data /models
USER app

EXPOSE 8443
CMD ["uvicorn", "interviewer.web.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8443", \
     "--ssl-keyfile", "/certs/server.key", "--ssl-certfile", "/certs/server.crt"]
