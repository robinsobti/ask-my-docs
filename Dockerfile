# syntax=docker/dockerfile:1.7

######
# Builder stage: install Python dependencies into a dedicated virtualenv.
######
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv

WORKDIR /app

RUN python -m venv $VENV_PATH \
    && $VENV_PATH/bin/pip install --no-cache-dir --upgrade pip

COPY requirements.txt ./
RUN $VENV_PATH/bin/pip install --no-cache-dir -r requirements.txt


######
# Runtime stage: copy only the venv + source, run as unprivileged user.
######
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv

WORKDIR /app

# Build metadata injected at build time (falls back to dev-friendly defaults).
ARG APP_BUILD_VERSION="dev"
ARG APP_BUILD_SHA="local"
ARG APP_BUILD_TIME="unknown"
ENV APP_BUILD_VERSION=${APP_BUILD_VERSION} \
    APP_BUILD_SHA=${APP_BUILD_SHA} \
    APP_BUILD_TIME=${APP_BUILD_TIME}

# Create a non-root user/group to run the service.
RUN groupadd --system appuser \
    && useradd --system --create-home --gid appuser appuser

COPY --from=builder $VENV_PATH $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

COPY . .

USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
