# Build stage: resolve the locked dependencies and install the package into a venv.
FROM python:3.12-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /src
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable


# Runtime stage: the venv, LightGBM's OpenMP runtime and nothing else.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin finqa \
    && mkdir -p /data \
    && chown finqa:finqa /data

COPY --from=build /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FINQA_SEARCH_DATA_DIR=/data \
    HF_HOME=/data/hf

USER 10001
WORKDIR /home/finqa
VOLUME ["/data"]

ENTRYPOINT ["finqa-search"]
CMD ["--help"]
