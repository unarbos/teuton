# syntax=docker/dockerfile:1.7
# Single Locus image, three roles via entrypoint command at runtime.
# Tags: $DOCKER_USER/locus:miner, :validator, :auditor are all the same image
# (Watchtower polls per tag so independent roll-outs work).
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/root/.local/bin:/usr/local/bin:/usr/bin:/bin

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git openssh-client \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        build-essential pkg-config libssl-dev \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python -m pip install --upgrade pip wheel \
    && rm -rf /var/lib/apt/lists/*

# bittensor-wallet (latent-to fork at feat/roman/add-ed25519-support) is built
# from source via maturin, which needs a Rust toolchain.
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

COPY locus_core ./locus_core
COPY locus_runtime ./locus_runtime
COPY locus_orchestrator ./locus_orchestrator
COPY locus_miner ./locus_miner
COPY locus_validator ./locus_validator
COPY locus_tasks ./locus_tasks
COPY locus_legacy_v2 ./locus_legacy_v2
COPY bench ./bench
COPY scripts ./scripts

RUN /root/.local/bin/uv pip install --system -e ".[subnet,data]"

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ARG GIT_SHA=unknown
ARG BUILD_TIME=unknown
ENV LOCUS_GIT_SHA=$GIT_SHA \
    LOCUS_BUILD_TIME=$BUILD_TIME

# Run id baked at build time. Watchtower-driven rollouts read this at start;
# the entrypoint falls back to it when no explicit LOCUS_RUN_ID is supplied
# via compose. Override at build via `--build-arg LOCUS_RUN_ID=...`.
ARG LOCUS_RUN_ID=""
ENV LOCUS_BAKED_RUN_ID=$LOCUS_RUN_ID

LABEL org.opencontainers.image.source="https://github.com/locus" \
      org.opencontainers.image.description="Locus v3 unified runtime image"

ENTRYPOINT ["/entrypoint.sh"]
CMD ["banner"]
