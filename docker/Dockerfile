# syntax=docker/dockerfile:1.7
# Single Teuton image, three roles via entrypoint command at runtime.
# Tags: $DOCKER_USER/teuton:miner, :validator, :auditor are all the same image
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

COPY teuton_core ./teuton_core
COPY teuton_runtime ./teuton_runtime
COPY teuton_orchestrator ./teuton_orchestrator
COPY teuton_miner ./teuton_miner
COPY teuton_validator ./teuton_validator
COPY teuton_tasks ./teuton_tasks
COPY teuton_legacy_v2 ./teuton_legacy_v2
COPY bench ./bench
COPY scripts ./scripts

RUN /root/.local/bin/uv pip install --system -e ".[subnet,data]"

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ARG GIT_SHA=unknown
ARG BUILD_TIME=unknown
ENV TEUTON_GIT_SHA=$GIT_SHA \
    TEUTON_BUILD_TIME=$BUILD_TIME

# Run id baked at build time. Watchtower-driven rollouts read this at start;
# the entrypoint falls back to it when no explicit TEUTON_RUN_ID is supplied
# via compose. Override at build via `--build-arg TEUTON_RUN_ID=...`.
ARG TEUTON_RUN_ID=""
ENV TEUTON_BAKED_RUN_ID=$TEUTON_RUN_ID

# Public fleet defaults. These are safe to bake into public miner images:
# they identify where the miner should look and which protocol defaults to use.
# Secrets and per-miner hotkeys stay runtime-only.
ARG S3_BUCKET=""
ARG S3_REGION="us-east-1"
ARG S3_ENDPOINT_URL=""
ARG TEUTON_NETUID="3"
ARG BT_NETWORK="finney"
ARG TEUTON_GRANT_MODE="presigned"
ARG TEUTON_ASSIGNMENT_CRYPTO="ed25519"
ARG TEUTON_DISCOVERY_BACKEND="bucket"
ARG TEUTON_MINER_POLL_INTERVAL="0.05"
ARG TEUTON_OWNER_HOTKEY=""
ENV TEUTON_BAKED_S3_BUCKET=$S3_BUCKET \
    TEUTON_BAKED_S3_REGION=$S3_REGION \
    TEUTON_BAKED_S3_ENDPOINT_URL=$S3_ENDPOINT_URL \
    TEUTON_BAKED_NETUID=$TEUTON_NETUID \
    TEUTON_BAKED_BT_NETWORK=$BT_NETWORK \
    TEUTON_BAKED_GRANT_MODE=$TEUTON_GRANT_MODE \
    TEUTON_BAKED_ASSIGNMENT_CRYPTO=$TEUTON_ASSIGNMENT_CRYPTO \
    TEUTON_BAKED_DISCOVERY_BACKEND=$TEUTON_DISCOVERY_BACKEND \
    TEUTON_BAKED_MINER_POLL_INTERVAL=$TEUTON_MINER_POLL_INTERVAL \
    TEUTON_BAKED_OWNER_HOTKEY=$TEUTON_OWNER_HOTKEY

LABEL org.opencontainers.image.source="https://github.com/unarbos/teuton" \
      org.opencontainers.image.description="Teuton v3 unified runtime image"

ENTRYPOINT ["/entrypoint.sh"]
CMD ["banner"]
