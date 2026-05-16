#!/usr/bin/env bash
# Build the Teuton runtime image and push it under three Docker Hub tags so
# Watchtower on each role can independently roll out updates:
#   $DOCKER_USER/teuton:miner
#   $DOCKER_USER/teuton:validator
#   $DOCKER_USER/teuton:auditor
#
# Usage:
#   ./scripts/build_push.sh                       # builds + pushes all three tags
#   ./scripts/build_push.sh miner                 # only the :miner tag
#   ./scripts/build_push.sh validator auditor
#   ./scripts/build_push.sh --run-id sn3-2026-05-15  # bake run id into image
#   ./scripts/build_push.sh --run-id-file /tmp/teuton_sn3_run_id miner
#
# The --run-id (or --run-id-file) value is baked into the image as
# TEUTON_BAKED_RUN_ID. When Watchtower then pulls the new image and restarts
# the container, the miner/validator/auditor entrypoint resolves
# TEUTON_RUN_ID to that baked value (unless the host has an explicit RUN_ID /
# TEUTON_RUN_ID in /root/teuton/.env). This is how you flip the whole fleet
# onto a new run without touching any host.
#
# Auth comes from Doppler arbos/dev. We require DOCKER_USER and DOCKER_PAT to
# be present in that config. The PAT is fed to `docker login --password-stdin`,
# never echoed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v doppler >/dev/null; then
    echo "error: doppler CLI not found" >&2; exit 1
fi

RUN_ID=""
RUN_ID_FILE=""
POSITIONAL=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --run-id)
            RUN_ID="${2:-}"; shift 2;;
        --run-id=*)
            RUN_ID="${1#*=}"; shift;;
        --run-id-file)
            RUN_ID_FILE="${2:-}"; shift 2;;
        --run-id-file=*)
            RUN_ID_FILE="${1#*=}"; shift;;
        -h|--help)
            sed -n '1,30p' "$0"; exit 0;;
        --) shift; POSITIONAL+=("$@"); break;;
        -*) echo "error: unknown flag '$1'" >&2; exit 2;;
        *)  POSITIONAL+=("$1"); shift;;
    esac
done

# Resolve baked run id: explicit flag > file > env > fleet.json default file.
if [ -z "$RUN_ID" ]; then
    if [ -n "$RUN_ID_FILE" ]; then
        [ -f "$RUN_ID_FILE" ] || { echo "error: --run-id-file does not exist: $RUN_ID_FILE" >&2; exit 1; }
        RUN_ID=$(tr -d '[:space:]' < "$RUN_ID_FILE")
    elif [ -n "${TEUTON_RUN_ID:-}" ]; then
        RUN_ID="$TEUTON_RUN_ID"
    elif [ -f "/tmp/teuton_sn3_run_id" ]; then
        RUN_ID=$(tr -d '[:space:]' < /tmp/teuton_sn3_run_id)
    fi
fi

ALL_TAGS=(miner validator auditor)
if [ "${#POSITIONAL[@]}" -eq 0 ]; then
    TAGS=("${ALL_TAGS[@]}")
else
    TAGS=("${POSITIONAL[@]}")
fi

for t in "${TAGS[@]}"; do
    case "$t" in
        miner|validator|auditor) ;;
        *) echo "error: unknown tag '$t' (expected miner|validator|auditor)" >&2; exit 1 ;;
    esac
done

GIT_SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo dev)
BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Use a dedicated buildx builder so concurrent runs don't fight over cache.
BUILDER_NAME=teuton-builder
if ! docker buildx inspect "$BUILDER_NAME" >/dev/null 2>&1; then
    docker buildx create --name "$BUILDER_NAME" --driver docker-container --use >/dev/null
else
    docker buildx use "$BUILDER_NAME"
fi
docker buildx inspect --bootstrap >/dev/null

doppler run --project arbos --config dev -- bash -lc '
set -euo pipefail
test -n "${DOCKER_USER:-}" || { echo "DOCKER_USER missing from Doppler" >&2; exit 1; }
test -n "${DOCKER_PAT:-}"  || { echo "DOCKER_PAT missing from Doppler"  >&2; exit 1; }
echo "$DOCKER_PAT" | docker login -u "$DOCKER_USER" --password-stdin
'

TAG_ARGS=()
for t in "${TAGS[@]}"; do
    TAG_ARGS+=("-t" "$(doppler secrets get DOCKER_USER --plain --project arbos --config dev)/teuton:${t}")
done

# Always also push :latest for convenience when we touch all three.
if [ "${#TAGS[@]}" -eq 3 ]; then
    TAG_ARGS+=("-t" "$(doppler secrets get DOCKER_USER --plain --project arbos --config dev)/teuton:latest")
fi

if [ -n "$RUN_ID" ]; then
    echo "[build_push] git=${GIT_SHA} build_time=${BUILD_TIME} tags=${TAGS[*]} baked_run_id=${RUN_ID}"
else
    echo "[build_push] git=${GIT_SHA} build_time=${BUILD_TIME} tags=${TAGS[*]} baked_run_id=<none>"
    echo "[build_push] note: no --run-id supplied; image will rely on host RUN_ID/TEUTON_RUN_ID at runtime."
fi
# --provenance=false / --sbom=false: skip OCI attestations so the resulting
# push is a plain Docker v2 single-platform manifest (no image-index wrapper).
# Watchtower 1.7.x reads Docker v2 manifests; without this it picks up a
# stale digest because its HEAD request lacks OCI Accept headers.
docker buildx build \
    --platform linux/amd64 \
    --file docker/Dockerfile \
    --build-arg GIT_SHA="$GIT_SHA" \
    --build-arg BUILD_TIME="$BUILD_TIME" \
    --build-arg TEUTON_RUN_ID="$RUN_ID" \
    --provenance=false \
    --sbom=false \
    --output type=registry,oci-mediatypes=false \
    "${TAG_ARGS[@]}" \
    .

echo "[build_push] done."
