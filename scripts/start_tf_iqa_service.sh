#!/usr/bin/env bash
set -euo pipefail

IMAGE="${JUDGE_VISION_TF_IMAGE:-nvcr.io/nvidia/tensorflow:24.02-tf2-py3}"
CONTAINER_NAME="${JUDGE_VISION_TF_CONTAINER:-judge-tf-iqa}"
HOST_PORT="${JUDGE_VISION_TF_PORT:-5105}"
CONTAINER_PORT=5005

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${IMAGEWORKS_MODEL_ROOT:-$HOME/ai-models/weights}"

# Normalize model root: if it already ends with weights, mount the parent
if [[ "$(basename "$MODEL_ROOT")" == "weights" ]]; then
  HOST_MODELS_DIR="$(cd "$MODEL_ROOT/.." && pwd)"
else
  HOST_MODELS_DIR="$(cd "$MODEL_ROOT" && pwd)"
fi

echo "[tf-iqa] Starting container ${CONTAINER_NAME} (image: ${IMAGE})"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -v "${PROJECT_ROOT}:${PROJECT_ROOT}" \
  -v "${HOST_MODELS_DIR}:/root/ai-models" \
  -w "${PROJECT_ROOT}" \
  -e PYTHONPATH="${PROJECT_ROOT}/src" \
  -e IMAGEWORKS_MODEL_ROOT=/root/ai-models \
  -e TFHUB_CACHE_DIR=/root/ai-models/weights/judge-iqa/musiq/tfhub-cache \
  -e TFHUB_CACHE=/root/ai-models/weights/judge-iqa/musiq/tfhub-cache \
  -e JUDGE_VISION_INSIDE_CONTAINER=1 \
  -e TF_CPP_MIN_LOG_LEVEL=2 \
  "${IMAGE}" \
  bash -lc "pip install -q --no-cache-dir 'numpy<2.0' 'tensorflow-hub==0.15.0' tomli && python3 src/imageworks/apps/judge_vision/tf_inference_service.py serve --host 0.0.0.0 --port ${CONTAINER_PORT}"

echo "[tf-iqa] Service listening on http://localhost:${HOST_PORT}"
