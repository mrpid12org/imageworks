#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${JUDGE_VISION_TF_CONTAINER:-judge-tf-iqa}"

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "[tf-iqa] Stopping ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
else
  echo "[tf-iqa] Container ${CONTAINER_NAME} not running."
fi
