#!/usr/bin/env bash
set -euo pipefail

IMAGE="${JUDGE_VISION_TF_IMAGE:-nvcr.io/nvidia/tensorflow:24.02-tf2-py3}"
echo "Pulling Judge Vision TensorFlow backend image: ${IMAGE}"
docker pull "${IMAGE}"
