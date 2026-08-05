#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"

IMAGE="${IMAGE:-jeremwan/tokenspeed:rocm7.2.4-torch2.11-profiler-glm-mi355x-${SOURCE_COMMIT:0:8}}"
CONTAINER="${CONTAINER:-jeremwan-ar-rmsnorm-profiler-mi355x-${SOURCE_COMMIT:0:8}}"
MODEL_PATH="${MODEL_PATH:-/data/models/glm-5.2-fp8}"
MODEL_SOURCE="$(realpath "${MODEL_PATH}")"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "image does not exist: ${IMAGE}" >&2
  exit 1
fi
if docker container inspect "${CONTAINER}" >/dev/null 2>&1; then
  echo "refusing to alter existing container: ${CONTAINER}" >&2
  exit 1
fi
if [[ ! -r "${MODEL_SOURCE}/model.safetensors.index.json" ]]; then
  echo "model is missing or unreadable: ${MODEL_SOURCE}" >&2
  exit 1
fi

IMAGE_OWNER="$(
  docker image inspect --format \
    '{{ index .Config.Labels "io.tokenspeed.owner" }}' "${IMAGE}"
)"
IMAGE_REVISION="$(
  docker image inspect --format \
    '{{ index .Config.Labels "org.opencontainers.image.revision" }}' "${IMAGE}"
)"
if [[ "${IMAGE_OWNER}" != "jeremwan" || "${IMAGE_REVISION}" != "${SOURCE_COMMIT}" ]]; then
  echo "refusing image with unexpected owner/revision: owner=${IMAGE_OWNER} revision=${IMAGE_REVISION}" >&2
  exit 1
fi

docker run --detach \
  --name "${CONTAINER}" \
  --label io.tokenspeed.owner=jeremwan \
  --label io.tokenspeed.purpose=ar-rmsnorm-glm-profiler \
  --ipc=host \
  --network=host \
  --pid=host \
  --privileged \
  --security-opt seccomp=unconfined \
  --shm-size 32g \
  --ulimit memlock=-1:-1 \
  --env MODEL_PATH=/data/models/glm-5.2-fp8 \
  --env CONTAINER_REPO_ROOT=/home/jeremwan/tokenspeed \
  --mount "type=bind,src=${REPO_ROOT},dst=/home/jeremwan/tokenspeed" \
  --mount "type=bind,src=${MODEL_SOURCE},dst=/data/models/glm-5.2-fp8,readonly" \
  --workdir /home/jeremwan/tokenspeed \
  "${IMAGE}" >/dev/null

printf 'container=%s\nimage=%s\n' "${CONTAINER}" "${IMAGE}"
printf 'export TOKENSPEED_CONTAINER=%q\n' "${CONTAINER}"
printf 'export CONTAINER_REPO_ROOT=%q\n' /home/jeremwan/tokenspeed
printf 'export MODEL_PATH=%q\n' /data/models/glm-5.2-fp8
