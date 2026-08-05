#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
BASE_IMAGE="lightseekorg/tokenspeed-runner-amd@sha256:cc64fe47dc2254677c2ad8ab7e66b851e003ce423d87f2f884f6ba385948f110"
BASE_DIGEST="${BASE_IMAGE##*@}"
SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
SOURCE_BRANCH="$(git -C "${REPO_ROOT}" branch --show-current)"
SOURCE_DATE_EPOCH="$(git -C "${REPO_ROOT}" show -s --format=%ct HEAD)"
IMAGE="${IMAGE:-jeremwan/tokenspeed:rocm7.2.4-torch2.11-profiler-glm-mi355x-${SOURCE_COMMIT:0:8}}"
MAX_JOBS="${MAX_JOBS:-32}"

if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "refusing to replace existing image: ${IMAGE}" >&2
  exit 1
fi

if ! docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
  timeout --signal=TERM --kill-after=30s 15m docker pull "${BASE_IMAGE}"
fi

timeout --signal=TERM --kill-after=30s 15m \
  docker build \
    --file "${SCRIPT_DIR}/Dockerfile" \
    --tag "${IMAGE}" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "BASE_IMAGE_DIGEST=${BASE_DIGEST}" \
    --build-arg "MAX_JOBS=${MAX_JOBS}" \
    --build-arg "SOURCE_COMMIT=${SOURCE_COMMIT}" \
    --build-arg "SOURCE_BRANCH=${SOURCE_BRANCH}" \
    --build-arg "SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}" \
    "${REPO_ROOT}"

printf 'image=%s\n' "${IMAGE}"
docker image inspect --format \
  'id={{.Id}} owner={{index .Config.Labels "io.tokenspeed.owner"}} revision={{index .Config.Labels "org.opencontainers.image.revision"}} base={{index .Config.Labels "io.tokenspeed.base.digest"}}' \
  "${IMAGE}"
