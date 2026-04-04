#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_REPO="${IMAGE_REPO:-ghcr.io/yanghu819/parameter-golf-h100-runtime}"
IMAGE_TAG="${IMAGE_TAG:-torch291-cu128-fa3}"
IMAGE_REF="${IMAGE_REPO}:${IMAGE_TAG}"
PLATFORM="${PLATFORM:-linux/amd64}"
GITHUB_USERNAME="${GITHUB_USERNAME:-yanghu819}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

if [[ -z "$GITHUB_TOKEN" ]]; then
    echo "GITHUB_TOKEN is required to push ${IMAGE_REF}" >&2
    exit 1
fi

echo "$GITHUB_TOKEN" | docker login ghcr.io -u "$GITHUB_USERNAME" --password-stdin
docker buildx build \
    --platform "$PLATFORM" \
    --file "$ROOT_DIR/docker/h100-runtime.Dockerfile" \
    --tag "$IMAGE_REF" \
    --push \
    "$ROOT_DIR"

echo "pushed_image=${IMAGE_REF}"
