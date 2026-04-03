#!/usr/bin/env bash

set -euo pipefail

RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"
RUNPOD_API_BASE="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
RUNPOD_NAME="${RUNPOD_NAME:-parameter-golf-hy-4090}"
RUNPOD_GPU_TYPE="${RUNPOD_GPU_TYPE:-NVIDIA GeForce RTX 4090}"
RUNPOD_GPU_COUNT="${RUNPOD_GPU_COUNT:-1}"
RUNPOD_IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04}"
RUNPOD_CONTAINER_DISK_GB="${RUNPOD_CONTAINER_DISK_GB:-50}"
RUNPOD_VOLUME_GB="${RUNPOD_VOLUME_GB:-120}"
RUNPOD_PORTS="${RUNPOD_PORTS:-22/tcp,8888/http}"
RUNPOD_VOLUME_MOUNT_PATH="${RUNPOD_VOLUME_MOUNT_PATH:-/workspace}"
REPO_OWNER="${REPO_OWNER:-yanghu819}"
REPO_NAME="${REPO_NAME:-parameter-golf-hy}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/$REPO_NAME}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/gh_yanghu819_ed25519}"
SSH_PUBLIC_KEY_PATH="${SSH_PUBLIC_KEY_PATH:-$SSH_KEY_PATH.pub}"

need() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing dependency: $1" >&2
        exit 1
    fi
}

need curl
need jq
need ssh

if [[ -z "$RUNPOD_API_KEY" ]]; then
    echo "RUNPOD_API_KEY is not set." >&2
    exit 1
fi

api() {
    local method="$1"
    local path="$2"
    local data="${3:-}"
    if [[ -n "$data" ]]; then
        curl -fsSL -X "$method" \
            -H "Authorization: Bearer $RUNPOD_API_KEY" \
            -H "Content-Type: application/json" \
            "$RUNPOD_API_BASE$path" \
            -d "$data"
        return
    fi

    curl -fsSL -X "$method" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        "$RUNPOD_API_BASE$path"
}

ports_json() {
    jq -nc --arg raw "$RUNPOD_PORTS" '$raw | split(",")'
}

pod_json() {
    api GET "/pods/$1"
}

pod_ip() {
    pod_json "$1" | jq -r '.publicIp // empty'
}

pod_ssh_port() {
    pod_json "$1" | jq -r '.portMappings["22"] // empty'
}

ssh_target() {
    local pod_id="$1"
    local ip port
    ip="$(pod_ip "$pod_id")"
    port="$(pod_ssh_port "$pod_id")"
    if [[ -z "$ip" || -z "$port" ]]; then
        echo "Unable to resolve SSH endpoint for pod $pod_id" >&2
        exit 1
    fi
    printf '%s %s\n' "$ip" "$port"
}

ssh_run() {
    local pod_id="$1"
    shift
    local ip port
    read -r ip port < <(ssh_target "$pod_id")
    ssh -i "$SSH_KEY_PATH" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -p "$port" \
        "root@$ip" \
        "$@"
}

usage() {
    cat <<'EOF'
Usage:
  bash runpod.sh list
  bash runpod.sh get POD_ID
  bash runpod.sh create [pod_name]
  bash runpod.sh start POD_ID
  bash runpod.sh stop POD_ID
  bash runpod.sh delete POD_ID
  bash runpod.sh ssh POD_ID
  bash runpod.sh bootstrap POD_ID
  bash runpod.sh sync POD_ID
  bash runpod.sh download POD_ID
  bash runpod.sh train POD_ID
  bash runpod.sh autostop POD_ID 2h

Environment overrides:
  RUNPOD_GPU_TYPE="NVIDIA GeForce RTX 4090"
  RUNPOD_IMAGE="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
  REPO_OWNER=yanghu819
  REPO_NAME=parameter-golf-hy
EOF
}

create_pod() {
    local pod_name="${1:-$RUNPOD_NAME}"
    local payload
    local public_key=""
    if [[ -f "$SSH_PUBLIC_KEY_PATH" ]]; then
        public_key="$(tr -d '\n' < "$SSH_PUBLIC_KEY_PATH")"
    fi
    payload="$(
        jq -nc \
            --arg name "$pod_name" \
            --arg image "$RUNPOD_IMAGE" \
            --arg gpuType "$RUNPOD_GPU_TYPE" \
            --arg mountPath "$RUNPOD_VOLUME_MOUNT_PATH" \
            --arg publicKey "$public_key" \
            --argjson gpuCount "$RUNPOD_GPU_COUNT" \
            --argjson containerDisk "$RUNPOD_CONTAINER_DISK_GB" \
            --argjson volumeDisk "$RUNPOD_VOLUME_GB" \
            --argjson ports "$(ports_json)" \
            '{
                name: $name,
                imageName: $image,
                gpuCount: $gpuCount,
                gpuTypeIds: [$gpuType],
                containerDiskInGb: $containerDisk,
                volumeInGb: $volumeDisk,
                volumeMountPath: $mountPath,
                ports: $ports,
                env: {
                    PUBLIC_KEY: $publicKey
                }
            }'
    )"
    api POST "/pods" "$payload" | jq -r '{id, name, desiredStatus, publicIp, portMappings}'
}

case "${1:-}" in
    list)
        api GET "/pods" | jq -r '.[] | [.id, .name, .desiredStatus, (.publicIp // "-"), (.portMappings["22"] // "-")] | @tsv'
        ;;
    get)
        [[ $# -eq 2 ]] || { usage; exit 1; }
        pod_json "$2" | jq
        ;;
    create)
        create_pod "${2:-}"
        ;;
    start)
        [[ $# -eq 2 ]] || { usage; exit 1; }
        api POST "/pods/$2/start" | jq
        ;;
    stop)
        [[ $# -eq 2 ]] || { usage; exit 1; }
        api POST "/pods/$2/stop" | jq
        ;;
    delete)
        [[ $# -eq 2 ]] || { usage; exit 1; }
        api DELETE "/pods/$2" >/dev/null
        echo "deleted $2"
        ;;
    ssh)
        [[ $# -ge 2 ]] || { usage; exit 1; }
        shift
        pod_id="$1"
        shift
        if [[ $# -eq 0 ]]; then
            read -r ip port < <(ssh_target "$pod_id")
            exec ssh -i "$SSH_KEY_PATH" \
                -o StrictHostKeyChecking=no \
                -o UserKnownHostsFile=/dev/null \
                -p "$port" \
                "root@$ip"
        fi
        ssh_run "$pod_id" "$@"
        ;;
    bootstrap)
        [[ $# -eq 2 ]] || { usage; exit 1; }
        ssh_run "$2" "bash -lc '
            set -euo pipefail
            if [[ ! -d \"$REMOTE_ROOT/.git\" ]]; then
                git clone https://github.com/$REPO_OWNER/$REPO_NAME.git \"$REMOTE_ROOT\"
            fi
            git -C \"$REMOTE_ROOT\" fetch origin main
            git -C \"$REMOTE_ROOT\" checkout main
            git -C \"$REMOTE_ROOT\" pull --ff-only origin main
            cd \"$REMOTE_ROOT\"
            bash setup.sh
        '"
        ;;
    sync)
        [[ $# -eq 2 ]] || { usage; exit 1; }
        ssh_run "$2" "bash -lc '
            set -euo pipefail
            git -C \"$REMOTE_ROOT\" fetch origin main
            git -C \"$REMOTE_ROOT\" pull --ff-only origin main
        '"
        ;;
    download)
        [[ $# -eq 2 ]] || { usage; exit 1; }
        ssh_run "$2" "bash -lc '
            set -euo pipefail
            cd \"$REMOTE_ROOT\"
            bash down.sh
        '"
        ;;
    train)
        [[ $# -eq 2 ]] || { usage; exit 1; }
        ssh_run "$2" "bash -lc '
            set -euo pipefail
            cd \"$REMOTE_ROOT\"
            bash run.sh
        '"
        ;;
    autostop)
        [[ $# -eq 3 ]] || { usage; exit 1; }
        ssh_run "$2" "bash -lc '
            set -euo pipefail
            export RUNPOD_POD_ID=\"$2\"
            nohup bash -lc \"sleep $3; runpodctl stop pod \\\$RUNPOD_POD_ID\" >/tmp/runpod-autostop.log 2>&1 &
            echo scheduled_autostop_after=$3
        '"
        ;;
    ""|-h|--help|help)
        usage
        ;;
    *)
        usage
        exit 1
        ;;
esac
