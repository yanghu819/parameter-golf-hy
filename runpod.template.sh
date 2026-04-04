#!/usr/bin/env bash

set -euo pipefail

RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"
RUNPOD_API_BASE="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
RUNPOD_PROFILE="${RUNPOD_PROFILE:-}"
RUNPOD_NAME="${RUNPOD_NAME:-}"
RUNPOD_GPU_TYPE="${RUNPOD_GPU_TYPE:-}"
RUNPOD_GPU_COUNT="${RUNPOD_GPU_COUNT:-}"
RUNPOD_IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04}"
RUNPOD_CONTAINER_DISK_GB="${RUNPOD_CONTAINER_DISK_GB:-50}"
RUNPOD_VOLUME_GB="${RUNPOD_VOLUME_GB:-}"
RUNPOD_PORTS="${RUNPOD_PORTS:-22/tcp,8888/http}"
RUNPOD_VOLUME_MOUNT_PATH="${RUNPOD_VOLUME_MOUNT_PATH:-/workspace}"
RUNPOD_NETWORK_VOLUME_ID="${RUNPOD_NETWORK_VOLUME_ID:-}"
RUNPOD_DATA_CENTER_ID="${RUNPOD_DATA_CENTER_ID:-}"
RUNPOD_VOLUME_NAME="${RUNPOD_VOLUME_NAME:-}"
REPO_OWNER="${REPO_OWNER:-yanghu819}"
REPO_NAME="${REPO_NAME:-parameter-golf-hy}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/$REPO_NAME}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/gh_yanghu819_ed25519}"
SSH_PUBLIC_KEY_PATH="${SSH_PUBLIC_KEY_PATH:-$SSH_KEY_PATH.pub}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H100_RECORD_DIR="${H100_RECORD_DIR:-records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072}"
H100_VENV_DIR="${H100_VENV_DIR:-$REMOTE_ROOT/.venv-h100}"
H100_TRAIN_SHARDS="${H100_TRAIN_SHARDS:-80}"

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

set_if_empty() {
    local var_name="$1"
    local value="$2"
    if [[ -z "${!var_name:-}" ]]; then
        printf -v "$var_name" '%s' "$value"
    fi
}

apply_profile() {
    case "${1:-}" in
        "" )
            ;;
        cheap-smoke)
            set_if_empty RUNPOD_NAME "$REPO_NAME-cheap-smoke"
            set_if_empty RUNPOD_GPU_TYPE "NVIDIA GeForce RTX 4090"
            set_if_empty RUNPOD_GPU_COUNT "1"
            set_if_empty RUNPOD_DATA_CENTER_ID "US-IL-1"
            set_if_empty RUNPOD_VOLUME_GB "25"
            set_if_empty RUNPOD_VOLUME_NAME "$REPO_NAME-cheap-us-il-1"
            ;;
        h100-prep)
            set_if_empty RUNPOD_NAME "$REPO_NAME-h100-prep"
            set_if_empty RUNPOD_GPU_TYPE "NVIDIA GeForce RTX 4090"
            set_if_empty RUNPOD_GPU_COUNT "1"
            set_if_empty RUNPOD_DATA_CENTER_ID "CA-MTL-1"
            set_if_empty RUNPOD_VOLUME_GB "45"
            set_if_empty RUNPOD_VOLUME_NAME "$REPO_NAME-h100-ca-mtl-1"
            ;;
        h100-single)
            set_if_empty RUNPOD_NAME "$REPO_NAME-h100-single"
            set_if_empty RUNPOD_GPU_TYPE "NVIDIA H100 80GB HBM3"
            set_if_empty RUNPOD_GPU_COUNT "1"
            set_if_empty RUNPOD_DATA_CENTER_ID "CA-MTL-1"
            set_if_empty RUNPOD_VOLUME_GB "45"
            set_if_empty RUNPOD_VOLUME_NAME "$REPO_NAME-h100-ca-mtl-1"
            ;;
        h100-formal)
            set_if_empty RUNPOD_NAME "$REPO_NAME-h100-formal"
            set_if_empty RUNPOD_GPU_TYPE "NVIDIA H100 80GB HBM3"
            set_if_empty RUNPOD_GPU_COUNT "8"
            set_if_empty RUNPOD_DATA_CENTER_ID "CA-MTL-1"
            set_if_empty RUNPOD_VOLUME_GB "45"
            set_if_empty RUNPOD_VOLUME_NAME "$REPO_NAME-h100-ca-mtl-1"
            ;;
        *)
            echo "Unknown RUNPOD_PROFILE=$1" >&2
            exit 1
            ;;
    esac
}

apply_profile "$RUNPOD_PROFILE"

RUNPOD_NAME="${RUNPOD_NAME:-parameter-golf-hy-4090}"
RUNPOD_GPU_TYPE="${RUNPOD_GPU_TYPE:-NVIDIA GeForce RTX 4090}"
RUNPOD_GPU_COUNT="${RUNPOD_GPU_COUNT:-1}"
RUNPOD_VOLUME_GB="${RUNPOD_VOLUME_GB:-120}"
RUNPOD_VOLUME_NAME="${RUNPOD_VOLUME_NAME:-$REPO_NAME-volume}"

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

graphql() {
    local query="$1"
    curl -fsSL "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
        -H "content-type: application/json" \
        --data "$(jq -nc --arg query "$query" '{query: $query}')"
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

resolve_target_sha() {
    local explicit_sha="${1:-}"
    if [[ -n "$explicit_sha" ]]; then
        printf '%s\n' "$explicit_sha"
        return
    fi
    git -C "$LOCAL_ROOT" rev-parse HEAD
}

wait_for_ssh() {
    local pod_id="$1"
    local timeout_sec="${2:-300}"
    local deadline=$(( $(date +%s) + timeout_sec ))
    while (( $(date +%s) < deadline )); do
        if ssh_run "$pod_id" "true" >/dev/null 2>&1; then
            echo "ssh_ready $pod_id"
            return
        fi
        sleep 5
    done
    echo "Timed out waiting for SSH on pod $pod_id" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage:
  bash runpod.sh list
  bash runpod.sh stock GPU_TYPE
  bash runpod.sh volumes
  bash runpod.sh volume-create [name]
  bash runpod.sh volume-delete VOLUME_ID
  bash runpod.sh get POD_ID
  bash runpod.sh create [pod_name]
  bash runpod.sh wait POD_ID [timeout_sec]
  bash runpod.sh start POD_ID
  bash runpod.sh stop POD_ID
  bash runpod.sh terminate POD_ID
  bash runpod.sh ssh POD_ID
  bash runpod.sh bootstrap POD_ID [commit_sha]
  bash runpod.sh sync POD_ID [commit_sha]
  bash runpod.sh download POD_ID [commit_sha]
  bash runpod.sh train POD_ID [commit_sha]
  bash runpod.sh h100-prep POD_ID [commit_sha]
  bash runpod.sh h100-preflight POD_ID [commit_sha]
  bash runpod.sh h100-formal POD_ID [commit_sha]
  bash runpod.sh autostop POD_ID 2h

Environment overrides:
  RUNPOD_PROFILE=cheap-smoke|h100-prep|h100-single|h100-formal
  RUNPOD_GPU_TYPE="NVIDIA GeForce RTX 4090"
  RUNPOD_IMAGE="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
  RUNPOD_NETWORK_VOLUME_ID=...
  RUNPOD_DATA_CENTER_ID=US-IL-1
  RUNPOD_VOLUME_NAME=parameter-golf-hy-h100-ca-mtl-1
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
    if [[ -n "$RUNPOD_NETWORK_VOLUME_ID" ]]; then
        if [[ -z "$RUNPOD_DATA_CENTER_ID" ]]; then
            echo "RUNPOD_DATA_CENTER_ID is required when RUNPOD_NETWORK_VOLUME_ID is set" >&2
            exit 1
        fi
        payload="$(
            jq -nc \
                --arg name "$pod_name" \
                --arg image "$RUNPOD_IMAGE" \
                --arg gpuType "$RUNPOD_GPU_TYPE" \
                --arg mountPath "$RUNPOD_VOLUME_MOUNT_PATH" \
                --arg publicKey "$public_key" \
                --arg networkVolumeId "$RUNPOD_NETWORK_VOLUME_ID" \
                --arg dataCenterId "$RUNPOD_DATA_CENTER_ID" \
                --argjson gpuCount "$RUNPOD_GPU_COUNT" \
                --argjson containerDisk "$RUNPOD_CONTAINER_DISK_GB" \
                --argjson ports "$(ports_json)" \
                '{
                    name: $name,
                    imageName: $image,
                    gpuCount: $gpuCount,
                    gpuTypeIds: [$gpuType],
                    containerDiskInGb: $containerDisk,
                    volumeMountPath: $mountPath,
                    networkVolumeId: $networkVolumeId,
                    dataCenterIds: [$dataCenterId],
                    ports: $ports,
                    env: {
                        PUBLIC_KEY: $publicKey
                    }
                }'
        )"
    else
        payload="$(
            jq -nc \
                --arg name "$pod_name" \
                --arg image "$RUNPOD_IMAGE" \
                --arg gpuType "$RUNPOD_GPU_TYPE" \
                --arg mountPath "$RUNPOD_VOLUME_MOUNT_PATH" \
                --arg publicKey "$public_key" \
                --arg dataCenterId "$RUNPOD_DATA_CENTER_ID" \
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
                    dataCenterIds: (if $dataCenterId == "" then [] else [$dataCenterId] end),
                    env: {
                        PUBLIC_KEY: $publicKey
                    }
                }'
        )"
    fi
    api POST "/pods" "$payload" | jq -r '{id, name, desiredStatus, publicIp, portMappings}'
}

create_volume() {
    local volume_name="${1:-$RUNPOD_VOLUME_NAME}"
    if [[ -z "$RUNPOD_DATA_CENTER_ID" ]]; then
        echo "RUNPOD_DATA_CENTER_ID is required to create a network volume" >&2
        exit 1
    fi
    if (( RUNPOD_VOLUME_GB >= 50 )); then
        echo "Network volumes must stay under 50GB; got RUNPOD_VOLUME_GB=$RUNPOD_VOLUME_GB" >&2
        exit 1
    fi
    api POST "/networkvolumes" "$(
        jq -nc \
            --arg name "$volume_name" \
            --arg dataCenterId "$RUNPOD_DATA_CENTER_ID" \
            --argjson size "$RUNPOD_VOLUME_GB" \
            '{
                name: $name,
                dataCenterId: $dataCenterId,
                size: $size
            }'
    )" | jq -r '{id, name, dataCenterId, size}'
}

case "${1:-}" in
    list)
        api GET "/pods" | jq -r '.[] | [.id, .name, .desiredStatus, (.publicIp // "-"), (.portMappings["22"] // "-")] | @tsv'
        ;;
    stock)
        [[ $# -eq 2 ]] || { usage; exit 1; }
        graphql "query { gpuTypes(input: { id: \"$2\" }) { id displayName lowestPrice(input: { gpuCount: $RUNPOD_GPU_COUNT, dataCenterId: \"$RUNPOD_DATA_CENTER_ID\", secureCloud: true, minDisk: $RUNPOD_CONTAINER_DISK_GB, minMemoryInGb: 8, minVcpuCount: 2, globalNetwork: false }) { uninterruptablePrice stockStatus maxUnreservedGpuCount availableGpuCounts } } }" | jq
        ;;
    volumes)
        api GET "/networkvolumes" | jq -r '.[] | [.id, .name, .dataCenterId, .size] | @tsv'
        ;;
    volume-create)
        [[ $# -le 2 ]] || { usage; exit 1; }
        create_volume "${2:-}"
        ;;
    volume-delete)
        [[ $# -eq 2 ]] || { usage; exit 1; }
        api DELETE "/networkvolumes/$2" >/dev/null
        echo "deleted_volume $2"
        ;;
    get)
        [[ $# -eq 2 ]] || { usage; exit 1; }
        pod_json "$2" | jq
        ;;
    create)
        create_pod "${2:-}"
        ;;
    wait)
        [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 1; }
        wait_for_ssh "$2" "${3:-300}"
        ;;
    start)
        [[ $# -eq 2 ]] || { usage; exit 1; }
        api POST "/pods/$2/start" | jq
        ;;
    stop)
        [[ $# -eq 2 ]] || { usage; exit 1; }
        api POST "/pods/$2/stop" | jq
        ;;
    terminate|delete)
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
        [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 1; }
        target_sha="$(resolve_target_sha "${3:-}")"
        ssh_run "$2" "bash -lc '
            set -euo pipefail
            if [[ ! -d \"$REMOTE_ROOT/.git\" ]]; then
                git clone https://github.com/$REPO_OWNER/$REPO_NAME.git \"$REMOTE_ROOT\"
            fi
            git -C \"$REMOTE_ROOT\" fetch origin main
            git -C \"$REMOTE_ROOT\" checkout --detach \"$target_sha\"
            cd \"$REMOTE_ROOT\"
            bash setup.sh
        '"
        ;;
    sync)
        [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 1; }
        target_sha="$(resolve_target_sha "${3:-}")"
        ssh_run "$2" "bash -lc '
            set -euo pipefail
            git -C \"$REMOTE_ROOT\" fetch origin main
            git -C \"$REMOTE_ROOT\" checkout --detach \"$target_sha\"
        '"
        ;;
    download)
        [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 1; }
        target_sha="$(resolve_target_sha "${3:-}")"
        ssh_run "$2" "bash -lc '
            set -euo pipefail
            git -C \"$REMOTE_ROOT\" fetch origin main
            git -C \"$REMOTE_ROOT\" checkout --detach \"$target_sha\"
            cd \"$REMOTE_ROOT\"
            bash down.sh
        '"
        ;;
    train)
        [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 1; }
        target_sha="$(resolve_target_sha "${3:-}")"
        ssh_run "$2" "bash -lc '
            set -euo pipefail
            git -C \"$REMOTE_ROOT\" fetch origin main
            git -C \"$REMOTE_ROOT\" checkout --detach \"$target_sha\"
            cd \"$REMOTE_ROOT\"
            bash run.sh
        '"
        ;;
    h100-prep)
        [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 1; }
        target_sha="$(resolve_target_sha "${3:-}")"
        ssh_run "$2" "bash -lc '
            set -euo pipefail
            if [[ ! -d \"$REMOTE_ROOT/.git\" ]]; then
                git clone https://github.com/$REPO_OWNER/$REPO_NAME.git \"$REMOTE_ROOT\"
            fi
            git -C \"$REMOTE_ROOT\" fetch origin main
            git -C \"$REMOTE_ROOT\" checkout --detach \"$target_sha\"
            cd \"$REMOTE_ROOT\"
            bash \"$H100_RECORD_DIR/setup_h100.sh\"
            PYTHON_BIN=\"$H100_VENV_DIR/bin/python\" TRAIN_SHARDS=\"$H100_TRAIN_SHARDS\" bash down.sh
        '"
        ;;
    h100-preflight)
        [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 1; }
        target_sha="$(resolve_target_sha "${3:-}")"
        ssh_run "$2" "bash -lc '
            set -euo pipefail
            if [[ ! -d \"$REMOTE_ROOT/.git\" ]]; then
                git clone https://github.com/$REPO_OWNER/$REPO_NAME.git \"$REMOTE_ROOT\"
            fi
            git -C \"$REMOTE_ROOT\" fetch origin main
            git -C \"$REMOTE_ROOT\" checkout --detach \"$target_sha\"
            cd \"$REMOTE_ROOT\"
            MODE=preflight \
            NPROC_PER_NODE=\"${NPROC_PER_NODE:-$RUNPOD_GPU_COUNT}\" \
            VENV_PYTHON=\"$H100_VENV_DIR/bin/python\" \
            bash \"$H100_RECORD_DIR/run_h100.sh\"
        '"
        ;;
    h100-formal)
        [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 1; }
        target_sha="$(resolve_target_sha "${3:-}")"
        ssh_run "$2" "bash -lc '
            set -euo pipefail
            if [[ ! -d \"$REMOTE_ROOT/.git\" ]]; then
                git clone https://github.com/$REPO_OWNER/$REPO_NAME.git \"$REMOTE_ROOT\"
            fi
            git -C \"$REMOTE_ROOT\" fetch origin main
            git -C \"$REMOTE_ROOT\" checkout --detach \"$target_sha\"
            cd \"$REMOTE_ROOT\"
            MODE=formal \
            NPROC_PER_NODE=\"${NPROC_PER_NODE:-$RUNPOD_GPU_COUNT}\" \
            VENV_PYTHON=\"$H100_VENV_DIR/bin/python\" \
            bash \"$H100_RECORD_DIR/run_h100.sh\"
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
