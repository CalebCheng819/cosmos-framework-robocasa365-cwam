#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 ARTIFACT SOURCE_FULL_RECEIPT STAGE RCLONE_PREFIX WORKERS" >&2
  exit 2
fi

ARTIFACT="$(realpath "$1")"
SOURCE_FULL_RECEIPT="$(realpath "$2")"
STAGE="$(realpath "$3")"
RCLONE_PREFIX="$4"
WORKERS="$5"

if [[ "$WORKERS" != "1" && "$WORKERS" != "2" ]]; then
  echo "WORKERS must be 1 or 2" >&2
  exit 2
fi
if [[ ! -d "$ARTIFACT/model" || ! -f "$SOURCE_FULL_RECEIPT" || ! -d "$STAGE/model" ]]; then
  echo "artifact, source receipt, or transfer staging path is missing" >&2
  exit 2
fi

upload_identity() {
  local source_path="$1"
  local relative_path="$2"
  local staged_path="$STAGE/$relative_path"
  local remote_path="$RCLONE_PREFIX/$relative_path"
  mkdir -p "$(dirname "$staged_path")"
  if [[ ! -e "$staged_path" ]]; then
    cp --reflink=auto --preserve=mode,timestamps "$source_path" "$staged_path"
  fi
  local source_size source_sha staged_size staged_sha
  source_size="$(stat -c '%s' "$source_path")"
  source_sha="$(sha256sum "$source_path" | awk '{print $1}')"
  staged_size="$(stat -c '%s' "$staged_path")"
  staged_sha="$(sha256sum "$staged_path" | awk '{print $1}')"
  if [[ "$source_size" != "$staged_size" || "$source_sha" != "$staged_sha" ]]; then
    echo "identity staging mismatch for $relative_path" >&2
    return 1
  fi
  rclone copyto "$staged_path" "$remote_path" --transfers 1 --checkers 1 --retries 5 --low-level-retries 10
  local remote_size
  remote_size="$(rclone lsl "$remote_path" | awk 'NR == 1 {print $1}')"
  if [[ "$remote_size" != "$staged_size" ]]; then
    echo "remote size mismatch for $relative_path: local=$staged_size remote=$remote_size" >&2
    return 1
  fi
  printf '%s  %s\n' "$staged_sha" "$(basename "$relative_path")" > "$staged_path.sha256"
  rclone copyto "$staged_path.sha256" "$remote_path.sha256" --transfers 1 --checkers 1 --retries 5
  printf '{"event":"object_uploaded","encoding":"identity","object":"%s","raw_size":%s,"transfer_size":%s,"raw_sha256":"%s","transfer_sha256":"%s"}\n' \
    "$relative_path" "$source_size" "$staged_size" "$source_sha" "$staged_sha"
}

compress_upload() {
  local shard="$1"
  local source_path="$ARTIFACT/model/$shard"
  local relative_path="model/$shard.gz"
  local staged_path="$STAGE/$relative_path"
  local remote_path="$RCLONE_PREFIX/$relative_path"
  local temporary="$STAGE/model/.$shard.gz.incomplete-$$"
  local raw_size raw_sha transfer_size transfer_sha
  raw_size="$(stat -c '%s' "$source_path")"
  raw_sha="$(sha256sum "$source_path" | awk '{print $1}')"
  if [[ ! -e "$staged_path" ]]; then
    if [[ -e "$temporary" ]]; then
      echo "temporary output already exists: $temporary" >&2
      return 1
    fi
    pigz -1 -n -p 1 -c "$source_path" > "$temporary"
    sync -f "$temporary"
    mv "$temporary" "$staged_path"
  fi
  transfer_size="$(stat -c '%s' "$staged_path")"
  transfer_sha="$(sha256sum "$staged_path" | awk '{print $1}')"
  printf '%s  %s\n' "$transfer_sha" "$(basename "$relative_path")" > "$staged_path.sha256"
  rclone copyto "$staged_path" "$remote_path" --transfers 1 --checkers 1 --retries 5 --low-level-retries 10
  local remote_size
  remote_size="$(rclone lsl "$remote_path" | awk 'NR == 1 {print $1}')"
  if [[ "$remote_size" != "$transfer_size" ]]; then
    echo "remote size mismatch for $relative_path: local=$transfer_size remote=$remote_size" >&2
    return 1
  fi
  rclone copyto "$staged_path.sha256" "$remote_path.sha256" --transfers 1 --checkers 1 --retries 5
  printf '{"event":"object_uploaded","encoding":"gzip-pigz-1-n","object":"%s","raw_object":"model/%s","raw_size":%s,"transfer_size":%s,"raw_sha256":"%s","transfer_sha256":"%s"}\n' \
    "$relative_path" "$shard" "$raw_size" "$transfer_size" "$raw_sha" "$transfer_sha"
}

SMALL_FILES=(
  "SHA256SUMS"
  "config.yaml"
  "ema-bf16-export-receipt.json"
  "ema-bf16-key-manifest.json"
  "ema-bf16-validation.json"
  "exporter-source.py"
  "serving-checkpoint-manifest.json"
  "model/.metadata"
)
for relative_path in "${SMALL_FILES[@]}"; do
  upload_identity "$ARTIFACT/$relative_path" "$relative_path"
done
upload_identity "$SOURCE_FULL_RECEIPT" "source-dcp-full-verification.json"

worker() {
  local worker_index="$1"
  local shard_index
  for ((shard_index=worker_index; shard_index<8; shard_index+=WORKERS)); do
    compress_upload "__${shard_index}_0.distcp"
  done
}

pids=()
for ((worker_index=0; worker_index<WORKERS; worker_index++)); do
  worker "$worker_index" > "$STAGE/upload-worker-$worker_index.jsonl" 2> "$STAGE/upload-worker-$worker_index.stderr" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" != "0" ]]; then
  echo "one or more upload workers failed" >&2
  exit "$status"
fi

echo '{"event":"all_objects_uploaded","status":"succeeded"}'
