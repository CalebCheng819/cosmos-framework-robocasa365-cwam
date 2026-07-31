#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SHARD_INDEX TUNNEL_PORT" >&2
  exit 2
fi

SHARD_INDEX="$1"
TUNNEL_PORT="$2"
if [[ ! "$SHARD_INDEX" =~ ^[0-7]$ ]]; then
  echo "SHARD_INDEX must be 0..7" >&2
  exit 2
fi
EXPECTED_PORT="$((8060 + SHARD_INDEX))"
if [[ "$TUNNEL_PORT" != "$EXPECTED_PORT" ]]; then
  echo "TUNNEL_PORT must be $EXPECTED_PORT for shard $SHARD_INDEX" >&2
  exit 2
fi

TASK_CONFIG="/home/chengjuntao/.codex-tmp/cwam_dsw974_ssh_config"
H_ALIAS="h-eailabagent"
DSW_ALIAS="cwam-dsw974"
CEPH_PREFIX="h-ceph:chengjuntao/cwam-dsw974-transfer-20260731/ema-serving-dcp/model"
NVME_ROOT="/tmp/cwam-dsw974-transfer/ema-serving-dcp-gzip"
CPFS_MODEL="/mnt/workspace/cwam_cosmos3_robocasa365_eval/inputs/checkpoints/.R14-iter_000010000-serving-ema-bf16-v1.incomplete-transfer-20260731/model"
SHARD="__${SHARD_INDEX}_0.distcp"
GZIP_OBJECT="${SHARD}.gz"
GZIP_PARTIAL="${NVME_ROOT}/${GZIP_OBJECT}.partial"
RAW_PARTIAL="${NVME_ROOT}/raw/${SHARD}.partial-20260731-v1"
CPFS_PARTIAL="${CPFS_MODEL}/.${SHARD}.partial-transfer-20260731"
CPFS_FINAL="${CPFS_MODEL}/${SHARD}"
RECEIPT="${NVME_ROOT}/receipts/${SHARD}.json"
LOCK="/home/chengjuntao/.codex-tmp/cwam-dsw974-shard-${SHARD_INDEX}.lock"

RAW_SIZES=(
  3794555281 3794555281 3794555281 3794555281
  3794555281 3794555281 3794555281 3794555281
)
RAW_SHAS=(
  3725441fc882ba66867347055529441ecc0288b0fa006f49544036adf97d6667
  eeee9e22e47b1b6b084e955dd4f2fdb8476431fb6a52f275fd8a09c15d3b86c4
  7b4a66e6f7b4ea6ccac73648bc3746471641ed96eb32fc86be60b9c57c4058ac
  b4ebc4f7ea45d42749452203de7df42cfa6777da8c9f9e2130d2c31de6c27ee5
  399d1cfd38eaa231df77a13e83ae972d44c50086af15d8b3247855754ae6e3ef
  0651e25684f14f93ceded471f142d797508d041604588069cc697d173815a9c3
  88e6ccc90b7052883c2a1fb83a432411c60d3bfdf86514fe9aa040ccb9a7d988
  87faa9874fd8cd0cc1f1db7541ea6ca29747b358d189c4cf8700c5057c75b622
)
GZIP_SIZES=(
  3057124317 3057212765 3057316072 3057260139
  3057271705 3057547695 3055974746 3052742447
)
GZIP_SHAS=(
  4687cdfe7c3d83e0b09a7b2746eef4db3e3fee64be0e66781dbee782d1c528b3
  b0079da56e8d995c2382f4c8fc719e1fc66cda8147414d793afce24f5d2f0ba1
  a59dc8f668daa36c3bd09ae09ba36715194b98780ced88709c5e57104f2cf82b
  cc5e4f95fc13fa033ef50663dfc22358ce3a590decef0fff6a9fcbabce2bac2a
  7ed2e3e66791eae34c2053805a425e88f3eda5f2e768b79fcf59ff4b18303f63
  145d28d556a6887089e8ac3e097106e1ba22a48b3d8d1f499e279099f09f6c93
  a86fb74500d7831858dfb141f4546ebb974dbb52ccc698f40cd2851204c4a9d5
  91c307bc85c4674296fa2c5bcb71a4ee69f1e3ea36be5102d3bd02b18b8bd00b
)
RAW_SIZE="${RAW_SIZES[$SHARD_INDEX]}"
RAW_SHA="${RAW_SHAS[$SHARD_INDEX]}"
GZIP_SIZE="${GZIP_SIZES[$SHARD_INDEX]}"
GZIP_SHA="${GZIP_SHAS[$SHARD_INDEX]}"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another driver already owns shard $SHARD_INDEX" >&2
  exit 3
fi

dsw() {
  ssh -F "$TASK_CONFIG" -o BatchMode=yes "$DSW_ALIAS" "$@"
}

remote_size() {
  dsw "if test -f '$1'; then stat -c '%s' '$1'; else printf '0\\n'; fi"
}

echo "phase=preflight shard=$SHARD port=$TUNNEL_PORT at=$(date -Is)"
dsw "set -eu
mkdir -p '$NVME_ROOT/raw' '$NVME_ROOT/receipts' '$CPFS_MODEL'
test -f '$CPFS_MODEL/.metadata'
test \"\$(stat -c '%a' '$CPFS_MODEL/.metadata')\" = 444
current=\$(if test -f '$GZIP_PARTIAL'; then stat -c '%s' '$GZIP_PARTIAL'; else printf '0'; fi)
test \"\$current\" -le '$GZIP_SIZE'
if test -e '$CPFS_FINAL'; then
  test -f '$CPFS_FINAL'
fi"

attempt=0
while [[ "$(remote_size "$GZIP_PARTIAL")" -lt "$GZIP_SIZE" ]]; do
  attempt="$((attempt + 1))"
  if [[ "$attempt" -gt 12 ]]; then
    echo "download attempts exhausted for $SHARD" >&2
    exit 4
  fi
  before="$(remote_size "$GZIP_PARTIAL")"
  echo "phase=download shard=$SHARD attempt=$attempt before_bytes=$before at=$(date -Is)"
  set +e
  ssh -o BatchMode=yes "$H_ALIAS" \
    "env 'RCLONE_CONFIG_H-CEPH_ENDPOINT=http://127.0.0.1:${TUNNEL_PORT}' \
      rclone link --expire 2h '${CEPH_PREFIX}/${GZIP_OBJECT}' --log-level ERROR" |
    ssh -F "$TASK_CONFIG" -o BatchMode=yes "$DSW_ALIAS" \
      "umask 077
       awk '{print \"url = \\\"\" \$0 \"\\\"\"}' |
       curl --silent --show-error -fL -C - --retry 5 --retry-delay 2 \
         --connect-timeout 15 --speed-limit 1024 --speed-time 90 \
         -o '$GZIP_PARTIAL' -K -"
  transfer_status="$?"
  set -e
  after="$(remote_size "$GZIP_PARTIAL")"
  echo "phase=download_attempt_done shard=$SHARD attempt=$attempt status=$transfer_status after_bytes=$after at=$(date -Is)"
  if [[ "$after" -gt "$GZIP_SIZE" ]]; then
    echo "download exceeded expected size for $SHARD: $after > $GZIP_SIZE" >&2
    exit 5
  fi
  if [[ "$transfer_status" != 0 && "$after" -eq "$before" ]]; then
    sleep 3
  fi
done

echo "phase=gzip_verify shard=$SHARD at=$(date -Is)"
dsw "set -eu
test \"\$(stat -c '%s' '$GZIP_PARTIAL')\" = '$GZIP_SIZE'
actual=\$(sha256sum '$GZIP_PARTIAL' | awk '{print \$1}')
test \"\$actual\" = '$GZIP_SHA'"

echo "phase=decompress shard=$SHARD at=$(date -Is)"
dsw "set -eu
if test -e '$RAW_PARTIAL'; then
  test -f '$RAW_PARTIAL'
  test \"\$(stat -c '%s' '$RAW_PARTIAL')\" = '$RAW_SIZE'
  actual=\$(sha256sum '$RAW_PARTIAL' | awk '{print \$1}')
  test \"\$actual\" = '$RAW_SHA'
else
  umask 077
  pigz -dc '$GZIP_PARTIAL' > '$RAW_PARTIAL'
  sync -f '$RAW_PARTIAL'
fi
test \"\$(stat -c '%s' '$RAW_PARTIAL')\" = '$RAW_SIZE'
actual=\$(sha256sum '$RAW_PARTIAL' | awk '{print \$1}')
test \"\$actual\" = '$RAW_SHA'"

echo "phase=cpfs_copy shard=$SHARD at=$(date -Is)"
dsw "set -eu
if test -e '$CPFS_FINAL'; then
  test -f '$CPFS_FINAL'
  test \"\$(stat -c '%s' '$CPFS_FINAL')\" = '$RAW_SIZE'
  actual=\$(sha256sum '$CPFS_FINAL' | awk '{print \$1}')
  test \"\$actual\" = '$RAW_SHA'
  test \"\$(stat -c '%a' '$CPFS_FINAL')\" = 444
else
  if test -e '$CPFS_PARTIAL'; then
    test -f '$CPFS_PARTIAL'
    test \"\$(stat -c '%s' '$CPFS_PARTIAL')\" -le '$RAW_SIZE'
  fi
  rsync --partial --append-verify --no-whole-file --chmod=F600 '$RAW_PARTIAL' '$CPFS_PARTIAL'
  sync -f '$CPFS_PARTIAL'
  test \"\$(stat -c '%s' '$CPFS_PARTIAL')\" = '$RAW_SIZE'
  actual=\$(sha256sum '$CPFS_PARTIAL' | awk '{print \$1}')
  test \"\$actual\" = '$RAW_SHA'
  chmod 0444 '$CPFS_PARTIAL'
  mv -T '$CPFS_PARTIAL' '$CPFS_FINAL'
fi"

echo "phase=receipt shard=$SHARD at=$(date -Is)"
dsw "python3 - '$RECEIPT' '$SHARD' '$TUNNEL_PORT' '$GZIP_PARTIAL' '$GZIP_SIZE' '$GZIP_SHA' \
  '$RAW_PARTIAL' '$RAW_SIZE' '$RAW_SHA' '$CPFS_FINAL' <<'PY'
import datetime
import json
import os
import sys
import uuid

(
    receipt_path,
    shard,
    tunnel_port,
    gzip_path,
    gzip_size,
    gzip_sha,
    raw_path,
    raw_size,
    raw_sha,
    cpfs_path,
) = sys.argv[1:]
value = {
    'schema_version': 1,
    'kind': 'cwam_ema_dcp_shard_transfer_receipt',
    'status': 'passed',
    'completed_at': datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z'),
    'shard': shard,
    'tunnel_port': int(tunnel_port),
    'gzip': {
        'path': gzip_path,
        'size_bytes': int(gzip_size),
        'sha256': gzip_sha,
        'verified': True,
        'retained': True,
    },
    'raw_nvme': {
        'path': raw_path,
        'size_bytes': int(raw_size),
        'sha256': raw_sha,
        'verified': True,
        'retained': True,
    },
    'cpfs': {
        'path': cpfs_path,
        'size_bytes': int(raw_size),
        'sha256': raw_sha,
        'verified_after_copy': True,
        'atomic_rename': True,
        'mode': '0444',
    },
}
payload = (json.dumps(value, indent=2, sort_keys=True) + '\\n').encode()
if os.path.exists(receipt_path):
    current = json.load(open(receipt_path, encoding='utf-8'))
    for key in ('schema_version', 'kind', 'status', 'shard', 'tunnel_port', 'gzip', 'raw_nvme', 'cpfs'):
        if current.get(key) != value[key]:
            raise SystemExit(f'receipt already exists with a different {key}: {receipt_path}')
else:
    temporary = f'{receipt_path}.incomplete-{uuid.uuid4().hex}'
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, 'wb', closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, receipt_path)
    finally:
        os.close(descriptor)
PY"

echo "phase=complete shard=$SHARD raw_size=$RAW_SIZE raw_sha=$RAW_SHA at=$(date -Is)"
