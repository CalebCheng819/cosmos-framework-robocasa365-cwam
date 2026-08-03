#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Download one manifest-bound EMA checkpoint object from H HDD2 to a hidden
# CPFS transaction stage on the Alibaba DSW host. Presigned URLs only flow over
# stdin and are never printed or placed in a remote process argument.

set -Eeuo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

if [[ $# -ne 8 ]]; then
    echo "usage: $0 MANIFEST EXPECTED_MANIFEST_SHA256 OBJECT_PATH LANE RCLONE_PREFIX CPFS_STAGE NVME_ROOT RECEIPT_ROOT" >&2
    exit 2
fi

readonly MANIFEST="$1"
readonly EXPECTED_MANIFEST_SHA256="$2"
readonly OBJECT_PATH="$3"
readonly LANE="$4"
readonly RCLONE_PREFIX="$5"
readonly CPFS_STAGE="$6"
readonly NVME_ROOT="$7"
readonly RECEIPT_ROOT="$8"
readonly SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly CONTRACT_HELPER="$SCRIPT_ROOT/cwam_ema_transfer_contract.py"
readonly H_ALIAS="${CWAM_H_SSH_ALIAS:-h-eailabagent}"
readonly DSW_ALIAS="${CWAM_DSW_SSH_ALIAS:-cwam-dsw974}"
readonly DSW_SSH_CONFIG="${CWAM_DSW_SSH_CONFIG:-/home/chengjuntao/.codex-tmp/cwam_dsw974_ssh_config}"
readonly ENDPOINT_ENV=RCLONE_CONFIG_EAILAB-HDD2_ENDPOINT

die() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ "$EXPECTED_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "invalid manifest SHA-256"
[[ "$LANE" =~ ^[0-7]$ ]] || die "lane must be 0..7"
readonly PORT="$((8060 + LANE))"
[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || die "manifest is missing or symlinked"
[[ -f "$CONTRACT_HELPER" && ! -L "$CONTRACT_HELPER" ]] || die "contract helper is missing"
[[ -f "$DSW_SSH_CONFIG" && ! -L "$DSW_SSH_CONFIG" ]] || die "DSW SSH config is missing"
[[ "$RCLONE_PREFIX" =~ ^eailab-hdd2:eailab/chengjuntao/[A-Za-z0-9._/-]+$ ]] ||
    die "RCLONE_PREFIX must stay below eailab-hdd2:eailab/chengjuntao/"
for remote_path in "$CPFS_STAGE" "$NVME_ROOT" "$RECEIPT_ROOT"; do
    [[ "$remote_path" =~ ^/[A-Za-z0-9._/-]+$ && "$remote_path" != *"//"* ]] ||
        die "unsafe remote path: $remote_path"
done
[[ "$CPFS_STAGE" == /mnt/workspace/cwam_cosmos3_robocasa365_eval/inputs/checkpoints/.S1-* ]] ||
    die "CPFS stage is outside the reserved hidden S1 checkpoint path"
[[ "$NVME_ROOT" == /tmp/cwam-e35s1-* ]] || die "NVMe root is outside the reserved S1 path"
[[ "$RECEIPT_ROOT" == /mnt/workspace/cwam_cosmos3_robocasa365_eval/control/checkpoint-transfers/S1-* ]] ||
    die "receipt root is outside the reserved S1 transfer path"

ITEM_TSV="$({
    python3 "$CONTRACT_HELPER" \
        --manifest "$MANIFEST" \
        --expected-sha256 "$EXPECTED_MANIFEST_SHA256" \
        --expected-remote-prefix "$RCLONE_PREFIX" \
        --object-path "$OBJECT_PATH" \
        --output-format tsv
} 2>/dev/null)" || die "transfer manifest item lookup failed"
readonly ITEM_TSV

IFS=$'\t' read -r ITEM_OBJECT ENCODING RAW_OBJECT RAW_SIZE RAW_SHA TRANSFER_SIZE TRANSFER_SHA EXTRA <<<"$ITEM_TSV"
[[ -z "${EXTRA:-}" && "$ITEM_OBJECT" == "$OBJECT_PATH" ]] || die "invalid item lookup result"
[[ "$RAW_SIZE" =~ ^[0-9]+$ && "$TRANSFER_SIZE" =~ ^[0-9]+$ ]] || die "invalid item sizes"
[[ "$RAW_SHA" =~ ^[0-9a-f]{64}$ && "$TRANSFER_SHA" =~ ^[0-9a-f]{64}$ ]] ||
    die "invalid item hashes"

case "$ITEM_OBJECT" in
    SHA256SUMS) readonly CPFS_FINAL="$CPFS_STAGE/H-SHA256SUMS" ;;
    serving-checkpoint-manifest.json)
        readonly CPFS_FINAL="$CPFS_STAGE/h-serving-checkpoint-manifest.json"
        ;;
    exporter-source.py) readonly CPFS_FINAL="$RECEIPT_ROOT/exporter-source.py" ;;
    *) readonly CPFS_FINAL="$CPFS_STAGE/$RAW_OBJECT" ;;
esac

readonly OBJECT_REMOTE="$RCLONE_PREFIX/$ITEM_OBJECT"
readonly TRANSFER_PARTIAL="$NVME_ROOT/objects/$ITEM_OBJECT.partial"
readonly RAW_PARTIAL="$NVME_ROOT/raw/$RAW_OBJECT.partial"
readonly CPFS_FINAL_PARENT="$(dirname -- "$CPFS_FINAL")"
readonly CPFS_PARTIAL="$CPFS_FINAL_PARENT/.$(basename -- "$CPFS_FINAL").partial-$LANE"
readonly RECEIPT="$RECEIPT_ROOT/${ITEM_OBJECT//\//_}.json"
readonly LOCK="/home/chengjuntao/.codex-tmp/cwam-e35s1-transfer-${EXPECTED_MANIFEST_SHA256:0:12}-${ITEM_OBJECT//\//_}.lock"

exec 9>"$LOCK"
flock -n 9 || die "another transfer process owns $OBJECT_PATH"

h() {
    timeout --foreground --signal=TERM --kill-after=15s 14400s \
        ssh -o BatchMode=yes -o ConnectTimeout=30 "$H_ALIAS" "$@"
}

dsw() {
    timeout --foreground --signal=TERM --kill-after=15s 14400s \
        ssh -F "$DSW_SSH_CONFIG" -o BatchMode=yes -o ConnectTimeout=30 "$DSW_ALIAS" "$@"
}

remote_size() {
    dsw "if test -f '$1'; then stat -c '%s' '$1'; else printf '0\\n'; fi"
}

write_receipt() {
    dsw "python3 - '$RECEIPT' '$ITEM_OBJECT' '$ENCODING' '$RAW_SIZE' '$RAW_SHA' '$TRANSFER_SIZE' '$TRANSFER_SHA' '$CPFS_FINAL' '$EXPECTED_MANIFEST_SHA256' <<'PY'
import datetime
import json
import os
import sys
import uuid

output, object_path, encoding, raw_size, raw_sha, transfer_size, transfer_sha, final, manifest_sha = sys.argv[1:]
value = {
    'schema_version': 1,
    'kind': 'cwam_ema_dcp_object_transfer_receipt',
    'status': 'passed',
    'completed_at': datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z'),
    'object_path': object_path,
    'encoding': encoding,
    'raw_size_bytes': int(raw_size),
    'raw_sha256': raw_sha,
    'transfer_size_bytes': int(transfer_size),
    'transfer_sha256': transfer_sha,
    'cpfs_path': final,
    'cpfs_mode': '0444',
    'transfer_manifest_sha256': manifest_sha,
}
payload = (json.dumps(value, indent=2, sort_keys=True) + '\\n').encode()
if os.path.exists(output):
    current = json.load(open(output, encoding='utf-8'))
    for key in value.keys() - {'completed_at'}:
        if current.get(key) != value[key]:
            raise SystemExit(f'existing receipt differs at {key}: {output}')
else:
    temporary = f'{output}.incomplete-{uuid.uuid4().hex}'
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, 'wb', closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, output)
    finally:
        os.close(descriptor)
PY"
}

echo "phase=preflight object=$ITEM_OBJECT lane=$LANE port=$PORT at=$(date -u +%FT%TZ)"
dsw "set -eu
mkdir -p '$CPFS_STAGE/model' '$NVME_ROOT/objects/model' '$NVME_ROOT/raw/model' '$RECEIPT_ROOT'
chmod 0755 '$CPFS_STAGE' '$CPFS_STAGE/model' '$RECEIPT_ROOT'
chmod 0700 '$NVME_ROOT' '$NVME_ROOT/objects' '$NVME_ROOT/objects/model' '$NVME_ROOT/raw' '$NVME_ROOT/raw/model'
test ! -L '$CPFS_STAGE'
test ! -L '$NVME_ROOT'
current=\$(if test -f '$TRANSFER_PARTIAL'; then stat -c '%s' '$TRANSFER_PARTIAL'; else printf '0'; fi)
test \"\$current\" -le '$TRANSFER_SIZE'
if test \"\$current\" = '$TRANSFER_SIZE' &&
    test \"\$(sha256sum '$TRANSFER_PARTIAL' | awk '{print \$1}')\" != '$TRANSFER_SHA'; then
  rm -f -- '$TRANSFER_PARTIAL'
fi"

if dsw "test -e '$CPFS_FINAL'"; then
    dsw "set -eu
test -f '$CPFS_FINAL'
test ! -L '$CPFS_FINAL'
test \"\$(stat -c '%s' '$CPFS_FINAL')\" = '$RAW_SIZE'
test \"\$(sha256sum '$CPFS_FINAL' | awk '{print \$1}')\" = '$RAW_SHA'
test \"\$(stat -c '%a' '$CPFS_FINAL')\" = 444"
    write_receipt
    echo "phase=complete_existing object=$ITEM_OBJECT bytes=$RAW_SIZE sha256=$RAW_SHA at=$(date -u +%FT%TZ)"
    exit 0
fi

attempt=0
while [[ "$(remote_size "$TRANSFER_PARTIAL")" -lt "$TRANSFER_SIZE" ]]; do
    attempt="$((attempt + 1))"
    [[ "$attempt" -le 20 ]] || die "download attempt budget exhausted for $ITEM_OBJECT"
    before="$(remote_size "$TRANSFER_PARTIAL")"
    echo "phase=download object=$ITEM_OBJECT attempt=$attempt before_bytes=$before at=$(date -u +%FT%TZ)"
    set +e
    h "env '$ENDPOINT_ENV=http://127.0.0.1:$PORT' rclone link --expire 4h '$OBJECT_REMOTE' --log-level ERROR" |
        python3 -c '
import json
import sys
from urllib.parse import urlsplit

port = int(sys.argv[1])
url = sys.stdin.read().strip()
parsed = urlsplit(url)
if not (
    parsed.scheme == "http"
    and parsed.hostname == "127.0.0.1"
    and parsed.port == port
    and "\\n" not in url
    and "\\r" not in url
):
    raise SystemExit("invalid presigned URL endpoint")
print("url = " + json.dumps(url))
' "$PORT" |
        dsw "umask 077
curl --silent --show-error -fL -C - --retry 5 --retry-delay 2 --connect-timeout 15 --speed-limit 1024 --speed-time 90 -o '$TRANSFER_PARTIAL' -K -"
    pipeline_status=("${PIPESTATUS[@]}")
    set -e
    after="$(remote_size "$TRANSFER_PARTIAL")"
    echo "phase=download_attempt_done object=$ITEM_OBJECT attempt=$attempt h_status=${pipeline_status[0]} validator_status=${pipeline_status[1]} dsw_status=${pipeline_status[2]} after_bytes=$after at=$(date -u +%FT%TZ)"
    [[ "$after" -le "$TRANSFER_SIZE" ]] || die "download exceeded expected size"
    if [[ "$after" == "$before" ]]; then
        sleep 3
    fi
done

echo "phase=transfer_verify object=$ITEM_OBJECT at=$(date -u +%FT%TZ)"
dsw "set -eu
test \"\$(stat -c '%s' '$TRANSFER_PARTIAL')\" = '$TRANSFER_SIZE'
test \"\$(sha256sum '$TRANSFER_PARTIAL' | awk '{print \$1}')\" = '$TRANSFER_SHA'"

if [[ "$ENCODING" == gzip-pigz-1-n ]]; then
    echo "phase=decompress object=$ITEM_OBJECT at=$(date -u +%FT%TZ)"
    dsw "set -eu
if test -e '$RAW_PARTIAL'; then
  test -f '$RAW_PARTIAL'
  test ! -L '$RAW_PARTIAL'
  if test \"\$(stat -c '%s' '$RAW_PARTIAL')\" != '$RAW_SIZE' ||
      test \"\$(sha256sum '$RAW_PARTIAL' | awk '{print \$1}')\" != '$RAW_SHA'; then
    rm -f -- '$RAW_PARTIAL'
  fi
fi
if test ! -e '$RAW_PARTIAL'; then
  pigz -dc '$TRANSFER_PARTIAL' > '$RAW_PARTIAL'
  sync -f '$RAW_PARTIAL'
fi
test \"\$(stat -c '%s' '$RAW_PARTIAL')\" = '$RAW_SIZE'
test \"\$(sha256sum '$RAW_PARTIAL' | awk '{print \$1}')\" = '$RAW_SHA'"
else
    [[ "$ENCODING" == identity ]] || die "unsupported encoding: $ENCODING"
    dsw "set -eu
test '$RAW_SIZE' = '$TRANSFER_SIZE'
test '$RAW_SHA' = '$TRANSFER_SHA'"
fi

readonly COPY_SOURCE="$([[ "$ENCODING" == identity ]] && printf '%s' "$TRANSFER_PARTIAL" || printf '%s' "$RAW_PARTIAL")"
echo "phase=cpfs_copy object=$ITEM_OBJECT destination=$CPFS_FINAL at=$(date -u +%FT%TZ)"
dsw "set -eu
mkdir -p '$CPFS_FINAL_PARENT'
if test -e '$CPFS_PARTIAL'; then
  test -f '$CPFS_PARTIAL'
  test \"\$(stat -c '%s' '$CPFS_PARTIAL')\" -le '$RAW_SIZE'
fi
rsync --partial --append-verify --no-whole-file --chmod=F600 '$COPY_SOURCE' '$CPFS_PARTIAL'
sync -f '$CPFS_PARTIAL'
test \"\$(stat -c '%s' '$CPFS_PARTIAL')\" = '$RAW_SIZE'
test \"\$(sha256sum '$CPFS_PARTIAL' | awk '{print \$1}')\" = '$RAW_SHA'
chmod 0444 '$CPFS_PARTIAL'
mv -T '$CPFS_PARTIAL' '$CPFS_FINAL'"

write_receipt

echo "phase=complete object=$ITEM_OBJECT raw_size=$RAW_SIZE raw_sha=$RAW_SHA at=$(date -u +%FT%TZ)"
