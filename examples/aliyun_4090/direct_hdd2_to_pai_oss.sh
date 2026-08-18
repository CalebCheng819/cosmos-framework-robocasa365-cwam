#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  direct_hdd2_to_pai_oss.sh preflight
  direct_hdd2_to_pai_oss.sh probe <source> <destination> <expected-bytes>
  direct_hdd2_to_pai_oss.sh copy <manifest.tsv> [parallelism]
  direct_hdd2_to_pai_oss.sh delete <destination>

The script reads one Alibaba Cloud STS credential JSON object from stdin. Copy
manifests contain three tab-separated fields per line:

  source-rclone-path  destination-rclone-path  expected-bytes

The destination remote is named pai-cwam and is restricted by the temporary
STS credential. Validation uses paths, file counts, and exact byte sizes only.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

MODE="$1"
shift

RCLONE="${RCLONE:-/usr/bin/rclone}"
H_RCLONE_CONFIG="${H_RCLONE_CONFIG:-/root/.config/rclone/rclone.conf}"
PAI_ENDPOINT="${PAI_ENDPOINT:-oss-cn-beijing.aliyuncs.com}"
PAI_PREFLIGHT_ROOT="${PAI_PREFLIGHT_ROOT:-pai-cwam:pjlab-bjpai-manip/chengjuntao}"

case "$MODE" in
  preflight)
    [[ $# -eq 0 ]] || { usage >&2; exit 2; }
    ;;
  probe)
    [[ $# -eq 3 ]] || { usage >&2; exit 2; }
    ;;
  copy)
    [[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }
    ;;
  delete)
    [[ $# -eq 1 ]] || { usage >&2; exit 2; }
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

[[ -x "$RCLONE" ]] || { echo "rclone is not executable: $RCLONE" >&2; exit 3; }
[[ -r "$H_RCLONE_CONFIG" ]] || { echo "H rclone config is not readable: $H_RCLONE_CONFIG" >&2; exit 3; }

umask 077
CREDENTIAL_JSON="$(mktemp /dev/shm/cwam-b2-b3-pai-sts.XXXXXX.json)"
RCLONE_CONFIG="$(mktemp /dev/shm/cwam-b2-b3-rclone.XXXXXX.conf)"
cleanup() {
  truncate -s 0 "$CREDENTIAL_JSON" "$RCLONE_CONFIG" 2>/dev/null || true
  unlink "$CREDENTIAL_JSON" 2>/dev/null || true
  unlink "$RCLONE_CONFIG" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
chmod 0600 "$CREDENTIAL_JSON" "$RCLONE_CONFIG"
cat >"$CREDENTIAL_JSON"

python3 - "$CREDENTIAL_JSON" "$H_RCLONE_CONFIG" "$RCLONE_CONFIG" "$PAI_ENDPOINT" <<'PY'
import configparser
import json
import os
import sys

credential_path, source_config_path, output_config_path, endpoint = sys.argv[1:]
with open(credential_path, encoding="utf-8") as handle:
    credential = json.load(handle)
required = ("AccessKeyId", "AccessKeySecret", "SecurityToken")
missing = [key for key in required if not credential.get(key)]
if missing:
    raise SystemExit(f"STS credential is missing fields: {missing}")

config = configparser.ConfigParser(interpolation=None)
with open(source_config_path, encoding="utf-8") as handle:
    config.read_file(handle)
if "pai-cwam" in config:
    raise SystemExit("temporary remote name pai-cwam already exists in the H config")
config["pai-cwam"] = {
    "type": "s3",
    "provider": "Alibaba",
    "access_key_id": credential["AccessKeyId"],
    "secret_access_key": credential["AccessKeySecret"],
    "session_token": credential["SecurityToken"],
    "endpoint": endpoint,
    "location_constraint": "cn-beijing",
    "acl": "private",
}
with open(output_config_path, "w", encoding="utf-8") as handle:
    config.write(handle)
os.chmod(output_config_path, 0o600)
PY
truncate -s 0 "$CREDENTIAL_JSON"

rclone_cmd() {
  "$RCLONE" --config "$RCLONE_CONFIG" "$@"
}

exact_size() {
  local path="$1"
  rclone_cmd size "$path" --json | python3 -c '
import json, sys
d = json.load(sys.stdin)
if int(d.get("count", -1)) != 1:
    raise SystemExit(f"expected one object, got {d}")
print(int(d["bytes"]))
'
}

run_copy() {
  local source="$1"
  local destination="$2"
  local expected_bytes="$3"
  local source_bytes destination_bytes

  source_bytes="$(exact_size "$source")"
  [[ "$source_bytes" == "$expected_bytes" ]] || {
    echo "source size mismatch: $source expected=$expected_bytes actual=$source_bytes" >&2
    return 11
  }
  if destination_bytes="$(exact_size "$destination" 2>/dev/null)"; then
    if [[ "$destination_bytes" == "$expected_bytes" ]]; then
      printf 'copy_skip_complete\t%s\t%s\n' "$destination" "$expected_bytes"
      return 0
    fi
    echo "refusing wrong-sized destination: $destination expected=$expected_bytes actual=$destination_bytes" >&2
    return 12
  fi

  rclone_cmd copyto "$source" "$destination" \
    --transfers 1 \
    --checkers 4 \
    --multi-thread-streams "${MULTI_THREAD_STREAMS:-8}" \
    --multi-thread-cutoff "${MULTI_THREAD_CUTOFF:-256M}" \
    --s3-upload-concurrency "${S3_UPLOAD_CONCURRENCY:-8}" \
    --s3-chunk-size "${S3_CHUNK_SIZE:-64M}" \
    --retries "${RCLONE_RETRIES:-12}" \
    --low-level-retries "${RCLONE_LOW_LEVEL_RETRIES:-30}" \
    --stats 60s \
    --stats-one-line

  destination_bytes="$(exact_size "$destination")"
  [[ "$destination_bytes" == "$expected_bytes" ]] || {
    echo "destination size mismatch: $destination expected=$expected_bytes actual=$destination_bytes" >&2
    return 13
  }
  printf 'copy_complete\t%s\t%s\n' "$destination" "$expected_bytes"
}

export RCLONE RCLONE_CONFIG MULTI_THREAD_STREAMS MULTI_THREAD_CUTOFF
export S3_UPLOAD_CONCURRENCY S3_CHUNK_SIZE RCLONE_RETRIES RCLONE_LOW_LEVEL_RETRIES
export -f rclone_cmd exact_size run_copy

case "$MODE" in
  preflight)
    rclone_cmd lsd "$PAI_PREFLIGHT_ROOT" --max-depth 1 >/dev/null
    printf 'pai_oss_connectivity=true\n'
    ;;
  probe)
    source="$1"
    destination="$2"
    expected_bytes="$3"
    [[ "$expected_bytes" =~ ^[1-9][0-9]*$ ]] || { echo "invalid expected bytes" >&2; exit 2; }
    started="$(date +%s)"
    rclone_cmd cat "$source" --count "$expected_bytes" --retries 3 --low-level-retries 5 |
      rclone_cmd rcat "$destination" --size "$expected_bytes" \
        --s3-upload-concurrency "${S3_UPLOAD_CONCURRENCY:-8}" \
        --s3-chunk-size "${S3_CHUNK_SIZE:-16M}" \
        --retries 3 --low-level-retries 5
    actual_bytes="$(exact_size "$destination")"
    [[ "$actual_bytes" == "$expected_bytes" ]] || {
      echo "probe destination size mismatch: expected=$expected_bytes actual=$actual_bytes" >&2
      exit 13
    }
    elapsed="$(( $(date +%s) - started ))"
    printf 'probe_complete\tbytes=%s\telapsed_seconds=%s\tdestination=%s\n' \
      "$actual_bytes" "$elapsed" "$destination"
    ;;
  copy)
    manifest="$1"
    parallelism="${2:-4}"
    [[ -s "$manifest" ]] || { echo "copy manifest is missing or empty: $manifest" >&2; exit 4; }
    [[ "$parallelism" =~ ^[1-9][0-9]*$ ]] || { echo "invalid parallelism: $parallelism" >&2; exit 2; }
    python3 - "$manifest" <<'PY' | xargs -0 -n 3 -P "$parallelism" bash -c 'run_copy "$1" "$2" "$3"' _
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    for line_number, raw in enumerate(handle, 1):
        line = raw.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 3 or not fields[2].isdigit() or int(fields[2]) <= 0:
            raise SystemExit(f"invalid manifest line {line_number}")
        for field in fields:
            sys.stdout.write(field + "\0")
PY
    ;;
  delete)
    rclone_cmd deletefile "$1"
    printf 'deleted\t%s\n' "$1"
    ;;
esac
