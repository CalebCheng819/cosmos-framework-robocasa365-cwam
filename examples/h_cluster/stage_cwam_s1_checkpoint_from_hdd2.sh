#!/usr/bin/env bash
# Stage the S1 EMA serving checkpoint from H object storage onto personal GPFS.

set -Eeuo pipefail
umask 022

: "${S1_SOURCE_REMOTE:=eailab-hdd2:eailab/chengjuntao/cwam-dlc-s1-transfer-20260803/ema-serving-dcp}"
: "${S1_TARGET_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/staging/cwam-serving-checkpoints/S1-R3-iter_000010000-serving-ema-bf16-v1}"

readonly STAGING_ROOT="${S1_TARGET_ROOT}.incomplete"
readonly COMPRESSED_ROOT="$STAGING_ROOT/compressed"
readonly MODEL_ROOT="$STAGING_ROOT/model"

expected_sizes=(
    3796677222
    3796677158
    3796675645
    3796675645
    3796675645
    3796675645
    3796675645
    3796675645
)

verify_complete() {
    local root="$1"
    local index
    [[ -f "$root/config.yaml" ]] || return 1
    [[ "$(stat -c %s "$root/config.yaml")" == 21049 ]] || return 1
    [[ -f "$root/model/.metadata" ]] || return 1
    [[ "$(stat -c %s "$root/model/.metadata")" == 1356571 ]] || return 1
    for index in "${!expected_sizes[@]}"; do
        [[ -f "$root/model/__${index}_0.distcp" ]] || return 1
        [[ "$(stat -c %s "$root/model/__${index}_0.distcp")" == "${expected_sizes[$index]}" ]] || return 1
    done
}

if [[ -e "$S1_TARGET_ROOT" ]]; then
    verify_complete "$S1_TARGET_ROOT" || {
        printf 'ERROR: existing target is incomplete: %s\n' "$S1_TARGET_ROOT" >&2
        exit 1
    }
    printf 'S1_CHECKPOINT_ALREADY_READY target=%s\n' "$S1_TARGET_ROOT"
    exit 0
fi

mkdir -p "$COMPRESSED_ROOT/model" "$MODEL_ROOT"
rclone copy "$S1_SOURCE_REMOTE" "$COMPRESSED_ROOT" \
    --include '/config.yaml' \
    --include '/serving-checkpoint-manifest.json' \
    --include '/model/.metadata' \
    --include '/model/*.distcp.gz' \
    --transfers 8 \
    --checkers 16 \
    --stats 30s \
    --stats-one-line

cp -p "$COMPRESSED_ROOT/config.yaml" "$STAGING_ROOT/config.yaml"
cp -p "$COMPRESSED_ROOT/serving-checkpoint-manifest.json" "$STAGING_ROOT/serving-checkpoint-manifest.json"
cp -p "$COMPRESSED_ROOT/model/.metadata" "$MODEL_ROOT/.metadata"

decompress_shard() {
    local index="$1"
    local expected_size
    case "$index" in
        0) expected_size=3796677222 ;;
        1) expected_size=3796677158 ;;
        2 | 3 | 4 | 5 | 6 | 7) expected_size=3796675645 ;;
        *) printf 'ERROR: invalid shard index: %s\n' "$index" >&2; return 1 ;;
    esac
    local source="$COMPRESSED_ROOT/model/__${index}_0.distcp.gz"
    local target="$MODEL_ROOT/__${index}_0.distcp"
    local temporary="${target}.partial"
    if [[ -f "$target" && "$(stat -c %s "$target")" == "$expected_size" ]]; then
        return 0
    fi
    rm -f "$temporary"
    gzip -dc -- "$source" >"$temporary"
    [[ "$(stat -c %s "$temporary")" == "$expected_size" ]] || {
        printf 'ERROR: shard %s size mismatch after decompression\n' "$index" >&2
        return 1
    }
    mv -f "$temporary" "$target"
}
export -f decompress_shard
export COMPRESSED_ROOT MODEL_ROOT
printf '%s\n' "${!expected_sizes[@]}" | xargs -r -n1 -P8 bash -c 'decompress_shard "$1"' _

verify_complete "$STAGING_ROOT" || {
    printf 'ERROR: staged S1 checkpoint failed metadata and size validation\n' >&2
    exit 1
}
rm -rf "$COMPRESSED_ROOT"
find "$STAGING_ROOT" -type f -exec chmod 0444 {} +
find "$STAGING_ROOT" -type d -exec chmod 0555 {} +
mv "$STAGING_ROOT" "$S1_TARGET_ROOT"
printf 'S1_CHECKPOINT_READY target=%s model_shards=8\n' "$S1_TARGET_ROOT"
