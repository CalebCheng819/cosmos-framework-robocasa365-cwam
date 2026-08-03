#!/usr/bin/env bash
# Publish the two relocation libraries used by the Cosmos environment wrapper.

set -Eeuo pipefail
umask 077

: "${RUNTIME_CLOSURE_HELPER:?Set the verified runtime closure helper path}"
: "${RUNTIME_SOURCE_STAGING:?Set the DSW node-local staging directory}"
: "${DLC_PROJECT_ROOT:?Set the canonical CPFS evaluation project root}"
: "${PUBLICATION_NONCE:?Set a unique publication nonce}"

readonly ARTIFACT_NAME=cosmos-r21-dlc-runtime-v1
readonly PRIVATE_GLIBC=cosmos-r21-private-glibc235-v1
readonly GCC_RUNTIME=cosmos-r21-gcc13-runtime-v3
readonly DESTINATION="$DLC_PROJECT_ROOT/inputs/runtime/$ARTIFACT_NAME"
readonly MANIFEST="$DLC_PROJECT_ROOT/inputs/config/$ARTIFACT_NAME-manifest.json"
readonly RECEIPT="$DLC_PROJECT_ROOT/control/receipts/$ARTIFACT_NAME-publication-v2.json"
readonly DESTINATION_STAGE="${DESTINATION}.incomplete-${PUBLICATION_NONCE}"
readonly MANIFEST_STAGE="${MANIFEST}.incomplete-${PUBLICATION_NONCE}"
readonly RECEIPT_STAGE="${RECEIPT}.incomplete-${PUBLICATION_NONCE}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ "$PUBLICATION_NONCE" =~ ^[A-Za-z0-9._-]+$ ]] || die "publication nonce is invalid"
[[ -f "$RUNTIME_CLOSURE_HELPER" && ! -L "$RUNTIME_CLOSURE_HELPER" ]] ||
    die "runtime closure helper must be a regular non-symlink file"
RUNTIME_SOURCE_STAGING="$(realpath -e -- "$RUNTIME_SOURCE_STAGING")"
DLC_PROJECT_ROOT="$(realpath -e -- "$DLC_PROJECT_ROOT")"
[[ "$DLC_PROJECT_ROOT" == /mnt/workspace/cwam_cosmos3_robocasa365_eval ]] ||
    die "DLC_PROJECT_ROOT differs from the audited CPFS path"
for name in "$PRIVATE_GLIBC" "$GCC_RUNTIME"; do
    [[ -d "$RUNTIME_SOURCE_STAGING/$name" && ! -L "$RUNTIME_SOURCE_STAGING/$name" ]] ||
        die "missing source runtime directory: $name"
done
for path in \
    "$DESTINATION" \
    "$MANIFEST" \
    "$RECEIPT" \
    "$DESTINATION_STAGE" \
    "$MANIFEST_STAGE" \
    "$RECEIPT_STAGE"; do
    [[ ! -e "$path" && ! -L "$path" ]] || die "refusing existing publication path: $path"
done

readonly LOCAL_ROOT="$(mktemp --directory /tmp/cwam-dlc-runtime-publication.XXXXXX)"
cleanup() {
    [[ "$LOCAL_ROOT" == /tmp/cwam-dlc-runtime-publication.* ]] ||
        die "refusing to clean unexpected local path: $LOCAL_ROOT"
    rm --one-file-system -rf -- "$LOCAL_ROOT"
}
trap cleanup EXIT

readonly SOURCE_COPY="$LOCAL_ROOT/source"
readonly MANIFEST_FIRST="$LOCAL_ROOT/manifest-first.json"
readonly MANIFEST_SECOND="$LOCAL_ROOT/manifest-second.json"
mkdir --mode=0700 "$SOURCE_COPY"
cp --archive "$RUNTIME_SOURCE_STAGING/$PRIVATE_GLIBC" "$SOURCE_COPY/"
cp --archive "$RUNTIME_SOURCE_STAGING/$GCC_RUNTIME" "$SOURCE_COPY/"

python3 "$RUNTIME_CLOSURE_HELPER" build \
    --root "$SOURCE_COPY" \
    --artifact-name "$ARTIFACT_NAME" \
    --output "$MANIFEST_FIRST" >/dev/null
sleep 5
python3 "$RUNTIME_CLOSURE_HELPER" build \
    --root "$SOURCE_COPY" \
    --artifact-name "$ARTIFACT_NAME" \
    --output "$MANIFEST_SECOND" >/dev/null
cmp --silent "$MANIFEST_FIRST" "$MANIFEST_SECOND" ||
    die "runtime closure changed between two complete scans"

readonly MANIFEST_SHA256="$(sha256sum "$MANIFEST_FIRST" | cut -d ' ' -f1)"
readonly TREE_SHA256="$(
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tree_sha256"])' \
        "$MANIFEST_FIRST"
)"

mkdir --parents --mode=0755 \
    "$(dirname -- "$DESTINATION")" \
    "$(dirname -- "$MANIFEST")" \
    "$(dirname -- "$RECEIPT")"
mkdir --mode=0700 "$DESTINATION_STAGE"
cp --archive "$SOURCE_COPY/." "$DESTINATION_STAGE/"
python3 "$RUNTIME_CLOSURE_HELPER" verify \
    --root "$DESTINATION_STAGE" \
    --manifest "$MANIFEST_FIRST" \
    --expected-manifest-sha256 "$MANIFEST_SHA256" \
    --expected-tree-sha256 "$TREE_SHA256" \
    --phase publication >/dev/null
sync -f -- "$DESTINATION_STAGE"
mv -T -- "$DESTINATION_STAGE" "$DESTINATION"

cp --preserve=mode,timestamps -- "$MANIFEST_FIRST" "$MANIFEST_STAGE"
chmod 0444 "$MANIFEST_STAGE"
mv -T -- "$MANIFEST_STAGE" "$MANIFEST"
sync -f -- "$MANIFEST"
# A newly renamed CPFS inode can briefly return different pathname/open
# metadata. Every retry still performs a complete byte-level tree verification.
publication_verified=0
for ((attempt = 1; attempt <= 10; attempt++)); do
    if python3 "$RUNTIME_CLOSURE_HELPER" verify \
        --root "$DESTINATION" \
        --manifest "$MANIFEST" \
        --expected-manifest-sha256 "$MANIFEST_SHA256" \
        --expected-tree-sha256 "$TREE_SHA256" \
        --phase publication \
        --output "$RECEIPT_STAGE" >/dev/null; then
        publication_verified=1
        break
    fi
    [[ ! -e "$RECEIPT_STAGE" && ! -L "$RECEIPT_STAGE" ]] ||
        die "failed verification unexpectedly created a receipt"
    sleep "$attempt"
done
[[ "$publication_verified" == 1 ]] || die "published runtime closure did not stabilize"
mv -T -- "$RECEIPT_STAGE" "$RECEIPT"

readonly RECEIPT_SHA256="$(sha256sum "$RECEIPT" | cut -d ' ' -f1)"
python3 - "$MANIFEST" "$RECEIPT" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
receipt = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
print("RUNTIME_CLOSURE_PUBLISHED")
print(f"manifest_sha256={receipt['manifest_sha256']}")
print(f"tree_sha256={receipt['tree_sha256']}")
print(f"entry_count={manifest['entry_count']}")
print(f"regular_file_count={manifest['regular_file_count']}")
print(f"symlink_count={manifest['symlink_count']}")
print(f"total_regular_file_bytes={manifest['total_regular_file_bytes']}")
PY
printf 'root=%s\nmanifest=%s\nreceipt=%s\nreceipt_sha256=%s\n' \
    "$DESTINATION" "$MANIFEST" "$RECEIPT" "$RECEIPT_SHA256"
