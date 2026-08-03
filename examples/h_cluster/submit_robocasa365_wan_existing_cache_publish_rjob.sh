#!/usr/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -Eeuo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

readonly PINNED_IMAGE="registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
readonly QUOTA_GROUP="eailabagent_gpu"
readonly RECOVERY_NODE="node/gpu-l-lg-cmc-h-h200-0298.host.h.pjlab.org.cn"
readonly PERSONAL_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao"
readonly OUTPUT_ROOT="$PERSONAL_ROOT/outputs/robocasa365-wan-fullpad-oss"
readonly SNAPSHOT_ROOT="$PERSONAL_ROOT/run_snapshots"
readonly PERSONAL_MOUNT="gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs"
readonly PUBLIC_MOUNT="gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public"
readonly RJOB_BIN="/usr/local/bin/rjob"
readonly PYTHON3_BIN="/usr/bin/python3"
readonly SHA256SUM_BIN="/usr/bin/sha256sum"
readonly REALPATH_BIN="/usr/bin/realpath"
readonly BASH_BIN="/usr/bin/bash"

: "${COSMOS_SOURCE:?Set COSMOS_SOURCE to the immutable recovery snapshot.}"
: "${PUBLISH_RUN_ID:?Set PUBLISH_RUN_ID.}"
: "${SOURCE_LATENT_CACHE_RUN_ID:?Set SOURCE_LATENT_CACHE_RUN_ID.}"
: "${RJOB_NAME:?Set RJOB_NAME.}"
: "${ROBOCASA365_OSS_REMOTE:?Set the exact guarded HDD2 target prefix.}"
: "${EXPECTED_SOURCE_COMMIT:?Set EXPECTED_SOURCE_COMMIT.}"
: "${EXPECTED_CACHE_SOURCE_COMMIT:?Set EXPECTED_CACHE_SOURCE_COMMIT.}"
: "${EXPECTED_SNAPSHOT_MANIFEST_SHA256:?Set EXPECTED_SNAPSHOT_MANIFEST_SHA256.}"
: "${EXPECTED_CACHE_ID:?Set EXPECTED_CACHE_ID.}"
: "${EXPECTED_CACHE_MANIFEST_SHA256:?Set EXPECTED_CACHE_MANIFEST_SHA256.}"
: "${RCLONE_BIN:?Set the pinned GPFS rclone binary.}"
: "${EXPECTED_RCLONE_SHA256:?Set EXPECTED_RCLONE_SHA256.}"
: "${RJOB_PREDICT_ONLY:=1}"
: "${COSMOS_ENV:=$PERSONAL_ROOT/envs/cosmos-framework-09f23119-cu128-train}"

[[ "$#" -eq 0 ]] || { echo "ERROR: command-line arguments are forbidden" >&2; exit 1; }
[[ "$RJOB_PREDICT_ONLY" =~ ^[01]$ ]] || { echo "ERROR: RJOB_PREDICT_ONLY must be 0 or 1" >&2; exit 1; }
for run_id in "$PUBLISH_RUN_ID" "$SOURCE_LATENT_CACHE_RUN_ID"; do
  [[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: one run ID is invalid" >&2; exit 1; }
done
[[ "$PUBLISH_RUN_ID" != "$SOURCE_LATENT_CACHE_RUN_ID" ]] || { echo "ERROR: publish run ID must be fresh" >&2; exit 1; }
[[ "$RJOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ && "${#RJOB_NAME}" -le 50 ]] ||
  { echo "ERROR: invalid rjob name" >&2; exit 1; }
for commit in "$EXPECTED_SOURCE_COMMIT" "$EXPECTED_CACHE_SOURCE_COMMIT"; do
  [[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: one source commit is invalid" >&2; exit 1; }
done
for digest in \
  "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" \
  "$EXPECTED_CACHE_MANIFEST_SHA256" \
  "$EXPECTED_RCLONE_SHA256"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || { echo "ERROR: one required SHA-256 is invalid" >&2; exit 1; }
done
[[ "$EXPECTED_CACHE_ID" =~ ^robocasa365-wan-[0-9a-f]{24}$ ]] || { echo "ERROR: invalid cache ID" >&2; exit 1; }
[[ "$ROBOCASA365_OSS_REMOTE" == eailab-hdd2:eailab/chengjuntao/* ]] ||
  { echo "ERROR: OSS target must stay below eailab/chengjuntao" >&2; exit 1; }

for tool in "$RJOB_BIN" "$PYTHON3_BIN" "$SHA256SUM_BIN" "$REALPATH_BIN" "$BASH_BIN"; do
  [[ -x "$tool" ]] || { echo "ERROR: required tool is unavailable: $tool" >&2; exit 1; }
done

sha256_file() {
  local digest remainder
  read -r digest remainder < <("$SHA256SUM_BIN" "$1")
  printf '%s\n' "$digest"
}

COSMOS_SOURCE_REAL="$("$REALPATH_BIN" -e -- "$COSMOS_SOURCE")"
COSMOS_ENV_REAL="$("$REALPATH_BIN" -e -- "$COSMOS_ENV")"
RCLONE_REAL="$("$REALPATH_BIN" -e -- "$RCLONE_BIN")"
[[ "$COSMOS_SOURCE" == "$COSMOS_SOURCE_REAL" && "$COSMOS_SOURCE_REAL" == "$SNAPSHOT_ROOT"/* ]] ||
  { echo "ERROR: COSMOS_SOURCE must be one immutable personal-GPFS snapshot" >&2; exit 1; }
[[ "$COSMOS_ENV" == "$COSMOS_ENV_REAL" ]] || { echo "ERROR: environment path must be canonical" >&2; exit 1; }
[[ "$RCLONE_BIN" == "$RCLONE_REAL" && "$(sha256_file "$RCLONE_REAL")" == "$EXPECTED_RCLONE_SHA256" ]] ||
  { echo "ERROR: pinned rclone path or checksum mismatch" >&2; exit 1; }

SNAPSHOT_MANIFEST="$COSMOS_SOURCE_REAL/SNAPSHOT_MANIFEST.json"
PROVENANCE_HELPER="$COSMOS_SOURCE_REAL/examples/h_cluster/cwam_formal_provenance.py"
RECOVERY_RUNNER="$COSMOS_SOURCE_REAL/examples/h_cluster/publish_existing_robocasa365_wan_fullpad_oss.sh"
OSS_PUBLISHER="$COSMOS_SOURCE_REAL/examples/h_cluster/publish_robocasa365_wan_cache_to_oss.py"
for file in "$SNAPSHOT_MANIFEST" "$PROVENANCE_HELPER" "$RECOVERY_RUNNER" "$OSS_PUBLISHER"; do
  [[ -s "$file" && ! -L "$file" ]] || { echo "ERROR: snapshot file missing or unsafe: $file" >&2; exit 1; }
done
[[ "$(sha256_file "$SNAPSHOT_MANIFEST")" == "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" ]] ||
  { echo "ERROR: snapshot manifest checksum mismatch" >&2; exit 1; }
"$PYTHON3_BIN" "$PROVENANCE_HELPER" verify-snapshot \
  --snapshot-root "$COSMOS_SOURCE_REAL" \
  --allowed-root "$SNAPSHOT_ROOT" \
  --expected-manifest-sha256 "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" >/dev/null
"$PYTHON3_BIN" - "$SNAPSHOT_MANIFEST" "$EXPECTED_SOURCE_COMMIT" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
if manifest.get("source_head") != sys.argv[2] or manifest.get("source_dirty") is not False:
    raise SystemExit("snapshot does not bind the expected clean publisher commit")
PY

ENV_MARKER="$COSMOS_ENV_REAL/.cwam_env_ready.json"
RUNTIME_PYTHON="$COSMOS_ENV_REAL/bin/python"
[[ -s "$ENV_MARKER" && -x "$RUNTIME_PYTHON" ]] || { echo "ERROR: audited Cosmos environment is incomplete" >&2; exit 1; }
EXPECTED_RECOVERY_RUNNER_SHA256="$(sha256_file "$RECOVERY_RUNNER")"
EXPECTED_OSS_PUBLISHER_SHA256="$(sha256_file "$OSS_PUBLISHER")"

WORK_ROOT="$OUTPUT_ROOT/$PUBLISH_RUN_ID"
SOURCE_CACHE_ROOT="/nvme/cwam-robocasa365-wan-fullpad/$SOURCE_LATENT_CACHE_RUN_ID/cache"
printf -v Q_SOURCE %q "$COSMOS_SOURCE_REAL"
printf -v Q_ENV %q "$COSMOS_ENV_REAL"
printf -v Q_PUBLISH_RUN_ID %q "$PUBLISH_RUN_ID"
printf -v Q_SOURCE_RUN_ID %q "$SOURCE_LATENT_CACHE_RUN_ID"
printf -v Q_WORK_ROOT %q "$WORK_ROOT"
printf -v Q_SOURCE_CACHE_ROOT %q "$SOURCE_CACHE_ROOT"
printf -v Q_REMOTE %q "$ROBOCASA365_OSS_REMOTE"
printf -v Q_RCLONE %q "$RCLONE_REAL"
printf -v Q_RECOVERY_RUNNER %q "$RECOVERY_RUNNER"
printf -v Q_RUNTIME_PYTHON %q "$RUNTIME_PYTHON"
CAPACITY_PROBE_CODE='import sys; sys.path.insert(0, sys.argv[1]); from examples.h_cluster.publish_robocasa365_wan_cache_to_oss import _capacity_gate; result = _capacity_gate(857279422464); print(f"OSS_CAPACITY_PREFLIGHT current_gb={result['\''current_used_gb'\'']:.2f} expected_gb={result['\''expected_used_gb'\'']:.2f} remaining_gb={result['\''remaining_to_threshold_gb'\'']:.2f}")'
printf -v Q_CAPACITY_PROBE_CODE %q "$CAPACITY_PROBE_CODE"

COMMAND_LINES=(
  "set -Eeuo pipefail"
  "unset BASH_ENV ENV PYTHONHOME PYTHONOPTIMIZE"
  "export PYTHONDONTWRITEBYTECODE=1"
  "export PYTHONNOUSERSITE=1"
  "export COSMOS_SOURCE=$Q_SOURCE"
  "export COSMOS_ENV=$Q_ENV"
  "export PUBLISH_RUN_ID=$Q_PUBLISH_RUN_ID"
  "export SOURCE_LATENT_CACHE_RUN_ID=$Q_SOURCE_RUN_ID"
  "export ROBOCASA365_OSS_REMOTE=$Q_REMOTE"
  "export ROBOCASA365_OSS_WORK_ROOT=$Q_WORK_ROOT"
  "export RCLONE_BIN=$Q_RCLONE"
  "export EXPECTED_SOURCE_COMMIT=$EXPECTED_SOURCE_COMMIT"
  "export EXPECTED_CACHE_SOURCE_COMMIT=$EXPECTED_CACHE_SOURCE_COMMIT"
  "export EXPECTED_RECOVERY_RUNNER_SHA256=$EXPECTED_RECOVERY_RUNNER_SHA256"
  "export EXPECTED_OSS_PUBLISHER_SHA256=$EXPECTED_OSS_PUBLISHER_SHA256"
  "export EXPECTED_RCLONE_SHA256=$EXPECTED_RCLONE_SHA256"
  "export EXPECTED_CACHE_ID=$EXPECTED_CACHE_ID"
  "export EXPECTED_CACHE_MANIFEST_SHA256=$EXPECTED_CACHE_MANIFEST_SHA256"
  "/usr/bin/test -d $Q_SOURCE_CACHE_ROOT"
  "/usr/bin/test -x /usr/bin/openssl"
  "/usr/bin/openssl cms -help >/dev/null 2>&1"
  "[[ \$(/usr/bin/sha256sum $Q_RCLONE | /usr/bin/cut -d' ' -f1) == $EXPECTED_RCLONE_SHA256 ]]"
  "$Q_RCLONE version"
  "$Q_RCLONE help flags | /usr/bin/grep -q -- --s3-upload-concurrency"
  "PYTHONPATH=$Q_SOURCE $Q_RUNTIME_PYTHON -B -c $Q_CAPACITY_PROBE_CODE $Q_SOURCE"
  "exec $BASH_BIN $Q_RECOVERY_RUNNER"
)
WORKER_COMMAND="$(printf '%s\n' "${COMMAND_LINES[@]}")"
WORKER_COMMAND_SHA256="$(printf '%s' "$WORKER_COMMAND" | "$SHA256SUM_BIN" | /usr/bin/cut -d' ' -f1)"

RJOB_ARGS=(
  submit
  --name "$RJOB_NAME"
  --group "$QUOTA_GROUP"
  --charged-group "$QUOTA_GROUP"
  --private-machine group
  --priority 9
  --task-type normal
  --replicas 1
  --restart-policy never
  --termination-grace-period-seconds 60
  --backoff_limit 1
  --enable-sshd
  --image "$PINNED_IMAGE"
  --cpu 16
  --gpu 0
  --memory 64000
  --positive-tags "$RECOVERY_NODE"
  --store-host-nvme
  --mount "$PERSONAL_MOUNT" "$PUBLIC_MOUNT"
)
if [[ "$RJOB_PREDICT_ONLY" == 1 ]]; then
  RJOB_ARGS+=(--predict-only true)
fi

echo "PUBLISH_RUN_ID=$PUBLISH_RUN_ID"
echo "SOURCE_LATENT_CACHE_RUN_ID=$SOURCE_LATENT_CACHE_RUN_ID"
echo "RJOB_NAME=$RJOB_NAME"
echo "RJOB_PREDICT_ONLY=$RJOB_PREDICT_ONLY"
echo "PUBLISHER_SOURCE_COMMIT=$EXPECTED_SOURCE_COMMIT"
echo "CACHE_SOURCE_COMMIT=$EXPECTED_CACHE_SOURCE_COMMIT"
echo "SNAPSHOT_MANIFEST_SHA256=$EXPECTED_SNAPSHOT_MANIFEST_SHA256"
echo "WORKER_COMMAND_SHA256=$WORKER_COMMAND_SHA256"
echo "LATENT_CONTRACT=stage=full_padding shape=48x5x30x52 image_size=480x832x256x768"
echo "OSS_REMOTE=$ROBOCASA365_OSS_REMOTE"

if [[ "$RJOB_PREDICT_ONLY" == 1 ]]; then
  "$RJOB_BIN" "${RJOB_ARGS[@]}" -- /usr/bin/env -u BASH_ENV -u ENV "$BASH_BIN" --noprofile --norc -c "$WORKER_COMMAND"
  echo "RJOB_PREDICTION_COMPLETE reservation_created=false"
  exit 0
fi

/usr/bin/mkdir -p -- "$OUTPUT_ROOT"
if ! /usr/bin/mkdir --mode=0700 -- "$WORK_ROOT"; then
  echo "ERROR: duplicate publish-run reservation exists: $WORK_ROOT" >&2
  exit 1
fi
INTENT="$WORK_ROOT/submission-intent.json"
"$PYTHON3_BIN" - "$INTENT" \
  "$PUBLISH_RUN_ID" "$SOURCE_LATENT_CACHE_RUN_ID" "$RJOB_NAME" \
  "$EXPECTED_SOURCE_COMMIT" "$EXPECTED_CACHE_SOURCE_COMMIT" \
  "$EXPECTED_SNAPSHOT_MANIFEST_SHA256" "$WORKER_COMMAND_SHA256" \
  "$ROBOCASA365_OSS_REMOTE" "$EXPECTED_CACHE_ID" "$EXPECTED_CACHE_MANIFEST_SHA256" <<'PY'
import datetime
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "kind": "robocasa365_wan_full_padding_existing_cache_publish_intent",
    "status": "submitting",
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "publish_run_id": sys.argv[2],
    "source_cache_run_id": sys.argv[3],
    "showname": sys.argv[4],
    "publisher_source_commit": sys.argv[5],
    "cache_source_commit": sys.argv[6],
    "snapshot_manifest_sha256": sys.argv[7],
    "worker_command_sha256": sys.argv[8],
    "oss_remote": sys.argv[9],
    "cache_id": sys.argv[10],
    "cache_manifest_sha256": sys.argv[11],
    "resources": {"replicas": 1, "gpu": 0, "cpu": 16, "memory_mib": 64000},
    "recovery_node": "gpu-l-lg-cmc-h-h200-0298.host.h.pjlab.org.cn",
    "latent_contract": {
        "stage": "full_padding",
        "shape": [48, 5, 30, 52],
        "image_size": [480, 832, 256, 768],
        "storage_dtype": "bfloat16",
    },
    "vae_generation_enabled": False,
}
with target.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.chmod(target, 0o444)
PY

SUBMIT_LOG="$WORK_ROOT/rjob-submit.log"
set +e
"$RJOB_BIN" "${RJOB_ARGS[@]}" -- /usr/bin/env -u BASH_ENV -u ENV "$BASH_BIN" --noprofile --norc -c "$WORKER_COMMAND" \
  > >( /usr/bin/tee "$SUBMIT_LOG" ) 2>&1
SUBMIT_STATUS=$?
set -e
[[ "$SUBMIT_STATUS" -eq 0 ]] || { echo "ERROR: rjob submission failed; reservation retained" >&2; exit 1; }
/usr/bin/chmod 0444 "$SUBMIT_LOG"
SCHEDULER_NAME="$("$PYTHON3_BIN" - "$SUBMIT_LOG" "$RJOB_NAME" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(errors="replace")
matches = re.findall(r"created rjob_name:\s*([a-z0-9](?:[-a-z0-9]*[a-z0-9])?)", text)
if len(matches) != 1 or not matches[0].startswith(sys.argv[2] + "-"):
    raise SystemExit(f"cannot bind one scheduler name: {matches!r}")
print(matches[0])
PY
)"
SCHEDULER_READBACK="$WORK_ROOT/rjob-get.log"
"$RJOB_BIN" get "$SCHEDULER_NAME" >"$SCHEDULER_READBACK" 2>&1
/usr/bin/chmod 0444 "$SCHEDULER_READBACK"
printf '%s\n' "$SCHEDULER_NAME" >"$WORK_ROOT/scheduler-rjob-name.txt"
/usr/bin/chmod 0444 "$WORK_ROOT/scheduler-rjob-name.txt"
echo "RJOB_SUBMISSION_ACCEPTED scheduler_name=$SCHEDULER_NAME work_root=$WORK_ROOT"
