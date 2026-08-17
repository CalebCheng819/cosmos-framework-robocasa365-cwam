#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Run on the H workspace after the worker has published its one-time
# recipient certificate. No plaintext credential is written to GPFS.

set -Eeuo pipefail
umask 077

: "${CONTROL_ROOT:=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/experiments/cwam-b2-20260811/control}"
: "${EXPERIMENT_LABEL:=B2}"
: "${LOCAL_RCLONE_CONFIG:=/root/.config/rclone/rclone.conf}"

HANDOFF_ROOT="$CONTROL_ROOT/rclone-handoff"
CERTIFICATE="$HANDOFF_ROOT/recipient.pem"
DESTINATION="$HANDOFF_ROOT/rclone-config.cms.der"
TEMPORARY="$HANDOFF_ROOT/.rclone-config.cms.der.tmp-$$"

[[ -s "$HANDOFF_ROOT/request.ready" && -s "$CERTIFICATE" ]] || {
    echo "ERROR: $EXPERIMENT_LABEL worker has not published a complete rclone handoff request" >&2
    exit 1
}
[[ -f "$LOCAL_RCLONE_CONFIG" && ! -L "$LOCAL_RCLONE_CONFIG" ]] || {
    echo "ERROR: local rclone config is missing or unsafe" >&2
    exit 1
}
[[ ! -e "$DESTINATION" ]] || {
    echo "ERROR: refusing to replace an existing encrypted rclone handoff" >&2
    exit 1
}
/usr/bin/rclone listremotes --config "$LOCAL_RCLONE_CONFIG" | grep -Fx 'eailab-hdd2:' >/dev/null

trap 'rm -f "$TEMPORARY"' EXIT INT TERM
/usr/bin/openssl cms -encrypt -binary -aes-256-cbc -outform DER \
    -in "$LOCAL_RCLONE_CONFIG" \
    -out "$TEMPORARY" \
    "$CERTIFICATE"
chmod 0400 "$TEMPORARY"
mv "$TEMPORARY" "$DESTINATION"
trap - EXIT INT TERM

echo "${EXPERIMENT_LABEL}_RCLONE_HANDOFF_COMPLETE destination=$DESTINATION"
