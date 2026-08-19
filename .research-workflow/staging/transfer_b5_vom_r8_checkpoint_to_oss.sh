#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://208.64.254.190/B5-VOM-R8-I20000-EMA-BF16-v1"
TOKEN_FILE="/mnt/workspace/cwam_cosmos3_robocasa365_eval/control/transfer-direct-h/relay.token"
STAGE_ROOT="/mnt/workspace/cwam_cosmos3_robocasa365_eval/control/transfer-staging/B5-VOM-R8-I20000-EMA-BF16-v1"
DEST_ROOT="/oss-chengjuntao/cpfs-user-chengjuntao/cwam_cosmos3_robocasa365_eval/inputs/checkpoints/B5-VOM-R8-I20000-EMA-BF16-v1"

test -r "${TOKEN_FILE}"
mountpoint -q /mnt/workspace
mountpoint -q /oss-chengjuntao
test -w /mnt/workspace
test -w /oss-chengjuntao
mkdir -p "${STAGE_ROOT}/model" "${DEST_ROOT}/model"

TOKEN="$(cat "${TOKEN_FILE}")"
export BASE_URL TOKEN STAGE_ROOT DEST_ROOT

transfer_one() {
  local relative="$1"
  local expected="$2"
  local stage="${STAGE_ROOT}/${relative}"
  local destination="${DEST_ROOT}/${relative}"
  local uploading="${destination}.uploading"

  mkdir -p "$(dirname "${stage}")" "$(dirname "${destination}")"
  if [[ -f "${destination}" && "$(stat -c %s "${destination}")" == "${expected}" ]]; then
    printf 'already_complete\t%s\t%s\n' "${relative}" "${expected}"
    return 0
  fi
  if [[ -e "${destination}" ]]; then
    printf 'wrong_existing_destination\t%s\t%s\n' "${relative}" "$(stat -c %s "${destination}")" >&2
    return 1
  fi

  curl --fail --silent --show-error --location --insecure \
    --header "Authorization: Bearer ${TOKEN}" \
    --retry 100 --retry-delay 5 \
    --connect-timeout 30 --continue-at - \
    --output "${stage}.partial" "${BASE_URL}/${relative}"
  [[ "$(stat -c %s "${stage}.partial")" == "${expected}" ]]
  mv "${stage}.partial" "${stage}"

  rm -f "${uploading}"
  cp --reflink=never "${stage}" "${uploading}"
  [[ "$(stat -c %s "${uploading}")" == "${expected}" ]]
  chmod 0444 "${uploading}"
  mv "${uploading}" "${destination}"
  printf 'complete\t%s\t%s\n' "${relative}" "${expected}"
}
export -f transfer_one

cat <<'EOF' | xargs -P 8 -n 2 bash -eo pipefail -c 'transfer_one "$0" "$1"'
config.yaml 24339
ema-bf16-metadata-receipt.json 2247
serving-checkpoint-size-manifest.json 1003
model/.metadata 1783145
model/__0_0.distcp 3847111406
model/__1_0.distcp 3847107525
model/__2_0.distcp 3847099361
model/__3_0.distcp 3847093624
model/__4_0.distcp 3847086118
model/__5_0.distcp 3847074544
model/__6_0.distcp 3847063569
model/__7_0.distcp 3847056447
EOF

python3 - "${DEST_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {
    "config.yaml": 24339,
    "ema-bf16-metadata-receipt.json": 2247,
    "serving-checkpoint-size-manifest.json": 1003,
    "model/.metadata": 1783145,
    "model/__0_0.distcp": 3847111406,
    "model/__1_0.distcp": 3847107525,
    "model/__2_0.distcp": 3847099361,
    "model/__3_0.distcp": 3847093624,
    "model/__4_0.distcp": 3847086118,
    "model/__5_0.distcp": 3847074544,
    "model/__6_0.distcp": 3847063569,
    "model/__7_0.distcp": 3847056447,
}
observed = {
    str(path.relative_to(root)): path.stat().st_size
    for path in root.rglob("*")
    if path.is_file()
}
if observed != expected:
    raise SystemExit(json.dumps({"expected": expected, "observed": observed}, indent=2, sort_keys=True))
print(json.dumps({
    "status": "complete",
    "file_count": len(observed),
    "total_bytes": sum(observed.values()),
    "root": str(root),
}, sort_keys=True))
PY
