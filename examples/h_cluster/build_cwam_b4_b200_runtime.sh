#!/usr/bin/env bash
set -Eeuo pipefail

readonly IMAGE="cwam-b4-runtime:ngc25.09-ffmpeg6"
readonly ROOT="/data0/wam/cwam-b4"
readonly SOURCE="${ROOT}/source/CWAM-COSMOS3-B4-ADAPTIVE-DSELECT-R1-20260814"
readonly ENV_DIR="${ROOT}/env/cosmos-cu130"
readonly DOCKERFILE="${SOURCE}/examples/h_cluster/Dockerfile.cwam_b4_b200_runtime"
readonly EVIDENCE="${ROOT}/evidence"
readonly BUILD_LOG="${EVIDENCE}/runtime-image-build.log"
readonly SELF_TEST_LOG="${EVIDENCE}/runtime-image-self-test.log"

test -s "${DOCKERFILE}"
test -x "${ENV_DIR}/bin/python"
mkdir -p "${EVIDENCE}"

if ! sudo docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  sudo docker build \
    --file "${DOCKERFILE}" \
    --tag "${IMAGE}" \
    "${SOURCE}" 2>&1 | tee "${BUILD_LOG}"
fi

sudo docker run --rm \
  -v /data0:/data0 \
  "${IMAGE}" bash -lc "
    set -Eeuo pipefail
    ffmpeg -version | head -n 1
    '${ENV_DIR}/bin/python' - <<'PY'
import torch
from torchcodec.decoders import VideoDecoder

print(f'torch={torch.__version__}')
print(f'torchcodec_decoder={VideoDecoder.__module__}.{VideoDecoder.__name__}')
print('runtime_image_self_test=PASS')
PY
  " 2>&1 | tee "${SELF_TEST_LOG}"

