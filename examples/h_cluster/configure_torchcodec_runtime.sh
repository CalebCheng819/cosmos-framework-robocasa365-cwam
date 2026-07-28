#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Expose the pinned PyAV wheel's bundled FFmpeg 8 libraries under the standard
# sonames expected by TorchCodec.  PyAV intentionally installs auditwheel-
# hashed filenames; TorchCodec dlopens the public FFmpeg sonames instead.
#
# Source this file from the H-cluster job shell before importing TorchCodec.
# The compatibility directory is process-local under /tmp and is inherited by
# torchrun ranks through LD_LIBRARY_PATH.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "ERROR: source configure_torchcodec_runtime.sh; do not execute it directly" >&2
    exit 2
fi

: "${COSMOS_ENV:?Set COSMOS_ENV to the pinned Cosmos training environment.}"

CWAM_TORCHCODEC_SITE_PACKAGES="$COSMOS_ENV/lib/python3.13/site-packages"
CWAM_TORCHCODEC_AV_LIBS="$CWAM_TORCHCODEC_SITE_PACKAGES/av.libs"
CWAM_TORCHCODEC_NVIDIA_ROOT="$CWAM_TORCHCODEC_SITE_PACKAGES/nvidia"
test -d "$CWAM_TORCHCODEC_AV_LIBS"
test -d "$CWAM_TORCHCODEC_NVIDIA_ROOT"
test -f "$CWAM_TORCHCODEC_SITE_PACKAGES/torchcodec/libtorchcodec_core8.so"
test -e "$CWAM_TORCHCODEC_NVIDIA_ROOT/npp/lib/libnppicc.so.12"

# TorchCodec's CUDA build links against the CUDA wheels installed in this
# exact Python environment.  Login/job shells do not expose those wheel-local
# libraries; the wrapper sanitizes inherited LD_LIBRARY_PATH to the audited H
# driver entries before this helper prepends the pinned wheel directories.
# Discover only the pinned ``nvidia/<component>/lib`` directories, in stable
# order, and carry them into every torchrun child.
CWAM_TORCHCODEC_NVIDIA_LIBRARY_DIRS=()
mapfile -t CWAM_TORCHCODEC_NVIDIA_LIBRARY_DIRS < <(
    find "$CWAM_TORCHCODEC_NVIDIA_ROOT" \
        -mindepth 2 -maxdepth 2 -type d -name lib -print |
        LC_ALL=C sort
)
if [[ "${#CWAM_TORCHCODEC_NVIDIA_LIBRARY_DIRS[@]}" -eq 0 ]]; then
    echo "ERROR: no pinned NVIDIA wheel library directories found under $CWAM_TORCHCODEC_NVIDIA_ROOT" >&2
    return 1
fi
CWAM_TORCHCODEC_NVIDIA_LIBRARY_PATH="$(
    IFS=:
    echo "${CWAM_TORCHCODEC_NVIDIA_LIBRARY_DIRS[*]}"
)"
export NVRTC_HOME="${NVRTC_HOME:-$CWAM_TORCHCODEC_NVIDIA_ROOT/cuda_nvrtc}"
export CURAND_HOME="${CURAND_HOME:-$CWAM_TORCHCODEC_NVIDIA_ROOT/curand}"

CWAM_TORCHCODEC_FFMPEG_DIR="$(
    mktemp -d "${TMPDIR:-/tmp}/cwam-torchcodec-ffmpeg8.XXXXXX"
)"

_cwam_link_ffmpeg_library() {
    local soname="$1"
    local pattern="$2"
    local matches=()
    mapfile -t matches < <(
        find "$CWAM_TORCHCODEC_AV_LIBS" -maxdepth 1 -type f -name "$pattern" -print
    )
    if [[ "${#matches[@]}" -ne 1 ]]; then
        echo "ERROR: expected one PyAV library for $soname, found ${#matches[@]} using $pattern" >&2
        return 1
    fi
    ln -s "${matches[0]}" "$CWAM_TORCHCODEC_FFMPEG_DIR/$soname"
}

_cwam_link_ffmpeg_library libavutil.so.60 "libavutil-*.so.60.*"
_cwam_link_ffmpeg_library libavcodec.so.62 "libavcodec-*.so.62.*"
_cwam_link_ffmpeg_library libavformat.so.62 "libavformat-*.so.62.*"
_cwam_link_ffmpeg_library libavdevice.so.62 "libavdevice-*.so.62.*"
_cwam_link_ffmpeg_library libavfilter.so.11 "libavfilter-*.so.11.*"
_cwam_link_ffmpeg_library libswscale.so.9 "libswscale-*.so.9.*"
_cwam_link_ffmpeg_library libswresample.so.6 "libswresample-*.so.6.*"
unset -f _cwam_link_ffmpeg_library

export CWAM_TORCHCODEC_FFMPEG_DIR
export CWAM_TORCHCODEC_NVIDIA_ROOT
export CWAM_TORCHCODEC_NVIDIA_LIBRARY_PATH
export LD_LIBRARY_PATH="$CWAM_TORCHCODEC_FFMPEG_DIR:$CWAM_TORCHCODEC_AV_LIBS:$CWAM_TORCHCODEC_NVIDIA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PYTHONNOUSERSITE=1 "$COSMOS_ENV/bin/python" - <<'PY'
import importlib.metadata as md

import av
from torchcodec.decoders import VideoDecoder

torchcodec_version = md.version("torchcodec")
assert av.__version__ == "16.1.0", av.__version__
assert torchcodec_version == "0.10.0+cu128", torchcodec_version
print(
    "TORCHCODEC_RUNTIME_READY",
    f"av={av.__version__}",
    f"torchcodec={torchcodec_version}",
    f"decoder={VideoDecoder.__name__}",
)
PY
