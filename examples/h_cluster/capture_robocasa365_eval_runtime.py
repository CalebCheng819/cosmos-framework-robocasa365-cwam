#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Capture a fail-closed runtime receipt for formal RoboCasa365 evaluation.

The helper is intentionally a separate process from both the policy server and
the evaluator.  It interrogates each pinned Python interpreter with isolated
``argv`` subprocesses, verifies both full Python-environment manifests, binds
the four source checkouts to exact clean commits, and cross-checks PyTorch's GPU
inventory against ``nvidia-smi``.  It also creates and releases one real
RoboCasa EGL context under each evaluator shard's exact GPU environment before
publishing a run-identity-bound immutable receipt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RECEIPT_KIND = "cwam_robocasa365_eval_runtime_receipt"
SCHEMA_VERSION = 1
COMMAND_TIMEOUT_SECONDS = 120.0
MAX_INPUT_JSON_BYTES = 16 * 1024 * 1024
MAX_COMMAND_OUTPUT_BYTES = 4 * 1024 * 1024
ENVIRONMENT_MANIFEST_KIND = "cwam_python_environment_manifest"
ENVIRONMENT_MANIFEST_HELPER_REL = Path("examples/h_cluster/cwam_python_environment_manifest.py")
H_DRIVER_LIBRARY_PATH = "/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_DRIVER_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){1,3}$")
_NORMALIZED_GPU_UUID_RE = re.compile(r"^[0-9a-f]{32}$")

_COSMOS_PROBE = r"""
import contextlib
import json
import pathlib
import sys

sys.dont_write_bytecode = True
source_root = pathlib.Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(source_root))

with contextlib.redirect_stdout(sys.stderr):
    import cosmos_framework
    import torch


def resolved_module_path(module):
    value = getattr(module, "__file__", None)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{module.__name__} has no concrete __file__")
    return str(pathlib.Path(value).resolve(strict=True))


gpus = []
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        device_uuid = getattr(properties, "uuid", None)
        if isinstance(device_uuid, bytes):
            device_uuid = device_uuid.hex()
        elif device_uuid is not None:
            device_uuid = str(device_uuid)
        gpus.append(
            {
                "index": index,
                "name": str(properties.name),
                "uuid": device_uuid,
                "capability": [int(properties.major), int(properties.minor)],
                "total_memory_bytes": int(properties.total_memory),
            }
        )

payload = {
    "python": {
        "version": sys.version,
        "executable": str(pathlib.Path(sys.executable).resolve(strict=True)),
        "prefix": str(pathlib.Path(sys.prefix).resolve(strict=True)),
        "base_prefix": str(pathlib.Path(sys.base_prefix).resolve(strict=True)),
    },
    "torch": {
        "version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "module_path": resolved_module_path(torch),
    },
    "cosmos_framework": {
        "module_path": resolved_module_path(cosmos_framework),
    },
    "gpus": gpus,
}
print(json.dumps(payload, sort_keys=True, allow_nan=False))
"""

_ROBOCASA_PROBE = r"""
import contextlib
import importlib
import importlib.metadata
import importlib.util
import json
import pathlib
import sys

sys.dont_write_bytecode = True
demian_import_root = pathlib.Path(sys.argv[1]).resolve(strict=True)
robocasa_source_root = pathlib.Path(sys.argv[2]).resolve(strict=True)
robosuite_source_root = pathlib.Path(sys.argv[3]).resolve(strict=True)
sys.path[:0] = [
    str(demian_import_root),
    str(robocasa_source_root),
    str(robosuite_source_root),
]

distribution_names = ("numpy", "mujoco", "gymnasium", "robocasa", "robosuite")
# Read authoritative distribution metadata before importing any project code.
# In particular, never trust a mutable numpy.__version__ value.
distribution_versions = {
    name: importlib.metadata.version(name)
    for name in distribution_names
}


def resolved_module_path(module):
    value = getattr(module, "__file__", None)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{module.__name__} has no concrete __file__")
    return str(pathlib.Path(value).resolve(strict=True))


packages = {}
with contextlib.redirect_stdout(sys.stderr):
    for name in distribution_names:
        module = importlib.import_module(name)
        reported = getattr(module, "__version__", None)
        packages[name] = {
            "distribution_version": distribution_versions[name],
            "module_reported_version": str(reported) if reported is not None else None,
            "version_source": "importlib.metadata",
            "module_path": resolved_module_path(module),
        }

demian_evaluator_spec = importlib.util.find_spec("eval_utils.run_robocasa_eval")
if demian_evaluator_spec is None or not isinstance(demian_evaluator_spec.origin, str):
    raise RuntimeError("DeMiAn evaluator has no concrete import origin")

payload = {
    "python": {
        "version": sys.version,
        "executable": str(pathlib.Path(sys.executable).resolve(strict=True)),
        "prefix": str(pathlib.Path(sys.prefix).resolve(strict=True)),
        "base_prefix": str(pathlib.Path(sys.base_prefix).resolve(strict=True)),
    },
    "packages": packages,
    "demian_evaluator": {
        "module_path": str(pathlib.Path(demian_evaluator_spec.origin).resolve(strict=True)),
    },
}
print(json.dumps(payload, sort_keys=True, allow_nan=False))
"""

_ROBOCASA_EGL_PROBE = r"""
import contextlib
import json
import os
import pathlib
import sys

sys.dont_write_bytecode = True
robosuite_source_root = pathlib.Path(sys.argv[1]).resolve(strict=True)
shard_index = int(sys.argv[2])
expected_device = str(shard_index)
sys.path.insert(0, str(robosuite_source_root))

expected_environment = {
    "CUDA_VISIBLE_DEVICES": expected_device,
    "MUJOCO_EGL_DEVICE_ID": expected_device,
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
}
for name, expected in expected_environment.items():
    actual = os.environ.get(name)
    if actual != expected:
        raise RuntimeError(f"{name} mismatch: expected={expected!r}, actual={actual!r}")

context = None
with contextlib.redirect_stdout(sys.stderr):
    import robosuite
    from robosuite.renderers.context.egl_context import EGLGLContext

    context = EGLGLContext(max_width=64, max_height=64, device_id=shard_index)
    try:
        context.make_current()
    finally:
        context.free()

payload = {
    "status": "passed",
    "shard_index": shard_index,
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "mujoco_egl_device_id": os.environ["MUJOCO_EGL_DEVICE_ID"],
    "mujoco_gl": os.environ["MUJOCO_GL"],
    "pyopengl_platform": os.environ["PYOPENGL_PLATFORM"],
    "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
    "egl_vendor_library_dirs": os.environ.get("__EGL_VENDOR_LIBRARY_DIRS"),
    "robosuite_module_path": str(pathlib.Path(robosuite.__file__).resolve(strict=True)),
}
print(json.dumps(payload, sort_keys=True, allow_nan=False))
"""


class RuntimeReceiptError(RuntimeError):
    """A runtime capture, validation, or publication contract failed."""


class RuntimeReceiptValidationError(RuntimeReceiptError):
    """Caller input or observed runtime state violated the receipt contract."""


class RuntimeReceiptCommandError(RuntimeReceiptError):
    """A bounded subprocess failed or returned malformed output."""


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
ExecutableResolver = Callable[[str], str | None]


def _reject_incomplete_path(path: Path, label: str) -> None:
    for component in path.parts:
        if ".incomplete" in component.lower():
            raise RuntimeReceiptValidationError(
                f"{label} contains a forbidden .incomplete path component: {component!r}"
            )
        if any(ord(character) < 32 or ord(character) == 127 for character in component):
            raise RuntimeReceiptValidationError(f"{label} contains a control character: {component!r}")


def _lexical_absolute(path: str | os.PathLike[str], label: str) -> Path:
    raw = os.fspath(path)
    if not raw:
        raise RuntimeReceiptValidationError(f"{label} must not be empty")
    supplied = Path(raw)
    lexical = Path(os.path.abspath(raw))
    if supplied != lexical:
        raise RuntimeReceiptValidationError(f"{label} must be a normalized absolute path: {supplied}")
    _reject_incomplete_path(lexical, label)
    return lexical


def _canonical_path(
    path: str | os.PathLike[str],
    label: str,
    *,
    kind: str,
    allow_leaf_symlink: bool = False,
) -> Path:
    lexical = _lexical_absolute(path, label)
    try:
        resolved = lexical.resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeReceiptValidationError(f"{label} does not exist: {lexical}") from error
    except (OSError, RuntimeError) as error:
        raise RuntimeReceiptValidationError(f"cannot resolve {label} {lexical}: {error}") from error
    if lexical != resolved:
        leaf_only = (
            allow_leaf_symlink and lexical.is_symlink() and lexical.parent.resolve(strict=True) == lexical.parent
        )
        if not leaf_only:
            raise RuntimeReceiptValidationError(
                f"{label} must contain no symlink components: provided={lexical}, resolved={resolved}"
            )
    _reject_incomplete_path(resolved, label)
    try:
        metadata = resolved.lstat()
    except OSError as error:
        raise RuntimeReceiptValidationError(f"cannot inspect {label} {resolved}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise RuntimeReceiptValidationError(f"{label} must resolve to a non-symlink path: {resolved}")
    if kind == "file" and not stat.S_ISREG(metadata.st_mode):
        raise RuntimeReceiptValidationError(f"{label} must be a regular file: {resolved}")
    if kind == "directory" and not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeReceiptValidationError(f"{label} must be a directory: {resolved}")
    return resolved


def _stable_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (int(metadata.st_dev), int(metadata.st_ino), int(metadata.st_mode))


def _read_stable_file(path: Path, label: str) -> bytes:
    canonical = _canonical_path(path, label, kind="file")
    before = canonical.stat()
    if before.st_size > MAX_INPUT_JSON_BYTES:
        raise RuntimeReceiptValidationError(
            f"{label} exceeds the {MAX_INPUT_JSON_BYTES}-byte safety limit: {canonical}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(canonical, flags)
    except OSError as error:
        raise RuntimeReceiptValidationError(f"cannot safely open {label} {canonical}: {error}") from error
    try:
        opened = os.fstat(descriptor)
        if _stable_identity(before) != _stable_identity(opened):
            raise RuntimeReceiptValidationError(f"{label} changed before reading: {canonical}")
        chunks: list[bytes] = []
        remaining = MAX_INPUT_JSON_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after_read = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = canonical.stat()
    identity = _stable_identity(before)
    if identity != _stable_identity(after_read) or identity != _stable_identity(after_path):
        raise RuntimeReceiptValidationError(f"{label} changed while reading: {canonical}")
    if len(payload) > MAX_INPUT_JSON_BYTES:
        raise RuntimeReceiptValidationError(
            f"{label} exceeds the {MAX_INPUT_JSON_BYTES}-byte safety limit: {canonical}"
        )
    return payload


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeReceiptValidationError(f"JSON contains a duplicate object key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise RuntimeReceiptValidationError(f"JSON contains a forbidden non-finite constant: {value}")


def _strict_json_bytes(payload: bytes, label: str) -> Any:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise RuntimeReceiptValidationError(f"{label} is not valid UTF-8: {error}") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise RuntimeReceiptValidationError(f"{label} is not valid JSON: {error}") from error


def _strict_json_text(payload: str, label: str) -> Any:
    return _strict_json_bytes(payload.encode("utf-8"), label)


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeReceiptValidationError(f"{label} must be a JSON object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise RuntimeReceiptValidationError(
            f"{label} keys mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(repr(key) for key in observed - expected)}"
        )


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RuntimeReceiptValidationError(f"{label} must be a non-empty trimmed string")
    return value


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RuntimeReceiptValidationError(f"{label} must be an integer >= {minimum}, got {value!r}")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuntimeReceiptValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _require_commit(value: object, label: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise RuntimeReceiptValidationError(f"{label} must be a full lowercase 40-character Git commit")
    return value


def _require_run_id(value: object, label: str) -> str:
    if not isinstance(value, str) or _RUN_ID_RE.fullmatch(value) is None:
        raise RuntimeReceiptValidationError(f"{label} must contain only ASCII letters, digits, '.', '_', or '-'")
    return value


def _require_capture_nonce(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuntimeReceiptValidationError("capture_nonce must be exactly 64 lowercase hexadecimal characters")
    return value


def _prepare_environment_manifest(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    expected = _require_sha256(expected_sha256, f"expected {label} SHA256")
    canonical = _canonical_path(path, label, kind="file")
    payload = _read_stable_file(canonical, label)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise RuntimeReceiptValidationError(
            f"{label} SHA256 mismatch: expected={expected}, actual={actual}, path={canonical}"
        )
    manifest = _require_mapping(_strict_json_bytes(payload, label), label)
    if manifest.get("kind") != ENVIRONMENT_MANIFEST_KIND or manifest.get("schema_version") != 1:
        raise RuntimeReceiptValidationError(f"{label} must be a schema-1 {ENVIRONMENT_MANIFEST_KIND!r} document")
    return {
        "path": str(canonical),
        "size_bytes": len(payload),
        "sha256": actual,
        "expected_sha256": expected,
        "kind": ENVIRONMENT_MANIFEST_KIND,
        "schema_version": 1,
    }


def _resolve_executable(name: str, resolver: ExecutableResolver) -> Path:
    candidate = resolver(name)
    if candidate is None:
        raise RuntimeReceiptValidationError(f"required executable is unavailable on PATH: {name}")
    executable = _canonical_path(candidate, f"{name} executable", kind="file", allow_leaf_symlink=True)
    if not os.access(executable, os.X_OK):
        raise RuntimeReceiptValidationError(f"{name} executable is not executable: {executable}")
    return executable


def _python_launch_and_resolved(path: Path, label: str) -> tuple[Path, Path]:
    launch = _lexical_absolute(path, label)
    parent = _canonical_path(launch.parent, f"{label} parent", kind="directory")
    if launch.parent != parent:
        raise RuntimeReceiptValidationError(f"{label} parent must contain no symlink components")
    try:
        leaf_metadata = launch.lstat()
        resolved = launch.resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeReceiptValidationError(f"{label} does not exist: {launch}") from error
    except (OSError, RuntimeError) as error:
        raise RuntimeReceiptValidationError(f"cannot resolve {label} {launch}: {error}") from error
    if not (stat.S_ISREG(leaf_metadata.st_mode) or stat.S_ISLNK(leaf_metadata.st_mode)):
        raise RuntimeReceiptValidationError(f"{label} must be a regular file or a leaf symlink: {launch}")
    resolved = _canonical_path(resolved, f"resolved {label}", kind="file")
    if not os.access(launch, os.X_OK) or not os.access(resolved, os.X_OK):
        raise RuntimeReceiptValidationError(f"{label} is not executable: {launch}")
    return launch, resolved


def _run_command(
    argv: Sequence[str],
    *,
    label: str,
    runner: CommandRunner,
    timeout_seconds: float,
    environment_overrides: Mapping[str, str] | None = None,
) -> str:
    if isinstance(argv, (str, bytes)) or not argv or any(not isinstance(item, str) or not item for item in argv):
        raise RuntimeReceiptValidationError(f"{label} command must be a non-empty argv sequence")
    environment = os.environ.copy()
    environment.update(
        {
            "LC_ALL": "C",
            "LANG": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    if environment_overrides is not None:
        if any(
            not isinstance(key, str) or not key or not isinstance(value, str)
            for key, value in environment_overrides.items()
        ):
            raise RuntimeReceiptValidationError(f"{label} environment overrides must be non-empty strings")
        environment.update(environment_overrides)
    try:
        result = runner(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=False,
            env=environment,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeReceiptCommandError(f"{label} timed out after {timeout_seconds:g} seconds") from error
    except OSError as error:
        raise RuntimeReceiptCommandError(f"cannot execute {label}: {error}") from error
    if type(result.returncode) is not int:
        raise RuntimeReceiptCommandError(f"{label} returned an invalid process status")
    stdout = result.stdout if isinstance(result.stdout, str) else ""
    stderr = result.stderr if isinstance(result.stderr, str) else ""
    if result.returncode != 0:
        detail = stderr.strip()[-2000:] or stdout.strip()[-2000:] or "no subprocess diagnostics"
        raise RuntimeReceiptCommandError(f"{label} failed with exit code {result.returncode}: {detail}")
    if len(stdout.encode("utf-8")) > MAX_COMMAND_OUTPUT_BYTES:
        raise RuntimeReceiptCommandError(f"{label} stdout exceeds the {MAX_COMMAND_OUTPUT_BYTES}-byte safety limit")
    if len(stderr.encode("utf-8")) > MAX_COMMAND_OUTPUT_BYTES:
        raise RuntimeReceiptCommandError(f"{label} stderr exceeds the {MAX_COMMAND_OUTPUT_BYTES}-byte safety limit")
    return stdout


def _git_source_record(
    *,
    name: str,
    root: Path,
    expected_commit: str,
    git: Path,
    runner: CommandRunner,
    timeout_seconds: float,
) -> dict[str, Any]:
    expected = _require_commit(expected_commit, f"expected {name} commit")
    canonical = _canonical_path(root, f"{name} source root", kind="directory")
    prefix = [str(git), "-c", f"safe.directory={canonical}", "-C", str(canonical)]
    actual_output = _run_command(
        [*prefix, "rev-parse", "--verify", "HEAD"],
        label=f"{name} git rev-parse",
        runner=runner,
        timeout_seconds=timeout_seconds,
    )
    lines = actual_output.splitlines()
    if len(lines) != 1:
        raise RuntimeReceiptValidationError(f"{name} git rev-parse must return exactly one line")
    actual = _require_commit(lines[0].strip(), f"actual {name} commit")
    if actual != expected:
        raise RuntimeReceiptValidationError(
            f"{name} source commit mismatch: expected={expected}, actual={actual}, root={canonical}"
        )
    status = _run_command(
        [*prefix, "status", "--porcelain=v1", "--untracked-files=all"],
        label=f"{name} git status",
        runner=runner,
        timeout_seconds=timeout_seconds,
    )
    if status.strip():
        raise RuntimeReceiptValidationError(f"{name} source checkout is dirty: {canonical}")
    return {
        "root": str(canonical),
        "expected_commit": expected,
        "actual_commit": actual,
        "clean": True,
    }


def _validate_environment_report(value: object, label: str) -> dict[str, Any]:
    report = _require_mapping(value, label)
    _require_exact_keys(
        report,
        frozenset(
            {
                "status",
                "interpreter_resolved_path",
                "interpreter_sha256",
                "python_version",
                "distribution_count",
                "distributions_sha256",
            }
        ),
        label,
    )
    if report["status"] != "passed":
        raise RuntimeReceiptValidationError(f"{label}.status must be 'passed'")
    interpreter = _canonical_path(
        _require_string(report["interpreter_resolved_path"], f"{label}.interpreter_resolved_path"),
        f"{label}.interpreter_resolved_path",
        kind="file",
    )
    _require_sha256(report["interpreter_sha256"], f"{label}.interpreter_sha256")
    _require_string(report["python_version"], f"{label}.python_version")
    _require_int(report["distribution_count"], f"{label}.distribution_count", minimum=1)
    _require_sha256(report["distributions_sha256"], f"{label}.distributions_sha256")
    normalized = dict(report)
    normalized["interpreter_resolved_path"] = str(interpreter)
    return normalized


def _verify_environment(
    *,
    python: Path,
    helper: Path,
    manifest: Path,
    label: str,
    runner: CommandRunner,
    timeout_seconds: float,
) -> dict[str, Any]:
    stdout = _run_command(
        [str(python), "-I", str(helper), "verify", "--manifest", str(manifest)],
        label=f"{label} environment-manifest verification",
        runner=runner,
        timeout_seconds=timeout_seconds,
    )
    return _validate_environment_report(_strict_json_text(stdout, f"{label} environment report"), label)


def _validate_python_record(value: object, label: str, requested_python: Path) -> dict[str, str]:
    record = _require_mapping(value, label)
    _require_exact_keys(record, frozenset({"version", "executable", "prefix", "base_prefix"}), label)
    version = _require_string(record["version"], f"{label}.version")
    executable = _canonical_path(
        _require_string(record["executable"], f"{label}.executable"),
        f"{label}.executable",
        kind="file",
    )
    prefix = _canonical_path(
        _require_string(record["prefix"], f"{label}.prefix"),
        f"{label}.prefix",
        kind="directory",
    )
    base_prefix = _canonical_path(
        _require_string(record["base_prefix"], f"{label}.base_prefix"),
        f"{label}.base_prefix",
        kind="directory",
    )
    if executable != requested_python:
        raise RuntimeReceiptValidationError(
            f"{label}.executable mismatch: expected={requested_python}, actual={executable}"
        )
    return {
        "version": version,
        "executable": str(executable),
        "prefix": str(prefix),
        "base_prefix": str(base_prefix),
    }


def _canonical_reported_module_path(value: object, label: str) -> Path:
    return _canonical_path(_require_string(value, label), label, kind="file")


def _normalize_gpu_uuid(value: object, label: str) -> str:
    text = _require_string(value, label)
    normalized = text.lower()
    if normalized.startswith("gpu-"):
        normalized = normalized[4:]
    normalized = normalized.replace("-", "")
    if _NORMALIZED_GPU_UUID_RE.fullmatch(normalized) is None:
        raise RuntimeReceiptValidationError(f"{label} is not a valid NVIDIA GPU UUID: {text!r}")
    return normalized


def _validate_cosmos_probe(
    value: object,
    *,
    requested_python: Path,
    cosmos_root: Path,
    num_gpus: int,
) -> dict[str, Any]:
    probe = _require_mapping(value, "Cosmos runtime probe")
    _require_exact_keys(probe, frozenset({"python", "torch", "cosmos_framework", "gpus"}), "Cosmos runtime probe")
    python_record = _validate_python_record(probe["python"], "Cosmos runtime probe.python", requested_python)

    torch_record = _require_mapping(probe["torch"], "Cosmos runtime probe.torch")
    _require_exact_keys(
        torch_record,
        frozenset({"version", "cuda_version", "cuda_available", "cuda_device_count", "module_path"}),
        "Cosmos runtime probe.torch",
    )
    torch_version = _require_string(torch_record["version"], "Cosmos runtime probe.torch.version")
    torch_cuda = _require_string(torch_record["cuda_version"], "Cosmos runtime probe.torch.cuda_version")
    if torch_record["cuda_available"] is not True:
        raise RuntimeReceiptValidationError("Cosmos runtime probe reports CUDA unavailable")
    device_count = _require_int(
        torch_record["cuda_device_count"],
        "Cosmos runtime probe.torch.cuda_device_count",
        minimum=1,
    )
    if device_count != num_gpus:
        raise RuntimeReceiptValidationError(
            f"Cosmos visible GPU count mismatch: expected={num_gpus}, actual={device_count}"
        )
    torch_module_path = _canonical_reported_module_path(
        torch_record["module_path"],
        "Cosmos runtime probe.torch.module_path",
    )

    cosmos_record = _require_mapping(probe["cosmos_framework"], "Cosmos runtime probe.cosmos_framework")
    _require_exact_keys(cosmos_record, frozenset({"module_path"}), "Cosmos runtime probe.cosmos_framework")
    cosmos_module_path = _canonical_reported_module_path(
        cosmos_record["module_path"],
        "Cosmos runtime probe.cosmos_framework.module_path",
    )
    expected_cosmos_module = _canonical_path(
        cosmos_root / "cosmos_framework" / "__init__.py",
        "expected cosmos_framework module",
        kind="file",
    )
    if cosmos_module_path != expected_cosmos_module:
        raise RuntimeReceiptValidationError(
            "cosmos_framework resolved outside the exact Cosmos source root: "
            f"expected={expected_cosmos_module}, actual={cosmos_module_path}"
        )

    raw_gpus = probe["gpus"]
    if not isinstance(raw_gpus, list) or len(raw_gpus) != num_gpus:
        raise RuntimeReceiptValidationError(
            f"Cosmos GPU inventory length mismatch: expected={num_gpus}, "
            f"actual={len(raw_gpus) if isinstance(raw_gpus, list) else 'non-list'}"
        )
    gpus: list[dict[str, Any]] = []
    normalized_uuids: set[str] = set()
    for expected_index, raw_gpu in enumerate(raw_gpus):
        gpu = _require_mapping(raw_gpu, f"Cosmos GPU {expected_index}")
        _require_exact_keys(
            gpu,
            frozenset({"index", "name", "uuid", "capability", "total_memory_bytes"}),
            f"Cosmos GPU {expected_index}",
        )
        index = _require_int(gpu["index"], f"Cosmos GPU {expected_index}.index")
        if index != expected_index:
            raise RuntimeReceiptValidationError(
                f"Cosmos GPU indices must be contiguous: expected={expected_index}, actual={index}"
            )
        name = _require_string(gpu["name"], f"Cosmos GPU {expected_index}.name")
        if "H200" not in name.upper():
            raise RuntimeReceiptValidationError(f"Cosmos GPU {expected_index} is not H200: {name!r}")
        normalized_uuid = _normalize_gpu_uuid(gpu["uuid"], f"Cosmos GPU {expected_index}.uuid")
        if normalized_uuid in normalized_uuids:
            raise RuntimeReceiptValidationError(f"Cosmos GPU UUID is duplicated: {gpu['uuid']!r}")
        normalized_uuids.add(normalized_uuid)
        capability = gpu["capability"]
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or any(type(item) is not int or item < 0 for item in capability)
        ):
            raise RuntimeReceiptValidationError(f"Cosmos GPU {expected_index}.capability must be [major, minor]")
        if capability != [9, 0]:
            raise RuntimeReceiptValidationError(
                f"Cosmos H200 GPU {expected_index} must report compute capability [9, 0], got {capability!r}"
            )
        total_memory = _require_int(
            gpu["total_memory_bytes"],
            f"Cosmos GPU {expected_index}.total_memory_bytes",
            minimum=1,
        )
        gpus.append(
            {
                "index": index,
                "name": name,
                "uuid": _require_string(gpu["uuid"], f"Cosmos GPU {expected_index}.uuid"),
                "normalized_uuid": normalized_uuid,
                "capability": capability,
                "total_memory_bytes": total_memory,
            }
        )
    return {
        "python": python_record,
        "torch": {
            "version": torch_version,
            "cuda_version": torch_cuda,
            "cuda_available": True,
            "cuda_device_count": device_count,
            "module_path": str(torch_module_path),
        },
        "cosmos_framework": {"module_path": str(cosmos_module_path)},
        "gpus": gpus,
    }


def _validate_robocasa_probe(
    value: object,
    *,
    requested_python: Path,
    demian_root: Path,
    robocasa_root: Path,
    robosuite_root: Path,
) -> dict[str, Any]:
    probe = _require_mapping(value, "RoboCasa runtime probe")
    _require_exact_keys(
        probe,
        frozenset({"python", "packages", "demian_evaluator"}),
        "RoboCasa runtime probe",
    )
    python_record = _validate_python_record(probe["python"], "RoboCasa runtime probe.python", requested_python)
    packages = _require_mapping(probe["packages"], "RoboCasa runtime probe.packages")
    expected_packages = frozenset({"numpy", "mujoco", "gymnasium", "robocasa", "robosuite"})
    _require_exact_keys(packages, expected_packages, "RoboCasa runtime probe.packages")

    normalized_packages: dict[str, Any] = {}
    for name in sorted(expected_packages):
        package = _require_mapping(packages[name], f"RoboCasa runtime probe.packages.{name}")
        _require_exact_keys(
            package,
            frozenset({"distribution_version", "module_reported_version", "version_source", "module_path"}),
            f"RoboCasa runtime probe.packages.{name}",
        )
        distribution_version = _require_string(
            package["distribution_version"],
            f"RoboCasa runtime probe.packages.{name}.distribution_version",
        )
        reported_version = package["module_reported_version"]
        if reported_version is not None:
            reported_version = _require_string(
                reported_version,
                f"RoboCasa runtime probe.packages.{name}.module_reported_version",
            )
        if package["version_source"] != "importlib.metadata":
            raise RuntimeReceiptValidationError(
                f"RoboCasa runtime probe.packages.{name}.version_source must be importlib.metadata"
            )
        module_path = _canonical_reported_module_path(
            package["module_path"],
            f"RoboCasa runtime probe.packages.{name}.module_path",
        )
        normalized_packages[name] = {
            "distribution_version": distribution_version,
            "module_reported_version": reported_version,
            "version_source": "importlib.metadata",
            "module_path": str(module_path),
        }

    expected_source_modules = {
        "robocasa": _canonical_path(
            robocasa_root / "robocasa" / "__init__.py",
            "expected robocasa module",
            kind="file",
        ),
        "robosuite": _canonical_path(
            robosuite_root / "robosuite" / "__init__.py",
            "expected robosuite module",
            kind="file",
        ),
    }
    for name, expected_module in expected_source_modules.items():
        actual_module = Path(normalized_packages[name]["module_path"])
        if actual_module != expected_module:
            raise RuntimeReceiptValidationError(
                f"{name} resolved outside its exact source root: expected={expected_module}, actual={actual_module}"
            )

    demian_evaluator = _require_mapping(
        probe["demian_evaluator"],
        "RoboCasa runtime probe.demian_evaluator",
    )
    _require_exact_keys(
        demian_evaluator,
        frozenset({"module_path"}),
        "RoboCasa runtime probe.demian_evaluator",
    )
    demian_evaluator_path = _canonical_reported_module_path(
        demian_evaluator["module_path"],
        "RoboCasa runtime probe.demian_evaluator.module_path",
    )
    expected_demian_evaluator = _canonical_path(
        demian_root / "demian_wam" / "eval_utils" / "run_robocasa_eval.py",
        "expected DeMiAn evaluator module",
        kind="file",
    )
    if demian_evaluator_path != expected_demian_evaluator:
        raise RuntimeReceiptValidationError(
            "DeMiAn evaluator resolved outside the exact DeMiAn source root: "
            f"expected={expected_demian_evaluator}, actual={demian_evaluator_path}"
        )

    return {
        "python": python_record,
        "packages": normalized_packages,
        "demian_evaluator": {"module_path": str(demian_evaluator_path)},
    }


def _validate_robocasa_egl_probe(
    value: object,
    *,
    shard_index: int,
    robosuite_root: Path,
) -> dict[str, Any]:
    label = f"RoboCasa EGL probe shard {shard_index}"
    probe = _require_mapping(value, label)
    _require_exact_keys(
        probe,
        frozenset(
            {
                "status",
                "shard_index",
                "cuda_visible_devices",
                "mujoco_egl_device_id",
                "mujoco_gl",
                "pyopengl_platform",
                "ld_library_path",
                "egl_vendor_library_dirs",
                "robosuite_module_path",
            }
        ),
        label,
    )
    if probe["status"] != "passed" or probe["shard_index"] != shard_index:
        raise RuntimeReceiptValidationError(f"{label} did not pass with the expected shard index")
    expected_device = str(shard_index)
    expected_values = {
        "cuda_visible_devices": expected_device,
        "mujoco_egl_device_id": expected_device,
        "mujoco_gl": "egl",
        "pyopengl_platform": "egl",
        "ld_library_path": H_DRIVER_LIBRARY_PATH,
    }
    for key, expected in expected_values.items():
        if probe[key] != expected:
            raise RuntimeReceiptValidationError(f"{label}.{key} mismatch: expected={expected!r}, actual={probe[key]!r}")
    egl_vendor_library_dirs = probe["egl_vendor_library_dirs"]
    if egl_vendor_library_dirs is not None:
        egl_vendor_library_dirs = _require_string(
            egl_vendor_library_dirs,
            f"{label}.egl_vendor_library_dirs",
        )
    robosuite_module_path = _canonical_reported_module_path(
        probe["robosuite_module_path"],
        f"{label}.robosuite_module_path",
    )
    expected_robosuite_module = _canonical_path(
        robosuite_root / "robosuite" / "__init__.py",
        f"{label} expected robosuite module",
        kind="file",
    )
    if robosuite_module_path != expected_robosuite_module:
        raise RuntimeReceiptValidationError(
            f"{label} imported robosuite outside the exact source root: "
            f"expected={expected_robosuite_module}, actual={robosuite_module_path}"
        )
    return {
        "status": "passed",
        "shard_index": shard_index,
        **expected_values,
        "egl_vendor_library_dirs": egl_vendor_library_dirs,
        "robosuite_module_path": str(robosuite_module_path),
    }


def _parse_nvidia_smi_inventory(stdout: str, *, num_gpus: int, executable: Path) -> dict[str, Any]:
    try:
        rows = list(csv.reader(stdout.splitlines(), skipinitialspace=True))
    except csv.Error as error:
        raise RuntimeReceiptValidationError(f"nvidia-smi returned malformed CSV: {error}") from error
    if len(rows) != num_gpus:
        raise RuntimeReceiptValidationError(f"nvidia-smi GPU count mismatch: expected={num_gpus}, actual={len(rows)}")
    gpus: list[dict[str, Any]] = []
    drivers: set[str] = set()
    normalized_uuids: set[str] = set()
    for expected_index, row in enumerate(rows):
        if len(row) != 4:
            raise RuntimeReceiptValidationError(
                f"nvidia-smi row {expected_index} must have four fields, got {len(row)}"
            )
        try:
            index = int(row[0].strip())
        except ValueError as error:
            raise RuntimeReceiptValidationError(
                f"nvidia-smi row {expected_index} has a non-integer index: {row[0]!r}"
            ) from error
        if index != expected_index:
            raise RuntimeReceiptValidationError(
                f"nvidia-smi indices must be contiguous: expected={expected_index}, actual={index}"
            )
        gpu_uuid = row[1].strip()
        normalized_uuid = _normalize_gpu_uuid(gpu_uuid, f"nvidia-smi GPU {index}.uuid")
        if normalized_uuid in normalized_uuids:
            raise RuntimeReceiptValidationError(f"nvidia-smi GPU UUID is duplicated: {gpu_uuid!r}")
        normalized_uuids.add(normalized_uuid)
        name = _require_string(row[2].strip(), f"nvidia-smi GPU {index}.name")
        if "H200" not in name.upper():
            raise RuntimeReceiptValidationError(f"nvidia-smi GPU {index} is not H200: {name!r}")
        driver = row[3].strip()
        if _DRIVER_VERSION_RE.fullmatch(driver) is None:
            raise RuntimeReceiptValidationError(f"nvidia-smi GPU {index} has an invalid driver version: {driver!r}")
        drivers.add(driver)
        gpus.append(
            {
                "index": index,
                "uuid": gpu_uuid,
                "normalized_uuid": normalized_uuid,
                "name": name,
                "driver_version": driver,
            }
        )
    if len(drivers) != 1:
        raise RuntimeReceiptValidationError(f"nvidia-smi GPUs disagree on driver version: {sorted(drivers)}")
    return {
        "executable": str(executable),
        "driver_version": next(iter(drivers)),
        "gpus": gpus,
    }


def _cross_validate_gpu_inventories(
    cosmos_gpus: Sequence[Mapping[str, Any]],
    nvidia_gpus: Sequence[Mapping[str, Any]],
) -> None:
    cosmos_by_uuid = {gpu["normalized_uuid"]: gpu for gpu in cosmos_gpus}
    nvidia_by_uuid = {gpu["normalized_uuid"]: gpu for gpu in nvidia_gpus}
    if set(cosmos_by_uuid) != set(nvidia_by_uuid):
        raise RuntimeReceiptValidationError(
            "PyTorch and nvidia-smi GPU UUID sets differ: "
            f"torch={sorted(cosmos_by_uuid)}, nvidia_smi={sorted(nvidia_by_uuid)}"
        )
    for gpu_uuid in sorted(cosmos_by_uuid):
        torch_name = " ".join(str(cosmos_by_uuid[gpu_uuid]["name"]).split()).casefold()
        nvidia_name = " ".join(str(nvidia_by_uuid[gpu_uuid]["name"]).split()).casefold()
        if torch_name != nvidia_name:
            raise RuntimeReceiptValidationError(
                f"PyTorch and nvidia-smi GPU names differ for UUID {gpu_uuid}: "
                f"torch={cosmos_by_uuid[gpu_uuid]['name']!r}, "
                f"nvidia_smi={nvidia_by_uuid[gpu_uuid]['name']!r}"
            )


def _utc_timestamp(clock: Callable[[], datetime]) -> str:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise RuntimeReceiptValidationError("capture clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json_atomic_exclusive(path: Path, payload: object) -> str:
    """Publish complete read-only JSON atomically, without following links."""

    target = _lexical_absolute(path, "output path")
    parent = _canonical_path(target.parent, "output parent", kind="directory")
    if target.parent != parent:
        raise RuntimeReceiptValidationError(f"output parent must be a canonical non-symlink directory: {target.parent}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    # Defense in depth: ensure our serializer emitted one strict JSON document.
    _strict_json_bytes(encoded, "serialized receipt")
    digest = hashlib.sha256(encoded).hexdigest()

    parent_identity = _directory_identity(parent.stat())
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_descriptor = os.open(parent, directory_flags)
    temporary_name = f".{target.name}.{uuid.uuid4().hex}.tmp"
    temporary_created = False
    target_linked = False
    try:
        if _directory_identity(os.fstat(directory_descriptor)) != parent_identity:
            raise RuntimeReceiptValidationError(f"output parent changed before publication: {parent}")
        try:
            os.stat(target.name, dir_fd=directory_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise RuntimeReceiptValidationError(f"output already exists; refusing to overwrite: {target}")

        open_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary_name, open_flags, 0o600, dir_fd=directory_descriptor)
        temporary_created = True
        try:
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written <= 0:
                    raise RuntimeReceiptValidationError("short write while creating runtime receipt")
                offset += written
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(
                temporary_name,
                target.name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            target_linked = True
        except FileExistsError as error:
            raise RuntimeReceiptValidationError(
                f"output appeared concurrently; refusing to overwrite: {target}"
            ) from error
        os.unlink(temporary_name, dir_fd=directory_descriptor)
        temporary_created = False
        os.fsync(directory_descriptor)
        try:
            current_parent_identity = _directory_identity(parent.stat())
        except OSError as error:
            raise RuntimeReceiptValidationError(
                f"output parent became unavailable during publication: {parent}"
            ) from error
        if current_parent_identity != parent_identity:
            raise RuntimeReceiptValidationError(f"output parent changed during publication: {parent}")
    except Exception:
        if target_linked:
            try:
                os.unlink(target.name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass
        raise
    finally:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass
        os.close(directory_descriptor)
    return digest


def capture_runtime_receipt(
    *,
    cosmos_python: Path,
    robocasa_python: Path,
    cosmos_environment_manifest: Path,
    expected_cosmos_environment_manifest_sha256: str,
    robocasa_environment_manifest: Path,
    expected_robocasa_environment_manifest_sha256: str,
    cosmos_source_root: Path,
    expected_cosmos_commit: str,
    demian_source_root: Path,
    expected_demian_commit: str,
    robocasa_source_root: Path,
    expected_robocasa_commit: str,
    robosuite_source_root: Path,
    expected_robosuite_commit: str,
    experiment_id: str,
    logical_run_id: str,
    capture_nonce: str,
    num_gpus: int,
    output: Path,
    runner: CommandRunner = subprocess.run,
    executable_resolver: ExecutableResolver = shutil.which,
    timeout_seconds: float = COMMAND_TIMEOUT_SECONDS,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> tuple[dict[str, Any], str]:
    """Validate the complete live runtime and atomically publish its receipt."""

    if type(num_gpus) is not int or num_gpus < 1:
        raise RuntimeReceiptValidationError(f"num_gpus must be a positive integer, got {num_gpus!r}")
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
        raise RuntimeReceiptValidationError("timeout_seconds must be positive")
    experiment_id = _require_run_id(experiment_id, "experiment_id")
    logical_run_id = _require_run_id(logical_run_id, "logical_run_id")
    capture_nonce = _require_capture_nonce(capture_nonce)
    driver_library_path = os.environ.get("LD_LIBRARY_PATH")
    if driver_library_path != H_DRIVER_LIBRARY_PATH:
        raise RuntimeReceiptValidationError(
            "LD_LIBRARY_PATH must be the exact audited H driver allowlist: "
            f"expected={H_DRIVER_LIBRARY_PATH!r}, actual={driver_library_path!r}"
        )
    egl_vendor_library_dirs = os.environ.get("__EGL_VENDOR_LIBRARY_DIRS")
    if egl_vendor_library_dirs is not None:
        egl_vendor_library_dirs = _require_string(
            egl_vendor_library_dirs,
            "__EGL_VENDOR_LIBRARY_DIRS",
        )
    # Fail before launching any subprocess if the output target is obviously unsafe.
    output_path = _lexical_absolute(output, "output path")
    output_parent = _canonical_path(output_path.parent, "output parent", kind="directory")
    if output_path.parent != output_parent:
        raise RuntimeReceiptValidationError("output parent must be canonical and contain no symlinks")
    try:
        output_path.lstat()
    except FileNotFoundError:
        pass
    else:
        raise RuntimeReceiptValidationError(f"output already exists; refusing to overwrite: {output_path}")

    cosmos_python_launch, cosmos_python_resolved = _python_launch_and_resolved(
        cosmos_python,
        "Cosmos Python",
    )
    robocasa_python_launch, robocasa_python_resolved = _python_launch_and_resolved(
        robocasa_python,
        "RoboCasa Python",
    )

    cosmos_manifest = _prepare_environment_manifest(
        cosmos_environment_manifest,
        expected_cosmos_environment_manifest_sha256,
        "Cosmos environment manifest",
    )
    robocasa_manifest = _prepare_environment_manifest(
        robocasa_environment_manifest,
        expected_robocasa_environment_manifest_sha256,
        "RoboCasa environment manifest",
    )

    git = _resolve_executable("git", executable_resolver)
    nvidia_smi = _resolve_executable("nvidia-smi", executable_resolver)
    source_specs = (
        ("cosmos", cosmos_source_root, expected_cosmos_commit),
        ("demian", demian_source_root, expected_demian_commit),
        ("robocasa", robocasa_source_root, expected_robocasa_commit),
        ("robosuite", robosuite_source_root, expected_robosuite_commit),
    )
    sources = {
        name: _git_source_record(
            name=name,
            root=root,
            expected_commit=commit,
            git=git,
            runner=runner,
            timeout_seconds=float(timeout_seconds),
        )
        for name, root, commit in source_specs
    }
    cosmos_root = Path(sources["cosmos"]["root"])
    demian_root = Path(sources["demian"]["root"])
    robocasa_root = Path(sources["robocasa"]["root"])
    robosuite_root = Path(sources["robosuite"]["root"])
    manifest_helper = _canonical_path(
        cosmos_root / ENVIRONMENT_MANIFEST_HELPER_REL,
        "Python environment manifest helper",
        kind="file",
    )

    environment_verification = {
        "cosmos": _verify_environment(
            python=cosmos_python_launch,
            helper=manifest_helper,
            manifest=Path(cosmos_manifest["path"]),
            label="Cosmos",
            runner=runner,
            timeout_seconds=float(timeout_seconds),
        ),
        "robocasa": _verify_environment(
            python=robocasa_python_launch,
            helper=manifest_helper,
            manifest=Path(robocasa_manifest["path"]),
            label="RoboCasa",
            runner=runner,
            timeout_seconds=float(timeout_seconds),
        ),
    }

    cosmos_stdout = _run_command(
        [str(cosmos_python_launch), "-I", "-c", _COSMOS_PROBE, str(cosmos_root)],
        label="Cosmos runtime probe",
        runner=runner,
        timeout_seconds=float(timeout_seconds),
    )
    cosmos_runtime = _validate_cosmos_probe(
        _strict_json_text(cosmos_stdout, "Cosmos runtime probe stdout"),
        requested_python=cosmos_python_resolved,
        cosmos_root=cosmos_root,
        num_gpus=num_gpus,
    )

    robocasa_stdout = _run_command(
        [
            str(robocasa_python_launch),
            "-I",
            "-c",
            _ROBOCASA_PROBE,
            str(demian_root / "demian_wam"),
            str(robocasa_root),
            str(robosuite_root),
        ],
        label="RoboCasa runtime probe",
        runner=runner,
        timeout_seconds=float(timeout_seconds),
    )
    robocasa_runtime = _validate_robocasa_probe(
        _strict_json_text(robocasa_stdout, "RoboCasa runtime probe stdout"),
        requested_python=robocasa_python_resolved,
        demian_root=demian_root,
        robocasa_root=robocasa_root,
        robosuite_root=robosuite_root,
    )
    robocasa_egl = []
    for shard_index in range(num_gpus):
        egl_stdout = _run_command(
            [
                str(robocasa_python_launch),
                "-I",
                "-c",
                _ROBOCASA_EGL_PROBE,
                str(robosuite_root),
                str(shard_index),
            ],
            label=f"RoboCasa EGL probe shard {shard_index}",
            runner=runner,
            timeout_seconds=float(timeout_seconds),
            environment_overrides={
                "CUDA_VISIBLE_DEVICES": str(shard_index),
                "MUJOCO_EGL_DEVICE_ID": str(shard_index),
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
            },
        )
        robocasa_egl.append(
            _validate_robocasa_egl_probe(
                _strict_json_text(
                    egl_stdout,
                    f"RoboCasa EGL probe shard {shard_index} stdout",
                ),
                shard_index=shard_index,
                robosuite_root=robosuite_root,
            )
        )

    if environment_verification["cosmos"]["interpreter_resolved_path"] != cosmos_runtime["python"]["executable"]:
        raise RuntimeReceiptValidationError(
            "Cosmos environment verification and runtime probe used different interpreters"
        )
    if environment_verification["robocasa"]["interpreter_resolved_path"] != robocasa_runtime["python"]["executable"]:
        raise RuntimeReceiptValidationError(
            "RoboCasa environment verification and runtime probe used different interpreters"
        )

    nvidia_stdout = _run_command(
        [
            str(nvidia_smi),
            "--query-gpu=index,uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        label="nvidia-smi inventory",
        runner=runner,
        timeout_seconds=float(timeout_seconds),
    )
    nvidia_inventory = _parse_nvidia_smi_inventory(
        nvidia_stdout,
        num_gpus=num_gpus,
        executable=nvidia_smi,
    )
    _cross_validate_gpu_inventories(cosmos_runtime["gpus"], nvidia_inventory["gpus"])

    manifests_after = {
        "cosmos": _prepare_environment_manifest(
            cosmos_environment_manifest,
            expected_cosmos_environment_manifest_sha256,
            "Cosmos environment manifest",
        ),
        "robocasa": _prepare_environment_manifest(
            robocasa_environment_manifest,
            expected_robocasa_environment_manifest_sha256,
            "RoboCasa environment manifest",
        ),
    }
    manifests_before = {"cosmos": cosmos_manifest, "robocasa": robocasa_manifest}
    if manifests_after != manifests_before:
        raise RuntimeReceiptValidationError("environment manifest identity changed during runtime capture")

    environment_verification_after = {
        "cosmos": _verify_environment(
            python=cosmos_python_launch,
            helper=manifest_helper,
            manifest=Path(cosmos_manifest["path"]),
            label="Cosmos",
            runner=runner,
            timeout_seconds=float(timeout_seconds),
        ),
        "robocasa": _verify_environment(
            python=robocasa_python_launch,
            helper=manifest_helper,
            manifest=Path(robocasa_manifest["path"]),
            label="RoboCasa",
            runner=runner,
            timeout_seconds=float(timeout_seconds),
        ),
    }
    if environment_verification_after != environment_verification:
        raise RuntimeReceiptValidationError("Python environment identity changed during runtime capture")

    sources_after = {
        name: _git_source_record(
            name=name,
            root=root,
            expected_commit=commit,
            git=git,
            runner=runner,
            timeout_seconds=float(timeout_seconds),
        )
        for name, root, commit in source_specs
    }
    if sources_after != sources:
        raise RuntimeReceiptValidationError("source identity changed during runtime capture")

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": RECEIPT_KIND,
        "status": "passed",
        "captured_at_utc": _utc_timestamp(clock),
        "driver_environment": {
            "ld_library_path": driver_library_path,
            "egl_vendor_library_dirs": egl_vendor_library_dirs,
        },
        "inputs": {
            "experiment_id": experiment_id,
            "logical_run_id": logical_run_id,
            "capture_nonce": capture_nonce,
            "num_gpus": num_gpus,
            "interpreters": {
                "cosmos": {
                    "launch_path": str(cosmos_python_launch),
                    "resolved_path": str(cosmos_python_resolved),
                },
                "robocasa": {
                    "launch_path": str(robocasa_python_launch),
                    "resolved_path": str(robocasa_python_resolved),
                },
            },
            "environment_manifests": {
                "cosmos": cosmos_manifest,
                "robocasa": robocasa_manifest,
            },
            "sources": sources,
        },
        "environment_verification": environment_verification,
        "cosmos_runtime": cosmos_runtime,
        "robocasa_runtime": robocasa_runtime,
        "robocasa_egl": robocasa_egl,
        "nvidia_smi": nvidia_inventory,
        "validation": {
            "environment_manifests_match_live_interpreters": True,
            "source_commits_match_and_are_clean": True,
            "source_module_paths_match_exact_roots": True,
            "cuda_available": True,
            "gpu_count_matches": True,
            "all_gpus_are_h200_compute_capability_9_0": True,
            "torch_and_nvidia_smi_uuid_sets_match": True,
            "h_driver_library_path_is_exact": True,
            "robocasa_egl_contexts_created_per_shard": True,
        },
    }
    digest = write_json_atomic_exclusive(output_path, receipt)
    return receipt, digest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmos-python", type=Path, required=True)
    parser.add_argument("--robocasa-python", type=Path, required=True)
    parser.add_argument("--cosmos-environment-manifest", type=Path, required=True)
    parser.add_argument("--expected-cosmos-environment-manifest-sha256", required=True)
    parser.add_argument("--robocasa-environment-manifest", type=Path, required=True)
    parser.add_argument("--expected-robocasa-environment-manifest-sha256", required=True)
    parser.add_argument("--cosmos-source-root", type=Path, required=True)
    parser.add_argument("--expected-cosmos-commit", required=True)
    parser.add_argument("--demian-source-root", type=Path, required=True)
    parser.add_argument("--expected-demian-commit", required=True)
    parser.add_argument("--robocasa-source-root", type=Path, required=True)
    parser.add_argument("--expected-robocasa-commit", required=True)
    parser.add_argument("--robosuite-source-root", type=Path, required=True)
    parser.add_argument("--expected-robosuite-commit", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--logical-run-id", required=True)
    parser.add_argument("--capture-nonce", required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=COMMAND_TIMEOUT_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        receipt, digest = capture_runtime_receipt(
            cosmos_python=arguments.cosmos_python,
            robocasa_python=arguments.robocasa_python,
            cosmos_environment_manifest=arguments.cosmos_environment_manifest,
            expected_cosmos_environment_manifest_sha256=(arguments.expected_cosmos_environment_manifest_sha256),
            robocasa_environment_manifest=arguments.robocasa_environment_manifest,
            expected_robocasa_environment_manifest_sha256=(arguments.expected_robocasa_environment_manifest_sha256),
            cosmos_source_root=arguments.cosmos_source_root,
            expected_cosmos_commit=arguments.expected_cosmos_commit,
            demian_source_root=arguments.demian_source_root,
            expected_demian_commit=arguments.expected_demian_commit,
            robocasa_source_root=arguments.robocasa_source_root,
            expected_robocasa_commit=arguments.expected_robocasa_commit,
            robosuite_source_root=arguments.robosuite_source_root,
            expected_robosuite_commit=arguments.expected_robosuite_commit,
            experiment_id=arguments.experiment_id,
            logical_run_id=arguments.logical_run_id,
            capture_nonce=arguments.capture_nonce,
            num_gpus=arguments.num_gpus,
            output=arguments.output,
            timeout_seconds=arguments.timeout_seconds,
        )
    except RuntimeReceiptError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        "ROBOCASA365_EVAL_RUNTIME_RECEIPT_CAPTURED "
        f"gpus={len(receipt['cosmos_runtime']['gpus'])} "
        f"driver={receipt['nvidia_smi']['driver_version']} "
        f"sha256={digest} path={arguments.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
