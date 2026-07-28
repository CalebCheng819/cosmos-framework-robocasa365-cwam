#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Create and verify strict manifests for the active Python environment.

Run this helper *with the interpreter being attested*.  The manifest binds the
fully resolved interpreter file, its full SHA-256, the Python runtime version,
the canonical environment prefixes, and the complete distribution inventory
reported by :mod:`importlib.metadata`.

Virtual-environment launchers are commonly symlinks.  They are accepted only as
the process-launch alias: the alias is resolved, compared with ``/proc/self/exe``
when available, and never persisted as trusted identity.  Every path stored in
the manifest, and the manifest file consumed by ``verify``, must itself be
canonical and contain no symlink or ``.incomplete`` component.

The distribution inventory intentionally binds installed metadata names and
versions, not every importable source or shared-library byte.  Source snapshots,
container images, CUDA drivers, and native libraries require separate evidence.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import importlib.metadata
import json
import os
import re
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MANIFEST_KIND = "cwam_python_environment_manifest"
HASH_CHUNK_BYTES = 8 * 1024 * 1024
MAX_MANIFEST_BYTES = 32 * 1024 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DISTRIBUTION_NAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_CANONICAL_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
_IMPLEMENTATION_RE = re.compile(r"^[a-z0-9_]+$")
_CACHE_TAG_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_NORMALIZE_NAME_RE = re.compile(r"[-_.]+")

_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "interpreter",
        "distribution_count",
        "distributions",
        "distributions_sha256",
    }
)
_INTERPRETER_KEYS = frozenset(
    {
        "resolved_path",
        "size_bytes",
        "sha256",
        "python_version",
        "version_info",
        "implementation",
        "cache_tag",
        "prefix",
        "base_prefix",
    }
)
_VERSION_INFO_KEYS = frozenset({"major", "minor", "micro", "releaselevel", "serial"})
_DISTRIBUTION_KEYS = frozenset({"name", "version"})


class PythonEnvironmentManifestError(RuntimeError):
    """Raised when an environment manifest contract is violated."""


def _stable_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _reject_incomplete_path(path: Path, label: str) -> None:
    for component in path.parts:
        if ".incomplete" in component.lower():
            raise PythonEnvironmentManifestError(f"{label} contains a forbidden .incomplete component: {component!r}")
        if any(ord(character) < 32 or ord(character) == 127 for character in component):
            raise PythonEnvironmentManifestError(f"{label} contains a control character: {component!r}")


def _lexical_absolute_path(path: str | os.PathLike[str], label: str) -> Path:
    raw_path = os.fspath(path)
    if not raw_path:
        raise PythonEnvironmentManifestError(f"{label} must not be empty")
    lexical = Path(os.path.abspath(os.path.expanduser(raw_path)))
    _reject_incomplete_path(lexical, label)
    return lexical


def _require_canonical_path(
    path: str | os.PathLike[str],
    label: str,
    *,
    expected_kind: str,
) -> tuple[Path, os.stat_result]:
    lexical = _lexical_absolute_path(path, label)
    try:
        resolved = lexical.resolve(strict=True)
    except FileNotFoundError as error:
        raise PythonEnvironmentManifestError(f"{label} does not exist: {lexical}") from error
    except (OSError, RuntimeError) as error:
        raise PythonEnvironmentManifestError(f"cannot resolve {label} {lexical}: {error}") from error
    if lexical != resolved:
        raise PythonEnvironmentManifestError(
            f"{label} must be canonical and contain no symlink components: provided={lexical}, resolved={resolved}"
        )
    _reject_incomplete_path(resolved, label)
    try:
        metadata = resolved.lstat()
    except OSError as error:
        raise PythonEnvironmentManifestError(f"cannot inspect {label} {resolved}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise PythonEnvironmentManifestError(f"{label} must not be a symlink: {resolved}")
    if expected_kind == "file" and not stat.S_ISREG(metadata.st_mode):
        raise PythonEnvironmentManifestError(f"{label} must be a regular file: {resolved}")
    if expected_kind == "directory" and not stat.S_ISDIR(metadata.st_mode):
        raise PythonEnvironmentManifestError(f"{label} must be a directory: {resolved}")
    return resolved, metadata


def _hash_canonical_file(path: Path, label: str) -> tuple[int, str]:
    canonical, before = _require_canonical_path(path, label, expected_kind="file")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(canonical, flags)
    except OSError as error:
        raise PythonEnvironmentManifestError(f"cannot safely open {label} {canonical}: {error}") from error

    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise PythonEnvironmentManifestError(f"{label} must be a regular file: {canonical}")
        if _stable_identity(before) != _stable_identity(opened):
            raise PythonEnvironmentManifestError(f"{label} changed before hashing: {canonical}")
        while chunk := os.read(descriptor, HASH_CHUNK_BYTES):
            digest.update(chunk)
        after_read = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    _, after_path = _require_canonical_path(canonical, label, expected_kind="file")
    identity = _stable_identity(before)
    if identity != _stable_identity(after_read) or identity != _stable_identity(after_path):
        raise PythonEnvironmentManifestError(f"{label} changed while hashing: {canonical}")
    return int(before.st_size), digest.hexdigest()


def _current_interpreter_path() -> Path:
    if not sys.executable:
        raise PythonEnvironmentManifestError("sys.executable is empty")
    launch_path = _lexical_absolute_path(sys.executable, "interpreter launch path")
    try:
        resolved = launch_path.resolve(strict=True)
    except FileNotFoundError as error:
        raise PythonEnvironmentManifestError(f"interpreter launch path does not exist: {launch_path}") from error
    except (OSError, RuntimeError) as error:
        raise PythonEnvironmentManifestError(
            f"cannot resolve interpreter launch path {launch_path}: {error}"
        ) from error
    _reject_incomplete_path(resolved, "resolved interpreter path")
    canonical, metadata = _require_canonical_path(
        resolved,
        "resolved interpreter path",
        expected_kind="file",
    )
    if metadata.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH) == 0:
        raise PythonEnvironmentManifestError(f"resolved interpreter is not executable: {canonical}")

    proc_executable = Path("/proc/self/exe")
    if proc_executable.exists():
        try:
            proc_resolved = proc_executable.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise PythonEnvironmentManifestError(f"cannot resolve /proc/self/exe: {error}") from error
        if proc_resolved != canonical:
            raise PythonEnvironmentManifestError(
                "sys.executable does not resolve to the running /proc/self/exe: "
                f"sys.executable={canonical}, proc={proc_resolved}"
            )
    return canonical


def _runtime_directory(path: str, label: str) -> str:
    canonical, _ = _require_canonical_path(path, label, expected_kind="directory")
    return str(canonical)


def _clean_metadata_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PythonEnvironmentManifestError(f"{label} must be a non-empty, trimmed string")
    if not value.isascii():
        raise PythonEnvironmentManifestError(f"{label} must contain only ASCII characters: {value!r}")
    if any(ord(character) < 33 or ord(character) == 127 for character in value):
        raise PythonEnvironmentManifestError(f"{label} contains whitespace/control characters: {value!r}")
    return value


def _canonical_distribution_name(name: object, label: str) -> tuple[str, str]:
    raw_name = _clean_metadata_string(name, label)
    if _DISTRIBUTION_NAME_RE.fullmatch(raw_name) is None:
        raise PythonEnvironmentManifestError(f"{label} is not a valid distribution name: {raw_name!r}")
    canonical = _NORMALIZE_NAME_RE.sub("-", raw_name).lower()
    if _CANONICAL_NAME_RE.fullmatch(canonical) is None:
        raise PythonEnvironmentManifestError(f"{label} cannot be canonicalized safely: {raw_name!r}")
    return canonical, raw_name


def _collect_distribution_map_once() -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    try:
        distributions = list(importlib.metadata.distributions())
    except Exception as error:
        raise PythonEnvironmentManifestError(f"cannot enumerate importlib.metadata distributions: {error}") from error

    for index, distribution in enumerate(distributions):
        try:
            raw_name_value = distribution.metadata.get("Name")
            raw_version_value = distribution.version
        except Exception as error:
            raise PythonEnvironmentManifestError(
                f"cannot inspect importlib.metadata distribution {index}: {error}"
            ) from error
        canonical_name, raw_name = _canonical_distribution_name(
            raw_name_value,
            f"distribution[{index}].Name",
        )
        version = _clean_metadata_string(raw_version_value, f"distribution[{index}].Version")
        if canonical_name in records:
            previous = records[canonical_name]
            raise PythonEnvironmentManifestError(
                "importlib.metadata contains a duplicate canonical distribution name: "
                f"{canonical_name!r}, first={previous!r}, duplicate={{'name': {raw_name!r}, "
                f"'version': {version!r}}}"
            )
        records[canonical_name] = {"name": raw_name, "version": version}
    return {name: records[name] for name in sorted(records)}


def _distributions_sha256(distributions: Mapping[str, Mapping[str, str]]) -> str:
    payload = json.dumps(
        distributions,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _version_info_record() -> dict[str, str | int]:
    version = sys.version_info
    return {
        "major": int(version.major),
        "minor": int(version.minor),
        "micro": int(version.micro),
        "releaselevel": str(version.releaselevel),
        "serial": int(version.serial),
    }


def _collect_interpreter_record() -> dict[str, Any]:
    interpreter = _current_interpreter_path()
    size_bytes, sha256 = _hash_canonical_file(interpreter, "resolved interpreter")
    version_info = _version_info_record()
    cache_tag = sys.implementation.cache_tag
    if not isinstance(cache_tag, str) or _CACHE_TAG_RE.fullmatch(cache_tag) is None:
        raise PythonEnvironmentManifestError(f"sys.implementation.cache_tag is invalid: {cache_tag!r}")
    implementation = sys.implementation.name
    if not isinstance(implementation, str) or _IMPLEMENTATION_RE.fullmatch(implementation) is None:
        raise PythonEnvironmentManifestError(f"sys.implementation.name is invalid: {implementation!r}")
    return {
        "resolved_path": str(interpreter),
        "size_bytes": size_bytes,
        "sha256": sha256,
        "python_version": (f"{version_info['major']}.{version_info['minor']}.{version_info['micro']}"),
        "version_info": version_info,
        "implementation": implementation,
        "cache_tag": cache_tag,
        "prefix": _runtime_directory(sys.prefix, "sys.prefix"),
        "base_prefix": _runtime_directory(sys.base_prefix, "sys.base_prefix"),
    }


def create_python_environment_manifest() -> dict[str, Any]:
    """Capture a stable manifest for the Python process executing this helper."""

    interpreter_before = _collect_interpreter_record()
    distributions_before = _collect_distribution_map_once()
    distributions_after = _collect_distribution_map_once()
    interpreter_after = _collect_interpreter_record()
    if interpreter_before != interpreter_after:
        raise PythonEnvironmentManifestError("interpreter identity changed while capturing the environment")
    if distributions_before != distributions_after:
        raise PythonEnvironmentManifestError("distribution inventory changed while capturing the environment")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "interpreter": interpreter_after,
        "distribution_count": len(distributions_after),
        "distributions": distributions_after,
        "distributions_sha256": _distributions_sha256(distributions_after),
    }


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted((repr(key) for key in observed - expected))
        raise PythonEnvironmentManifestError(f"{label} keys mismatch: missing={missing}, extra={extra}")


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PythonEnvironmentManifestError(f"{label} must be a JSON object")
    return value


def _require_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise PythonEnvironmentManifestError(f"{label} must be a non-negative integer, got {value!r}")
    return value


def _require_positive_int(value: object, label: str) -> int:
    integer = _require_nonnegative_int(value, label)
    if integer == 0:
        raise PythonEnvironmentManifestError(f"{label} must be positive")
    return integer


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PythonEnvironmentManifestError(f"{label} must be a lowercase SHA-256, got {value!r}")
    return value


def _validated_version_info(value: object) -> dict[str, str | int]:
    version = _require_mapping(value, "manifest.interpreter.version_info")
    _require_exact_keys(version, _VERSION_INFO_KEYS, "manifest.interpreter.version_info")
    major = _require_nonnegative_int(version["major"], "manifest.interpreter.version_info.major")
    minor = _require_nonnegative_int(version["minor"], "manifest.interpreter.version_info.minor")
    micro = _require_nonnegative_int(version["micro"], "manifest.interpreter.version_info.micro")
    serial = _require_nonnegative_int(version["serial"], "manifest.interpreter.version_info.serial")
    releaselevel = version["releaselevel"]
    if releaselevel not in {"alpha", "beta", "candidate", "final"}:
        raise PythonEnvironmentManifestError(
            "manifest.interpreter.version_info.releaselevel must be alpha, beta, candidate, or final"
        )
    return {
        "major": major,
        "minor": minor,
        "micro": micro,
        "releaselevel": releaselevel,
        "serial": serial,
    }


def _validated_interpreter(value: object) -> dict[str, Any]:
    interpreter = _require_mapping(value, "manifest.interpreter")
    _require_exact_keys(interpreter, _INTERPRETER_KEYS, "manifest.interpreter")

    resolved_path_value = interpreter["resolved_path"]
    if not isinstance(resolved_path_value, str):
        raise PythonEnvironmentManifestError("manifest.interpreter.resolved_path must be a string")
    resolved_path, _ = _require_canonical_path(
        resolved_path_value,
        "manifest.interpreter.resolved_path",
        expected_kind="file",
    )
    size_bytes = _require_positive_int(interpreter["size_bytes"], "manifest.interpreter.size_bytes")
    sha256 = _require_sha256(interpreter["sha256"], "manifest.interpreter.sha256")
    version_info = _validated_version_info(interpreter["version_info"])
    expected_python_version = f"{version_info['major']}.{version_info['minor']}.{version_info['micro']}"
    python_version = interpreter["python_version"]
    if python_version != expected_python_version:
        raise PythonEnvironmentManifestError(
            "manifest.interpreter.python_version does not match version_info: "
            f"expected={expected_python_version!r}, actual={python_version!r}"
        )
    implementation = interpreter["implementation"]
    if not isinstance(implementation, str) or _IMPLEMENTATION_RE.fullmatch(implementation) is None:
        raise PythonEnvironmentManifestError("manifest.interpreter.implementation is invalid")
    cache_tag = interpreter["cache_tag"]
    if not isinstance(cache_tag, str) or _CACHE_TAG_RE.fullmatch(cache_tag) is None:
        raise PythonEnvironmentManifestError("manifest.interpreter.cache_tag is invalid")

    prefix_value = interpreter["prefix"]
    base_prefix_value = interpreter["base_prefix"]
    if not isinstance(prefix_value, str) or not isinstance(base_prefix_value, str):
        raise PythonEnvironmentManifestError("manifest interpreter prefixes must be strings")
    prefix, _ = _require_canonical_path(
        prefix_value,
        "manifest.interpreter.prefix",
        expected_kind="directory",
    )
    base_prefix, _ = _require_canonical_path(
        base_prefix_value,
        "manifest.interpreter.base_prefix",
        expected_kind="directory",
    )
    return {
        "resolved_path": str(resolved_path),
        "size_bytes": size_bytes,
        "sha256": sha256,
        "python_version": python_version,
        "version_info": version_info,
        "implementation": implementation,
        "cache_tag": cache_tag,
        "prefix": str(prefix),
        "base_prefix": str(base_prefix),
    }


def _validated_distributions(value: object) -> dict[str, dict[str, str]]:
    distributions = _require_mapping(value, "manifest.distributions")
    validated: dict[str, dict[str, str]] = {}
    for canonical_name, raw_record in distributions.items():
        if not isinstance(canonical_name, str) or _CANONICAL_NAME_RE.fullmatch(canonical_name) is None:
            raise PythonEnvironmentManifestError(f"manifest distribution key is not canonical: {canonical_name!r}")
        record = _require_mapping(raw_record, f"manifest.distributions[{canonical_name!r}]")
        _require_exact_keys(
            record,
            _DISTRIBUTION_KEYS,
            f"manifest.distributions[{canonical_name!r}]",
        )
        computed_name, raw_name = _canonical_distribution_name(
            record["name"],
            f"manifest.distributions[{canonical_name!r}].name",
        )
        if computed_name != canonical_name:
            raise PythonEnvironmentManifestError(
                f"manifest distribution key/name mismatch: key={canonical_name!r}, name={raw_name!r}"
            )
        version = _clean_metadata_string(
            record["version"],
            f"manifest.distributions[{canonical_name!r}].version",
        )
        validated[canonical_name] = {"name": raw_name, "version": version}
    return {name: validated[name] for name in sorted(validated)}


def _validated_manifest(value: object) -> dict[str, Any]:
    manifest = _require_mapping(value, "manifest")
    _require_exact_keys(manifest, _MANIFEST_KEYS, "manifest")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != SCHEMA_VERSION:
        raise PythonEnvironmentManifestError(
            f"manifest.schema_version must be {SCHEMA_VERSION}, got {manifest['schema_version']!r}"
        )
    if manifest["kind"] != MANIFEST_KIND:
        raise PythonEnvironmentManifestError(f"manifest.kind must be {MANIFEST_KIND!r}, got {manifest['kind']!r}")
    interpreter = _validated_interpreter(manifest["interpreter"])
    distributions = _validated_distributions(manifest["distributions"])
    distribution_count = _require_nonnegative_int(
        manifest["distribution_count"],
        "manifest.distribution_count",
    )
    if distribution_count != len(distributions):
        raise PythonEnvironmentManifestError(
            f"manifest.distribution_count mismatch: expected={len(distributions)}, actual={distribution_count}"
        )
    expected_inventory_sha = _require_sha256(
        manifest["distributions_sha256"],
        "manifest.distributions_sha256",
    )
    actual_inventory_sha = _distributions_sha256(distributions)
    if expected_inventory_sha != actual_inventory_sha:
        raise PythonEnvironmentManifestError(
            "manifest.distributions_sha256 mismatch: "
            f"expected={expected_inventory_sha}, computed={actual_inventory_sha}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "interpreter": interpreter,
        "distribution_count": distribution_count,
        "distributions": distributions,
        "distributions_sha256": actual_inventory_sha,
    }


def _inventory_drift(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> str:
    expected_distributions = expected["distributions"]
    actual_distributions = actual["distributions"]
    missing = sorted(set(expected_distributions) - set(actual_distributions))
    extra = sorted(set(actual_distributions) - set(expected_distributions))
    changed = sorted(
        name
        for name in set(expected_distributions) & set(actual_distributions)
        if expected_distributions[name] != actual_distributions[name]
    )
    interpreter_fields = sorted(
        key for key in expected["interpreter"] if expected["interpreter"][key] != actual["interpreter"].get(key)
    )
    return (
        f"interpreter_fields={interpreter_fields}, missing={missing[:10]}, extra={extra[:10]}, changed={changed[:10]}"
    )


def verify_python_environment_manifest(manifest: object) -> dict[str, Any]:
    """Verify that the live Python process exactly matches ``manifest``."""

    expected = _validated_manifest(manifest)
    actual = create_python_environment_manifest()
    if actual != expected:
        raise PythonEnvironmentManifestError(
            f"live Python environment does not match the manifest: {_inventory_drift(expected, actual)}"
        )
    return {
        "status": "passed",
        "interpreter_resolved_path": actual["interpreter"]["resolved_path"],
        "interpreter_sha256": actual["interpreter"]["sha256"],
        "python_version": actual["interpreter"]["python_version"],
        "distribution_count": actual["distribution_count"],
        "distributions_sha256": actual["distributions_sha256"],
    }


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PythonEnvironmentManifestError(f"manifest JSON contains a duplicate object key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise PythonEnvironmentManifestError(f"manifest JSON contains a forbidden constant: {value}")


def _read_manifest_bytes(path: str | os.PathLike[str]) -> bytes:
    canonical, before = _require_canonical_path(path, "manifest path", expected_kind="file")
    if before.st_size > MAX_MANIFEST_BYTES:
        raise PythonEnvironmentManifestError(f"manifest exceeds {MAX_MANIFEST_BYTES} bytes: {canonical}")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(canonical, flags)
    except OSError as error:
        raise PythonEnvironmentManifestError(f"cannot safely open manifest {canonical}: {error}") from error
    try:
        opened = os.fstat(descriptor)
        if _stable_identity(before) != _stable_identity(opened):
            raise PythonEnvironmentManifestError(f"manifest changed before reading: {canonical}")
        chunks: list[bytes] = []
        remaining = MAX_MANIFEST_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(HASH_CHUNK_BYTES, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after_read = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(payload) > MAX_MANIFEST_BYTES:
        raise PythonEnvironmentManifestError(f"manifest exceeds {MAX_MANIFEST_BYTES} bytes: {canonical}")
    _, after_path = _require_canonical_path(canonical, "manifest path", expected_kind="file")
    identity = _stable_identity(before)
    if identity != _stable_identity(after_read) or identity != _stable_identity(after_path):
        raise PythonEnvironmentManifestError(f"manifest changed while reading: {canonical}")
    return payload


def load_python_environment_manifest(
    manifest_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Load a strict UTF-8 JSON manifest without following symlinks."""

    payload = _read_manifest_bytes(manifest_path)
    if payload.startswith(codecs.BOM_UTF8):
        raise PythonEnvironmentManifestError("manifest JSON must not contain a UTF-8 BOM")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise PythonEnvironmentManifestError(f"manifest is not valid UTF-8: {error}") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise PythonEnvironmentManifestError(f"manifest is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise PythonEnvironmentManifestError("manifest JSON root must be an object")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("create", help="print a manifest for this Python environment")
    verify = subparsers.add_parser("verify", help="verify this Python environment")
    verify.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "create":
            value = create_python_environment_manifest()
        else:
            manifest = load_python_environment_manifest(args.manifest)
            value = verify_python_environment_manifest(manifest)
    except PythonEnvironmentManifestError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
