#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Fail-closed input binding for the RoboCasa365 Cosmos3 formal continuation.

The continuation is deliberately derived from three immutable inputs:

* one published Wan latent cache;
* one passed throughput matrix and its ``best_observed_cell_id``; and
* the complete R6 ``iter_000001000`` model/optimizer/scheduler/trainer DCP.

This module validates those inputs, records their cryptographic identities, and
creates a fresh run directory whose only initial checkpoint is an absolute
symlink to the audited R6 checkpoint.  It never consumes the throughput probe's
fixed-index trace: that trace exists only to compare benchmark cells fairly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
CHECKPOINT_ITERATION = 1_000
FINAL_ITERATION = 10_000
CHECKPOINT_COMPONENTS = ("model", "optim", "scheduler", "trainer")
CHECKPOINT_FINGERPRINT_ALGORITHM = "sha256-full-dcp-file-inventory-v1"
CELL_ID_PATTERN = re.compile(
    r"^workers-(?P<workers>4|8|12|16)__storage-(?P<storage>gpfs|nvme)"
    r"__ac-(?P<activation_checkpointing>selective|none)$"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ContinuationError(ValueError):
    """Raised when a formal-continuation input violates its immutable contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ContinuationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ContinuationError(f"expected a regular non-symlink file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    after = path.stat()
    if (
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ContinuationError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _load_json_and_sha256(path: Path, label: str) -> tuple[dict[str, Any], str]:
    digest = _sha256_file(path)
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContinuationError(f"{label} is not valid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContinuationError(f"{label} must contain one JSON object: {path}")
    return value, digest


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContinuationError(f"{label} must be an object")
    return value


def _canonical_directory(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(path))
    if path != absolute or path.is_symlink() or not path.is_dir():
        raise ContinuationError(f"{label} must be an absolute non-symlink directory: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ContinuationError(f"{label} must already be canonical: {path}")
    return resolved


def _safe_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContinuationError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or str(path) != value:
        raise ContinuationError(f"{label} is not a safe normalized relative path: {value!r}")
    return value


def validate_cache_binding(
    root: Path,
    *,
    expected_success_sha256: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Bind the exact completion marker and manifest selected for training."""

    root = _canonical_directory(root, "latent-cache root")
    expected_success_sha256 = _require_sha256(
        expected_success_sha256,
        "expected cache _SUCCESS SHA-256",
    )
    expected_manifest_sha256 = _require_sha256(
        expected_manifest_sha256,
        "expected cache manifest SHA-256",
    )
    success_path = root / "_SUCCESS.json"
    manifest_path = root / "manifest.json"
    success, success_sha256 = _load_json_and_sha256(success_path, "cache _SUCCESS")
    manifest, manifest_sha256 = _load_json_and_sha256(manifest_path, "cache manifest")
    if success_sha256 != expected_success_sha256:
        raise ContinuationError(
            "cache _SUCCESS SHA-256 mismatch: "
            f"expected={expected_success_sha256}, actual={success_sha256}"
        )
    if manifest_sha256 != expected_manifest_sha256:
        raise ContinuationError(
            "cache manifest SHA-256 mismatch: "
            f"expected={expected_manifest_sha256}, actual={manifest_sha256}"
        )
    if success.get("schema_version") != 1 or success.get("status") != "complete":
        raise ContinuationError("cache _SUCCESS must be schema-v1 and complete")
    if _safe_relative_path(success.get("manifest_path"), "_SUCCESS.manifest_path") != "manifest.json":
        raise ContinuationError("cache _SUCCESS must point exactly to manifest.json")
    if success.get("manifest_sha256") != manifest_sha256:
        raise ContinuationError("cache _SUCCESS does not bind the selected manifest")
    if manifest.get("schema_version") != 1:
        raise ContinuationError("cache manifest must use schema_version=1")
    cache_id = manifest.get("cache_id")
    if not isinstance(cache_id, str) or not cache_id or success.get("cache_id") != cache_id:
        raise ContinuationError("cache ID differs between _SUCCESS and manifest")
    equivalence = _require_mapping(manifest.get("equivalence"), "cache manifest equivalence")
    if equivalence.get("status") != "passed" or equivalence.get("torch_equal") is not True:
        raise ContinuationError("cache has no passed bitwise equivalence gate")
    if not isinstance(manifest.get("shards"), list) or not manifest["shards"]:
        raise ContinuationError("cache manifest has no shards")
    if not isinstance(manifest.get("splits"), dict) or not manifest["splits"]:
        raise ContinuationError("cache manifest has no split mappings")
    return {
        "root": str(root),
        "cache_id": cache_id,
        "success_path": str(success_path),
        "success_sha256": success_sha256,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
    }


def validate_throughput_matrix(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Validate and materialize the one matrix-selected training configuration."""

    absolute = Path(os.path.abspath(path))
    if path != absolute or path.is_symlink() or path.resolve(strict=True) != absolute:
        raise ContinuationError(
            "throughput matrix path must be canonical, absolute, and non-symlink"
        )
    expected_sha256 = _require_sha256(expected_sha256, "expected throughput matrix SHA-256")
    matrix, matrix_sha256 = _load_json_and_sha256(path, "throughput matrix")
    if matrix_sha256 != expected_sha256:
        raise ContinuationError(
            "throughput matrix SHA-256 mismatch: "
            f"expected={expected_sha256}, actual={matrix_sha256}"
        )
    if matrix.get("schema_version") != 1:
        raise ContinuationError("throughput matrix must use schema_version=1")
    if matrix.get("status") not in {"passed", "passed_with_no_ac_fallback"}:
        raise ContinuationError(f"throughput matrix did not pass: {matrix.get('status')!r}")
    if matrix.get("sample_trace_consistent_across_cells") is not True:
        raise ContinuationError("throughput matrix cells did not use one consistent comparison trace")
    best_cell_id = matrix.get("best_observed_cell_id")
    match = CELL_ID_PATTERN.fullmatch(str(best_cell_id))
    if match is None:
        raise ContinuationError(f"invalid best_observed_cell_id: {best_cell_id!r}")
    raw_cells = matrix.get("cells")
    if not isinstance(raw_cells, list):
        raise ContinuationError("throughput matrix cells must be a list")
    candidates = [
        cell
        for cell in raw_cells
        if isinstance(cell, dict) and cell.get("cell_id") == best_cell_id
    ]
    if len(candidates) != 1:
        raise ContinuationError(
            "throughput matrix must contain exactly one selected cell: "
            f"best={best_cell_id!r}, matches={len(candidates)}"
        )
    selected = candidates[0]
    expected_identity: Mapping[str, object] = {
        "cell_id": best_cell_id,
        "workers_per_rank": int(match.group("workers")),
        "node_total_workers": int(match.group("workers")) * 8,
        "storage": match.group("storage"),
        "activation_checkpointing": match.group("activation_checkpointing"),
        "status": "passed",
    }
    for key, expected in expected_identity.items():
        if selected.get(key) != expected:
            raise ContinuationError(
                f"selected throughput cell {key} mismatch: "
                f"expected={expected!r}, actual={selected.get(key)!r}"
            )
    timing = _require_mapping(selected.get("timing"), "selected cell timing")
    mean_step_seconds = timing.get("mean_step_seconds")
    if (
        not isinstance(mean_step_seconds, (int, float))
        or not math.isfinite(float(mean_step_seconds))
        or float(mean_step_seconds) <= 0
    ):
        raise ContinuationError("selected cell mean_step_seconds must be finite and positive")
    passed_timings: list[tuple[float, str]] = []
    for index, cell in enumerate(raw_cells):
        if not isinstance(cell, dict):
            raise ContinuationError(f"throughput matrix cell {index} must be an object")
        cell_id = cell.get("cell_id")
        cell_match = CELL_ID_PATTERN.fullmatch(str(cell_id))
        if cell_match is None:
            raise ContinuationError(f"invalid throughput matrix cell_id: {cell_id!r}")
        expected_cell_identity = {
            "workers_per_rank": int(cell_match.group("workers")),
            "node_total_workers": int(cell_match.group("workers")) * 8,
            "storage": cell_match.group("storage"),
            "activation_checkpointing": cell_match.group("activation_checkpointing"),
        }
        for key, expected in expected_cell_identity.items():
            if cell.get(key) != expected:
                raise ContinuationError(
                    f"throughput cell {cell_id!r} {key} mismatch: "
                    f"expected={expected!r}, actual={cell.get(key)!r}"
                )
        if cell.get("status") != "passed":
            continue
        cell_timing = _require_mapping(
            cell.get("timing"),
            f"throughput cell {cell_id!r} timing",
        )
        seconds = cell_timing.get("mean_step_seconds")
        if (
            not isinstance(seconds, (int, float))
            or not math.isfinite(float(seconds))
            or float(seconds) <= 0
        ):
            raise ContinuationError(
                f"passed throughput cell {cell_id!r} has invalid mean_step_seconds"
            )
        passed_timings.append((float(seconds), str(cell_id)))
    if len({cell_id for _, cell_id in passed_timings}) != len(passed_timings):
        raise ContinuationError("passed throughput matrix cell IDs are not unique")
    recomputed_ranking = [
        cell_id for _, cell_id in sorted(passed_timings, key=lambda item: item[0])
    ]
    ranking = matrix.get("ranking_by_mean_step_seconds")
    if not isinstance(ranking, list):
        raise ContinuationError("throughput matrix has no ranking_by_mean_step_seconds")
    declared_ranking = [
        item.get("cell_id") if isinstance(item, dict) else None for item in ranking
    ]
    if declared_ranking != recomputed_ranking or recomputed_ranking[:1] != [best_cell_id]:
        raise ContinuationError(
            "throughput matrix best cell/ranking does not match passed-cell timing"
        )
    return {
        "matrix_summary_path": str(path),
        "matrix_summary_sha256": matrix_sha256,
        "matrix_status": matrix["status"],
        "best_observed_cell_id": str(best_cell_id),
        "workers_per_rank": int(match.group("workers")),
        "node_total_workers": int(match.group("workers")) * 8,
        "storage": match.group("storage"),
        "activation_checkpointing": match.group("activation_checkpointing"),
        "mean_step_seconds": float(mean_step_seconds),
    }


def fingerprint_checkpoint(path: Path) -> dict[str, Any]:
    """Hash every R6 DCP byte under an exact component/file-layout contract."""

    path = _canonical_directory(path, "R6 checkpoint")
    if path.name != "iter_000001000":
        raise ContinuationError("resume checkpoint must be named iter_000001000")
    children = sorted(child.name for child in path.iterdir())
    if children != sorted(CHECKPOINT_COMPONENTS):
        raise ContinuationError(
            "R6 checkpoint component set mismatch: "
            f"expected={sorted(CHECKPOINT_COMPONENTS)}, actual={children}"
        )
    files: list[dict[str, Any]] = []
    for component in CHECKPOINT_COMPONENTS:
        component_dir = path / component
        if component_dir.is_symlink() or not component_dir.is_dir():
            raise ContinuationError(f"R6 checkpoint component is unsafe: {component_dir}")
        component_files = sorted(component_dir.iterdir(), key=lambda item: item.name)
        names = [item.name for item in component_files]
        if names.count(".metadata") != 1 or not any(name.endswith(".distcp") for name in names):
            raise ContinuationError(
                f"R6 checkpoint component lacks metadata or DCP shards: {component_dir}"
            )
        for file_path in component_files:
            if (
                file_path.is_symlink()
                or not file_path.is_file()
                or file_path.stat().st_size <= 0
                or (file_path.name != ".metadata" and file_path.suffix != ".distcp")
            ):
                raise ContinuationError(f"R6 checkpoint contains an unsafe entry: {file_path}")
            files.append(
                {
                    "path": file_path.relative_to(path).as_posix(),
                    "size_bytes": file_path.stat().st_size,
                    "sha256": _sha256_file(file_path),
                }
            )
    canonical_inventory = {
        "algorithm": CHECKPOINT_FINGERPRINT_ALGORITHM,
        "checkpoint_iteration": CHECKPOINT_ITERATION,
        "files": files,
    }
    encoded = json.dumps(
        canonical_inventory,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "path": str(path),
        "iteration": CHECKPOINT_ITERATION,
        "components": list(CHECKPOINT_COMPONENTS),
        "fingerprint_algorithm": CHECKPOINT_FINGERPRINT_ALGORITHM,
        "fingerprint_sha256": hashlib.sha256(encoded).hexdigest(),
        "file_count": len(files),
        "total_bytes": sum(int(item["size_bytes"]) for item in files),
        "files": files,
    }


def build_contract(args: argparse.Namespace) -> dict[str, Any]:
    if args.world_size != 8:
        raise ContinuationError("formal continuation requires exactly eight ranks")
    if args.run_name != args.experiment_id:
        raise ContinuationError("formal continuation run name must equal experiment ID")
    if not re.fullmatch(r"CWAM-E35-[A-Za-z0-9._-]+", args.experiment_id):
        raise ContinuationError("formal continuation experiment ID must use the CWAM-E35 family")
    cache = validate_cache_binding(
        args.cache_root,
        expected_success_sha256=args.cache_success_sha256,
        expected_manifest_sha256=args.cache_manifest_sha256,
    )
    throughput = validate_throughput_matrix(
        args.matrix_summary,
        expected_sha256=args.matrix_summary_sha256,
    )
    resume = fingerprint_checkpoint(args.resume_checkpoint)
    expected_fingerprint = _require_sha256(
        args.resume_checkpoint_fingerprint_sha256,
        "expected R6 checkpoint fingerprint",
    )
    if resume["fingerprint_sha256"] != expected_fingerprint:
        raise ContinuationError(
            "R6 checkpoint fingerprint mismatch: "
            f"expected={expected_fingerprint}, actual={resume['fingerprint_sha256']}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "kind": "cwam_cosmos3_formal_continuation_inputs",
        "validated_at": _utc_now(),
        "experiment_id": args.experiment_id,
        "run_name": args.run_name,
        "world_size": args.world_size,
        "cache": cache,
        "throughput": throughput,
        "resume": resume,
        "protocol": {
            "initial_iteration": CHECKPOINT_ITERATION,
            "max_iter": FINAL_ITERATION,
            "resume_existing_run": True,
            "allow_data_stream_replay": True,
            "data_stream_resume": "restarted_from_beginning_not_exact_loader_cursor",
            "checkpoint_load_training_state": True,
            "checkpoint_keys_to_skip_loading": [],
            "checkpoint_keys_not_to_resume": ["dataloader"],
            "checkpoint_strict_resume": True,
            "fixed_index_trace_path": None,
        },
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any], *, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if exclusive:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_run(
    *,
    output_root: Path,
    run_name: str,
    contract_path: Path,
    expected_contract_sha256: str,
) -> dict[str, Any]:
    """Atomically create one new run with an exact R6 checkpoint symlink."""

    output_root = _canonical_directory(output_root, "formal output root")
    expected_contract_sha256 = _require_sha256(
        expected_contract_sha256,
        "expected continuation-contract SHA-256",
    )
    contract, contract_sha256 = _load_json_and_sha256(
        contract_path,
        "formal continuation contract",
    )
    if contract_sha256 != expected_contract_sha256:
        raise ContinuationError(
            "continuation-contract SHA-256 mismatch: "
            f"expected={expected_contract_sha256}, actual={contract_sha256}"
        )
    if (
        contract.get("schema_version") != 1
        or contract.get("status") != "passed"
        or contract.get("kind") != "cwam_cosmos3_formal_continuation_inputs"
        or contract.get("run_name") != run_name
    ):
        raise ContinuationError("continuation contract is not a passed binding for this run")
    resume = _require_mapping(contract.get("resume"), "continuation contract resume")
    resume_path = Path(str(resume.get("path")))
    resume_path = _canonical_directory(resume_path, "R6 checkpoint")
    if resume_path.name != "iter_000001000":
        raise ContinuationError("continuation contract no longer points to iter_000001000")
    declared_files = resume.get("files")
    if not isinstance(declared_files, list) or not declared_files:
        raise ContinuationError("continuation contract has no R6 checkpoint file inventory")
    observed_inventory: list[tuple[str, int]] = []
    for component in CHECKPOINT_COMPONENTS:
        component_dir = resume_path / component
        if component_dir.is_symlink() or not component_dir.is_dir():
            raise ContinuationError(f"R6 checkpoint component changed or is unsafe: {component_dir}")
        for file_path in sorted(component_dir.iterdir(), key=lambda item: item.name):
            if file_path.is_symlink() or not file_path.is_file() or file_path.stat().st_size <= 0:
                raise ContinuationError(f"R6 checkpoint entry changed or is unsafe: {file_path}")
            observed_inventory.append(
                (file_path.relative_to(resume_path).as_posix(), file_path.stat().st_size)
            )
    declared_inventory = [
        (str(item.get("path")), int(item.get("size_bytes", -1)))
        for item in declared_files
        if isinstance(item, dict)
    ]
    if observed_inventory != declared_inventory:
        raise ContinuationError("R6 checkpoint inventory changed after contract validation")
    canonical_root = output_root / "cosmos3_action_robocasa365" / "action_sft"
    canonical_root.mkdir(parents=True, exist_ok=True)
    canonical_root = _canonical_directory(canonical_root, "formal run parent")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_name):
        raise ContinuationError(f"unsupported formal run name: {run_name!r}")
    run_dir = canonical_root / run_name
    if run_dir.exists() or run_dir.is_symlink():
        raise ContinuationError(f"formal continuation run already exists: {run_dir}")
    stage = canonical_root / f".{run_name}.incomplete-{os.getpid()}-{uuid.uuid4().hex}"
    stage.mkdir(mode=0o755)
    checkpoints = stage / "checkpoints"
    checkpoints.mkdir(mode=0o755)
    seed_name = f"iter_{CHECKPOINT_ITERATION:09d}"
    (checkpoints / seed_name).symlink_to(resume_path, target_is_directory=True)
    (checkpoints / "latest_checkpoint.txt").write_text(f"{seed_name}\n")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared",
        "prepared_at": _utc_now(),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "seed_checkpoint_link": str(run_dir / "checkpoints" / seed_name),
        "seed_checkpoint_target": str(resume_path),
        "seed_checkpoint_fingerprint_algorithm": resume.get("fingerprint_algorithm"),
        "seed_checkpoint_fingerprint_sha256": resume.get("fingerprint_sha256"),
        "continuation_contract": str(contract_path),
        "continuation_contract_sha256": contract_sha256,
    }
    _write_json_atomic(stage / "continuation-seed.json", receipt, exclusive=True)
    os.replace(stage, run_dir)
    return receipt


def _print(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fingerprint = commands.add_parser("fingerprint-checkpoint")
    fingerprint.add_argument("--checkpoint", type=Path, required=True)

    validate = commands.add_parser("validate-inputs")
    validate.add_argument("--experiment-id", required=True)
    validate.add_argument("--run-name", required=True)
    validate.add_argument("--world-size", type=int, default=8)
    validate.add_argument("--cache-root", type=Path, required=True)
    validate.add_argument("--cache-success-sha256", required=True)
    validate.add_argument("--cache-manifest-sha256", required=True)
    validate.add_argument("--matrix-summary", type=Path, required=True)
    validate.add_argument("--matrix-summary-sha256", required=True)
    validate.add_argument("--resume-checkpoint", type=Path, required=True)
    validate.add_argument("--resume-checkpoint-fingerprint-sha256", required=True)
    validate.add_argument("--output", type=Path)

    prepare = commands.add_parser("prepare-run")
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--run-name", required=True)
    prepare.add_argument("--contract", type=Path, required=True)
    prepare.add_argument("--contract-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "fingerprint-checkpoint":
        payload = fingerprint_checkpoint(args.checkpoint)
    elif args.command == "validate-inputs":
        payload = build_contract(args)
        if args.output is not None:
            _write_json_atomic(args.output, payload, exclusive=True)
    else:
        payload = prepare_run(
            output_root=args.output_root,
            run_name=args.run_name,
            contract_path=args.contract,
            expected_contract_sha256=args.contract_sha256,
        )
    _print(payload)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContinuationError, OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
