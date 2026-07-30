#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fail-closed helpers for a deliberately non-formal one-episode quick eval.

This module never emits the formal RoboCasa365 summary/evidence schemas.  Its
asset check authenticates the immutable publisher manifest and receipt, then
checks only explicitly required live entries by metadata.  It intentionally
does not enumerate or content-hash the full asset payload.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import stat
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from . import cwam_robocasa_asset_manifest as asset_manifest
except ImportError:
    import cwam_robocasa_asset_manifest as asset_manifest

SCHEMA_VERSION = 1
QUICK_ASSET_RECEIPT_KIND = "cwam_robocasa365_quick_asset_metadata_receipt"
QUICK_RESULT_KIND = "cwam_robocasa365_raw_quick_result"
QUICK_LIFECYCLE = "raw_quick_nonsealed"
DISCLAIMER = (
    "Exploratory single-episode observation only. Not formal evidence, not a prior-gate "
    "artifact, and not suitable for model comparison or success-rate claims. Full asset "
    "tree enumeration and content hashing were skipped; same-size content drift and "
    "manifest-external files are not ruled out."
)

_INPUT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class QuickEvalError(RuntimeError):
    """Raised when a quick-eval input or raw result fails closed validation."""


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise QuickEvalError(f"{label} must be a JSON object with string keys")
    return value


def _require_sha256(value: object, label: str) -> str:
    try:
        return asset_manifest._require_sha256(value, label)
    except asset_manifest.RoboCasaAssetManifestError as error:
        raise QuickEvalError(str(error)) from error


def _read_stable_file(
    path: Path,
    label: str,
    *,
    maximum_bytes: int | None = None,
    expected_mode: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    try:
        payload, size_bytes, sha256 = asset_manifest._read_stable_regular_file(
            path,
            label,
            maximum_bytes=maximum_bytes,
            expected_mode=expected_mode,
        )
        canonical, _ = asset_manifest._require_path(path, label, expected_kind="file", expected_mode=expected_mode)
    except asset_manifest.RoboCasaAssetManifestError as error:
        raise QuickEvalError(str(error)) from error
    return payload, {
        "path": str(canonical),
        "size_bytes": size_bytes,
        "sha256": sha256,
    }


def _decode_json(payload: bytes, label: str) -> object:
    try:
        return asset_manifest._decode_json(payload, label)
    except asset_manifest.RoboCasaAssetManifestError as error:
        raise QuickEvalError(str(error)) from error


def _quick_markers() -> dict[str, Any]:
    return {
        "formal_eligible": False,
        "asset_full_tree_verification_skipped": True,
        "lifecycle": QUICK_LIFECYCLE,
        "disclaimer": DISCLAIMER,
    }


def _manifest_summary_matches_receipt(records: list[dict[str, Any]], receipt: Mapping[str, Any]) -> dict[str, int]:
    summary = asset_manifest._payload_summary(records)
    for summary_key, receipt_key in (
        ("directory_count", "payload_directory_count"),
        ("file_count", "payload_file_count"),
        ("symlink_count", "payload_symlink_count"),
        ("total_bytes", "payload_total_bytes"),
    ):
        if summary[summary_key] != receipt[receipt_key]:
            raise QuickEvalError(
                f"asset manifest {summary_key} does not match publisher receipt: "
                f"manifest={summary[summary_key]}, receipt={receipt[receipt_key]}"
            )
    return summary


def _parse_required_entry(specification: str) -> tuple[str, str]:
    kind, separator, raw_path = specification.partition(":")
    if separator != ":" or kind not in {"file", "directory"}:
        raise QuickEvalError("required asset entry must have the form file:relative/path or directory:relative/path")
    try:
        relative_path = asset_manifest._canonical_relative_path(raw_path, "required asset entry")
    except asset_manifest.RoboCasaAssetManifestError as error:
        raise QuickEvalError(str(error)) from error
    return kind, relative_path


def verify_quick_assets(
    *,
    asset_root: Path,
    expected_manifest_sha256: str,
    expected_publisher_receipt_sha256: str,
    expected_robocasa_commit: str,
    required_entries: list[str],
) -> dict[str, Any]:
    """Verify publisher provenance plus selected live metadata without full hashing."""

    expected_manifest_sha256 = _require_sha256(expected_manifest_sha256, "expected asset manifest SHA-256")
    expected_publisher_receipt_sha256 = _require_sha256(
        expected_publisher_receipt_sha256,
        "expected asset publisher receipt SHA-256",
    )
    try:
        root, root_metadata = asset_manifest._require_path(
            asset_root,
            "asset root",
            expected_kind="directory",
        )
        provenance_root, _ = asset_manifest._require_path(
            root / asset_manifest.PROVENANCE_DIRECTORY,
            "asset provenance directory",
            expected_kind="directory",
        )
    except asset_manifest.RoboCasaAssetManifestError as error:
        raise QuickEvalError(str(error)) from error

    receipt_path = provenance_root / Path(asset_manifest.PUBLISHER_RECEIPT_RELATIVE_PATH).name
    manifest_path = provenance_root / Path(asset_manifest.MANIFEST_RELATIVE_PATH).name
    receipt_bytes, receipt_descriptor = _read_stable_file(
        receipt_path,
        "asset publisher receipt",
        maximum_bytes=asset_manifest.MAX_RECEIPT_BYTES,
        expected_mode=asset_manifest.PROVENANCE_FILE_MODE,
    )
    if receipt_descriptor["sha256"] != expected_publisher_receipt_sha256:
        raise QuickEvalError(
            "asset publisher receipt SHA-256 mismatch: "
            f"expected={expected_publisher_receipt_sha256}, actual={receipt_descriptor['sha256']}"
        )
    try:
        receipt = asset_manifest._validate_publisher_receipt(
            _decode_json(receipt_bytes, "asset publisher receipt"),
            root,
        )
    except asset_manifest.RoboCasaAssetManifestError as error:
        raise QuickEvalError(str(error)) from error
    if receipt["git_commit"] != expected_robocasa_commit:
        raise QuickEvalError(
            "asset publisher receipt Git commit mismatch: "
            f"expected={expected_robocasa_commit}, actual={receipt['git_commit']}"
        )

    manifest_bytes, manifest_descriptor = _read_stable_file(
        manifest_path,
        "asset manifest",
        maximum_bytes=asset_manifest.MAX_MANIFEST_BYTES,
        expected_mode=asset_manifest.PROVENANCE_FILE_MODE,
    )
    if manifest_descriptor["sha256"] != expected_manifest_sha256:
        raise QuickEvalError(
            "asset manifest SHA-256 mismatch: "
            f"expected={expected_manifest_sha256}, actual={manifest_descriptor['sha256']}"
        )
    if manifest_descriptor["sha256"] != receipt["manifest_sha256"]:
        raise QuickEvalError("asset manifest SHA-256 does not match the authenticated publisher receipt")
    try:
        header, records = asset_manifest._parse_asset_manifest(manifest_bytes, root)
    except asset_manifest.RoboCasaAssetManifestError as error:
        raise QuickEvalError(str(error)) from error
    if header["excluded_top_level_paths"] != receipt["excluded_top_level_paths"]:
        raise QuickEvalError("asset manifest and publisher receipt exclusion policies do not match")
    manifest_summary = _manifest_summary_matches_receipt(records, receipt)

    records_by_path = {record["path"]: record for record in records}
    if not required_entries:
        raise QuickEvalError("at least one required asset entry must be specified")
    observed_required_entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for specification in required_entries:
        expected_kind, relative_path = _parse_required_entry(specification)
        if relative_path in seen:
            raise QuickEvalError(f"duplicate required asset entry: {relative_path}")
        seen.add(relative_path)
        record = records_by_path.get(relative_path)
        if record is None:
            raise QuickEvalError(f"required asset entry is absent from publisher manifest: {relative_path}")
        if record["type"] != expected_kind:
            raise QuickEvalError(
                f"required asset entry type mismatch in manifest: path={relative_path}, "
                f"expected={expected_kind}, actual={record['type']}"
            )
        live_path = root.joinpath(*PurePosixPath(relative_path).parts)
        try:
            canonical, metadata = asset_manifest._require_path(
                live_path,
                f"required asset entry {relative_path}",
                expected_kind=expected_kind,
            )
        except asset_manifest.RoboCasaAssetManifestError as error:
            raise QuickEvalError(str(error)) from error
        live_mode = stat.S_IMODE(metadata.st_mode)
        if live_mode != record["mode"]:
            raise QuickEvalError(
                f"required asset entry mode mismatch: path={relative_path}, "
                f"expected={oct(record['mode'])}, actual={oct(live_mode)}"
            )
        observed = {
            "path": relative_path,
            "absolute_path": str(canonical),
            "type": expected_kind,
            "mode": live_mode,
        }
        if expected_kind == "file":
            if metadata.st_size != record["size_bytes"]:
                raise QuickEvalError(
                    f"required asset entry size mismatch: path={relative_path}, "
                    f"expected={record['size_bytes']}, actual={metadata.st_size}"
                )
            observed["size_bytes"] = int(metadata.st_size)
            observed["content_sha256_verified"] = False
        observed_required_entries.append(observed)

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": QUICK_ASSET_RECEIPT_KIND,
        "status": "quick_metadata_checks_passed",
        **_quick_markers(),
        "asset_root": str(root),
        "asset_root_mode": stat.S_IMODE(root_metadata.st_mode),
        "manifest": manifest_descriptor,
        "publisher_receipt": receipt_descriptor,
        "publisher_generator_id": receipt["generator_id"],
        "publisher_generated_at_utc": receipt["generated_at_utc"],
        "robocasa_git_commit": receipt["git_commit"],
        "publisher_payload_summary": manifest_summary,
        "required_entries": observed_required_entries,
        "validation": {
            "publisher_manifest_sha256_verified": True,
            "publisher_receipt_sha256_verified": True,
            "publisher_schema_and_binding_verified": True,
            "publisher_git_commit_verified": True,
            "publisher_manifest_summary_verified": True,
            "required_asset_entry_metadata_verified": True,
            "asset_full_tree_enumerated": False,
            "asset_payload_content_hashes_verified": False,
            "manifest_external_entries_ruled_out": False,
        },
    }


def _require_int(value: object, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise QuickEvalError(f"{label} must be an integer >= {minimum}, got {value!r}")
    return value


def _require_rate(value: object, expected: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise QuickEvalError(f"{label} must be a finite numeric rate")
    if not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12):
        raise QuickEvalError(f"{label} mismatch: expected={expected}, actual={value!r}")


def _validate_args(
    value: object,
    *,
    expected_task: str,
    expected_seed: int,
    expected_run_name: str,
    expected_results_root: Path,
    expected_checkpoint: Path,
) -> dict[str, Any]:
    args_json = _require_mapping(value, "args.json")
    expected = {
        "task_shard": 0,
        "num_shards": 1,
        "episodes": 1,
        "episodes_start": 0,
        "max_steps": 400,
        "open_loop_horizon": 12,
        "seed": expected_seed,
        "checkpoint": str(expected_checkpoint),
        "run_name": expected_run_name,
        "results_dir": str(expected_results_root),
        "eval_dataset": None,
        "headless": True,
        "save_video": False,
        "camera_height": 256,
        "camera_width": 256,
        "host": "127.0.0.1",
        "port": 16000,
    }
    for key, expected_value in expected.items():
        actual_value = args_json.get(key)
        if actual_value != expected_value or (
            type(expected_value) in {bool, int} and type(actual_value) is not type(expected_value)
        ):
            raise QuickEvalError(f"args.json {key} mismatch: expected={expected_value!r}, actual={actual_value!r}")
    if args_json.get("tasks") != [expected_task]:
        raise QuickEvalError(f"args.json tasks must be exactly [{expected_task!r}]")
    for forbidden in (
        "dynamic_instructions_file",
        "gt_reasoning_file",
        "vlm_captions_file",
        "fg_captions_path",
    ):
        if args_json.get(forbidden) not in (None, ""):
            raise QuickEvalError(f"args.json enables forbidden non-causal input {forbidden}")
    return args_json


def _validate_result(value: object, *, expected_task: str) -> dict[str, Any]:
    results = _require_mapping(value, "results_shard0.json")
    if set(results) != {expected_task}:
        raise QuickEvalError(
            f"results_shard0.json task coverage mismatch: expected={[expected_task]!r}, actual={sorted(results)!r}"
        )
    task_result = _require_mapping(results[expected_task], f"results[{expected_task}]")
    if "error" in task_result:
        raise QuickEvalError(f"results[{expected_task}] contains evaluator error: {task_result['error']!r}")
    if task_result.get("complete") is not True:
        raise QuickEvalError(f"results[{expected_task}] is not complete")
    if task_result.get("split") != "atomic":
        raise QuickEvalError(f"results[{expected_task}] split must be 'atomic'")
    if type(task_result.get("episodes")) is not int or task_result["episodes"] != 1:
        raise QuickEvalError(f"results[{expected_task}] episodes must be exactly 1")
    episodes = task_result.get("episode_results")
    if not isinstance(episodes, list) or len(episodes) != 1:
        raise QuickEvalError(f"results[{expected_task}] must contain exactly one episode result")
    episode = _require_mapping(episodes[0], f"results[{expected_task}].episode_results[0]")
    success = episode.get("success")
    place_anytime = episode.get("place_anytime")
    if type(success) is not bool or type(place_anytime) is not bool:
        raise QuickEvalError("episode success and place_anytime must be strict booleans")
    steps = _require_int(episode.get("steps"), "episode steps", minimum=1)
    if steps > 400:
        raise QuickEvalError(f"episode steps exceeds max_steps=400: {steps}")
    instruction = episode.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip() or instruction != instruction.strip():
        raise QuickEvalError("episode instruction must be a non-empty trimmed string")
    action_log = episode.get("action_log")
    if not isinstance(action_log, list) or len(action_log) != steps:
        raise QuickEvalError(f"episode action_log must contain exactly {steps} actions")
    for step_index, action in enumerate(action_log):
        if not isinstance(action, (list, tuple)) or len(action) != 7:
            raise QuickEvalError(f"episode action {step_index} must have width 7")
        if any(
            isinstance(component, bool)
            or not isinstance(component, (int, float))
            or not math.isfinite(float(component))
            for component in action
        ):
            raise QuickEvalError(f"episode action {step_index} contains a non-finite or non-numeric component")

    successes = int(success)
    place_anytime_count = int(place_anytime)
    if type(task_result.get("successes")) is not int or task_result["successes"] != successes:
        raise QuickEvalError(f"results[{expected_task}] successes does not match episode.success")
    if (
        type(task_result.get("place_anytime_count")) is not int
        or task_result["place_anytime_count"] != place_anytime_count
    ):
        raise QuickEvalError(f"results[{expected_task}] place_anytime_count does not match the episode")
    _require_rate(task_result.get("success_rate"), float(successes), "task success_rate")
    _require_rate(task_result.get("place_anytime_rate"), float(place_anytime_count), "task place_anytime_rate")
    return {
        "task": expected_task,
        "success": success,
        "steps": steps,
        "instruction": instruction,
        "place_anytime": place_anytime,
        "errors": [],
    }


def _collect_input_descriptors(specifications: list[list[str]]) -> dict[str, dict[str, Any]]:
    descriptors: dict[str, dict[str, Any]] = {}
    for name, raw_path, expected_sha256 in specifications:
        if _INPUT_NAME_RE.fullmatch(name) is None:
            raise QuickEvalError(f"input name is invalid: {name!r}")
        if name in descriptors:
            raise QuickEvalError(f"duplicate input name: {name}")
        expected_sha256 = _require_sha256(expected_sha256, f"{name} expected SHA-256")
        _, descriptor = _read_stable_file(Path(raw_path), f"input {name}")
        if descriptor["sha256"] != expected_sha256:
            raise QuickEvalError(
                f"input {name} SHA-256 mismatch: expected={expected_sha256}, actual={descriptor['sha256']}"
            )
        descriptors[name] = descriptor
    if not descriptors:
        raise QuickEvalError("at least one immutable input binding is required")
    return dict(sorted(descriptors.items()))


def seal_quick_result(
    *,
    args_path: Path,
    results_path: Path,
    output: Path,
    expected_task: str,
    expected_seed: int,
    expected_run_name: str,
    expected_results_root: Path,
    expected_checkpoint: Path,
    experiment_id: str,
    logical_run_id: str,
    input_specifications: list[list[str]],
) -> dict[str, Any]:
    """Validate, bind, and locally freeze one raw quick result."""

    try:
        args_canonical, _ = asset_manifest._require_path(args_path, "args.json", expected_kind="file")
        results_canonical, _ = asset_manifest._require_path(
            results_path,
            "results_shard0.json",
            expected_kind="file",
        )
        results_directory, _ = asset_manifest._require_path(
            args_canonical.parent,
            "quick results directory",
            expected_kind="directory",
        )
        results_root, _ = asset_manifest._require_path(
            expected_results_root,
            "expected quick results root",
            expected_kind="directory",
        )
        checkpoint, _ = asset_manifest._require_path(
            expected_checkpoint,
            "expected serving checkpoint",
            expected_kind="directory",
        )
    except asset_manifest.RoboCasaAssetManifestError as error:
        raise QuickEvalError(str(error)) from error
    if results_directory != results_root / expected_run_name:
        raise QuickEvalError("raw results directory does not derive from expected results root and run name")
    if results_canonical.parent != results_directory:
        raise QuickEvalError("args.json and results_shard0.json must share one canonical results directory")
    if output.parent != results_directory or output.name != "quick-result.json":
        raise QuickEvalError("quick result output must be quick-result.json in the raw results directory")
    if output.exists() or output.is_symlink():
        raise QuickEvalError(f"refusing to overwrite quick result: {output}")

    os.chmod(args_canonical, 0o444, follow_symlinks=False)
    os.chmod(results_canonical, 0o444, follow_symlinks=False)
    args_bytes, args_descriptor = _read_stable_file(args_canonical, "args.json", expected_mode=0o444)
    results_bytes, results_descriptor = _read_stable_file(
        results_canonical,
        "results_shard0.json",
        expected_mode=0o444,
    )
    _validate_args(
        _decode_json(args_bytes, "args.json"),
        expected_task=expected_task,
        expected_seed=expected_seed,
        expected_run_name=expected_run_name,
        expected_results_root=results_root,
        expected_checkpoint=checkpoint,
    )
    episode = _validate_result(
        _decode_json(results_bytes, "results_shard0.json"),
        expected_task=expected_task,
    )
    inputs = _collect_input_descriptors(input_specifications)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": QUICK_RESULT_KIND,
        "status": "completed",
        **_quick_markers(),
        "experiment_id": experiment_id,
        "logical_run_id": logical_run_id,
        "protocol": {
            "stage": "one_episode_gate",
            "evaluation_class": "exploratory",
            "split": "atomic",
            "task_count": 1,
            "episodes_per_task": 1,
            "num_shards": 1,
            "seed": expected_seed,
            "max_steps": 400,
            "open_loop_horizon": 12,
        },
        "task": episode["task"],
        "seed": expected_seed,
        "success": episode["success"],
        "steps": episode["steps"],
        "errors": episode["errors"],
        "episode": {
            "instruction": episode["instruction"],
            "place_anytime": episode["place_anytime"],
        },
        "raw_artifacts": {
            "args_json": args_descriptor,
            "results_shard0_json": results_descriptor,
        },
        "inputs": inputs,
        "validation": {
            "evaluator_exit_zero": True,
            "single_task_complete": True,
            "single_episode_present": True,
            "episode_success_strict_boolean": True,
            "action_log_shape_and_finiteness_verified": True,
            "aggregate_counts_and_rates_recomputed": True,
            "raw_artifacts_sha256_bound": True,
            "formal_post_rollout_verification_run": False,
            "formal_summary_or_evidence_generated": False,
        },
    }
    try:
        asset_manifest.write_verification_receipt(output, payload)
        os.chmod(results_directory, 0o555, follow_symlinks=False)
    except asset_manifest.RoboCasaAssetManifestError as error:
        raise QuickEvalError(str(error)) from error

    _, sealed_args = _read_stable_file(args_canonical, "sealed args.json", expected_mode=0o444)
    _, sealed_results = _read_stable_file(
        results_canonical,
        "sealed results_shard0.json",
        expected_mode=0o444,
    )
    _, sealed_output = _read_stable_file(output, "sealed quick-result.json", expected_mode=0o444)
    if sealed_args != args_descriptor or sealed_results != results_descriptor:
        raise QuickEvalError("raw result changed while quick-result.json was being sealed")
    if stat.S_IMODE(results_directory.lstat().st_mode) != 0o555:
        raise QuickEvalError("quick results directory is not locally frozen mode 0555")
    payload["_output_descriptor"] = sealed_output
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_assets = subparsers.add_parser(
        "verify-assets",
        help="authenticate publisher metadata and selected live entries without a full-tree hash",
    )
    verify_assets.add_argument("--asset-root", type=Path, required=True)
    verify_assets.add_argument("--expected-manifest-sha256", required=True)
    verify_assets.add_argument("--expected-publisher-receipt-sha256", required=True)
    verify_assets.add_argument("--expected-robocasa-commit", required=True)
    verify_assets.add_argument("--required-entry", action="append", default=[])
    verify_assets.add_argument("--output", type=Path, required=True)

    seal_result = subparsers.add_parser(
        "seal-result",
        help="strictly validate and locally freeze one raw quick result",
    )
    seal_result.add_argument("--args-json", type=Path, required=True)
    seal_result.add_argument("--results-json", type=Path, required=True)
    seal_result.add_argument("--output", type=Path, required=True)
    seal_result.add_argument("--expected-task", required=True)
    seal_result.add_argument("--expected-seed", type=int, required=True)
    seal_result.add_argument("--expected-run-name", required=True)
    seal_result.add_argument("--expected-results-root", type=Path, required=True)
    seal_result.add_argument("--expected-checkpoint", type=Path, required=True)
    seal_result.add_argument("--experiment-id", required=True)
    seal_result.add_argument("--logical-run-id", required=True)
    seal_result.add_argument("--input", action="append", nargs=3, metavar=("NAME", "PATH", "SHA256"), default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "verify-assets":
            payload = verify_quick_assets(
                asset_root=args.asset_root,
                expected_manifest_sha256=args.expected_manifest_sha256,
                expected_publisher_receipt_sha256=args.expected_publisher_receipt_sha256,
                expected_robocasa_commit=args.expected_robocasa_commit,
                required_entries=args.required_entry,
            )
            digest = asset_manifest.write_verification_receipt(args.output, payload)
            print(
                "QUICK_NONFORMAL_ASSET_METADATA_VERIFIED "
                f"formal_eligible=false asset_full_tree_verification_skipped=true "
                f"sha256={digest} path={args.output}"
            )
            return 0

        payload = seal_quick_result(
            args_path=args.args_json,
            results_path=args.results_json,
            output=args.output,
            expected_task=args.expected_task,
            expected_seed=args.expected_seed,
            expected_run_name=args.expected_run_name,
            expected_results_root=args.expected_results_root,
            expected_checkpoint=args.expected_checkpoint,
            experiment_id=args.experiment_id,
            logical_run_id=args.logical_run_id,
            input_specifications=args.input,
        )
    except (QuickEvalError, asset_manifest.RoboCasaAssetManifestError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    descriptor = payload["_output_descriptor"]
    print(
        "QUICK_NONFORMAL_RESULT_READY "
        "formal_eligible=false asset_full_tree_verification_skipped=true "
        f"success={str(payload['success']).lower()} steps={payload['steps']} "
        f"sha256={descriptor['sha256']} path={descriptor['path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
