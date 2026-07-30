# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU-only contracts for the explicitly non-formal quick-eval path."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path, PurePosixPath

import pytest

from examples.h_cluster import cwam_robocasa365_quick_eval as quick_eval
from examples.h_cluster import cwam_robocasa_asset_manifest as asset_manifest

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_records(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    def scan(directory: Path, relative_base: PurePosixPath) -> None:
        for child in sorted(directory.iterdir(), key=lambda path: path.name):
            if not relative_base.parts and child.name in asset_manifest.EXCLUDED_TOP_LEVEL_PATHS:
                continue
            relative = relative_base / child.name
            metadata = child.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                records.append(
                    {
                        "mode": stat.S_IMODE(metadata.st_mode),
                        "path": relative.as_posix(),
                        "type": "directory",
                    }
                )
                scan(child, relative)
            else:
                records.append(
                    {
                        "mode": stat.S_IMODE(metadata.st_mode),
                        "path": relative.as_posix(),
                        "sha256": _sha256(child),
                        "size_bytes": metadata.st_size,
                        "type": "file",
                    }
                )

    scan(root, PurePosixPath())
    return records


@pytest.fixture
def published_tree(tmp_path: Path) -> Path:
    root = tmp_path / "robocasa-assets"
    (root / ".git").mkdir(parents=True)
    provenance = root / asset_manifest.PROVENANCE_DIRECTORY
    provenance.mkdir()
    (root / "robocasa" / "environments").mkdir(parents=True)
    (root / "robocasa" / "models" / "assets").mkdir(parents=True)
    init_file = root / "robocasa" / "__init__.py"
    init_file.write_text("fixture = True\n", encoding="utf-8")
    payload_file = root / "robocasa" / "models" / "assets" / "fixture.bin"
    payload_file.write_bytes(b"original")
    for path in (init_file, payload_file):
        path.chmod(0o644)
    for path in (
        root,
        root / ".git",
        provenance,
        root / "robocasa",
        root / "robocasa" / "environments",
        root / "robocasa" / "models",
        root / "robocasa" / "models" / "assets",
    ):
        path.chmod(0o755)

    records = _payload_records(root)
    header = {
        "excluded_top_level_paths": list(asset_manifest.EXCLUDED_TOP_LEVEL_PATHS),
        "kind": asset_manifest.MANIFEST_KIND,
        "payload_root": str(root),
        "schema_version": asset_manifest.SCHEMA_VERSION,
    }
    manifest = root / asset_manifest.MANIFEST_RELATIVE_PATH
    manifest.write_text(
        "".join(f"{json.dumps(value, separators=(',', ':'), sort_keys=True)}\n" for value in [header, *records]),
        encoding="utf-8",
    )
    manifest.chmod(0o444)
    source_root = tmp_path / "source-archive"
    receipt = {
        "excluded_top_level_paths": list(asset_manifest.EXCLUDED_TOP_LEVEL_PATHS),
        "generated_at_utc": "2026-07-31T00:00:00Z",
        "generator_id": asset_manifest.GENERATOR_ID,
        "git_commit": "a" * 40,
        "git_status_porcelain": "",
        "kind": asset_manifest.PUBLISHER_RECEIPT_KIND,
        "manifest_relative_path": asset_manifest.MANIFEST_RELATIVE_PATH,
        "manifest_sha256": _sha256(manifest),
        "payload_directory_count": sum(record["type"] == "directory" for record in records),
        "payload_file_count": sum(record["type"] == "file" for record in records),
        "payload_root": str(root),
        "payload_symlink_count": 0,
        "payload_total_bytes": sum(int(record["size_bytes"]) for record in records if record["type"] == "file"),
        "schema_version": asset_manifest.SCHEMA_VERSION,
        "source_archive_directory": str(source_root),
        "source_sha256sums_path": str(source_root / "SHA256SUMS"),
        "source_sha256sums_sha256": "b" * 64,
        "source_sizes_path": str(source_root / "SIZES"),
        "source_sizes_sha256": "c" * 64,
        "status": "passed",
    }
    publisher_receipt = root / asset_manifest.PUBLISHER_RECEIPT_RELATIVE_PATH
    publisher_receipt.write_text(f"{json.dumps(receipt, sort_keys=True)}\n", encoding="utf-8")
    publisher_receipt.chmod(0o444)
    return root


def _verify_assets(published_tree: Path) -> dict:
    return quick_eval.verify_quick_assets(
        asset_root=published_tree,
        expected_manifest_sha256=_sha256(published_tree / asset_manifest.MANIFEST_RELATIVE_PATH),
        expected_publisher_receipt_sha256=_sha256(published_tree / asset_manifest.PUBLISHER_RECEIPT_RELATIVE_PATH),
        expected_robocasa_commit="a" * 40,
        required_entries=[
            "directory:robocasa",
            "file:robocasa/__init__.py",
            "directory:robocasa/environments",
            "directory:robocasa/models/assets",
        ],
    )


def test_asset_receipt_is_explicitly_nonformal_and_does_not_full_hash(
    published_tree: Path,
) -> None:
    result = _verify_assets(published_tree)

    assert result["status"] == "quick_metadata_checks_passed"
    assert result["formal_eligible"] is False
    assert result["asset_full_tree_verification_skipped"] is True
    assert result["lifecycle"] == "raw_quick_nonsealed"
    assert result["validation"]["asset_full_tree_enumerated"] is False
    assert result["validation"]["asset_payload_content_hashes_verified"] is False
    assert all(entry.get("content_sha256_verified") is not True for entry in result["required_entries"])


def test_same_size_unrequired_asset_drift_is_deliberately_not_claimed(
    published_tree: Path,
) -> None:
    payload = published_tree / "robocasa" / "models" / "assets" / "fixture.bin"
    payload.write_bytes(b"mutated!")
    payload.chmod(0o644)

    result = _verify_assets(published_tree)

    assert result["asset_full_tree_verification_skipped"] is True
    assert result["validation"]["manifest_external_entries_ruled_out"] is False


def test_missing_required_asset_entry_fails_closed(published_tree: Path) -> None:
    (published_tree / "robocasa" / "__init__.py").unlink()

    with pytest.raises(quick_eval.QuickEvalError, match="does not exist"):
        _verify_assets(published_tree)


def _raw_result_fixture(tmp_path: Path, *, success: bool = False) -> dict[str, Path]:
    task = "OpenDrawer"
    run_name = "QUICK-NONFORMAL-E1-open-drawer"
    results_root = tmp_path / "quick-nonformal-results"
    results_dir = results_root / run_name
    results_dir.mkdir(parents=True)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    args_path = results_dir / "args.json"
    args_path.write_text(
        json.dumps(
            {
                "tasks": [task],
                "split": "all",
                "task_shard": 0,
                "num_shards": 1,
                "episodes": 1,
                "episodes_start": 0,
                "max_steps": 400,
                "open_loop_horizon": 12,
                "host": "127.0.0.1",
                "port": 16000,
                "checkpoint": str(checkpoint),
                "results_dir": str(results_root),
                "run_name": run_name,
                "save_video": False,
                "headless": True,
                "dynamic_instructions_file": None,
                "camera_height": 256,
                "camera_width": 256,
                "seed": 42,
                "gt_reasoning_file": None,
                "vlm_captions_file": None,
                "eval_dataset": None,
                "fg_captions_path": None,
            }
        ),
        encoding="utf-8",
    )
    results_path = results_dir / "results_shard0.json"
    results_path.write_text(
        json.dumps(
            {
                task: {
                    "success_rate": float(success),
                    "successes": int(success),
                    "place_anytime_rate": 1.0,
                    "place_anytime_count": 1,
                    "episodes": 1,
                    "episode_results": [
                        {
                            "success": success,
                            "place_anytime": True,
                            "steps": 1,
                            "instruction": "open the drawer",
                            "action_log": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                        }
                    ],
                    "split": "atomic",
                    "complete": True,
                }
            }
        ),
        encoding="utf-8",
    )
    immutable_input = tmp_path / "source-manifest.json"
    immutable_input.write_text('{"source": "fixture"}\n', encoding="utf-8")
    return {
        "args": args_path,
        "results": results_path,
        "output": results_dir / "quick-result.json",
        "results_root": results_root,
        "checkpoint": checkpoint,
        "input": immutable_input,
    }


def _seal_fixture(paths: dict[str, Path]) -> dict:
    return quick_eval.seal_quick_result(
        args_path=paths["args"],
        results_path=paths["results"],
        output=paths["output"],
        expected_task="OpenDrawer",
        expected_seed=42,
        expected_run_name="QUICK-NONFORMAL-E1-open-drawer",
        expected_results_root=paths["results_root"],
        expected_checkpoint=paths["checkpoint"],
        experiment_id="QUICK-NONFORMAL-E1",
        logical_run_id="QUICK-NONFORMAL-E1-open-drawer",
        input_specifications=[
            ["source_manifest", str(paths["input"]), _sha256(paths["input"])],
        ],
    )


def test_quick_result_uses_strict_episode_success_and_binds_raw_files(tmp_path: Path) -> None:
    paths = _raw_result_fixture(tmp_path, success=False)

    payload = _seal_fixture(paths)
    on_disk = json.loads(paths["output"].read_text(encoding="utf-8"))

    assert payload["success"] is False
    assert payload["episode"]["place_anytime"] is True
    assert payload["errors"] == []
    assert on_disk["formal_eligible"] is False
    assert on_disk["asset_full_tree_verification_skipped"] is True
    assert on_disk["lifecycle"] == "raw_quick_nonsealed"
    assert on_disk["raw_artifacts"]["args_json"]["sha256"] == _sha256(paths["args"])
    assert on_disk["raw_artifacts"]["results_shard0_json"]["sha256"] == _sha256(paths["results"])
    assert stat.S_IMODE(paths["args"].stat().st_mode) == 0o444
    assert stat.S_IMODE(paths["results"].stat().st_mode) == 0o444
    assert stat.S_IMODE(paths["output"].stat().st_mode) == 0o444
    assert stat.S_IMODE(paths["output"].parent.stat().st_mode) == 0o555


@pytest.mark.parametrize("mutation", ["incomplete", "error", "bad_action", "aggregate"])
def test_invalid_or_incomplete_raw_result_never_emits_quick_result(
    tmp_path: Path,
    mutation: str,
) -> None:
    paths = _raw_result_fixture(tmp_path, success=True)
    value = json.loads(paths["results"].read_text(encoding="utf-8"))
    task_result = value["OpenDrawer"]
    if mutation == "incomplete":
        task_result["complete"] = False
    elif mutation == "error":
        task_result["error"] = "environment failed"
    elif mutation == "bad_action":
        task_result["episode_results"][0]["action_log"][0] = [0.0] * 6
    else:
        task_result["successes"] = 0
    paths["results"].write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(quick_eval.QuickEvalError):
        _seal_fixture(paths)
    assert not paths["output"].exists()


def test_quick_runner_and_submitter_are_isolated_from_formal_artifacts() -> None:
    directory = Path(__file__).parent
    helper = (directory / "cwam_robocasa365_quick_eval.py").read_text(encoding="utf-8")
    runner = (directory / "run_cwam_cosmos3_robocasa365_quick_eval.sh").read_text(encoding="utf-8")
    submitter = (directory / "submit_cwam_cosmos3_robocasa365_quick_eval_rjob.sh").read_text(encoding="utf-8")
    formal_runner = (directory / "run_cwam_cosmos3_robocasa365_eval.sh").read_text(encoding="utf-8")

    for token in (
        "CWAM_QUICK_EVAL",
        "formal_eligible=false",
        "asset_full_tree_verification_skipped=true",
        "quick-nonformal-control",
        "quick-nonformal-results",
        "seal-result",
        "PINNED_WAN22_VAE_SHA256",
        "--input wan2pt2_vae",
    ):
        assert token in runner
    for forbidden in (
        "verify-post-rollout",
        "SUMMARIZE_ARGS",
        "summary.csv",
        "task_success_rates.csv",
        "evaluation_evidence.json",
        "require_accepted_submission_terminal_state",
    ):
        assert forbidden not in runner
    assert "_scan_payload_tree" not in helper
    assert "CWAM_QUICK_EVAL" not in formal_runner
    assert runner.index('wait "$EVALUATOR_PID"') < runner.index('stop_child_pids "$SERVER_PID"')
    assert runner.index('stop_child_pids "$SERVER_PID"') < runner.index("seal-result")
    assert 'readonly EVAL_NVME_PARENT="/nvme/.cwam-robocasa365-quick-eval"' in runner
    assert 'mkdir --mode=0700 -- "$EVAL_NVME_PARENT"' in runner
    assert '[[ -d "$EVAL_NVME_PARENT" && ! -L "$EVAL_NVME_PARENT" ]]' in runner
    assert '"$(realpath -e -- "$EVAL_NVME_PARENT")" == "$EVAL_NVME_PARENT"' in runner
    assert '"$(stat -c %u:%g -- "$EVAL_NVME_PARENT")" == 0:0' in runner
    assert '"$(stat -c %a -- "$EVAL_NVME_PARENT")" == 700' in runner
    assert '[[ ! -e "$EVAL_NVME_ROOT" && ! -L "$EVAL_NVME_ROOT" ]]' in runner
    assert 'mkdir --mode=0700 -- "$EVAL_NVME_ROOT"' in runner
    assert '/usr/bin/rm --one-file-system -rf -- "$EVAL_NVME_ROOT"' in runner
    assert '/usr/bin/rm --one-file-system -rf -- "$EVAL_NVME_PARENT"' not in runner

    assert "--predict-only" not in submitter
    assert "scheduler-readback" not in submitter
    assert submitter.count('"$RJOB_BIN" "${SUBMIT_ARGS[@]}"') == 1
    assert "--gpu 1" in submitter
    assert "--host-network" not in submitter
    assert "RJOB_HOST_NETWORK=false" in submitter
    assert "exec $BASH_BIN" in submitter
