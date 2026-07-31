# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU-only tests for the formal RoboCasa365 runtime receipt."""

from __future__ import annotations

import copy
import hashlib
import json
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from examples.h_cluster import capture_robocasa365_eval_runtime as runtime_receipt


def test_all_subprocess_probe_programs_compile() -> None:
    for name in ("_COSMOS_PROBE", "_ROBOCASA_PROBE", "_ROBOCASA_EGL_PROBE"):
        compile(getattr(runtime_receipt, name), f"{runtime_receipt.__file__}:{name}", "exec")


def test_graphics_probe_distinguishes_glvnd_frontends_from_vendor_abis() -> None:
    probe = runtime_receipt._ROBOCASA_EGL_PROBE
    assert "reported_library = pathlib.Path(os.fsdecode(info.dli_fname))" in probe
    assert "if not reported_library.is_absolute():" in probe
    assert "resolved_library = reported_library.resolve(strict=True)" in probe
    assert "descriptor = file_descriptor(resolved_library)" in probe
    assert '("libEGL", "libEGL.so.1", "eglGetDisplay", "egl_context_mapped")' in probe
    assert '("libEGL_nvidia", "libEGL_nvidia.so.0", "__egl_Main", "egl_context_mapped")' in probe
    assert '("libEGL_nvidia", "libEGL_nvidia.so.0", "eglQueryString", "egl_context_mapped")' not in probe
    assert '("libGLX", "libGLX.so.0", "glXGetProcAddress", "resolved_and_dlopen_verified")' in probe
    assert '("libGLX_nvidia", "libGLX_nvidia.so.0", "__glx_Main", "resolved_and_dlopen_verified")' in probe
    assert '("libGLX_nvidia", "libGLX_nvidia.so.0", "glXGetProcAddress", "resolved_and_dlopen_verified")' not in probe


def test_graphics_probe_binds_loader_path_before_third_party_imports() -> None:
    probe = runtime_receipt._ROBOCASA_EGL_PROBE
    capture = 'launch_ld_library_path = os.environ.get("LD_LIBRARY_PATH")'
    assert probe.index(capture) < probe.index("    import robosuite")
    assert '"ld_library_path": launch_ld_library_path' in probe
    assert '"ld_library_path": os.environ.get("LD_LIBRARY_PATH")' not in probe


def _write_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _write_json(path: Path, value: object) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def _file_descriptor(path: Path, **extra: object) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "supplied_path": str(path),
        "canonical_path": str(path.resolve(strict=True)),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        **extra,
    }


def _uuid(index: int) -> str:
    return f"GPU-00000000-0000-0000-0000-{index:012x}"


class FakeRunner:
    def __init__(self, fixture: dict[str, Any]) -> None:
        self.fixture = fixture
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.timeout_label: str | None = None
        self.failure_label: str | None = None

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        copied_argv = list(argv)
        self.calls.append((copied_argv, dict(kwargs)))
        label = self._label(copied_argv)
        if label == self.timeout_label:
            raise subprocess.TimeoutExpired(copied_argv, kwargs["timeout"])
        if label == self.failure_label:
            return subprocess.CompletedProcess(copied_argv, 7, "", "synthetic failure")
        if label.startswith("git-rev-parse:"):
            root = label.partition(":")[2]
            stdout = self.fixture["commits"][root] + "\n"
        elif label.startswith("git-status:"):
            root = label.partition(":")[2]
            stdout = self.fixture["statuses"].get(root, "")
        elif label == "cosmos-environment":
            stdout = json.dumps(self.fixture["cosmos_environment_report"])
        elif label == "robocasa-environment":
            stdout = json.dumps(self.fixture["robocasa_environment_report"])
        elif label == "cosmos-probe":
            stdout = json.dumps(self.fixture["cosmos_probe"])
        elif label == "robocasa-probe":
            stdout = json.dumps(self.fixture["robocasa_probe"])
        elif label.startswith("robocasa-egl:"):
            shard_index = int(label.partition(":")[2])
            stdout = json.dumps(self.fixture["robocasa_egl_probes"][shard_index])
        elif label == "nvidia-smi":
            stdout = self.fixture["nvidia_smi"]
        else:  # pragma: no cover - a new command must be explicitly modeled
            raise AssertionError(f"unexpected argv: {copied_argv!r}")
        return subprocess.CompletedProcess(copied_argv, 0, stdout, "")

    def _label(self, argv: list[str]) -> str:
        if Path(argv[0]) == self.fixture["git"]:
            root = argv[argv.index("-C") + 1]
            if "rev-parse" in argv:
                return f"git-rev-parse:{root}"
            return f"git-status:{root}"
        if Path(argv[0]) == self.fixture["nvidia_smi_executable"]:
            return "nvidia-smi"
        if argv[2] == "-c":
            if argv[3] == runtime_receipt._COSMOS_PROBE:
                return "cosmos-probe"
            if argv[3] == runtime_receipt._ROBOCASA_EGL_PROBE:
                return f"robocasa-egl:{argv[-2]}"
            return "robocasa-probe"
        if Path(argv[0]) == self.fixture["cosmos_python"]:
            return "cosmos-environment"
        return "robocasa-environment"


@pytest.fixture
def valid_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("LD_LIBRARY_PATH", runtime_receipt.H_DRIVER_LIBRARY_PATH)
    for name in runtime_receipt.FORBIDDEN_GRAPHICS_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    graphics_root = tmp_path / "graphics"
    graphics_root.mkdir()
    vendor_json = graphics_root / "10_nvidia.json"
    vendor_json.write_text(
        '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_receipt, "PINNED_EGL_VENDOR_JSON", str(vendor_json))
    monkeypatch.setenv("__EGL_VENDOR_LIBRARY_FILENAMES", str(vendor_json))
    monkeypatch.setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    library_specs = {
        "libEGL": ("libEGL.so.1", "eglGetDisplay", "egl_context_mapped", True),
        "libEGL_nvidia": (
            "libEGL_nvidia.so.0",
            "__egl_Main",
            "egl_context_mapped",
            True,
        ),
        "libGL": (
            "libGL.so.1",
            "glGetString",
            "resolved_and_dlopen_verified",
            False,
        ),
        "libGLX": (
            "libGLX.so.0",
            "glXGetProcAddress",
            "resolved_and_dlopen_verified",
            False,
        ),
        "libGLX_nvidia": (
            "libGLX_nvidia.so.0",
            "__glx_Main",
            "resolved_and_dlopen_verified",
            False,
        ),
    }
    graphics_libraries = {}
    for name, (soname, symbol, evidence_class, mapped) in library_specs.items():
        library = graphics_root / soname
        library.write_bytes(f"{name}-fixture".encode())
        graphics_libraries[name] = _file_descriptor(
            library,
            name=name,
            requested_soname=soname,
            symbol=symbol,
            evidence_class=evidence_class,
            was_mapped_after_egl_context=mapped,
        )
    graphics_userspace = {
        "egl_vendor_json": _file_descriptor(
            vendor_json,
            file_format_version="1.0.0",
            icd_library_path="libEGL_nvidia.so.0",
        ),
        "libraries": graphics_libraries,
        "gl_identity": {
            "vendor": "NVIDIA Corporation",
            "renderer": "NVIDIA H200",
        },
    }
    cosmos_python = _write_executable(tmp_path / "cosmos-env" / "bin" / "python")
    robocasa_python = _write_executable(tmp_path / "robocasa-env" / "bin" / "python")
    git = _write_executable(tmp_path / "tools" / "git")
    nvidia_smi = _write_executable(tmp_path / "tools" / "nvidia-smi")
    cosmos_prefix = cosmos_python.parents[1]
    robocasa_prefix = robocasa_python.parents[1]
    base_prefix = tmp_path / "base-python"
    base_prefix.mkdir()

    source_roots = {
        "cosmos": tmp_path / "sources" / "cosmos",
        "demian": tmp_path / "sources" / "demian",
        "robocasa": tmp_path / "sources" / "robocasa",
        "robosuite": tmp_path / "sources" / "robosuite",
    }
    commits = {
        str(source_roots["cosmos"]): "1" * 40,
        str(source_roots["demian"]): "2" * 40,
        str(source_roots["robocasa"]): "3" * 40,
        str(source_roots["robosuite"]): "4" * 40,
    }
    module_paths = {
        "cosmos_framework": source_roots["cosmos"] / "cosmos_framework" / "__init__.py",
        "environment_helper": (source_roots["cosmos"] / runtime_receipt.ENVIRONMENT_MANIFEST_HELPER_REL),
        "demian_evaluator": (source_roots["demian"] / "demian_wam" / "eval_utils" / "run_robocasa_eval.py"),
        "robocasa": source_roots["robocasa"] / "robocasa" / "__init__.py",
        "robosuite": source_roots["robosuite"] / "robosuite" / "__init__.py",
        "torch": cosmos_prefix / "lib" / "python3.11" / "site-packages" / "torch" / "__init__.py",
        "numpy": robocasa_prefix / "lib" / "python3.11" / "site-packages" / "numpy" / "__init__.py",
        "mujoco": robocasa_prefix / "lib" / "python3.11" / "site-packages" / "mujoco" / "__init__.py",
        "gymnasium": (robocasa_prefix / "lib" / "python3.11" / "site-packages" / "gymnasium" / "__init__.py"),
    }
    for root in source_roots.values():
        root.mkdir(parents=True)
    for path in module_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    asset_verification_value = {
        "schema_version": 1,
        "kind": runtime_receipt.ASSET_VERIFICATION_KIND,
        "status": "passed",
        "asset_root": str(source_roots["robocasa"]),
        "asset_root_mode": 0o755,
        "excluded_top_level_paths": [".cwam_provenance", ".git"],
        "manifest_path": str(source_roots["robocasa"] / ".cwam_provenance/asset-tree-manifest.jsonl"),
        "manifest_size_bytes": 10,
        "manifest_sha256": "1" * 64,
        "publisher_receipt_path": str(source_roots["robocasa"] / ".cwam_provenance/asset-tree-receipt.json"),
        "publisher_receipt_size_bytes": 10,
        "publisher_receipt_sha256": "2" * 64,
        "publisher_generator_id": "generator",
        "publisher_generated_at_utc": "2026-07-28T18:09:13Z",
        "robocasa_git_commit": "3" * 40,
        "payload_directory_count": 1,
        "payload_file_count": 1,
        "payload_symlink_count": 0,
        "payload_total_bytes": 1,
        "source_archive_directory": "/source",
        "source_sha256sums_sha256": "4" * 64,
        "source_sizes_sha256": "5" * 64,
    }
    asset_verification, asset_verification_sha = _write_json(
        tmp_path / "receipts" / "asset-verification.json",
        asset_verification_value,
    )
    eval_intent, eval_intent_sha = _write_json(
        tmp_path / "intent" / "eval-intent.json",
        {"kind": runtime_receipt.EVAL_INTENT_KIND, "schema_version": 1},
    )
    eval_intent.chmod(0o444)
    eval_intent_payload = eval_intent.read_bytes()
    eval_intent_descriptor = {
        "path": str(eval_intent.resolve(strict=True)),
        "size_bytes": len(eval_intent_payload),
        "sha256": eval_intent_sha,
    }
    repositories = {}
    for index, name in enumerate(("demian", "robocasa", "robosuite"), start=2):
        root = source_roots[name]
        commit = commits[str(root)]
        repositories[name] = {
            "root": str(root),
            "expected_commit": commit,
            "actual_commit": commit,
            "clean": True,
            "tracked_files": {
                "all_tracked_regular_files_match_head_blobs": True,
                "head_tree_oid": f"{index}" * 40,
                "object_format": "sha1",
                "tracked_entry_count": 1,
                "tracked_non_regular_entry_count": 0,
                "tracked_regular_file_count": 1,
                "tracked_regular_manifest_sha256": f"{index}" * 64,
                "tracked_regular_total_bytes": 1,
            },
        }
    preflight = {
        "asset_tree": asset_verification_value,
        "repositories": repositories,
        "validation": {
            "asset_tree_full_hash_match": True,
            "repository_heads_clean": True,
            "tracked_regular_files_match_head_blobs": True,
        },
    }
    preflight_payload = json.dumps(
        preflight,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    submitted_at = "2026-07-29T00:50:00Z"
    submission_verification_value = {
        "schema_version": 1,
        "kind": runtime_receipt.SUBMISSION_VERIFICATION_KIND,
        "status": "passed",
        "verified_at_utc": "2026-07-29T01:01:00Z",
        "identity": {
            "experiment_id": "CWAM-E35-R12-GATE",
            "logical_run_id": "CWAM-E35-R12-GATE-S42",
            "run_name": "CWAM-E35-R12-GATE-S42",
            "submission_nonce": "e" * 64,
            "submitted_at_utc": submitted_at,
        },
        "freshness": {
            "age_seconds": 660.0,
            "expires_at_utc": "2026-08-01T00:50:00Z",
            "max_age_seconds": 259200,
            "max_future_skew_seconds": runtime_receipt.MAX_FUTURE_SKEW_SECONDS,
            "scheduler_started_at_utc": "2026-07-29T01:00:00Z",
            "start_age_seconds": 60.0,
            "max_start_to_verification_seconds": (runtime_receipt.MAX_START_TO_CAPTURE_SECONDS),
        },
        "evaluation": {},
        "eval_intent_manifest": eval_intent_descriptor,
        "preflight": preflight,
        "preflight_sha256": hashlib.sha256(preflight_payload).hexdigest(),
        "scheduler": {},
        "reservation": {},
        "scheduler_readback_receipt": {},
        "scheduler_artifacts": {},
        "validation": {},
    }
    submission_verification, submission_verification_sha = _write_json(
        tmp_path / "receipts" / "submission-verification.json",
        submission_verification_value,
    )

    manifest_value = {
        "schema_version": 1,
        "kind": runtime_receipt.ENVIRONMENT_MANIFEST_KIND,
        "interpreter": {},
        "distributions": {},
    }
    cosmos_manifest, cosmos_manifest_sha = _write_json(
        tmp_path / "manifests" / "cosmos.json",
        manifest_value,
    )
    robocasa_manifest, robocasa_manifest_sha = _write_json(
        tmp_path / "manifests" / "robocasa.json",
        manifest_value,
    )
    cosmos_source_manifest, cosmos_source_manifest_sha = _write_json(
        source_roots["cosmos"] / "SNAPSHOT_MANIFEST.json",
        {
            "schema_version": 1,
            "source_head": commits[str(source_roots["cosmos"])],
            "source_dirty": False,
            "file_count": 0,
            "files": [],
        },
    )
    cosmos_source_manifest.chmod(0o444)
    gpus = [
        {
            "index": index,
            "name": "NVIDIA H200",
            "uuid": _uuid(index),
            "capability": [9, 0],
            "total_memory_bytes": 150_000_000_000,
        }
        for index in range(2)
    ]
    environment_base = {
        "status": "passed",
        "interpreter_sha256": "a" * 64,
        "python_version": "3.11.13",
        "distribution_count": 42,
        "distributions_sha256": "b" * 64,
    }
    packages = {
        "numpy": {
            "distribution_version": "2.2.5",
            # Simulate DeMiAn's mutable compatibility lie. It is recorded but
            # never used as the authoritative package version.
            "module_reported_version": "1.23.5",
            "version_source": "importlib.metadata",
            "module_path": str(module_paths["numpy"]),
        },
        "mujoco": {
            "distribution_version": "3.3.1",
            "module_reported_version": "3.3.1",
            "version_source": "importlib.metadata",
            "module_path": str(module_paths["mujoco"]),
        },
        "gymnasium": {
            "distribution_version": "1.1.1",
            "module_reported_version": "1.1.1",
            "version_source": "importlib.metadata",
            "module_path": str(module_paths["gymnasium"]),
        },
        "robocasa": {
            "distribution_version": "1.0.1",
            "module_reported_version": None,
            "version_source": "importlib.metadata",
            "module_path": str(module_paths["robocasa"]),
        },
        "robosuite": {
            "distribution_version": "1.5.1",
            "module_reported_version": "1.5.1",
            "version_source": "importlib.metadata",
            "module_path": str(module_paths["robosuite"]),
        },
    }
    return {
        "cosmos_python": cosmos_python,
        "robocasa_python": robocasa_python,
        "git": git,
        "nvidia_smi_executable": nvidia_smi,
        "source_roots": source_roots,
        "module_paths": module_paths,
        "commits": commits,
        "statuses": {},
        "cosmos_manifest": cosmos_manifest,
        "cosmos_manifest_sha": cosmos_manifest_sha,
        "cosmos_source_manifest": cosmos_source_manifest,
        "cosmos_source_manifest_sha": cosmos_source_manifest_sha,
        "robocasa_manifest": robocasa_manifest,
        "robocasa_manifest_sha": robocasa_manifest_sha,
        "cosmos_environment_report": {
            **environment_base,
            "interpreter_resolved_path": str(cosmos_python),
        },
        "robocasa_environment_report": {
            **environment_base,
            "interpreter_resolved_path": str(robocasa_python),
            "interpreter_sha256": "c" * 64,
            "distributions_sha256": "d" * 64,
        },
        "cosmos_probe": {
            "python": {
                "version": "3.11.13 (main, test)",
                "executable": str(cosmos_python),
                "prefix": str(cosmos_prefix),
                "base_prefix": str(base_prefix),
            },
            "torch": {
                "version": "2.10.0+cu128",
                "cuda_version": "12.8",
                "cuda_available": True,
                "cuda_device_count": 2,
                "module_path": str(module_paths["torch"]),
            },
            "cosmos_framework": {"module_path": str(module_paths["cosmos_framework"])},
            "gpus": gpus,
        },
        "robocasa_probe": {
            "python": {
                "version": "3.11.13 (main, test)",
                "executable": str(robocasa_python),
                "prefix": str(robocasa_prefix),
                "base_prefix": str(base_prefix),
            },
            "packages": packages,
            "demian_evaluator": {"module_path": str(module_paths["demian_evaluator"])},
        },
        "robocasa_egl_probes": [
            {
                "status": "passed",
                "shard_index": index,
                "cuda_visible_devices": str(index),
                "mujoco_egl_device_id": str(index),
                "mujoco_gl": "egl",
                "pyopengl_platform": "egl",
                "ld_library_path": runtime_receipt.H_DRIVER_LIBRARY_PATH,
                "egl_vendor_library_filenames": str(vendor_json),
                "glx_vendor_library_name": "nvidia",
                "robosuite_module_path": str(module_paths["robosuite"]),
                "graphics_userspace": graphics_userspace,
            }
            for index in range(2)
        ],
        "nvidia_smi": "\n".join(f"{index}, {_uuid(index)}, NVIDIA H200, 570.133.20" for index in range(2)) + "\n",
        "output": tmp_path / "output" / "runtime-receipt.json",
        "vendor_json": vendor_json,
        "graphics_userspace": graphics_userspace,
        "submission_verification": submission_verification,
        "submission_verification_sha": submission_verification_sha,
        "submission_verification_value": submission_verification_value,
        "eval_intent": eval_intent,
        "eval_intent_descriptor": eval_intent_descriptor,
        "preflight": preflight,
        "asset_verification": asset_verification,
        "asset_verification_sha": asset_verification_sha,
        "submitted_at": submitted_at,
    }


def _capture_arguments(fixture: dict[str, Any]) -> dict[str, Any]:
    fixture["output"].parent.mkdir(exist_ok=True)
    roots = fixture["source_roots"]
    commits = fixture["commits"]
    return {
        "cosmos_python": fixture["cosmos_python"],
        "robocasa_python": fixture["robocasa_python"],
        "cosmos_environment_manifest": fixture["cosmos_manifest"],
        "expected_cosmos_environment_manifest_sha256": fixture["cosmos_manifest_sha"],
        "robocasa_environment_manifest": fixture["robocasa_manifest"],
        "expected_robocasa_environment_manifest_sha256": fixture["robocasa_manifest_sha"],
        "cosmos_source_root": roots["cosmos"],
        "expected_cosmos_commit": commits[str(roots["cosmos"])],
        "cosmos_source_manifest": fixture["cosmos_source_manifest"],
        "expected_cosmos_source_manifest_sha256": fixture["cosmos_source_manifest_sha"],
        "demian_source_root": roots["demian"],
        "expected_demian_commit": commits[str(roots["demian"])],
        "robocasa_source_root": roots["robocasa"],
        "expected_robocasa_commit": commits[str(roots["robocasa"])],
        "robosuite_source_root": roots["robosuite"],
        "expected_robosuite_commit": commits[str(roots["robosuite"])],
        "experiment_id": "CWAM-E35-R12-GATE",
        "logical_run_id": "CWAM-E35-R12-GATE-S42",
        "capture_nonce": "e" * 64,
        "submitted_at_utc": fixture["submitted_at"],
        "submission_verification_receipt": fixture["submission_verification"],
        "expected_submission_verification_receipt_sha256": fixture["submission_verification_sha"],
        "asset_verification_receipt": fixture["asset_verification"],
        "expected_asset_verification_receipt_sha256": fixture["asset_verification_sha"],
        "egl_vendor_json": fixture["vendor_json"],
        "max_start_to_capture_seconds": runtime_receipt.MAX_START_TO_CAPTURE_SECONDS,
        "max_future_skew_seconds": runtime_receipt.MAX_FUTURE_SKEW_SECONDS,
        "num_gpus": 2,
        "output": fixture["output"],
        "clock": lambda: datetime(2026, 7, 29, 1, 2, 3, tzinfo=timezone.utc),
    }


def _rewrite_submission_verification(
    fixture: dict[str, Any],
    value: dict[str, Any],
) -> dict[str, Any]:
    _, digest = _write_json(fixture["submission_verification"], value)
    arguments = _capture_arguments(fixture)
    arguments["expected_submission_verification_receipt_sha256"] = digest
    return arguments


def _rehash_preflight(submission_value: dict[str, Any]) -> None:
    encoded = json.dumps(
        submission_value["preflight"],
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    submission_value["preflight_sha256"] = hashlib.sha256(encoded).hexdigest()


def _run_valid_capture(fixture: dict[str, Any]) -> tuple[dict[str, Any], str, FakeRunner]:
    runner = FakeRunner(fixture)

    def resolver(name: str) -> str:
        return str(fixture["git"] if name == "git" else fixture["nvidia_smi_executable"])

    receipt, digest = runtime_receipt.capture_runtime_receipt(
        **_capture_arguments(fixture),
        runner=runner,
        executable_resolver=resolver,
    )
    return receipt, digest, runner


def test_happy_path_publishes_read_only_receipt_and_uses_only_argv_subprocesses(
    valid_fixture: dict[str, Any],
) -> None:
    receipt, digest, runner = _run_valid_capture(valid_fixture)
    output = valid_fixture["output"]

    assert json.loads(output.read_text(encoding="utf-8")) == receipt
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert receipt["kind"] == runtime_receipt.RECEIPT_KIND
    assert receipt["status"] == "passed"
    assert receipt["schema_version"] == 2
    assert receipt["capture_started_at_utc"] == "2026-07-29T01:02:03Z"
    assert receipt["captured_at_utc"] == "2026-07-29T01:02:03Z"
    assert receipt["inputs"]["experiment_id"] == "CWAM-E35-R12-GATE"
    assert receipt["inputs"]["logical_run_id"] == "CWAM-E35-R12-GATE-S42"
    assert receipt["inputs"]["capture_nonce"] == "e" * 64
    assert receipt["inputs"]["submitted_at_utc"] == "2026-07-29T00:50:00Z"
    assert receipt["driver_environment"] == {
        "ld_library_path": runtime_receipt.H_DRIVER_LIBRARY_PATH,
        "egl_vendor_library_filenames": str(valid_fixture["vendor_json"]),
        "glx_vendor_library_name": "nvidia",
        "forbidden_environment_cleared": list(runtime_receipt.FORBIDDEN_GRAPHICS_ENVIRONMENT),
    }
    assert receipt["graphics_userspace"] == valid_fixture["graphics_userspace"]
    assert receipt["cosmos_runtime"]["torch"]["cuda_device_count"] == 2
    assert [record["shard_index"] for record in receipt["robocasa_egl"]] == [0, 1]
    assert receipt["nvidia_smi"]["driver_version"] == "570.133.20"
    numpy = receipt["robocasa_runtime"]["packages"]["numpy"]
    assert numpy["distribution_version"] == "2.2.5"
    assert numpy["module_reported_version"] == "1.23.5"
    assert numpy["version_source"] == "importlib.metadata"
    assert receipt["inputs"]["sources"]["cosmos"] == {
        "root": str(valid_fixture["source_roots"]["cosmos"]),
        "expected_commit": "1" * 40,
        "actual_commit": "1" * 40,
        "clean": True,
    }
    cosmos_root = str(valid_fixture["source_roots"]["cosmos"])
    assert not any("-C" in argv and argv[argv.index("-C") + 1] == cosmos_root for argv, _kwargs in runner.calls)
    assert len(runner.calls) == 21
    for argv, kwargs in runner.calls:
        assert isinstance(argv, list)
        assert kwargs["shell"] is False
        assert kwargs["timeout"] == runtime_receipt.COMMAND_TIMEOUT_SECONDS
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False
        if argv[3:4] == [runtime_receipt._ROBOCASA_EGL_PROBE]:
            shard = argv[-2]
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == shard
            assert kwargs["env"]["MUJOCO_EGL_DEVICE_ID"] == shard
            assert kwargs["env"]["MUJOCO_GL"] == "egl"
            assert kwargs["env"]["PYOPENGL_PLATFORM"] == "egl"
            assert kwargs["env"]["LD_LIBRARY_PATH"] == runtime_receipt.H_DRIVER_LIBRARY_PATH
            assert kwargs["env"]["__EGL_VENDOR_LIBRARY_FILENAMES"] == str(valid_fixture["vendor_json"])
            assert kwargs["env"]["__GLX_VENDOR_LIBRARY_NAME"] == "nvidia"


def test_existing_output_is_rejected_before_any_subprocess(valid_fixture: dict[str, Any]) -> None:
    valid_fixture["output"].parent.mkdir(exist_ok=True)
    valid_fixture["output"].write_text("owner data", encoding="utf-8")
    runner = FakeRunner(valid_fixture)
    with pytest.raises(runtime_receipt.RuntimeReceiptValidationError, match="refusing to overwrite"):
        runtime_receipt.capture_runtime_receipt(
            **_capture_arguments(valid_fixture),
            runner=runner,
            executable_resolver=lambda name: str(valid_fixture["git"]),
        )
    assert valid_fixture["output"].read_text(encoding="utf-8") == "owner data"
    assert runner.calls == []


def test_manifest_sha_mismatch_and_duplicate_keys_fail_before_commands(
    valid_fixture: dict[str, Any],
) -> None:
    runner = FakeRunner(valid_fixture)
    arguments = _capture_arguments(valid_fixture)
    arguments["expected_cosmos_environment_manifest_sha256"] = "0" * 64
    with pytest.raises(runtime_receipt.RuntimeReceiptValidationError, match="SHA256 mismatch"):
        runtime_receipt.capture_runtime_receipt(
            **arguments,
            runner=runner,
            executable_resolver=lambda name: str(valid_fixture["git"]),
        )
    assert runner.calls == []
    assert not valid_fixture["output"].exists()

    duplicate = (
        b'{"schema_version":1,"kind":"cwam_python_environment_manifest","kind":"cwam_python_environment_manifest"}\n'
    )
    valid_fixture["cosmos_manifest"].write_bytes(duplicate)
    arguments["expected_cosmos_environment_manifest_sha256"] = hashlib.sha256(duplicate).hexdigest()
    with pytest.raises(runtime_receipt.RuntimeReceiptValidationError, match="duplicate object key"):
        runtime_receipt.capture_runtime_receipt(
            **arguments,
            runner=runner,
            executable_resolver=lambda name: str(valid_fixture["git"]),
        )
    assert runner.calls == []
    assert not valid_fixture["output"].exists()


def test_cosmos_snapshot_manifest_is_commit_bound_and_rechecked(
    valid_fixture: dict[str, Any],
) -> None:
    arguments = _capture_arguments(valid_fixture)
    arguments["expected_cosmos_source_manifest_sha256"] = "0" * 64
    runner = FakeRunner(valid_fixture)
    with pytest.raises(
        runtime_receipt.RuntimeReceiptValidationError,
        match="snapshot manifest SHA256 mismatch",
    ):
        runtime_receipt.capture_runtime_receipt(
            **arguments,
            runner=runner,
            executable_resolver=lambda name: str(
                valid_fixture["git"] if name == "git" else valid_fixture["nvidia_smi_executable"]
            ),
        )
    assert runner.calls == []

    manifest = valid_fixture["cosmos_source_manifest"]
    manifest.chmod(0o644)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["source_head"] = "f" * 40
    _, digest = _write_json(manifest, value)
    manifest.chmod(0o444)
    arguments["expected_cosmos_source_manifest_sha256"] = digest
    with pytest.raises(
        runtime_receipt.RuntimeReceiptValidationError,
        match="snapshot source commit mismatch",
    ):
        runtime_receipt.capture_runtime_receipt(
            **arguments,
            runner=runner,
            executable_resolver=lambda name: str(
                valid_fixture["git"] if name == "git" else valid_fixture["nvidia_smi_executable"]
            ),
        )
    assert runner.calls == []
    assert not valid_fixture["output"].exists()


def test_cosmos_snapshot_manifest_drift_during_capture_is_rejected(
    valid_fixture: dict[str, Any],
) -> None:
    manifest = valid_fixture["cosmos_source_manifest"]

    class SnapshotDriftingRunner(FakeRunner):
        def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            result = super().__call__(argv, **kwargs)
            if self._label(argv) == "nvidia-smi":
                manifest.chmod(0o644)
                value = json.loads(manifest.read_text(encoding="utf-8"))
                value["extra"] = "drift"
                _write_json(manifest, value)
                manifest.chmod(0o444)
            return result

    with pytest.raises(
        runtime_receipt.RuntimeReceiptValidationError,
        match="snapshot manifest SHA256 mismatch",
    ):
        runtime_receipt.capture_runtime_receipt(
            **_capture_arguments(valid_fixture),
            runner=SnapshotDriftingRunner(valid_fixture),
            executable_resolver=lambda name: str(
                valid_fixture["git"] if name == "git" else valid_fixture["nvidia_smi_executable"]
            ),
        )
    assert not valid_fixture["output"].exists()


def test_invalid_run_identity_or_driver_environment_fails_before_commands(
    valid_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = FakeRunner(valid_fixture)
    arguments = _capture_arguments(valid_fixture)
    arguments["logical_run_id"] = "../escape"
    with pytest.raises(runtime_receipt.RuntimeReceiptValidationError, match="logical_run_id"):
        runtime_receipt.capture_runtime_receipt(
            **arguments,
            runner=runner,
            executable_resolver=lambda name: str(valid_fixture["git"]),
        )
    assert runner.calls == []

    arguments["logical_run_id"] = "CWAM-E35-R12-GATE-S42"
    arguments["capture_nonce"] = "A" * 64
    with pytest.raises(runtime_receipt.RuntimeReceiptValidationError, match="capture_nonce"):
        runtime_receipt.capture_runtime_receipt(
            **arguments,
            runner=runner,
            executable_resolver=lambda name: str(valid_fixture["git"]),
        )
    assert runner.calls == []

    arguments["capture_nonce"] = "e" * 64
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/untrusted")
    with pytest.raises(runtime_receipt.RuntimeReceiptValidationError, match="exact audited H driver allowlist"):
        runtime_receipt.capture_runtime_receipt(
            **arguments,
            runner=runner,
            executable_resolver=lambda name: str(valid_fixture["git"]),
        )
    assert runner.calls == []
    assert not valid_fixture["output"].exists()


def test_manifest_and_source_symlinks_and_incomplete_paths_are_rejected(
    valid_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    runner = FakeRunner(valid_fixture)
    arguments = _capture_arguments(valid_fixture)
    manifest_alias = valid_fixture["cosmos_manifest"].with_name("cosmos-alias.json")
    manifest_alias.symlink_to(valid_fixture["cosmos_manifest"].name)
    arguments["cosmos_environment_manifest"] = manifest_alias
    with pytest.raises(runtime_receipt.RuntimeReceiptValidationError, match="symlink"):
        runtime_receipt.capture_runtime_receipt(
            **arguments,
            runner=runner,
            executable_resolver=lambda name: str(valid_fixture["git"]),
        )
    assert runner.calls == []

    incomplete_manifest = tmp_path / ".incomplete-manifest" / "cosmos.json"
    incomplete_manifest.parent.mkdir()
    incomplete_manifest.write_bytes(valid_fixture["cosmos_manifest"].read_bytes())
    arguments["cosmos_environment_manifest"] = incomplete_manifest
    with pytest.raises(runtime_receipt.RuntimeReceiptValidationError, match=r"\.incomplete"):
        runtime_receipt.capture_runtime_receipt(
            **arguments,
            runner=runner,
            executable_resolver=lambda name: str(valid_fixture["git"]),
        )
    assert runner.calls == []

    arguments["cosmos_environment_manifest"] = valid_fixture["cosmos_manifest"]
    source_alias = tmp_path / "cosmos-source-alias"
    source_alias.symlink_to(valid_fixture["source_roots"]["cosmos"], target_is_directory=True)
    arguments["cosmos_source_root"] = source_alias
    with pytest.raises(runtime_receipt.RuntimeReceiptValidationError, match="symlink"):
        runtime_receipt.capture_runtime_receipt(
            **arguments,
            runner=runner,
            executable_resolver=lambda name: str(
                valid_fixture["git"] if name == "git" else valid_fixture["nvidia_smi_executable"]
            ),
        )
    assert runner.calls == []
    assert not valid_fixture["output"].exists()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda fixture: fixture["cosmos_probe"]["torch"].update(cuda_available=False),
            "CUDA unavailable",
        ),
        (
            lambda fixture: fixture["cosmos_probe"]["torch"].update(cuda_device_count=1),
            "visible GPU count mismatch",
        ),
        (
            lambda fixture: fixture["cosmos_probe"]["gpus"][0].update(name="NVIDIA A100"),
            "not H200",
        ),
        (
            lambda fixture: fixture["cosmos_probe"]["gpus"][0].update(capability=[8, 0]),
            "compute capability",
        ),
        (
            lambda fixture: fixture["cosmos_probe"]["gpus"][0].update(uuid=_uuid(9)),
            "UUID sets differ",
        ),
    ],
)
def test_gpu_contract_failures_never_publish(
    valid_fixture: dict[str, Any],
    mutation: Any,
    match: str,
) -> None:
    mutation(valid_fixture)
    runner = FakeRunner(valid_fixture)
    with pytest.raises(runtime_receipt.RuntimeReceiptValidationError, match=match):
        runtime_receipt.capture_runtime_receipt(
            **_capture_arguments(valid_fixture),
            runner=runner,
            executable_resolver=lambda name: str(
                valid_fixture["git"] if name == "git" else valid_fixture["nvidia_smi_executable"]
            ),
        )
    assert not valid_fixture["output"].exists()


def test_source_module_path_drift_and_dirty_git_are_rejected(
    valid_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside" / "robocasa" / "__init__.py"
    outside.parent.mkdir(parents=True)
    outside.write_text("", encoding="utf-8")
    valid_fixture["robocasa_probe"]["packages"]["robocasa"]["module_path"] = str(outside)
    runner = FakeRunner(valid_fixture)
    with pytest.raises(runtime_receipt.RuntimeReceiptValidationError, match="exact source root"):
        runtime_receipt.capture_runtime_receipt(
            **_capture_arguments(valid_fixture),
            runner=runner,
            executable_resolver=lambda name: str(
                valid_fixture["git"] if name == "git" else valid_fixture["nvidia_smi_executable"]
            ),
        )
    assert not valid_fixture["output"].exists()

    valid_fixture["robocasa_probe"]["packages"]["robocasa"]["module_path"] = str(
        valid_fixture["module_paths"]["robocasa"]
    )
    valid_fixture["statuses"][str(valid_fixture["source_roots"]["demian"])] = " M evaluator.py\n"
    runner = FakeRunner(valid_fixture)
    with pytest.raises(runtime_receipt.RuntimeReceiptValidationError, match="checkout is dirty"):
        runtime_receipt.capture_runtime_receipt(
            **_capture_arguments(valid_fixture),
            runner=runner,
            executable_resolver=lambda name: str(
                valid_fixture["git"] if name == "git" else valid_fixture["nvidia_smi_executable"]
            ),
        )
    assert not valid_fixture["output"].exists()


def test_environment_report_and_probe_json_are_strict(valid_fixture: dict[str, Any]) -> None:
    valid_fixture["cosmos_environment_report"]["unexpected"] = True
    runner = FakeRunner(valid_fixture)
    with pytest.raises(runtime_receipt.RuntimeReceiptValidationError, match="keys mismatch"):
        runtime_receipt.capture_runtime_receipt(
            **_capture_arguments(valid_fixture),
            runner=runner,
            executable_resolver=lambda name: str(
                valid_fixture["git"] if name == "git" else valid_fixture["nvidia_smi_executable"]
            ),
        )
    assert not valid_fixture["output"].exists()

    valid_fixture["cosmos_environment_report"].pop("unexpected")

    class DuplicateProbeRunner(FakeRunner):
        def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            result = super().__call__(argv, **kwargs)
            if self._label(argv) == "cosmos-probe":
                result.stdout = '{"python":{},"python":{}}\n'
            return result

    runner = DuplicateProbeRunner(valid_fixture)
    with pytest.raises(runtime_receipt.RuntimeReceiptValidationError, match="duplicate object key"):
        runtime_receipt.capture_runtime_receipt(
            **_capture_arguments(valid_fixture),
            runner=runner,
            executable_resolver=lambda name: str(
                valid_fixture["git"] if name == "git" else valid_fixture["nvidia_smi_executable"]
            ),
        )
    assert not valid_fixture["output"].exists()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("cuda_visible_devices", "0", "cuda_visible_devices mismatch"),
        ("mujoco_egl_device_id", "0", "mujoco_egl_device_id mismatch"),
        ("ld_library_path", "/tmp/untrusted", "ld_library_path mismatch"),
        ("robosuite_module_path", "/tmp/outside/robosuite/__init__.py", "does not exist"),
    ],
)
def test_per_shard_egl_probe_must_bind_exact_device_and_driver_environment(
    valid_fixture: dict[str, Any],
    field: str,
    value: str,
    match: str,
) -> None:
    valid_fixture["robocasa_egl_probes"][1][field] = value
    runner = FakeRunner(valid_fixture)
    with pytest.raises(runtime_receipt.RuntimeReceiptValidationError, match=match):
        runtime_receipt.capture_runtime_receipt(
            **_capture_arguments(valid_fixture),
            runner=runner,
            executable_resolver=lambda name: str(
                valid_fixture["git"] if name == "git" else valid_fixture["nvidia_smi_executable"]
            ),
        )
    assert not valid_fixture["output"].exists()


def test_subprocess_timeout_and_failure_never_publish(valid_fixture: dict[str, Any]) -> None:
    runner = FakeRunner(valid_fixture)
    runner.timeout_label = "cosmos-probe"
    with pytest.raises(runtime_receipt.RuntimeReceiptCommandError, match="timed out"):
        runtime_receipt.capture_runtime_receipt(
            **_capture_arguments(valid_fixture),
            runner=runner,
            executable_resolver=lambda name: str(
                valid_fixture["git"] if name == "git" else valid_fixture["nvidia_smi_executable"]
            ),
        )
    assert not valid_fixture["output"].exists()

    runner = FakeRunner(valid_fixture)
    runner.failure_label = "nvidia-smi"
    with pytest.raises(runtime_receipt.RuntimeReceiptCommandError, match="exit code 7"):
        runtime_receipt.capture_runtime_receipt(
            **_capture_arguments(valid_fixture),
            runner=runner,
            executable_resolver=lambda name: str(
                valid_fixture["git"] if name == "git" else valid_fixture["nvidia_smi_executable"]
            ),
        )
    assert not valid_fixture["output"].exists()


def test_source_becoming_dirty_during_capture_is_rejected(valid_fixture: dict[str, Any]) -> None:
    demian_root = str(valid_fixture["source_roots"]["demian"])

    class DriftingRunner(FakeRunner):
        def __init__(self, fixture: dict[str, Any]) -> None:
            super().__init__(fixture)
            self.demian_status_calls = 0

        def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            label = self._label(argv)
            if label == f"git-status:{demian_root}":
                self.demian_status_calls += 1
                if self.demian_status_calls == 2:
                    self.calls.append((list(argv), dict(kwargs)))
                    return subprocess.CompletedProcess(argv, 0, "?? __pycache__/drift.pyc\n", "")
            return super().__call__(argv, **kwargs)

    runner = DriftingRunner(valid_fixture)
    with pytest.raises(runtime_receipt.RuntimeReceiptValidationError, match="checkout is dirty"):
        runtime_receipt.capture_runtime_receipt(
            **_capture_arguments(valid_fixture),
            runner=runner,
            executable_resolver=lambda name: str(
                valid_fixture["git"] if name == "git" else valid_fixture["nvidia_smi_executable"]
            ),
        )
    assert runner.demian_status_calls == 2
    assert not valid_fixture["output"].exists()


def test_external_submission_and_asset_receipts_are_identity_bound(
    valid_fixture: dict[str, Any],
) -> None:
    submission_value = json.loads(valid_fixture["submission_verification"].read_text(encoding="utf-8"))
    submission_value["identity"]["submission_nonce"] = "f" * 64
    _, digest = _write_json(
        valid_fixture["submission_verification"],
        submission_value,
    )
    arguments = _capture_arguments(valid_fixture)
    arguments["expected_submission_verification_receipt_sha256"] = digest
    with pytest.raises(
        runtime_receipt.RuntimeReceiptValidationError,
        match="identity does not match",
    ):
        runtime_receipt.capture_runtime_receipt(
            **arguments,
            runner=FakeRunner(valid_fixture),
            executable_resolver=lambda name: str(
                valid_fixture["git"] if name == "git" else valid_fixture["nvidia_smi_executable"]
            ),
        )

    submission_value["identity"]["submission_nonce"] = "e" * 64
    _, restored_submission_digest = _write_json(
        valid_fixture["submission_verification"],
        submission_value,
    )
    asset_value = json.loads(valid_fixture["asset_verification"].read_text(encoding="utf-8"))
    asset_value["payload_symlink_count"] = 1
    _, asset_digest = _write_json(valid_fixture["asset_verification"], asset_value)
    arguments = _capture_arguments(valid_fixture)
    arguments["expected_submission_verification_receipt_sha256"] = restored_submission_digest
    arguments["expected_asset_verification_receipt_sha256"] = asset_digest
    with pytest.raises(
        runtime_receipt.RuntimeReceiptValidationError,
        match="zero symlinks",
    ):
        runtime_receipt.capture_runtime_receipt(
            **arguments,
            runner=FakeRunner(valid_fixture),
            executable_resolver=lambda name: str(
                valid_fixture["git"] if name == "git" else valid_fixture["nvidia_smi_executable"]
            ),
        )


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_submission_verification_expanded_top_level_schema_is_exact(
    valid_fixture: dict[str, Any],
    mutation: str,
) -> None:
    value = copy.deepcopy(valid_fixture["submission_verification_value"])
    if mutation == "missing":
        value.pop("preflight")
    else:
        value["unexpected"] = True
    arguments = _rewrite_submission_verification(valid_fixture, value)
    runner = FakeRunner(valid_fixture)
    with pytest.raises(runtime_receipt.RuntimeReceiptValidationError, match="keys mismatch"):
        runtime_receipt.capture_runtime_receipt(
            **arguments,
            runner=runner,
            executable_resolver=lambda name: str(
                valid_fixture["git"] if name == "git" else valid_fixture["nvidia_smi_executable"]
            ),
        )
    assert runner.calls == []
    assert not valid_fixture["output"].exists()


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("digest", "canonical SHA-256 mismatch"),
        ("asset", "asset tree"),
        ("validation", "validation"),
    ),
)
def test_submission_preflight_digest_asset_and_verdict_are_bound(
    valid_fixture: dict[str, Any],
    mutation: str,
    match: str,
) -> None:
    value = copy.deepcopy(valid_fixture["submission_verification_value"])
    if mutation == "digest":
        value["preflight"]["validation"]["repository_heads_clean"] = False
    elif mutation == "asset":
        value["preflight"]["asset_tree"]["payload_total_bytes"] += 1
        _rehash_preflight(value)
    else:
        value["preflight"]["validation"]["repository_heads_clean"] = False
        _rehash_preflight(value)
    arguments = _rewrite_submission_verification(valid_fixture, value)
    runner = FakeRunner(valid_fixture)
    with pytest.raises(runtime_receipt.RuntimeReceiptValidationError, match=match):
        runtime_receipt.capture_runtime_receipt(
            **arguments,
            runner=runner,
            executable_resolver=lambda name: str(
                valid_fixture["git"] if name == "git" else valid_fixture["nvidia_smi_executable"]
            ),
        )
    assert not valid_fixture["output"].exists()


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("root", "root"),
        ("commit", "commit"),
        ("clean", "clean"),
        ("tracked_schema", "keys mismatch"),
        ("tracked_verdict", "tracked"),
    ),
)
def test_submission_preflight_repository_evidence_is_live_source_bound(
    valid_fixture: dict[str, Any],
    mutation: str,
    match: str,
) -> None:
    value = copy.deepcopy(valid_fixture["submission_verification_value"])
    repository = value["preflight"]["repositories"]["demian"]
    if mutation == "root":
        repository["root"] = str(valid_fixture["source_roots"]["robocasa"])
    elif mutation == "commit":
        repository["actual_commit"] = "f" * 40
    elif mutation == "clean":
        repository["clean"] = False
    elif mutation == "tracked_schema":
        repository["tracked_files"].pop("head_tree_oid")
    else:
        repository["tracked_files"]["all_tracked_regular_files_match_head_blobs"] = False
    _rehash_preflight(value)
    arguments = _rewrite_submission_verification(valid_fixture, value)
    with pytest.raises(runtime_receipt.RuntimeReceiptValidationError, match=match):
        runtime_receipt.capture_runtime_receipt(
            **arguments,
            runner=FakeRunner(valid_fixture),
            executable_resolver=lambda name: str(
                valid_fixture["git"] if name == "git" else valid_fixture["nvidia_smi_executable"]
            ),
        )
    assert not valid_fixture["output"].exists()


@pytest.mark.parametrize(("mutation", "match"), (("hash", "descriptor"), ("mode", "0444")))
def test_eval_intent_descriptor_is_live_file_bound(
    valid_fixture: dict[str, Any],
    mutation: str,
    match: str,
) -> None:
    value = copy.deepcopy(valid_fixture["submission_verification_value"])
    if mutation == "hash":
        value["eval_intent_manifest"]["sha256"] = "f" * 64
    else:
        valid_fixture["eval_intent"].chmod(0o644)
    arguments = _rewrite_submission_verification(valid_fixture, value)
    runner = FakeRunner(valid_fixture)
    with pytest.raises(runtime_receipt.RuntimeReceiptValidationError, match=match):
        runtime_receipt.capture_runtime_receipt(
            **arguments,
            runner=runner,
            executable_resolver=lambda name: str(
                valid_fixture["git"] if name == "git" else valid_fixture["nvidia_smi_executable"]
            ),
        )
    assert runner.calls == []
    assert not valid_fixture["output"].exists()


def test_eval_intent_drift_during_runtime_capture_is_rejected(
    valid_fixture: dict[str, Any],
) -> None:
    intent = valid_fixture["eval_intent"]

    class IntentDriftingRunner(FakeRunner):
        def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            result = super().__call__(argv, **kwargs)
            if self._label(argv) == "nvidia-smi":
                payload = intent.read_bytes().replace(b"cwam_", b"xwam_", 1)
                intent.chmod(0o644)
                intent.write_bytes(payload)
                intent.chmod(0o444)
            return result

    with pytest.raises(runtime_receipt.RuntimeReceiptValidationError, match="descriptor"):
        runtime_receipt.capture_runtime_receipt(
            **_capture_arguments(valid_fixture),
            runner=IntentDriftingRunner(valid_fixture),
            executable_resolver=lambda name: str(
                valid_fixture["git"] if name == "git" else valid_fixture["nvidia_smi_executable"]
            ),
        )
    assert not valid_fixture["output"].exists()


def test_graphics_userspace_is_nvidia_cross_shard_and_live_file_bound(
    valid_fixture: dict[str, Any],
) -> None:
    valid_fixture["robocasa_egl_probes"][0]["graphics_userspace"]["gl_identity"]["vendor"] = "Mesa Project"
    with pytest.raises(runtime_receipt.RuntimeReceiptValidationError, match="not NVIDIA"):
        runtime_receipt.capture_runtime_receipt(
            **_capture_arguments(valid_fixture),
            runner=FakeRunner(valid_fixture),
            executable_resolver=lambda name: str(
                valid_fixture["git"] if name == "git" else valid_fixture["nvidia_smi_executable"]
            ),
        )


@pytest.mark.parametrize(
    ("library_name", "frontend_symbol"),
    [
        ("libEGL_nvidia", "eglQueryString"),
        ("libGLX_nvidia", "glXGetProcAddress"),
    ],
)
def test_runtime_receipt_rejects_frontend_symbol_as_vendor_abi(
    valid_fixture: dict[str, Any],
    library_name: str,
    frontend_symbol: str,
) -> None:
    valid_fixture["robocasa_egl_probes"][0]["graphics_userspace"]["libraries"][library_name]["symbol"] = frontend_symbol
    with pytest.raises(
        runtime_receipt.RuntimeReceiptValidationError,
        match=rf"libraries\.{library_name}\.symbol mismatch",
    ):
        runtime_receipt.capture_runtime_receipt(
            **_capture_arguments(valid_fixture),
            runner=FakeRunner(valid_fixture),
            executable_resolver=lambda name: str(
                valid_fixture["git"] if name == "git" else valid_fixture["nvidia_smi_executable"]
            ),
        )
    assert not valid_fixture["output"].exists()


def test_graphics_file_drift_during_capture_is_rejected(
    valid_fixture: dict[str, Any],
) -> None:
    library = Path(valid_fixture["graphics_userspace"]["libraries"]["libEGL"]["canonical_path"])

    class GraphicsDriftingRunner(FakeRunner):
        def __call__(
            self,
            argv: list[str],
            **kwargs: Any,
        ) -> subprocess.CompletedProcess[str]:
            result = super().__call__(argv, **kwargs)
            if self._label(argv) == "nvidia-smi":
                payload = library.read_bytes()
                library.write_bytes(b"X" * len(payload))
            return result

    with pytest.raises(
        runtime_receipt.RuntimeReceiptValidationError,
        match="does not match the live file",
    ):
        runtime_receipt.capture_runtime_receipt(
            **_capture_arguments(valid_fixture),
            runner=GraphicsDriftingRunner(valid_fixture),
            executable_resolver=lambda name: str(
                valid_fixture["git"] if name == "git" else valid_fixture["nvidia_smi_executable"]
            ),
        )


def test_graphics_soname_parent_alias_is_canonicalized_before_validation(tmp_path: Path) -> None:
    canonical_parent = tmp_path / "usr" / "lib"
    canonical_parent.mkdir(parents=True)
    canonical_library = canonical_parent / "libEGL.so.1.1.0"
    canonical_library.write_bytes(b"egl-fixture")
    soname = canonical_parent / "libEGL.so.1"
    soname.symlink_to(canonical_library.name)
    parent_alias = tmp_path / "lib"
    parent_alias.symlink_to(canonical_parent, target_is_directory=True)
    supplied_soname = parent_alias / soname.name

    descriptor = _file_descriptor(
        supplied_soname.resolve(strict=True),
        name="libEGL",
        requested_soname="libEGL.so.1",
        symbol="eglGetDisplay",
        evidence_class="egl_context_mapped",
        was_mapped_after_egl_context=True,
    )
    normalized = runtime_receipt._validate_graphics_file_descriptor(
        descriptor,
        "graphics_userspace.libraries.libEGL",
        extra_keys=frozenset(
            {
                "name",
                "requested_soname",
                "symbol",
                "evidence_class",
                "was_mapped_after_egl_context",
            }
        ),
    )

    assert normalized["supplied_path"] == str(canonical_library)
    assert normalized["canonical_path"] == str(canonical_library)
    assert normalized["requested_soname"] == "libEGL.so.1"


def test_graphics_validator_still_rejects_an_ancestor_symlink(tmp_path: Path) -> None:
    canonical_parent = tmp_path / "usr" / "lib"
    canonical_parent.mkdir(parents=True)
    canonical_library = canonical_parent / "libEGL.so.1.1.0"
    canonical_library.write_bytes(b"egl-fixture")
    parent_alias = tmp_path / "lib"
    parent_alias.symlink_to(canonical_parent, target_is_directory=True)
    supplied_alias = parent_alias / canonical_library.name
    descriptor = _file_descriptor(supplied_alias)
    descriptor["supplied_path"] = str(supplied_alias)

    with pytest.raises(
        runtime_receipt.RuntimeReceiptValidationError,
        match="must contain no symlink components",
    ):
        runtime_receipt._validate_graphics_file_descriptor(
            descriptor,
            "graphics_userspace.libraries.libEGL",
        )


def test_atomic_writer_refuses_symlink_incomplete_and_overwrite_paths(tmp_path: Path) -> None:
    parent = tmp_path / "real"
    parent.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(parent, target_is_directory=True)
    with pytest.raises(runtime_receipt.RuntimeReceiptValidationError, match="symlink"):
        runtime_receipt.write_json_atomic_exclusive(alias / "receipt.json", {"status": "passed"})
    with pytest.raises(runtime_receipt.RuntimeReceiptValidationError, match=r"\.incomplete"):
        runtime_receipt.write_json_atomic_exclusive(
            parent / "receipt.incomplete.json",
            {"status": "passed"},
        )

    target = parent / "receipt.json"
    digest = runtime_receipt.write_json_atomic_exclusive(target, {"status": "passed"})
    assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
    with pytest.raises(runtime_receipt.RuntimeReceiptValidationError, match="refusing to overwrite"):
        runtime_receipt.write_json_atomic_exclusive(target, {"status": "different"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "passed"}
    assert not list(parent.glob(".*.tmp"))


def test_leaf_python_symlink_is_invoked_but_bound_to_resolved_binary(
    valid_fixture: dict[str, Any],
) -> None:
    real_python = valid_fixture["cosmos_python"]
    link = real_python.parent / "cosmos-python-link"
    link.symlink_to(real_python.name)
    valid_fixture["cosmos_python"] = link
    # Both the manifest verifier and probe report the canonical executable.
    valid_fixture["cosmos_environment_report"]["interpreter_resolved_path"] = str(real_python)
    valid_fixture["cosmos_probe"]["python"]["executable"] = str(real_python)

    receipt, _, runner = _run_valid_capture(valid_fixture)
    assert receipt["inputs"]["interpreters"]["cosmos"] == {
        "launch_path": str(link),
        "resolved_path": str(real_python),
    }
    cosmos_commands = [argv for argv, _ in runner.calls if argv[0] == str(link)]
    assert len(cosmos_commands) == 3


def test_test_fixture_isolated_from_mutation(valid_fixture: dict[str, Any]) -> None:
    cloned = copy.deepcopy(valid_fixture)
    cloned["cosmos_probe"]["gpus"][0]["name"] = "mutated"
    assert valid_fixture["cosmos_probe"]["gpus"][0]["name"] == "NVIDIA H200"
