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
            return "cosmos-probe" if argv[3] == runtime_receipt._COSMOS_PROBE else "robocasa-probe"
        if Path(argv[0]) == self.fixture["cosmos_python"]:
            return "cosmos-environment"
        return "robocasa-environment"


@pytest.fixture
def valid_fixture(tmp_path: Path) -> dict[str, Any]:
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
    commits = {
        str(source_roots["cosmos"]): "1" * 40,
        str(source_roots["demian"]): "2" * 40,
        str(source_roots["robocasa"]): "3" * 40,
        str(source_roots["robosuite"]): "4" * 40,
    }
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
        "nvidia_smi": "\n".join(f"{index}, {_uuid(index)}, NVIDIA H200, 570.133.20" for index in range(2)) + "\n",
        "output": tmp_path / "output" / "runtime-receipt.json",
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
        "demian_source_root": roots["demian"],
        "expected_demian_commit": commits[str(roots["demian"])],
        "robocasa_source_root": roots["robocasa"],
        "expected_robocasa_commit": commits[str(roots["robocasa"])],
        "robosuite_source_root": roots["robosuite"],
        "expected_robosuite_commit": commits[str(roots["robosuite"])],
        "num_gpus": 2,
        "output": fixture["output"],
    }


def _run_valid_capture(fixture: dict[str, Any]) -> tuple[dict[str, Any], str, FakeRunner]:
    runner = FakeRunner(fixture)

    def resolver(name: str) -> str:
        return str(fixture["git"] if name == "git" else fixture["nvidia_smi_executable"])

    receipt, digest = runtime_receipt.capture_runtime_receipt(
        **_capture_arguments(fixture),
        runner=runner,
        executable_resolver=resolver,
        clock=lambda: datetime(2026, 7, 29, 1, 2, 3, tzinfo=timezone.utc),
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
    assert receipt["captured_at_utc"] == "2026-07-29T01:02:03Z"
    assert receipt["cosmos_runtime"]["torch"]["cuda_device_count"] == 2
    assert receipt["nvidia_smi"]["driver_version"] == "570.133.20"
    numpy = receipt["robocasa_runtime"]["packages"]["numpy"]
    assert numpy["distribution_version"] == "2.2.5"
    assert numpy["module_reported_version"] == "1.23.5"
    assert numpy["version_source"] == "importlib.metadata"
    assert len(runner.calls) == 23
    for argv, kwargs in runner.calls:
        assert isinstance(argv, list)
        assert kwargs["shell"] is False
        assert kwargs["timeout"] == runtime_receipt.COMMAND_TIMEOUT_SECONDS
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False


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
