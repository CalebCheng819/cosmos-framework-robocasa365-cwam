"""Structural tests for the B4 RoboCasa365 DLC launcher."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


LAUNCHER = Path(__file__).with_name("launch_cwam_b4_robocasa365_eval_dlc.sh")
SHELL_UTILS = Path(__file__).with_name("cwam_b4_eval_shell_utils.sh")

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _script() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_launcher_is_valid_bash() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)


def test_launcher_stages_validator_inputs_on_worker_local_storage() -> None:
    script = _script()

    assert 'readonly SOURCE_ENV_ARCHIVE="$OSS_ROOT/inputs/envs/archives/' in script
    assert 'readonly PUBLICATION_ENV_ARCHIVE="$INPUT_ROOT/envs/archives/' in script
    assert 'readonly LOCAL_ENV_ARCHIVE="$SCRATCH_ROOT/inputs/archives/' in script
    assert 'readonly LOCAL_ENV_RECEIPT="$SCRATCH_ROOT/control/receipts/' in script
    assert 'readonly LOCAL_ROBOCASA_ARCHIVE="$SCRATCH_ROOT/inputs/archives/' in script
    assert 'readonly LOCAL_ROBOCASA_OVERLAY="$SCRATCH_ROOT/inputs/archives/' in script
    assert 'chmod 0444 -- "$destination"' in script
    assert 'copy_regular_file "$SOURCE_ENV_ARCHIVE" "$LOCAL_ENV_ARCHIVE" 14736947200' in script
    assert 'copy_regular_file "$SOURCE_ENV_RECEIPT" "$LOCAL_ENV_RECEIPT" 2124' in script
    assert 'copy_regular_file "$SOURCE_ROBOCASA_ARCHIVE" "$LOCAL_ROBOCASA_ARCHIVE" 11101859840' in script
    assert 'copy_regular_file "$SOURCE_ROBOCASA_OVERLAY" "$LOCAL_ROBOCASA_OVERLAY" 2181120' in script


def test_launcher_preserves_environment_receipt_publication_binding() -> None:
    script = _script()

    assert '--archive "$LOCAL_ENV_ARCHIVE"' in script
    assert '--receipt "$LOCAL_ENV_RECEIPT"' in script
    assert '--expected-publication-archive "$PUBLICATION_ENV_ARCHIVE"' in script
    assert script.index('copy_regular_file "$SOURCE_ENV_ARCHIVE"') < script.index(
        'python3 "$ENV_HELPER" verify-extract-metadata'
    )


def test_launcher_validates_worker_local_robocasa_archives() -> None:
    script = _script()

    assert '--archive "$LOCAL_ROBOCASA_ARCHIVE" --expected-archive-bytes 11101859840' in script
    assert '--overlay "$LOCAL_ROBOCASA_OVERLAY" --expected-overlay-bytes 2181120' in script
    assert 'readonly REQUIRED_LOCAL_BYTES=161061273600' in script
    assert 'worker-local /tmp has less than 150 GiB available' in script


def test_launcher_prepares_exact_empty_extraction_stages() -> None:
    script = _script()

    assert 'prepare_empty_stage "$COSMOS_ENV"' in script
    assert 'prepare_empty_stage "$ROBOCASA_REPO"' in script
    assert '[[ -d "$stage" && ! -L "$stage" ]]' in script
    assert 'find "$stage" -mindepth 1 -maxdepth 1 -print -quit' in script
    assert script.index('prepare_empty_stage "$COSMOS_ENV"') < script.index(
        'python3 "$ENV_HELPER" verify-extract-metadata'
    )
    assert script.index('prepare_empty_stage "$ROBOCASA_REPO"') < script.index(
        'python3 "$ASSET_HELPER" verify-extract'
    )


def test_launcher_stages_cosmos_runtime_closure_at_wrapper_path() -> None:
    script = _script()

    assert "readonly LOCAL_RUNTIME_STAGING_ROOT=/tmp/cwam_cosmos3_robocasa365_eval_nvme/staging" in script
    assert '"$RUNTIME_CLOSURE_SOURCE/cosmos-r21-private-glibc235-v1"' in script
    assert '"$RUNTIME_CLOSURE_SOURCE/cosmos-r21-gcc13-runtime-v3"' in script
    assert '"$LOCAL_PRIVATE_GLIBC/root/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2"' in script
    assert '"$LOCAL_GCC_RUNTIME/runtime-root/lib/libstdc++.so.6.0.32"' in script
    assert script.index("stage_runtime_directory \\\n    cosmos_private_glibc") < script.index('bash "$RUNNER"')


def test_trusted_executable_accepts_in_root_symlink_and_rejects_escape(tmp_path: Path) -> None:
    trusted_root = tmp_path / "envs"
    target = trusted_root / "python" / "bin" / "python3.11"
    entrypoint = trusted_root / "robocasa" / "bin" / "python"
    target.parent.mkdir(parents=True)
    entrypoint.parent.mkdir(parents=True)
    target.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
    target.chmod(0o755)
    entrypoint.symlink_to(target)

    environment = {
        "UTILS": str(SHELL_UTILS),
        "ENTRYPOINT": str(entrypoint),
        "TRUSTED_ROOT": str(trusted_root),
    }
    accepted = subprocess.run(
        ["bash", "-c", 'source "$UTILS"; cwam_require_trusted_executable "$ENTRYPOINT" "$TRUSTED_ROOT"'],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stderr

    outside = tmp_path / "outside-python"
    outside.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
    outside.chmod(0o755)
    entrypoint.unlink()
    entrypoint.symlink_to(outside)
    rejected = subprocess.run(
        ["bash", "-c", 'source "$UTILS"; cwam_require_trusted_executable "$ENTRYPOINT" "$TRUSTED_ROOT"'],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "escapes trusted root" in rejected.stderr


def test_trusted_executable_rejects_broken_symlink(tmp_path: Path) -> None:
    trusted_root = tmp_path / "envs"
    entrypoint = trusted_root / "robocasa" / "bin" / "python"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.symlink_to(trusted_root / "missing-python")
    result = subprocess.run(
        ["bash", "-c", 'source "$UTILS"; cwam_require_trusted_executable "$ENTRYPOINT" "$TRUSTED_ROOT"'],
        env={
            "UTILS": str(SHELL_UTILS),
            "ENTRYPOINT": str(entrypoint),
            "TRUSTED_ROOT": str(trusted_root),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "missing, broken, or not executable" in result.stderr
