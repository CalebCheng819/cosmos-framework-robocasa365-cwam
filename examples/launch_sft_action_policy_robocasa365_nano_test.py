# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import os
import subprocess
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LAUNCHER = _REPO_ROOT / "examples/launch_sft_action_policy_robocasa365_nano.sh"
_B5_TOML = _REPO_ROOT / "examples/toml/sft_config/action_policy_robocasa365_b5_visual_object_memory_nano.toml"


def _touch(path: Path, payload: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def _base_env(tmp_path: Path, *, checkpoint_shards: int, checkpoint_json: bool) -> dict[str, str]:
    processed = tmp_path / "processed"
    source = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    tokenizer = tmp_path / "tokenizer"
    control_cache = tmp_path / "control-cache"
    latent_cache = tmp_path / "latent-cache"
    visual_cache = tmp_path / "visual-cache"
    bin_dir = tmp_path / "bin"

    for relative in (
        "_SUCCESS.json",
        "dataset_manifest.json",
        "splits/train.parquet",
        "splits/val.parquet",
        "splits/test.parquet",
        "stats/robocasa365_native_frame_wise_relative_rot6d.json",
    ):
        _touch(processed / relative)
    (source / "videos").mkdir(parents=True)
    _touch(checkpoint / "model/.metadata")
    for index in range(checkpoint_shards):
        _touch(checkpoint / f"model/__{index}_0.distcp")
    if checkpoint_json:
        _touch(checkpoint / "checkpoint.json")
    for filename in ("tokenizer_config.json", "tokenizer.json", "config.json"):
        _touch(tokenizer / filename)
    for cache in (control_cache, latent_cache, visual_cache):
        _touch(cache / "manifest.json")
        _touch(cache / "_SUCCESS.json")
    _touch(tmp_path / "Wan2.2_VAE.pth")
    torchrun = bin_dir / "torchrun"
    _touch(torchrun, "#!/usr/bin/env bash\nprintf 'TORCHRUN_ARG=%s\\n' \"$@\"\n")
    torchrun.chmod(0o755)

    return {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TOML_FILE": str(_B5_TOML),
        "BASE_CHECKPOINT_PATH": str(checkpoint),
        "WAN_VAE_PATH": str(tmp_path / "Wan2.2_VAE.pth"),
        "ROBOCASA365_ROOT": str(processed),
        "ROBOCASA365_SOURCE_ROOT": str(source),
        "QWEN3_VL_TOKENIZER_PATH": str(tokenizer),
        "ROBOCASA365_CONTROL_CACHE_ROOT": str(control_cache),
        "ROBOCASA365_STRONG_LATENT_CACHE_ROOT": str(latent_cache),
        "ROBOCASA365_VISUAL_HISTORY_CACHE_ROOT": str(visual_cache),
        "ROBOCASA365_STRONG_LATENT_CACHE_MODE": "required",
        "CWAM_STRONG_METADATA_ONLY_VIDEO_TOKENIZER": "1",
        "NPROC_PER_NODE": "8",
        "OUTPUT_ROOT": str(tmp_path / "output"),
    }


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_LAUNCHER)],
        cwd=_REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_b5_production_batch_contract_matches_toml_microbatch() -> None:
    with _B5_TOML.open("rb") as handle:
        config = tomllib.load(handle)

    samples_per_rank = config["dataloader_train"]["max_samples_per_batch"]
    grad_accum_iter = config["trainer"]["grad_accum_iter"]
    memory_gate = config["trainer"]["callbacks"]["production_batch_memory_gate"]

    assert samples_per_rank == 64
    assert memory_gate["expected_samples_per_microbatch"] == samples_per_rank
    assert memory_gate["expected_grad_accum_iter"] == grad_accum_iter


def test_b5_formal_requires_official_six_shard_checkpoint(tmp_path: Path) -> None:
    env = _base_env(tmp_path, checkpoint_shards=6, checkpoint_json=True)
    env["RUN_NAME"] = "CWAM-COSMOS3-B5-FORMAL"

    result = _run(env)

    assert result.returncode == 0, result.stdout
    assert "TORCHRUN_ARG=--nproc_per_node=8" in result.stdout
    assert "action_policy_robocasa365_b5_visual_object_memory_nano.toml" in result.stdout


def test_b5_b4_probe_warm_start_is_explicit_and_skips_only_new_memory(tmp_path: Path) -> None:
    env = _base_env(tmp_path, checkpoint_shards=8, checkpoint_json=False)
    env["RUN_NAME"] = "CWAM-COSMOS3-B5-PROBE-R1"
    env["CWAM_B5_RUN_CHECKPOINT_MODE"] = "b4"

    result = _run(env)

    assert result.returncode == 0, result.stdout
    assert "TORCHRUN_ARG=checkpoint.keys_to_skip_loading=[visual_object_memory,visual_memory_]" in result.stdout
    assert "TORCHRUN_ARG=optimizer.keys_to_select=[visual_object_memory,visual_memory_]" in result.stdout
    assert (
        "TORCHRUN_ARG=trainer.callbacks.action_gradient_evidence.families="
        "[visual_object_memory,visual_memory_]"
    ) in result.stdout
    assert "TORCHRUN_ARG=checkpoint.load_training_state=false" in result.stdout


def test_b5_b4_probe_can_train_the_complete_strong_action_expert(tmp_path: Path) -> None:
    env = _base_env(tmp_path, checkpoint_shards=8, checkpoint_json=False)
    env["RUN_NAME"] = "CWAM-COSMOS3-B5-PROBE-R2"
    env["CWAM_B5_RUN_CHECKPOINT_MODE"] = "b4"
    env["CWAM_B5_PROBE_TRAIN_SCOPE"] = "strong_action_expert"

    result = _run(env)

    assert result.returncode == 0, result.stdout
    assert "TORCHRUN_ARG=optimizer.keys_to_select=[strong_action_expert]" in result.stdout
    assert (
        "TORCHRUN_ARG=trainer.callbacks.action_gradient_evidence.families=[strong_action_expert]"
        in result.stdout
    )


def test_b5_r5_probe_loads_r4_and_trains_complete_strong_action_expert(tmp_path: Path) -> None:
    env = _base_env(tmp_path, checkpoint_shards=8, checkpoint_json=False)
    env["RUN_NAME"] = "CWAM-COSMOS3-B5-VOM-PROBE-R5"
    env["CWAM_B5_RUN_CHECKPOINT_MODE"] = "r4"

    result = _run(env)

    assert result.returncode == 0, result.stdout
    assert (
        "TORCHRUN_ARG=checkpoint.keys_to_skip_loading="
        "[visual_memory_delta_main,visual_memory_delta_conditioners]" in result.stdout
    )
    assert "TORCHRUN_ARG=optimizer.keys_to_select=[strong_action_expert]" in result.stdout
    assert (
        "TORCHRUN_ARG=trainer.callbacks.action_gradient_evidence.families=[strong_action_expert]"
        in result.stdout
    )
    assert "TORCHRUN_ARG=checkpoint.load_training_state=false" in result.stdout


def test_b5_run_checkpoint_mode_cannot_be_used_for_formal_name(tmp_path: Path) -> None:
    env = _base_env(tmp_path, checkpoint_shards=8, checkpoint_json=False)
    env["RUN_NAME"] = "CWAM-COSMOS3-B5-FORMAL"
    env["CWAM_B5_RUN_CHECKPOINT_MODE"] = "b4"

    result = _run(env)

    assert result.returncode != 0
    assert "run-checkpoint warm start is probe-only" in result.stdout


def test_b5_r4_formal_warm_start_requires_explicit_authorization(tmp_path: Path) -> None:
    env = _base_env(tmp_path, checkpoint_shards=8, checkpoint_json=False)
    env["RUN_NAME"] = "CWAM-COSMOS3-B5-VOM-FORMAL-R7-20K"
    env["CWAM_B5_RUN_CHECKPOINT_MODE"] = "r4"
    env["CWAM_B5_ALLOW_FORMAL_R4_WARM_START"] = "1"

    result = _run(env)

    assert result.returncode == 0, result.stdout
    assert (
        "TORCHRUN_ARG=checkpoint.keys_to_skip_loading="
        "[visual_memory_delta_main,visual_memory_delta_conditioners]" in result.stdout
    )
    assert "TORCHRUN_ARG=optimizer.keys_to_select=[strong_action_expert]" in result.stdout
    assert "TORCHRUN_ARG=checkpoint.load_training_state=false" in result.stdout


def test_b5_formal_authorization_does_not_allow_non_r4_checkpoint(tmp_path: Path) -> None:
    env = _base_env(tmp_path, checkpoint_shards=8, checkpoint_json=False)
    env["RUN_NAME"] = "CWAM-COSMOS3-B5-VOM-FORMAL-R7-20K"
    env["CWAM_B5_RUN_CHECKPOINT_MODE"] = "b4"
    env["CWAM_B5_ALLOW_FORMAL_R4_WARM_START"] = "1"

    result = _run(env)

    assert result.returncode != 0
    assert "run-checkpoint warm start is probe-only" in result.stdout
