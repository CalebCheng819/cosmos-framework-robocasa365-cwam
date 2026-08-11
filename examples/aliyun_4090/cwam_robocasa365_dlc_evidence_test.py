"""CPU tests for the Alibaba DLC RoboCasa365 formal-eval adapter."""

from __future__ import annotations

import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
H_HELPERS = HERE.parent / "h_cluster"
sys.path.insert(0, str(H_HELPERS))
sys.path.insert(0, str(HERE))

import cwam_dlc_runtime_closure as runtime_closure  # noqa: E402
import cwam_robocasa365_dlc_evidence as dlc_evidence  # noqa: E402
import cwam_robocasa365_dsw_evidence_test as dsw_test  # noqa: E402
import generate_cwam_dlc_egl_mapping_receipt as egl_receipt  # noqa: E402

RUNNER = HERE / "run_cwam_cosmos3_robocasa365_eval_dlc.sh"
LAUNCHER = HERE / "launch_cwam_cosmos3_robocasa365_eval_dlc.sh"
MATRIX = HERE / "run_mujoco_egl_mapping_matrix_dlc.sh"
RUNTIME_HELPER = HERE / "cwam_dlc_runtime_closure.py"
RUNTIME_PUBLISHER = HERE / "publish_cwam_dlc_runtime_closure.sh"
ENV_ARCHIVE_HELPER = HERE / "cwam_cosmos_env_archive.py"
MATCHED_HELPER = HERE / "cwam_s1_r14_matched_eval.py"

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _runtime_receipt() -> dict:
    value = deepcopy(dsw_test._runtime_receipt())
    value["kind"] = "cwam_robocasa365_alibaba_dlc_runtime_receipt"
    value["platform"] = {
        "provider": "Alibaba Cloud",
        "service": "PAI DLC",
        "region": "cn-beijing",
    }
    value["topology"].update(
        {
            "hostname": "dlc-test",
            "physical_gpu_count": 8,
            "distributed_world_size": 8,
        }
    )
    value["resources"].update(
        {
            "cpu_affinity_count": 126,
            "cpu_affinity_ids": list(range(126)),
            "memory_limit_bytes": 960 * (1024**3),
            "memory_limit_gib": 960.0,
            "gpus": [
                {
                    "visible_index": index,
                    "name": "NVIDIA GeForce RTX 4090",
                    "compute_capability": "8.9",
                    "sm": "sm89",
                    "total_memory_bytes": 48 * (1024**3),
                }
                for index in range(8)
            ],
        }
    )
    value["policy_server"]["distributed_world_size"] = 8
    value["policy_server"]["parallelism"] = {
        "dp_shard_size": 8,
        "dp_replicate_size": 1,
        "cp_size": 1,
        "cfgp_size": 1,
    }
    value["bindings"]["ema_selection"] = {
        "selected_tree": "net_ema",
        "included_prefix": "net_ema.",
        "excluded_prefix": "net.",
        "key_count": 809,
        "chunk_count": 6472,
        "source_dtype": "torch.float32",
        "target_dtype": "torch.bfloat16",
        "custom_load_planner": "CustomLoadPlanner(load_ema_to_reg=True)",
    }
    value["provenance"]["receipts"]["cosmos_environment_archive_publication"] = dsw_test._descriptor(
        "/receipts/cosmos_environment_archive_publication.json"
    )
    validation = value["validation"]
    for key in (
        "four_local_distributed_ranks",
        "four_visible_4090_gpus",
        "effective_cpu_affinity_is_128",
        "effective_memory_limit_is_400_gib",
        "parallelism_is_dp_shard4_only",
    ):
        validation.pop(key)
    validation.update(
        {
            "eight_local_distributed_ranks": True,
            "eight_visible_4090_gpus": True,
            "effective_cpu_affinity_is_126": True,
            "effective_memory_limit_is_960_gib": True,
            "parallelism_is_dp_shard8_only": True,
        }
    )
    return value


def test_dlc_runtime_receipt_requires_exact_fsdp8_topology() -> None:
    receipt = _runtime_receipt()
    validated = dlc_evidence.validate_runtime_receipt(receipt)
    assert validated["platform"]["service"] == "PAI DLC"
    assert validated["topology"]["physical_gpu_count"] == 8
    assert validated["policy_server"]["parallelism"]["dp_shard_size"] == 8

    stale = deepcopy(receipt)
    stale["topology"]["physical_gpu_count"] = 4
    with pytest.raises(ValueError, match="physical_gpu_count mismatch"):
        dlc_evidence.validate_runtime_receipt(stale)


def test_dlc_runtime_receipt_uses_checkpoint_bound_ema_key_count() -> None:
    receipt = _runtime_receipt()
    receipt["bindings"]["ema_selection"]["key_count"] = 822
    weight = receipt["policy_server"]["checkpoint_weight_receipt"]
    weight["selected_key_count"] = 822
    weight["selected_source_dtype_census"] = {"torch.bfloat16": 822}
    weight["ema_key_count"] = 822

    validated = dlc_evidence.validate_runtime_receipt(receipt)

    assert validated["bindings"]["ema_selection"]["key_count"] == 822


def test_dlc_runtime_receipt_rejects_wrong_allocation() -> None:
    receipt = _runtime_receipt()
    receipt["resources"]["cpu_affinity_count"] = 128
    with pytest.raises(ValueError, match="exactly 126 effective CPUs"):
        dlc_evidence.validate_runtime_receipt(receipt)

    receipt = _runtime_receipt()
    receipt["resources"]["gpus"][7]["compute_capability"] = "9.0"
    with pytest.raises(ValueError, match="compute capability 8.9"):
        dlc_evidence.validate_runtime_receipt(receipt)


def test_dlc_runner_pins_protocol_but_not_s1_export_hashes() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    for required in (
        "readonly PHYSICAL_GPU_COUNT=8",
        "readonly LOGICAL_NUM_SHARDS=8",
        "readonly TASK_COUNT=17",
        "readonly EPISODES_PER_TASK=20",
        "readonly EXPECTED_EPISODE_COUNT=340",
        "readonly EXPECTED_CPU_COUNT=126",
        "readonly EXPECTED_MEMORY_GIB=960",
        "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
        'readonly FORMAL_CHECKPOINTS_ROOT="$FORMAL_INPUT_ROOT/checkpoints"',
        "cosmos-r21-dlc-runtime-v1",
        "EXPECTED_RUNTIME_CLOSURE_MANIFEST_SHA256",
        "EXPECTED_RUNTIME_CLOSURE_TREE_SHA256",
        "PINNED_RUNTIME_CLOSURE_PUBLICATION_RECEIPT_SHA256",
        "COSMOS_ENV_ARCHIVE_PUBLICATION_RECEIPT",
        "EXPECTED_COSMOS_ENV_ARCHIVE_SHA256",
        "EXPECTED_COSMOS_ENV_ARCHIVE_PUBLICATION_RECEIPT_SHA256",
        "cwam_cosmos_env_archive.py",
        "stage_cosmos_environment_archive",
        "cosmos_environment_archive_publication",
        "runtime_closure_source_pre",
        "runtime_closure_local_pre",
        "runtime_closure_local_post",
        "runtime_closure_publication",
        "cwam_robocasa365_dlc_evidence.py",
        "cwam_s1_r14_matched_eval.py",
        'readonly S1_R14_MATCHED_MODE="${S1_R14_MATCHED_MODE:-0}"',
        "EVAL_TASK_ARGS=(--tasks CloseBlenderLid PickPlaceCounterToStove)",
        'R14_BASELINE_ARGS="$(canonical_existing_file "$R14_BASELINE_ARGS")"',
        '[[ -s "$COMPARISON_JSON" ]]',
    ):
        assert required in script
    for forbidden in (
        "R14-iter_000010000-serving-ema-bf16-v1",
        "PINNED_H_SERVING_MANIFEST_SHA256",
        "PINNED_SERVING_TREE_SHA256",
        "PINNED_EMA_EXPORT_SHA256",
        "robocasa-egl-device-mapping-dsw974.json",
        "FORMAL_COSMOS_ENV_SOURCE",
        "stage_directory cosmos_environment",
    ):
        assert forbidden not in script


def test_dlc_launcher_generates_same_host_egl_evidence() -> None:
    script = LAUNCHER.read_text(encoding="utf-8")
    assert "readonly GPU_COUNT=8" in script
    assert "run_mujoco_egl_mapping_matrix_dlc.sh" in script
    assert "generate_cwam_dlc_egl_mapping_receipt.py" in script
    assert 'export EXPECTED_EGL_MAPPING_RECEIPT_SHA256="$(sha256_file' in script
    assert 'require_immutable_source_file "$EGL_PROBE"' in script
    assert 'bash "$EGL_MATRIX"' in script
    assert 'exec bash "$FORMAL_RUNNER"' in script

    generator = (HERE / "generate_cwam_dlc_egl_mapping_receipt.py").read_text(encoding="utf-8")
    assert '"script": descriptor(probe, 0o444)' in generator
    assert '"script": descriptor(probe, 0o555)' not in generator

    matrix = MATRIX.read_text(encoding="utf-8")
    assert "readonly GPU_COUNT=8" in matrix
    assert "GPU_INDICES" in matrix
    assert "ALL_VISIBLE_DEVICES" in matrix
    assert "nvidia-smi pmon -s um -c 2 -d 1" in matrix
    assert '[[ -f "$PROBE" && ! -L "$PROBE" && "$(stat -c %a "$PROBE")" == 444 ]]' in matrix


def test_egl_generator_requires_exact_eight_gpu_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = "\n".join(f"{index}, GPU-{index:04d}, NVIDIA GeForce RTX 4090, 0000:{index:02x}:00.0" for index in range(8))
    monkeypatch.setattr(
        egl_receipt.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=rows),
    )
    inventory = egl_receipt.gpu_inventory(8)
    assert [row["physical_index"] for row in inventory] == list(range(8))
    assert egl_receipt.expected_environment("visible_all_physical_index", 7, 8) == (
        "0,1,2,3,4,5,6,7",
        "7",
    )
    with pytest.raises(ValueError, match="expected exact 4-GPU inventory"):
        egl_receipt.gpu_inventory(4)


def test_runtime_closure_manifest_survives_exact_relocation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "bin").mkdir(mode=0o755)
    executable = source / "bin" / "runtime"
    executable.write_bytes(b"runtime-bytes\n")
    executable.chmod(0o755)
    (source / "bin" / "runtime-link").symlink_to("runtime")
    (source / "loader-link").symlink_to("/lib64/ld-linux-x86-64.so.2")

    manifest = runtime_closure.build_manifest(source, "test-runtime-v1")
    manifest_path = tmp_path / "manifest.json"
    runtime_closure._write_json(manifest_path, manifest)
    manifest_sha256 = runtime_closure._sha256_file(manifest_path)
    destination.mkdir()
    subprocess.run(["cp", "--archive", f"{source}/.", f"{destination}/"], check=True)

    receipt = runtime_closure.verify(
        destination,
        manifest_path,
        manifest_sha256,
        manifest["tree_sha256"],
        "pre",
    )
    assert receipt["status"] == "passed"
    assert receipt["regular_file_count"] == 1
    assert receipt["symlink_count"] == 2

    executable_copy = destination / "bin" / "runtime"
    executable_copy.chmod(0o644)
    with pytest.raises(runtime_closure.RuntimeClosureError, match="differs"):
        runtime_closure.verify(
            destination,
            manifest_path,
            manifest_sha256,
            manifest["tree_sha256"],
            "post",
        )


@pytest.mark.parametrize("script", [RUNNER, LAUNCHER, MATRIX, RUNTIME_PUBLISHER])
def test_dlc_shell_scripts_parse(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_dlc_python_helpers_compile() -> None:
    for helper in (RUNTIME_HELPER, ENV_ARCHIVE_HELPER, MATCHED_HELPER):
        subprocess.run([sys.executable, "-m", "py_compile", str(helper)], check=True)
