from pathlib import Path


SUBMITTER = Path(__file__).with_name("submit_cwam_serving_ema_export_rjob.sh")


def test_cpu_export_does_not_request_gpu_or_host_network() -> None:
    script = SUBMITTER.read_text(encoding="utf-8")

    assert 'PINNED_QUOTA_GROUP="eailabagent_cpu_task"' in script
    assert "--gpu" not in script
    assert "--host-network" not in script
    assert "--private-machine" not in script


def test_export_packages_control_and_uses_explicit_gpfs_mounts() -> None:
    script = SUBMITTER.read_text(encoding="utf-8")

    assert '--folder "$CONTROL_ROOT"' in script
    assert 'worker_env+=("CONTROL_ROOT=.")' in script
    assert "gpfs://gpfs2/ailab-eailabagent-gpfs:" in script
    assert "gpfs://gpfs2/gpfs2-shared-public:" not in script
    assert '--mount "$PERSONAL_GPFS_MOUNT"' in script


def test_export_uses_non_private_cpu_task_scheduling_for_shared_handoff() -> None:
    script = SUBMITTER.read_text(encoding="utf-8")

    assert '--group "$PINNED_QUOTA_GROUP"' in script
    assert '--charged-group "$PINNED_QUOTA_GROUP"' in script
    assert "--private-machine" not in script


def test_cpu_export_uses_explicit_gpfs_workspace() -> None:
    script = SUBMITTER.read_text(encoding="utf-8")

    assert "--cpu 32" in script
    assert "--memory 250000" in script
    assert "--store-host-nvme" not in script
    assert "WORK_ROOT_BASE" in script
