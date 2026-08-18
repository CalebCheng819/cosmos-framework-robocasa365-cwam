from pathlib import Path


SUBMITTER = Path(__file__).with_name("submit_cwam_serving_ema_export_rjob.sh")


def test_single_gpu_export_does_not_request_host_network() -> None:
    script = SUBMITTER.read_text(encoding="utf-8")

    assert "--gpu 1" in script
    assert "--host-network" not in script


def test_export_packages_control_and_uses_explicit_gpfs_mounts() -> None:
    script = SUBMITTER.read_text(encoding="utf-8")

    assert '--folder "$CONTROL_ROOT"' in script
    assert 'worker_env+=("CONTROL_ROOT=.")' in script
    assert "gpfs://gpfs2/ailab-eailabagent-gpfs:" in script
    assert "gpfs://gpfs2/gpfs2-shared-public:" not in script
    assert '--mount "$PERSONAL_GPFS_MOUNT"' in script


def test_export_uses_normal_gpu_scheduling_for_shared_handoff() -> None:
    script = SUBMITTER.read_text(encoding="utf-8")

    assert "--private-machine" not in script
