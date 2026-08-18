from pathlib import Path


SUBMITTER = Path(__file__).with_name("submit_cwam_serving_ema_export_rjob.sh")


def test_single_gpu_export_does_not_request_host_network() -> None:
    script = SUBMITTER.read_text(encoding="utf-8")

    assert "--gpu 1" in script
    assert "--host-network" not in script
