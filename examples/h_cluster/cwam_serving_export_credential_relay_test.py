from pathlib import Path


ROOT = Path(__file__).resolve().parent


def test_exporter_publishes_public_certificate_marker_only() -> None:
    exporter = (ROOT / "export_cwam_serving_ema_bf16_from_hdd2.sh").read_text(encoding="utf-8")

    assert "CWAM_RCLONE_HANDOFF_REQUEST" in exporter
    assert 'base64 -w0 "$CERTIFICATE"' in exporter
    assert 'base64 -w0 "$PRIVATE_KEY"' not in exporter


def test_relay_packages_only_ciphertext_and_mounts_personal_gpfs() -> None:
    submitter = (ROOT / "submit_cwam_serving_export_credential_relay_rjob.sh").read_text(encoding="utf-8")
    worker = (ROOT / "relay_cwam_serving_export_rclone_config.sh").read_text(encoding="utf-8")

    assert '--folder "$CONTROL_ROOT"' in submitter
    assert "gpfs://gpfs2/ailab-eailabagent-gpfs:" in submitter
    assert "RCLONE_CONFIG" not in submitter
    assert "rclone-config.cms.der" in worker
    assert 'target.open("xb")' in worker
