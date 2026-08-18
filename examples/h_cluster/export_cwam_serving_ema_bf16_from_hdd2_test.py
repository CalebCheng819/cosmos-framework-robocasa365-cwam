from pathlib import Path


WORKER = Path(__file__).with_name("export_cwam_serving_ema_bf16_from_hdd2.sh")


def test_cpu_export_uses_persistent_gpfs_rclone_binary() -> None:
    script = WORKER.read_text(encoding="utf-8")

    assert 'readonly RCLONE_BIN="${RCLONE_BIN:-$PERSONAL_HANDOFF_ROOT/tools/rclone}"' in script
    assert '[[ -f "$RCLONE_BIN" && ! -L "$RCLONE_BIN" && -x "$RCLONE_BIN" ]]' in script
    assert script.count('"$RCLONE_BIN" --config "$RCLONE_CONFIG"') == 5
    assert "downloading source checkpoint to worker GPFS" in script
    assert "worker NVMe" not in script
