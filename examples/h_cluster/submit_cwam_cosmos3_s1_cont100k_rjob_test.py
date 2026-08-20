from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_submitter_pins_private_0252_and_eight_gpus() -> None:
    text = (ROOT / "examples/h_cluster/submit_cwam_cosmos3_s1_cont100k_rjob.sh").read_text()
    assert "--private-machine=group" in text
    assert "--gpu=8" in text
    assert "gpu-l-lg-cmc-h-h200-0252" in text
    assert "--store-host-nvme=true" in text
    assert "--restart-policy=never" in text
    assert "--group=eailabagent_gpu" in text
    assert "--task-type=normal" in text
    assert "--backoff_limit=1" in text
    assert text.count("--mount") == 1
    assert "--enable-sshd" not in text
    assert "WORKER_BOOTSTRAP_START" in text
    assert "/bin/bash --noprofile --norc" in text
    assert 'KUBEBRAIN_RJOB_NAME="$RJOB_NAME"' in text


def test_runner_restores_optimizer_but_restarts_scheduler() -> None:
    text = (ROOT / "examples/h_cluster/run_cwam_cosmos3_s1_cont100k.sh").read_text()
    assert "checkpoint.load_training_state=true" in text
    assert "checkpoint.keys_not_to_resume=[scheduler,dataloader]" in text
    assert "checkpoint.save_iter" not in text
    assert "--query-gpu=" in text


def test_runner_bootstraps_rclone_without_persisting_credentials() -> None:
    text = (ROOT / "examples/h_cluster/run_cwam_cosmos3_s1_cont100k.sh").read_text()
    assert "chengjuntao/tools/rclone" in text
    assert "/.auth/accesskey_id" in text
    assert "/.auth/accesskey_secret" in text
    assert 'chmod 600 "$RCLONE_CONFIG_PATH"' in text
    assert 'rm -f "$RCLONE_CONFIG_PATH"' in text
    assert '"$RCLONE_BIN" lsf "$SOURCE_CHECKPOINT_URI"' in text
    assert "access_key_id = %s" in text
    assert "secret_access_key = %s" in text
