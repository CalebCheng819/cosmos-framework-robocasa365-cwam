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
    provision = (ROOT / "examples/h_cluster/provision_cwam_s1_cont100k_rclone_config.sh").read_text()
    assert '$PERSONAL_ROOT/tools/rclone' in text
    assert "/dev/shm/cwam-s1-cont100k-rclone" in text
    assert "openssl cms -decrypt" in text
    assert "RCLONE_HANDOFF_WAITING" in text
    assert 'chmod 400 "$RCLONE_CONFIG_PATH"' in text
    assert 'rm -f "$RCLONE_CONFIG_PATH" "$RCLONE_PRIVATE_KEY"' in text
    assert '"$RCLONE_BIN" lsf "$SOURCE_CHECKPOINT_URI"' in text
    assert "/.auth/" not in text
    assert "openssl cms -encrypt" in provision
    assert "eailab-hdd2:eailab/chengjuntao/" in provision
    assert "access_key_id =" not in provision
    assert "secret_access_key =" not in provision
