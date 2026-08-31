from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_submitter_supports_selected_private_node_and_eight_gpus() -> None:
    text = (ROOT / "examples/h_cluster/submit_cwam_cosmos3_s1_cont100k_rjob.sh").read_text()
    assert "--private-machine=group" in text
    assert '--gpu="$GPU_COUNT"' in text
    assert '--cpu="$CPU_COUNT"' in text
    assert '--memory="$MEMORY_MIB"' in text
    assert 'NODE_TAG:=node/gpu-l-lg-cmc-h-h200-0252' in text
    assert '--positive-tags="$NODE_TAG"' in text
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
    assert 'RESUME_STEP="$RESUME_STEP"' in text
    assert 'CHECKPOINT_KEYS_NOT_TO_RESUME="$CHECKPOINT_KEYS_NOT_TO_RESUME"' in text
    assert 'GPU_COUNT="$GPU_COUNT"' in text
    assert 'GRAD_ACCUM_ITER="$GRAD_ACCUM_ITER"' in text


def test_runner_parameterizes_resume_step_and_training_state_components() -> None:
    text = (ROOT / "examples/h_cluster/run_cwam_cosmos3_s1_cont100k.sh").read_text()
    assert "checkpoint.load_training_state=true" in text
    assert ': "${RESUME_STEP:=10000}"' in text
    assert ': "${CHECKPOINT_KEYS_NOT_TO_RESUME:=[scheduler,dataloader]}"' in text
    assert ': "${GPU_COUNT:=8}"' in text
    assert ': "${GRAD_ACCUM_ITER:=2}"' in text
    assert 'readonly RESUME_ROOT="$RUN_ROOT/resume/$RESUME_TAG"' in text
    assert 'readonly ARCHIVE_START_STEP=$((RESUME_STEP + 10000))' in text
    assert '--start-step "$ARCHIVE_START_STEP"' in text
    assert "checkpoint.keys_not_to_resume=$CHECKPOINT_KEYS_NOT_TO_RESUME" in text
    assert "trainer.grad_accum_iter=$GRAD_ACCUM_ITER" in text
    assert (
        "trainer.callbacks.production_batch_memory_gate.expected_grad_accum_iter="
        "$GRAD_ACCUM_ITER" in text
    )
    assert (
        "trainer.callbacks.production_batch_memory_gate.allowed_world_sizes="
        "[$GPU_COUNT]" in text
    )
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


def test_runner_does_not_precreate_the_launcher_run_directory() -> None:
    text = (ROOT / "examples/h_cluster/run_cwam_cosmos3_s1_cont100k.sh").read_text()
    assert 'readonly RUN_ROOT="$NVME_ROOT/$EXPERIMENT_ID/$KUBEBRAIN_RJOB_NAME"' in text
    assert 'mkdir -p "$OUTPUT_ROOT" "$CHECKPOINTS_DIR"' not in text
    assert '[[ -d "$CHECKPOINTS_DIR" ]] && break' in text
    assert 'exec "$COSMOS_ENV/bin/python" "$ARCHIVER"' in text
