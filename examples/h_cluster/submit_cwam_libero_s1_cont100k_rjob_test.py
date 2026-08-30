from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_continuation_contract_is_step_500_to_100k() -> None:
    config = (ROOT / "examples/toml/sft_config/action_policy_libero_all_s1_cont100k_nano.toml").read_text()
    submit = (ROOT / "examples/h_cluster/submit_cwam_libero_s1_cont100k_rjob.sh").read_text()
    runner = (ROOT / "examples/h_cluster/run_cwam_libero_s1.sh").read_text()

    assert "cycle_lengths = [99500]" in config
    assert "f_start = [1.0]" in config
    assert "max_iter        = 100000" in config
    assert "save_iter = 1000" in config
    assert 'RESUME_STEP:=500' in submit
    assert "LIBERO_LATENT_CACHE_MODE=required" in submit
    assert "ALLOW_DATA_STREAM_REPLAY=1" in submit
    assert "ARCHIVE_START_STEP=1000" in submit
    assert "ARCHIVE_INTERVAL=1000" in submit
    assert "ARCHIVE_COMPACT_INTERVAL=10000" in submit
    assert '--compact-interval "$ARCHIVE_COMPACT_INTERVAL"' in runner
    assert "checkpoint.load_training_state=true" in runner
    assert "checkpoint.keys_not_to_resume=[scheduler,dataloader]" in runner
    assert 'CHECKPOINTS_DIR="$OUTPUT_ROOT/cwam_action_libero/' in runner
    assert "for component in model optim scheduler trainer" in runner
    assert "trainer.run_validation_on_start=false" in runner
    assert "ARCHIVER_HEALTHY" in runner
    assert "checkpoint archiver exited during training" in runner
    assert "--stability-seconds" in runner
    assert "--drain-seconds" in runner


def test_continuation_rejects_topology_or_resume_step_drift() -> None:
    submit = (ROOT / "examples/h_cluster/submit_cwam_libero_s1_cont100k_rjob.sh").read_text()

    assert '[[ "$NUM_GPUS" -eq 4 ]]' in submit
    assert '[[ "$RESUME_STEP" -eq 500 ]]' in submit
    assert "--positive-tags=\"$H254_TAG\"" in submit


def test_rescue_targets_the_actual_s1_output_tree_without_a_gpu() -> None:
    rescue = (ROOT / "examples/h_cluster/submit_cwam_libero_checkpoint_rescue_rjob.sh").read_text()

    assert "/output/cwam_action_libero/action_sft/" in rescue
    assert "--gpu=0" in rescue
    assert 'TARGET_STEP:=500' in rescue
    assert 'SOURCE_COMMIT:=07b6b60' in rescue
    assert 'REQUIRED_COMPONENTS = ("model", "optim", "scheduler", "trainer")' in (
        ROOT / "examples/h_cluster/archive_cwam_s1_cont100k.py"
    ).read_text()


def test_checkpoint_safe_probe_and_formal_continuation_contracts() -> None:
    probe = (ROOT / "examples/h_cluster/submit_cwam_libero_s1_checkpoint_probe_rjob.sh").read_text()
    formal = (
        ROOT / "examples/h_cluster/submit_cwam_libero_s1_checkpoint_safe_cont100k_rjob.sh"
    ).read_text()

    assert 'RESUME_STEP:=13000' in probe
    assert 'TARGET_STEP:=13001' in probe
    assert 'checkpoint.save_iter=1' in probe
    assert 'ARCHIVE_START_STEP="$TARGET_STEP"' in probe
    assert 'RESUME_STEP:=13001' in formal
    assert 'ARCHIVE_START_STEP=14000' in formal
    assert 'ARCHIVE_END_STEP=100000' in formal
    assert '--positive-tags="$H253_TAG"' in probe
    assert '--positive-tags="$H253_TAG"' in formal
