"""Static contract tests for the B2/C1 H-cluster training controller."""

from pathlib import Path


SCRIPT = Path(__file__).with_name("cwam_b2_smoke_then_train.sh")
LAUNCHER = SCRIPT.parents[1] / "launch_sft_action_policy_robocasa365_nano.sh"


def test_node_preflight_uses_scheduler_injected_physical_node() -> None:
    source = SCRIPT.read_text()

    assert 'actual_node_name="${KUBEBRAIN_NODE_NAME:-}"' in source
    assert '[[ -n "$actual_node_name" ]]' in source
    assert '[[ "$actual_node_name" == "$EXPECTED_HOST" ]]' in source
    assert '[[ "$(hostname -f)" == "$EXPECTED_HOST" ]]' not in source


def test_c1_submit_disables_python_bytecode_writes() -> None:
    source = SCRIPT.with_name("submit_cwam_c1_formal_rjob.sh").read_text()

    assert "export PYTHONDONTWRITEBYTECODE=1" in source


def test_c1_launcher_accepts_six_or_eight_ranks() -> None:
    source = LAUNCHER.read_text()

    assert '[[ "$NPROC_PER_NODE" -eq 6 || "$NPROC_PER_NODE" -eq 8 ]]' in source
    assert "C1 formal training requires exactly 6 or 8 ranks" in source
    assert '"model.config.parallelism.data_parallel_shard_degree=${NPROC_PER_NODE}"' in source
    assert '[[ "$NPROC_PER_NODE" -eq 6 ]] ||' not in source
