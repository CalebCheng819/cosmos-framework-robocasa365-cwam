"""Static contract tests for the B2/C1 H-cluster training controller."""

from pathlib import Path


SCRIPT = Path(__file__).with_name("cwam_b2_smoke_then_train.sh")


def test_node_preflight_uses_scheduler_injected_physical_node() -> None:
    source = SCRIPT.read_text()

    assert 'actual_node_name="${KUBEBRAIN_NODE_NAME:-}"' in source
    assert '[[ -n "$actual_node_name" ]]' in source
    assert '[[ "$actual_node_name" == "$EXPECTED_HOST" ]]' in source
    assert '[[ "$(hostname -f)" == "$EXPECTED_HOST" ]]' not in source
