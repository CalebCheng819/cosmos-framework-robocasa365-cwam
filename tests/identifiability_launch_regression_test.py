from pathlib import Path


def test_robocasa_launcher_uses_distinct_parent_and_child_demian_names() -> None:
    script = (
        Path(__file__).parents[1]
        / "examples"
        / "h_cluster"
        / "run_robocasa_s1_i60000_identifiability.sh"
    ).read_text(encoding="utf-8")
    assert 'readonly DEMIAN_REPO=' not in script
    assert 'readonly STAGED_DEMIAN_REPO=' in script
    assert 'DEMIAN_REPO="$STAGED_DEMIAN_REPO"' in script
