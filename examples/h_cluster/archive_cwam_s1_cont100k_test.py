from pathlib import Path

from examples.h_cluster.archive_cwam_s1_cont100k import checkpoint_name, checkpoint_ready, tree_stats


def test_checkpoint_ready_requires_all_dcp_components(tmp_path: Path) -> None:
    checkpoint = tmp_path / checkpoint_name(20_000)
    for component in ("model", "optim", "scheduler"):
        directory = checkpoint / component
        directory.mkdir(parents=True)
        (directory / ".metadata").write_bytes(b"metadata")
    assert not checkpoint_ready(checkpoint)
    trainer = checkpoint / "trainer"
    trainer.mkdir()
    (trainer / ".metadata").write_bytes(b"metadata")
    assert checkpoint_ready(checkpoint)
    assert tree_stats(checkpoint) == (4, 32)


def test_checkpoint_name_uses_nine_digit_cosmos_format() -> None:
    assert checkpoint_name(100_000) == "iter_000100000"
