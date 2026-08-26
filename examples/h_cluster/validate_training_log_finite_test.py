from pathlib import Path

import pytest

from examples.h_cluster.validate_training_log_finite import validate_training_log


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "train.log"
    path.write_text(text)
    return path


def test_accepts_finite_losses_with_skip_nan_configuration(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "Instantiating callback skip_nan_step: {'max_consecutive_nan': 100}\n"
        "clip_grad_norm/video/global: 25.25 (iteration 0)\n"
        "Iteration 1: Hit counter: 1/50 | Loss: 14.4650 | Time: 72.18s\n"
        "clip_grad_norm/video/global: 19.5 (iteration 1)\n"
        "Iteration 2: Hit counter: 2/50 | Loss: 15.8220 | Time: 0.52s\n",
    )

    assert validate_training_log(path, min_iterations=2) == (2, 2)


def test_ignores_multiline_wandb_summary_after_gradient_metric_name(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "clip_grad_norm/video/global: 19.0 (iteration 0)\n"
        "Iteration 1: Hit counter: 1/50 | Loss: 14.1950 | Time: 72.89s\n"
        "clip_grad_norm/video/global: 28.375 (iteration 1)\n"
        "Iteration 2: Hit counter: 2/50 | Loss: 15.8942 | Time: 0.54s\n"
        "wandb: clip_grad_norm/video/dp_replicate-dp_shard ▁█\n"
        "wandb:                                            +1,757 ...\n"
        "wandb: clip_grad_norm/video/dp_replicate-dp_shard 28.375\n",
    )

    assert validate_training_log(path, min_iterations=2) == (2, 2)


@pytest.mark.parametrize("value", ["nan", "NaN", "inf", "-inf"])
def test_rejects_nonfinite_loss(tmp_path: Path, value: str) -> None:
    path = _write(
        tmp_path,
        "clip_grad_norm/video/global: 1.0 (iteration 0)\n"
        f"Iteration 1: Hit counter: 1/50 | Loss: {value} | Time: 1.0s\n",
    )

    with pytest.raises(ValueError, match="non-finite loss"):
        validate_training_log(path, min_iterations=1)
