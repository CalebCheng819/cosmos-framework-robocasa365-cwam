#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMIT_SCRIPT = REPO_ROOT / "examples/h_cluster/submit_cwam_c1_formal_rjob.sh"


def test_resume_submission_records_dynamic_host_and_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source"
        worker_dir = source / "examples/h_cluster"
        worker_dir.mkdir(parents=True)
        for name in ("cwam_c1_prepare_assets.sh", "cwam_c1_smoke_then_train.sh"):
            path = worker_dir / name
            path.write_text("#!/usr/bin/env bash\nexit 0\n")
            path.chmod(0o755)

        revision = "test-revision"
        run_id = "CWAM-COSMOS3-C1-20260817-R7"
        (source / "GIT_REVISION").write_text(revision + "\n")
        (source / "RUN_ID").write_text(run_id + "\n")

        fake_bin = root / "bin"
        fake_bin.mkdir()
        fake_rjob = fake_bin / "rjob"
        fake_rjob.write_text(
            "#!/usr/bin/env bash\n"
            "set -Eeuo pipefail\n"
            "if [[ $1 == submit ]]; then\n"
            "  printf '%s\\n' \"$@\" > \"$FAKE_RJOB_ARGS\"\n"
            "  echo 'created rjob_name: cwam-c1-20260817-r7-12345678'\n"
            "elif [[ $1 == get ]]; then\n"
            "  echo 'rjob cwam-c1-20260817-r7-12345678 (showname=cwam-c1-20260817-r7): Inqueue'\n"
            "else\n"
            "  exit 2\n"
            "fi\n"
        )
        fake_rjob.chmod(0o755)

        expected_host = "gpu-l-lg-cmc-h-h200-0252.host.h.pjlab.org.cn"
        checkpoint_name = "iter_000007000"
        checkpoint_remote = (
            "eailab-hdd2:eailab/chengjuntao/experiments/cwam-cosmos3/C1/"
            "CWAM-COSMOS3-C1-20260817-R6/formal/run/checkpoints/iter_000007000"
        )
        control_root = root / "control"
        args_path = root / "rjob-args.txt"
        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{fake_bin}:{env['PATH']}",
                "FAKE_RJOB_ARGS": str(args_path),
                "COSMOS_SOURCE": str(source),
                "FORMAL_RUN_NAME": run_id,
                "GIT_REVISION": revision,
                "RJOB_NAME": "cwam-c1-20260817-r7",
                "EXPECTED_HOST": expected_host,
                "CONTROL_ROOT": str(control_root),
                "C1_GPU_COUNT": "8",
                "RESUME_FROM_OSS": "1",
                "RESUME_CHECKPOINT_REMOTE": checkpoint_remote,
                "RESUME_CHECKPOINT_NAME": checkpoint_name,
                "RESUME_SOURCE_RUN_NAME": "CWAM-COSMOS3-C1-20260817-R6",
            }
        )
        subprocess.run(["bash", str(SUBMIT_SCRIPT)], env=env, check=True)

        submitted = args_path.read_text()
        assert f"node/{expected_host}" in submitted
        assert "RESUME_FROM_OSS=1" in submitted
        assert checkpoint_remote in submitted
        assert checkpoint_name in submitted
        assert "WORLD_SIZE=8" in submitted
        assert "--gpu\n8" in submitted
        assert "--cpu\n128" in submitted
        assert "--memory\n1500000" in submitted

        receipt_path = control_root / "results" / f"{run_id}.submission.json"
        receipt = json.loads(receipt_path.read_text())
        assert receipt["status"] == "accepted"
        assert receipt["scheduler_rjob_name"] == "cwam-c1-20260817-r7-12345678"
        assert receipt["expected_host"] == expected_host
        assert receipt["gpu_count"] == 8
        assert receipt["cpu"] == 128
        assert receipt["memory_mib"] == 1500000
        assert receipt["resume_from_oss"] is True
        assert receipt["resume_checkpoint_remote"] == checkpoint_remote
        assert receipt["resume_checkpoint_name"] == checkpoint_name
        assert receipt["resume_source_run_id"] == "CWAM-COSMOS3-C1-20260817-R6"
        assert receipt["data_stream_replay_acknowledged"] is True


if __name__ == "__main__":
    test_resume_submission_records_dynamic_host_and_checkpoint()
