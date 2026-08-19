#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMIT_SCRIPT = REPO_ROOT / "examples/h_cluster/submit_cwam_v2_formal_rjob.sh"


def test_direct_v2_submission_records_r13_warmstart_contract() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source"
        worker_dir = source / "examples/h_cluster"
        worker_dir.mkdir(parents=True)
        for name in ("cwam_c1_prepare_assets.sh", "cwam_v2_smoke_then_train.sh"):
            path = worker_dir / name
            path.write_text("#!/usr/bin/env bash\nexit 0\n")
            path.chmod(0o755)

        revision = "test-v2-revision"
        run_id = "CWAM-COSMOS3-V2-20260819-R4"
        (source / "GIT_REVISION").write_text(revision + "\n")
        (source / "RUN_ID").write_text(run_id + "\n")

        r13 = root / "c1-r13"
        (r13 / "model").mkdir(parents=True)
        (r13 / "model/.metadata").write_text("metadata\n")

        fake_bin = root / "bin"
        fake_bin.mkdir()
        fake_rjob = fake_bin / "rjob"
        fake_rjob.write_text(
            "#!/usr/bin/env bash\n"
            "set -Eeuo pipefail\n"
            "if [[ $1 == submit ]]; then\n"
            "  printf '%s\\n' \"$@\" > \"$FAKE_RJOB_ARGS\"\n"
            "  echo 'created rjob_name: cwam-v2-0819-r4pm-12345678'\n"
            "elif [[ $1 == get ]]; then\n"
            "  echo 'rjob cwam-v2-0819-r4pm-12345678 (showname=cwam-v2-0819-r4pm): Inqueue'\n"
            "else\n"
            "  exit 2\n"
            "fi\n"
        )
        fake_rjob.chmod(0o755)

        expected_host = "gpu-l-lg-cmc-h-h200-0253.host.h.pjlab.org.cn"
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
                "RJOB_NAME": "cwam-v2-0819-r4pm",
                "EXPECTED_HOST": expected_host,
                "CONTROL_ROOT": str(control_root),
                "C1_R13_SERVING_CHECKPOINT": str(r13),
            }
        )
        subprocess.run(["bash", str(SUBMIT_SCRIPT)], env=env, check=True)

        submitted = args_path.read_text()
        assert f"node/{expected_host}" in submitted
        assert "--private-machine=group" in submitted
        assert "RESUME_FROM_OSS=0" in submitted
        assert "C1_INPUT_ROOT=/nvme/cwam-v2/inputs/shared" in submitted
        assert "SECRET_ROOT=/dev/shm/cwam-v2-rclone" in submitted
        assert "RCLONE_CONFIG=/dev/shm/cwam-v2-rclone/rclone.conf" in submitted
        assert str(r13) in submitted
        assert "WORLD_SIZE=8" in submitted
        assert "--gpu\n8" in submitted
        assert "--cpu\n128" in submitted
        assert "--memory\n1500000" in submitted
        assert "run_cwam_c1_diag_then_v2_h8.sh" not in submitted

        receipt_path = control_root / "results" / f"{run_id}.submission.json"
        receipt = json.loads(receipt_path.read_text())
        assert receipt["status"] == "accepted"
        assert receipt["scheduler_rjob_name"] == "cwam-v2-0819-r4pm-12345678"
        assert receipt["phase_order"] == ["cwam_v2_smoke", "cwam_v2_formal_training"]
        assert receipt["initialization"]["source"] == "C1 R13 serving EMA"
        assert receipt["initialization"]["source_model_iteration"] == 20000
        assert receipt["initialization"]["v2_optimizer_scheduler_restored"] is False
        assert receipt["initialization"]["v2_start_iteration"] == 0
        assert receipt["initialization"]["v2_absolute_target_iteration"] == 20000


if __name__ == "__main__":
    test_direct_v2_submission_records_r13_warmstart_contract()
