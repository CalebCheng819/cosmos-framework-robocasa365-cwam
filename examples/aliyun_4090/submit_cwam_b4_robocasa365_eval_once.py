#!/usr/bin/env python3
"""Audit and submit the B4 RoboCasa365 DLC evaluation exactly once."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

from alibabacloud_credentials.client import Client as CredentialClient
from alibabacloud_credentials.models import Config as CredentialConfig
from alibabacloud_pai_dlc20201203 import models
from alibabacloud_pai_dlc20201203.client import Client
from alibabacloud_tea_openapi.models import Config


EXPERIMENT_ID = "CWAM-DLC-ROB365-B4-I20000-8GPU-S42-R1-20260817"
LOGICAL_RUN_ID = "CWAM-DLC-ROB365-COSMOS3-B4-I20000-EMA-340EP-S42-R1-20260817"
DISPLAY_NAME = "cwam-b4-i20000-rob365-8g-s42-r1-20260817"
WORKSPACE_ID = "270969"
RESOURCE_ID = "quotaksvqq2oh2pg"
PROJECT_ROOT = pathlib.Path("/mnt/workspace/cwam_cosmos3_robocasa365_eval")
OSS_ROOT = pathlib.Path(
    "/oss-chengjuntao/cpfs-user-chengjuntao/cwam_cosmos3_robocasa365_eval"
)
CONTROL_ROOT = PROJECT_ROOT / "control" / "scripts" / f"{EXPERIMENT_ID}-control-v3"
LAUNCHER = CONTROL_ROOT / "launch_cwam_b4_robocasa365_eval_dlc.sh"
CHECKPOINT_ROOT = OSS_ROOT / "inputs" / "checkpoints" / (
    "B4-iter_000020000-serving-ema-bf16-metadata-v1"
)
SOURCE_ROOT = PROJECT_ROOT / "inputs" / "source" / "CWAM-B4-e7775ff-20260817"
OUTPUT_ROOT = OSS_ROOT / "runs" / EXPERIMENT_ID / LOGICAL_RUN_ID
RECEIPT = OSS_ROOT / "control" / "receipts" / f"{EXPERIMENT_ID}-dlc-submit.json"
SDK_PYTHON = pathlib.Path(
    "/mnt/workspace/tools/pai-control-py311/"
    "20260817-credentials1.0.10-dlc1.9.2/bin/python"
)
IMAGE = (
    "dsw-registry-vpc.cn-beijing.cr.aliyuncs.com/pai/"
    "pytorch:2.7.1-gpu-py310-cu128-ubuntu22.04-3995b779-1764350887"
)
SOURCE_COMMIT = "e7775ffcea0a2d577597fbeb05c8bb92905aba0f"
DATA_SOURCES = [
    {"DataSourceId": "d-a5mu77ymwjio71dkmw", "MountAccess": "RW", "MountPath": "/mnt/workspace"},
    {"DataSourceId": "d-n7rly4fll0q2z6v91h", "MountAccess": "RW", "MountPath": "/oss-chengjuntao"},
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def credentials_uri() -> str:
    document = json.loads(pathlib.Path("/root/.aliyun/config.json").read_text())
    profiles = document["profiles"]
    current = document["current"]
    profile = profiles[current] if isinstance(profiles, dict) else next(
        item for item in profiles if item.get("name") == current
    )
    uri = profile.get("credentials_uri") or profile.get("CredentialsURI") or profile.get("credentialsUri")
    require(bool(uri), "active Alibaba Cloud profile has no CredentialsURI")
    return str(uri)


def make_client() -> Client:
    credential = CredentialClient(
        CredentialConfig(type="credentials_uri", credentials_uri=credentials_uri())
    )
    return Client(Config(credential=credential, region_id="cn-beijing", endpoint="pai-dlc.cn-beijing.aliyuncs.com"))


def all_jobs(client: Client) -> list[dict]:
    result: list[dict] = []
    seen: set[str] = set()
    page = 1
    total = None
    while True:
        body = client.list_jobs(models.ListJobsRequest(
            workspace_id=WORKSPACE_ID, resource_id=RESOURCE_ID, page_number=page, page_size=100
        )).body.to_map()
        observed_total = int(body["TotalCount"])
        total = observed_total if total is None else total
        require(total == observed_total, "ListJobs changed during pagination")
        batch = body.get("Jobs") or []
        for job in batch:
            identifier = str(job.get("JobId") or job.get("job_id") or "")
            require(identifier and identifier not in seen, "invalid or duplicate DLC JobId")
            seen.add(identifier)
            result.append(job)
        if len(result) == total:
            return result
        require(batch and page < 1000, "ListJobs pagination ended early")
        page += 1


def verify_no_duplicate(client: Client) -> None:
    first = all_jobs(client)
    second = all_jobs(client)
    identity = lambda job: (str(job.get("JobId") or job.get("job_id")), str(job.get("DisplayName") or job.get("display_name")))
    require(sorted(map(identity, first)) == sorted(map(identity, second)), "ListJobs snapshot changed")
    matches = [job for job in second if identity(job)[1] == DISPLAY_NAME or EXPERIMENT_ID in str(job.get("Description") or "")]
    require(not matches, f"matching DLC job already exists: {matches}")


def mounted_target(path: pathlib.Path) -> str:
    return subprocess.check_output(["findmnt", "-n", "-T", str(path), "-o", "TARGET"], text=True).strip()


def control_commit() -> str:
    value = os.environ.get("EVAL_CONTROL_COMMIT", "")
    require(len(value) == 40 and all(char in "0123456789abcdef" for char in value), "invalid EVAL_CONTROL_COMMIT")
    return value


def validate_local() -> None:
    require(mounted_target(PROJECT_ROOT) == "/mnt/workspace", "CPFS mount missing")
    require(mounted_target(OSS_ROOT) == "/oss-chengjuntao", "OSS mount missing")
    require(SOURCE_ROOT.is_dir() and not SOURCE_ROOT.is_symlink(), "source snapshot missing")
    revision = subprocess.check_output(["git", "-c", f"safe.directory={SOURCE_ROOT}", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"], text=True).strip()
    require(revision == SOURCE_COMMIT, "source snapshot revision mismatch")
    require(CHECKPOINT_ROOT.is_dir() and not CHECKPOINT_ROOT.is_symlink(), "checkpoint missing")
    expected = {
        "config.yaml": 22139,
        "ema-bf16-metadata-receipt.json": 2482,
        "serving-checkpoint-size-manifest.json": 997,
        "model/.metadata": 1612962,
        "model/__0_0.distcp": 3828985322,
        "model/__1_0.distcp": 3828985194,
        "model/__2_0.distcp": 3828980056,
        "model/__3_0.distcp": 3828978543,
        "model/__4_0.distcp": 3828974063,
        "model/__5_0.distcp": 3828962489,
        "model/__6_0.distcp": 3828951514,
        "model/__7_0.distcp": 3828948488,
    }
    observed = {
        path.relative_to(CHECKPOINT_ROOT).as_posix(): path.stat().st_size
        for path in CHECKPOINT_ROOT.rglob("*") if path.is_file()
    }
    require(observed == expected, "checkpoint inventory differs from audited B4 serving artifact")
    for name in (
        "launch_cwam_b4_robocasa365_eval_dlc.sh",
        "run_cwam_b4_robocasa365_eval_dlc.sh",
        "run_cwam_b4_robocasa365_eval_shard_group.sh",
        "cwam_b4_robocasa365_eval.py",
        "cwam_cosmos_env_archive.py",
        "cwam_robocasa_asset_source_package.py",
        "probe_mujoco_egl_mapping.py",
    ):
        path = CONTROL_ROOT / name
        metadata = os.stat(path, follow_symlinks=False)
        require(stat.S_ISREG(metadata.st_mode), f"control file is not regular: {path}")
        require(stat.S_IMODE(metadata.st_mode) == 0o444, f"control file mode is not 0444: {path}")
    require(not os.path.lexists(OUTPUT_ROOT), f"output already exists: {OUTPUT_ROOT}")
    require(RECEIPT.parent.is_dir() and os.access(RECEIPT.parent, os.W_OK), "receipt parent unavailable")
    require(not os.path.lexists(RECEIPT), f"submission receipt exists: {RECEIPT}")


def request_body(commit: str) -> dict:
    shell = (
        "set -Eeuo pipefail; "
        f"export EVAL_CONTROL_COMMIT={commit}; "
        f"export EVAL_EXPERIMENT_ID={EXPERIMENT_ID}; "
        f"export EVAL_LOGICAL_RUN_ID={LOGICAL_RUN_ID}; "
        f"exec bash {LAUNCHER}"
    )
    return {
        "Accessibility": "PRIVATE",
        "CustomEnvs": [],
        "DataSources": DATA_SOURCES,
        "Description": f"{EXPERIMENT_ID}: B4 iter 20000 EMA, RoboCasa365 17x20 seed 42",
        "DisplayName": DISPLAY_NAME,
        "Envs": {
            "EVAL_CONTROL_COMMIT": commit,
            "EVAL_EXPERIMENT_ID": EXPERIMENT_ID,
            "EVAL_LOGICAL_RUN_ID": LOGICAL_RUN_ID,
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics,display",
        },
        "JobMaxRunningTimeMinutes": 8640,
        "JobSpecs": [{
            "ElasticSpotSpecs": [],
            "Image": IMAGE,
            "LocalMountSpecs": [],
            "PodCount": 1,
            "ResourceConfig": {"CPU": "126", "GPU": "8", "Memory": "960Gi", "SharedMemory": "960Gi"},
            "RestartPolicy": "Never",
            "StartupDependencies": [],
            "Type": "Worker",
        }],
        "JobType": "PyTorchJob",
        "Priority": 7,
        "ResourceId": RESOURCE_ID,
        "Settings": {
            "AllocateAllRDMADevices": False,
            "EnableCPUAffinity": True,
            "EnableErrorMonitoringInAIMaster": False,
            "EnableOssAppend": False,
            "EnableRDMA": False,
            "EnableSanityCheck": False,
            "Tags": {
                "experiment": "B4-iter20000-eval",
                "protocol": "robocasa365-17x20-seed42",
                "provenance": "git-path-time-no-new-hash",
                "topology": "1x8-physical-dual-4rank-policy",
            },
        },
        "SuccessPolicy": "AllWorkers",
        "UserCommand": f"/bin/bash -lc {json.dumps(shell)}",
        "WorkspaceId": WORKSPACE_ID,
    }


def fsync_parent(path: pathlib.Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_receipt(path: pathlib.Path, payload: dict, *, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    else:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = pathlib.Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    fsync_parent(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    require(os.path.realpath(sys.executable) == os.path.realpath(SDK_PYTHON), f"wrong SDK Python: {sys.executable}")
    validate_local()
    commit = control_commit()
    body = request_body(commit)
    request = models.CreateJobRequest()
    request.from_map(body)
    request.validate()
    require(request.to_map() == body, "SDK request roundtrip changed payload")
    client = make_client()
    verify_no_duplicate(client)
    base = {
        "schema_version": "metadata_no_hash_v1",
        "experiment_id": EXPERIMENT_ID,
        "logical_run_id": LOGICAL_RUN_ID,
        "display_name": DISPLAY_NAME,
        "workspace_id": WORKSPACE_ID,
        "resource_id": RESOURCE_ID,
        "control_commit": commit,
        "source_commit": SOURCE_COMMIT,
        "checkpoint_root": str(CHECKPOINT_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "request": body,
    }
    if args.audit_only:
        print(json.dumps({**base, "status": "AUDIT_PASSED", "audited_at_utc": utc_now()}, indent=2, sort_keys=True))
        return 0
    initial = {**base, "status": "CREATE_REQUEST_PENDING", "created_at_utc": utc_now()}
    write_receipt(RECEIPT, initial, exclusive=True)
    try:
        response = client.create_job(request).body.to_map()
        identifier = str(response.get("JobId") or response.get("job_id") or "")
        require(bool(identifier), f"CreateJob omitted JobId: {response}")
        readback = client.get_job(identifier).body.to_map()
        final = {
            **base,
            "status": "SUBMITTED",
            "submitted_at_utc": utc_now(),
            "job_id": identifier,
            "create_response": response,
            "get_job_readback": readback,
        }
        write_receipt(RECEIPT, final, exclusive=False)
        print(json.dumps(final, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        failed = {**initial, "status": "CREATE_REQUEST_FAILED", "failed_at_utc": utc_now(), "error": repr(error)}
        write_receipt(RECEIPT, failed, exclusive=False)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
