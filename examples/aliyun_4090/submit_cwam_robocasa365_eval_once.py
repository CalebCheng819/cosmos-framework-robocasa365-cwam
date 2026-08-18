#!/usr/bin/env python3
"""Audit and submit one configuration-bound RoboCasa365 DLC evaluation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import shlex
import stat
import subprocess
import sys
import tempfile
from typing import Any

from alibabacloud_credentials.client import Client as CredentialClient
from alibabacloud_credentials.models import Config as CredentialConfig
from alibabacloud_pai_dlc20201203 import models
from alibabacloud_pai_dlc20201203.client import Client
from alibabacloud_tea_openapi.models import Config


WORKSPACE_ID = "270969"
RESOURCE_ID = "quotaksvqq2oh2pg"
PROJECT_ROOT = pathlib.Path("/mnt/workspace/cwam_cosmos3_robocasa365_eval")
OSS_ROOT = pathlib.Path(
    "/oss-chengjuntao/cpfs-user-chengjuntao/cwam_cosmos3_robocasa365_eval"
)
SDK_PYTHON = pathlib.Path(
    "/mnt/workspace/tools/pai-control-py311/"
    "20260817-credentials1.0.10-dlc1.9.2/bin/python"
)
IMAGE = (
    "dsw-registry-vpc.cn-beijing.cr.aliyuncs.com/pai/"
    "pytorch:2.7.1-gpu-py310-cu128-ubuntu22.04-3995b779-1764350887"
)
DATA_SOURCES = [
    {"DataSourceId": "d-a5mu77ymwjio71dkmw", "MountAccess": "RW", "MountPath": "/mnt/workspace"},
    {"DataSourceId": "d-n7rly4fll0q2z6v91h", "MountAccess": "RW", "MountPath": "/oss-chengjuntao"},
]
CONTROL_FILES = (
    "launch_cwam_b4_robocasa365_eval_dlc.sh",
    "run_cwam_b4_robocasa365_eval_dlc.sh",
    "run_cwam_b4_robocasa365_eval_shard_group.sh",
    "cwam_b4_eval_shell_utils.sh",
    "cwam_b4_robocasa365_eval.py",
    "cwam_cosmos_env_archive.py",
    "cwam_robocasa_asset_source_package.py",
    "cwam_dlc_persist_eval_output.py",
    "probe_mujoco_egl_mapping.py",
    "model-spec.json",
)
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
NOTION_PAGE_ID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_object(path: pathlib.Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{label} must contain a JSON object")
    return value


def load_submission(path: pathlib.Path) -> dict[str, Any]:
    config = load_object(path, "submission config")
    require(config.get("kind") == "cwam_robocasa365_dlc_eval_submission", "invalid submission kind")
    for name in ("experiment_id", "logical_run_id", "display_name"):
        value = config.get(name)
        require(isinstance(value, str) and SAFE_ID.fullmatch(value), f"invalid {name}")
    require(isinstance(config.get("description"), str) and config["description"], "invalid description")
    require(
        isinstance(config.get("notion_page_id"), str)
        and NOTION_PAGE_ID.fullmatch(config["notion_page_id"]),
        "invalid notion_page_id",
    )
    source = config.get("source")
    checkpoint = config.get("checkpoint")
    evaluation = config.get("evaluation")
    tags = config.get("tags")
    require(isinstance(source, dict), "invalid source")
    require(isinstance(checkpoint, dict), "invalid checkpoint")
    require(isinstance(evaluation, dict), "invalid evaluation")
    require(isinstance(tags, dict) and tags, "invalid tags")
    require(REVISION.fullmatch(str(source.get("revision", ""))), "invalid source revision")
    require(isinstance(source.get("run_id"), str) and source["run_id"], "invalid source run_id")
    require(
        isinstance(source.get("directory"), str)
        and SAFE_ID.fullmatch(source["directory"]),
        "invalid source directory",
    )
    require(
        isinstance(checkpoint.get("name"), str)
        and SAFE_ID.fullmatch(checkpoint["name"]),
        "invalid checkpoint name",
    )
    inventory = checkpoint.get("inventory")
    require(isinstance(inventory, dict) and inventory, "invalid checkpoint inventory")
    require(
        all(isinstance(name, str) and type(size) is int and size > 0 for name, size in inventory.items()),
        "invalid checkpoint inventory entry",
    )
    require(
        isinstance(evaluation.get("run_name"), str)
        and SAFE_ID.fullmatch(evaluation["run_name"]),
        "invalid evaluation run_name",
    )
    require(
        type(evaluation.get("expected_ema_key_count")) is int
        and evaluation["expected_ema_key_count"] > 0,
        "invalid expected EMA key count",
    )
    require(all(isinstance(key, str) and isinstance(value, str) for key, value in tags.items()), "invalid tags")
    return config


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
    return Client(
        Config(
            credential=credential,
            region_id="cn-beijing",
            endpoint="pai-dlc.cn-beijing.aliyuncs.com",
        )
    )


def all_jobs(client: Client) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    page = 1
    total: int | None = None
    while True:
        body = client.list_jobs(
            models.ListJobsRequest(
                workspace_id=WORKSPACE_ID,
                resource_id=RESOURCE_ID,
                page_number=page,
                page_size=100,
            )
        ).body.to_map()
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


def stable_matching_jobs(client: Client, config: dict[str, Any]) -> list[dict[str, Any]]:
    first = all_jobs(client)
    second = all_jobs(client)
    identity = lambda job: (
        str(job.get("JobId") or job.get("job_id")),
        str(job.get("DisplayName") or job.get("display_name")),
    )
    require(sorted(map(identity, first)) == sorted(map(identity, second)), "ListJobs snapshot changed")
    return [
        job
        for job in second
        if identity(job)[1] == config["display_name"]
        or config["experiment_id"] in str(job.get("Description") or "")
    ]


def get_job_readback(client: Client, identifier: str) -> dict[str, Any]:
    return client.get_job(identifier, models.GetJobRequest(need_detail=True)).body.to_map()


def mounted_target(path: pathlib.Path) -> str:
    return subprocess.check_output(
        ["findmnt", "-n", "-T", str(path), "-o", "TARGET"], text=True
    ).strip()


def control_commit() -> str:
    value = os.environ.get("EVAL_CONTROL_COMMIT", "")
    require(bool(REVISION.fullmatch(value)), "invalid EVAL_CONTROL_COMMIT")
    return value


def source_revision(root: pathlib.Path) -> str:
    if (root / ".git").is_dir():
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={root}", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    marker = root / "GIT_REVISION"
    require(marker.is_file() and not marker.is_symlink(), "source GIT_REVISION marker missing")
    return marker.read_text(encoding="ascii").strip()


def resolved_paths(config: dict[str, Any]) -> dict[str, pathlib.Path]:
    experiment_id = config["experiment_id"]
    logical_run_id = config["logical_run_id"]
    return {
        "control_root": PROJECT_ROOT / "control" / "scripts" / f"{experiment_id}-control-v1",
        "source_root": PROJECT_ROOT / "inputs" / "source" / config["source"]["directory"],
        "checkpoint_root": OSS_ROOT / "inputs" / "checkpoints" / config["checkpoint"]["name"],
        "output_root": OSS_ROOT / "runs" / experiment_id / logical_run_id,
        "receipt": OSS_ROOT / "control" / "receipts" / f"{experiment_id}-dlc-submit.json",
    }


def validate_local(config: dict[str, Any], paths: dict[str, pathlib.Path], *, require_unsubmitted: bool) -> None:
    require(mounted_target(PROJECT_ROOT) == "/mnt/workspace", "CPFS mount missing")
    require(mounted_target(OSS_ROOT) == "/oss-chengjuntao", "OSS mount missing")
    source_root = paths["source_root"]
    require(source_root.is_dir() and not source_root.is_symlink(), "source snapshot missing")
    require(source_revision(source_root) == config["source"]["revision"], "source snapshot revision mismatch")
    run_id = source_root / "RUN_ID"
    require(run_id.is_file() and not run_id.is_symlink(), "source RUN_ID marker missing")
    require(run_id.read_text(encoding="utf-8").strip() == config["source"]["run_id"], "source run ID mismatch")

    checkpoint_root = paths["checkpoint_root"]
    require(checkpoint_root.is_dir() and not checkpoint_root.is_symlink(), "checkpoint missing")
    observed = {
        path.relative_to(checkpoint_root).as_posix(): path.stat().st_size
        for path in checkpoint_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    require(observed == config["checkpoint"]["inventory"], "checkpoint inventory differs from submission config")
    require(not any(path.is_symlink() for path in checkpoint_root.rglob("*")), "checkpoint contains a symlink")

    control_root = paths["control_root"]
    for name in CONTROL_FILES:
        path = control_root / name
        metadata = os.stat(path, follow_symlinks=False)
        require(stat.S_ISREG(metadata.st_mode), f"control file is not regular: {path}")
        require(stat.S_IMODE(metadata.st_mode) == 0o444, f"control file mode is not 0444: {path}")
    model_spec = load_object(control_root / "model-spec.json", "model spec")
    require(model_spec.get("checkpoint_name") == config["checkpoint"]["name"], "model spec checkpoint mismatch")
    require(
        model_spec.get("source")
        == {
            "revision": config["source"]["revision"],
            "run_id": config["source"]["run_id"],
        },
        "model spec source mismatch",
    )
    require(
        (model_spec.get("ema") or {}).get("key_count")
        == config["evaluation"]["expected_ema_key_count"],
        "model spec EMA key count mismatch",
    )
    expected_from_model_spec = {
        **{f"model/{name}": size for name, size in model_spec["artifact"]["model_files"].items()},
        **model_spec["artifact"]["aux_files"],
    }
    require(expected_from_model_spec == config["checkpoint"]["inventory"], "model spec inventory mismatch")

    receipt = paths["receipt"]
    require(receipt.parent.is_dir() and os.access(receipt.parent, os.W_OK), "receipt parent unavailable")
    if require_unsubmitted:
        require(not os.path.lexists(paths["output_root"]), f"output already exists: {paths['output_root']}")
        require(not os.path.lexists(receipt), f"submission receipt exists: {receipt}")


def request_body(config: dict[str, Any], paths: dict[str, pathlib.Path], commit: str) -> dict[str, Any]:
    exports = {
        "EVAL_CONTROL_COMMIT": commit,
        "EVAL_EXPERIMENT_ID": config["experiment_id"],
        "EVAL_LOGICAL_RUN_ID": config["logical_run_id"],
        "EVAL_CONTROL_ROOT": str(paths["control_root"]),
        "EVAL_SOURCE_ROOT": str(paths["source_root"]),
        "EVAL_SOURCE_COMMIT": config["source"]["revision"],
        "EVAL_SOURCE_RUN_ID": config["source"]["run_id"],
        "EVAL_CHECKPOINT_NAME": config["checkpoint"]["name"],
        "EVAL_RUN_NAME": config["evaluation"]["run_name"],
        "EVAL_EXPECTED_EMA_KEY_COUNT": str(config["evaluation"]["expected_ema_key_count"]),
        "EVAL_NOTION_PAGE_ID": config["notion_page_id"],
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics,display",
    }
    shell = "set -Eeuo pipefail; " + " ".join(
        f"export {name}={shlex.quote(value)};" for name, value in exports.items()
    ) + f" exec bash {shlex.quote(str(paths['control_root'] / CONTROL_FILES[0]))}"
    return {
        "Accessibility": "PRIVATE",
        "CustomEnvs": [],
        "DataSources": DATA_SOURCES,
        "Description": config["description"],
        "DisplayName": config["display_name"],
        "Envs": exports,
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
            "Tags": config["tags"],
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


def write_receipt(path: pathlib.Path, payload: dict[str, Any], *, exclusive: bool) -> None:
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


def load_receipt(path: pathlib.Path) -> dict[str, Any]:
    metadata = os.stat(path, follow_symlinks=False)
    require(stat.S_ISREG(metadata.st_mode), f"submission receipt is not regular: {path}")
    return load_object(path, "submission receipt")


def base_receipt(
    config: dict[str, Any],
    paths: dict[str, pathlib.Path],
    commit: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "metadata_no_hash_v1",
        "experiment_id": config["experiment_id"],
        "logical_run_id": config["logical_run_id"],
        "display_name": config["display_name"],
        "notion_page_id": config["notion_page_id"],
        "workspace_id": WORKSPACE_ID,
        "resource_id": RESOURCE_ID,
        "control_commit": commit,
        "source_commit": config["source"]["revision"],
        "source_run_id": config["source"]["run_id"],
        "checkpoint_root": str(paths["checkpoint_root"]),
        "output_root": str(paths["output_root"]),
        "request": body,
    }


def reconcile_existing(
    client: Client,
    config: dict[str, Any],
    paths: dict[str, pathlib.Path],
    base: dict[str, Any],
) -> dict[str, Any]:
    previous = load_receipt(paths["receipt"])
    allowed = {"CREATE_REQUEST_FAILED", "CREATE_ACCEPTED_READBACK_PENDING", "CREATE_ACCEPTED_READBACK_FAILED"}
    require(previous.get("status") in allowed, f"receipt cannot be reconciled: {previous.get('status')}")
    for key, value in base.items():
        require(previous.get(key) == value, f"receipt identity mismatch: {key}")
    matches = stable_matching_jobs(client, config)
    require(len(matches) == 1, f"expected exactly one matching DLC job, found {len(matches)}: {matches}")
    identifier = str(matches[0].get("JobId") or matches[0].get("job_id") or "")
    require(identifier, "matching job omitted JobId")
    previous_identifier = str(previous.get("job_id") or "")
    require(not previous_identifier or previous_identifier == identifier, "receipt JobId differs from ListJobs")
    final = {
        **base,
        "status": "SUBMITTED",
        "submitted_at_utc": previous.get("submitted_at_utc") or previous.get("created_at_utc"),
        "job_id": identifier,
        "create_response": previous.get("create_response"),
        "get_job_readback": get_job_readback(client, identifier),
        "reconciled_at_utc": utc_now(),
        "reconciled_from_status": previous["status"],
        "list_jobs_match": matches[0],
    }
    write_receipt(paths["receipt"], final, exclusive=False)
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--submit", action="store_true")
    mode.add_argument("--reconcile", action="store_true")
    args = parser.parse_args()
    require(os.path.realpath(sys.executable) == os.path.realpath(SDK_PYTHON), f"wrong SDK Python: {sys.executable}")
    config = load_submission(args.config)
    paths = resolved_paths(config)
    validate_local(config, paths, require_unsubmitted=not args.reconcile)
    commit = control_commit()
    body = request_body(config, paths, commit)
    request = models.CreateJobRequest()
    request.from_map(body)
    request.validate()
    require(request.to_map() == body, "SDK request roundtrip changed payload")
    client = make_client()
    base = base_receipt(config, paths, commit, body)
    if args.reconcile:
        final = reconcile_existing(client, config, paths, base)
        print(json.dumps(final, indent=2, sort_keys=True))
        return 0
    matches = stable_matching_jobs(client, config)
    require(not matches, f"matching DLC job already exists: {matches}")
    if args.audit_only:
        print(json.dumps({**base, "status": "AUDIT_PASSED", "audited_at_utc": utc_now()}, indent=2, sort_keys=True))
        return 0

    initial = {**base, "status": "CREATE_REQUEST_PENDING", "created_at_utc": utc_now()}
    write_receipt(paths["receipt"], initial, exclusive=True)
    response: dict[str, Any] | None = None
    identifier = ""
    try:
        response = client.create_job(request).body.to_map()
        identifier = str(response.get("JobId") or response.get("job_id") or "")
        require(identifier, f"CreateJob omitted JobId: {response}")
        accepted = {
            **initial,
            "status": "CREATE_ACCEPTED_READBACK_PENDING",
            "submitted_at_utc": utc_now(),
            "job_id": identifier,
            "create_response": response,
        }
        write_receipt(paths["receipt"], accepted, exclusive=False)
        final = {
            **base,
            "status": "SUBMITTED",
            "submitted_at_utc": accepted["submitted_at_utc"],
            "job_id": identifier,
            "create_response": response,
            "get_job_readback": get_job_readback(client, identifier),
        }
        write_receipt(paths["receipt"], final, exclusive=False)
        print(json.dumps(final, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        failed = {
            **initial,
            "status": "CREATE_ACCEPTED_READBACK_FAILED" if identifier else "CREATE_REQUEST_FAILED",
            "failed_at_utc": utc_now(),
            "error": repr(error),
        }
        if response is not None:
            failed["create_response"] = response
        if identifier:
            failed["job_id"] = identifier
        write_receipt(paths["receipt"], failed, exclusive=False)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
