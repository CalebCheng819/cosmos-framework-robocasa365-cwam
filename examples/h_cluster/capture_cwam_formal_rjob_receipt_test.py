# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import copy
import hashlib
import json
import stat
from pathlib import Path

import pytest

from examples.h_cluster.capture_cwam_formal_rjob_receipt import (
    ReceiptNotReady,
    ReceiptValidationError,
    capture_with_retries,
    fetch_scheduler_objects,
    load_expected_contract,
    run_capture,
    validate_and_normalize_receipt,
    validate_expected_contract,
    write_json_atomic_exclusive,
)

_SCHEDULER_NAME = "cwam-e35-formal-s42-20260728-12345678"
_SHOW_NAME = "cwam-e35-formal-s42-20260728"
_EXPERIMENT_ID = "CWAM-E35-ROB365-COSMOS3-NANO-8GPU-S42-20260728"
_SNAPSHOT_ROOT = (
    "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/"
    "run_snapshots/CWAM-E35-ROB365-COSMOS3-NANO-8GPU-S42-20260728"
)
_WRAPPER = f"{_SNAPSHOT_ROOT}/examples/h_cluster/run_cwam_cosmos3_nano_production_then_formal.sh"
_IMAGE = "registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
_NEGATIVE_TAG = "node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn"
_MOUNTS = [
    "gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs",
    "gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public",
]


def _expected() -> dict[str, object]:
    return {
        "schema_version": 1,
        "scheduler_rjob_name": _SCHEDULER_NAME,
        "showname": _SHOW_NAME,
        "quota_group": "eailabagent_gpu",
        "task_type": "normal",
        "priority": 9,
        "backoff_limit": 1,
        "task_name": "t0",
        "replicas": 1,
        "image": _IMAGE,
        "cpu": 128,
        "memory_mib": 1_500_000,
        "gpu": 8,
        "negative_tags": [_NEGATIVE_TAG],
        "private_machine": "group",
        "restart_policy": "Never",
        "termination_grace_period_seconds": 60,
        "mounts": list(_MOUNTS),
        "snapshot_root": _SNAPSHOT_ROOT,
        "wrapper_path": _WRAPPER,
        "experiment_id": _EXPERIMENT_ID,
        "command_sha256": hashlib.sha256(_formal_shell_script().encode()).hexdigest(),
    }


def _formal_shell_script() -> str:
    return "\n".join(
        [
            "set -Eeuo pipefail",
            f"export COSMOS_SOURCE={_SNAPSHOT_ROOT}",
            f"export FORMAL_EXPERIMENT_ID={_EXPERIMENT_ID}",
            f"export FORMAL_RUN_NAME={_EXPERIMENT_ID}",
            f"bash {_WRAPPER}",
        ]
    )


def _raw_rjob() -> dict[str, object]:
    labels = {
        "rjob.brainpp.cn/rjob-name": _SCHEDULER_NAME,
        "rjob.brainpp.cn/zs-task-type": "rjob-normal",
        "quotagroup.brainpp.cn/quotagroup": "",
        "scheduling.k8s.io/group-name": _SCHEDULER_NAME,
    }
    template_labels = {
        **labels,
        "quotagroup.brainpp.cn/quotagroup": "eailabagent_gpu",
        "rjob.brainpp.cn/rjob-task-type": "t0",
        "privatemachine.brainpp.cn/privatemachine": "group",
    }
    return {
        "apiVersion": "rjob.brainpp.cn/v1alpha1",
        "kind": "RJob",
        "metadata": {
            "name": _SCHEDULER_NAME,
            "uid": "6d7d6919-2bff-44fa-8b8f-4d5ec49460e2",
            "resourceVersion": "309123456",
            "creationTimestamp": "2026-07-28T00:00:00Z",
            "labels": labels,
            "annotations": {
                "kubebrain.brainpp.cn/showname": _SHOW_NAME,
                "rjob.brainpp.cn/zs-task-type": "rjob-normal",
                "scheduling.k8s.io/group-name": _SCHEDULER_NAME,
                "volcano.brainpp.cn/priority": "9",
                "unrelated": "scheduler-added",
            },
        },
        "spec": {
            "backoffLimit": 1,
            "taskSpecs": {
                "t0": {
                    "replicas": 1,
                    "negativeTags": [_NEGATIVE_TAG],
                    "template": {
                        "metadata": {
                            "labels": template_labels,
                            "annotations": {
                                "mount.brainpp.cn/gpfs": ";".join(_MOUNTS),
                                "unrelated": "scheduler-added",
                            },
                        },
                        "spec": {
                            "restartPolicy": "Never",
                            "terminationGracePeriodSeconds": 60,
                            "containers": [
                                {
                                    "name": "t0",
                                    "image": _IMAGE,
                                    "command": ["bash", "-lc", _formal_shell_script()],
                                    "resources": {
                                        "requests": {
                                            "cpu": "128",
                                            "memory": "1500000Mi",
                                            "nvidia.com/gpu": "8",
                                        },
                                        "limits": {
                                            "cpu": "128",
                                            "memory": "1500000Mi",
                                            "nvidia.com/gpu": "8",
                                        },
                                    },
                                }
                            ],
                        },
                    },
                }
            },
        },
        "status": {
            "phase": "Inqueue",
            "reason": "RJobInqueue",
            "message": "RJob is enqueue",
            "conditions": [
                {
                    "type": "Inqueue",
                    "status": "True",
                    "reason": "StartInqueue",
                    "lastTransitionTime": "2026-07-28T00:00:01Z",
                    "ignored": "not copied",
                }
            ],
            "taskStatuses": {"t0": {"pending": 1}},
        },
    }


def _raw_replica(*, phase: str = "Pending") -> dict[str, object]:
    replica_name = f"{_SCHEDULER_NAME}-abcde"
    return {
        "apiVersion": "rjob.brainpp.cn/v1alpha1",
        "kind": "Replica",
        "metadata": {
            "name": replica_name,
            "uid": "792a8db7-dc8d-4c62-8c7e-a5fb4494b564",
            "resourceVersion": "309123457",
            "creationTimestamp": "2026-07-28T00:00:01Z",
            "labels": {
                "rjob.brainpp.cn/rjob-name": _SCHEDULER_NAME,
                "rjob.brainpp.cn/rjob-task-type": "t0",
                "rjob.brainpp.cn/zs-task-type": "rjob-normal",
                "quotagroup.brainpp.cn/quotagroup": "eailabagent_gpu",
                "privatemachine.brainpp.cn/privatemachine": "group",
                "kubebrain.brainpp.cn/replica": "0",
            },
            "annotations": {
                "mount.brainpp.cn/gpfs": ";".join(_MOUNTS),
            },
        },
        "spec": {
            "type": "t0",
            "hostname": "rjob-formal-hostname-0",
            "restartPolicy": "Never",
            "terminationGracePeriodSeconds": 60,
            "containers": [
                {
                    "name": "t0",
                    "image": _IMAGE,
                    "command": [
                        "/kubebrain/worker-init",
                        "run",
                        "--enable-pid-monitor",
                        "--",
                        "bash",
                        "-lc",
                        _formal_shell_script(),
                    ],
                    "env": [{"name": "PLATFORM_INJECTED", "value": "1"}],
                    "resources": {
                        "requests": {
                            "cpu": "128",
                            "memory": "1500000Mi",
                            "nvidia.com/gpu": "8",
                        },
                        "limits": {
                            "cpu": "128",
                            "memory": "1500000Mi",
                            "nvidia.com/gpu": "8",
                            "mellanox.com/mlnx_rdma": "1",
                            "rdma/mlnx_shared": "8",
                        },
                    },
                    "volumeMounts": [{"name": "platform", "mountPath": "/kubebrain"}],
                }
            ],
            "volumes": [{"name": "platform", "emptyDir": {}}],
        },
        "status": {
            "phase": phase,
            "nodeName": None,
            "podIP": None,
            "conditions": [
                {
                    "type": "PodScheduled",
                    "status": "False",
                    "reason": "Unschedulable",
                    "message": "Insufficient nvidia.com/gpu",
                }
            ],
        },
    }


def _raw_replicas(*, phase: str = "Pending") -> dict[str, object]:
    return {
        "apiVersion": "rjob.brainpp.cn/v1alpha1",
        "kind": "ReplicaList",
        "items": [_raw_replica(phase=phase)],
    }


def _set_path(root: object, path: tuple[object, ...], value: object) -> None:
    current = root
    for key in path[:-1]:
        current = current[key]  # type: ignore[index]
    current[path[-1]] = value  # type: ignore[index]


def test_valid_receipt_normalizes_authoritative_fields_and_replica_identity() -> None:
    raw_rjob = _raw_rjob()
    raw_replicas = _raw_replicas()
    receipt = validate_and_normalize_receipt(
        raw_rjob,
        raw_replicas,
        _expected(),
        captured_at="2026-07-28T00:00:02+00:00",
    )

    assert receipt["kind"] == "cwam_formal_rjob_scheduler_receipt"
    assert receipt["validation"] == {
        "status": "passed",
        "expected_contract_schema_version": 1,
    }
    assert receipt["metadata"] == {
        "name": _SCHEDULER_NAME,
        "showname": _SHOW_NAME,
        "uid": "6d7d6919-2bff-44fa-8b8f-4d5ec49460e2",
        "resource_version": "309123456",
        "creation_timestamp": "2026-07-28T00:00:00Z",
    }
    assert receipt["spec"]["task"]["container"]["resources"] == {
        "requests_equal_core_limits": True,
        "cpu": 128,
        "memory_mib": 1_500_000,
        "gpu": 8,
        "platform_injected_limit_keys": [],
    }
    assert receipt["status"]["task_statuses"] == {"t0": {"pending": 1}}
    assert receipt["replica_names"] == [f"{_SCHEDULER_NAME}-abcde"]
    assert receipt["pod_names"] == ["rjob-formal-hostname-0"]
    assert receipt["replicas"][0]["pod_hostname"] == "rjob-formal-hostname-0"
    assert receipt["replicas"][0]["pod_name"] == "rjob-formal-hostname-0"
    assert receipt["replicas"][0]["worker_spec"]["resources"] == {
        "requests_equal_core_limits": True,
        "cpu": 128,
        "memory_mib": 1_500_000,
        "gpu": 8,
        "platform_injected_limit_keys": [
            "mellanox.com/mlnx_rdma",
            "rdma/mlnx_shared",
        ],
    }
    assert receipt["replicas"][0]["worker_spec"]["worker_init_prefix"] == [
        "/kubebrain/worker-init",
        "run",
        "--enable-pid-monitor",
        "--",
    ]
    assert receipt["replicas"][0]["status"]["phase"] == "Pending"
    assert "ignored" not in receipt["status"]["conditions"][0]
    assert (
        receipt["source_objects"]["rjob_raw_sha256"]
        == hashlib.sha256(json.dumps(raw_rjob, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )
    assert (
        receipt["source_objects"]["replicas_raw_sha256"]
        == hashlib.sha256(json.dumps(raw_replicas["items"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("metadata", "name"), "different-12345678"),
        (("metadata", "annotations", "kubebrain.brainpp.cn/showname"), "different"),
        (("metadata", "labels", "rjob.brainpp.cn/zs-task-type"), "rjob-idle"),
        (("metadata", "annotations", "volcano.brainpp.cn/priority"), "8"),
        (("spec", "backoffLimit"), 2),
        (("spec", "taskSpecs", "t0", "replicas"), 2),
        (("spec", "taskSpecs", "t0", "negativeTags"), ["node/wrong"]),
        (
            (
                "spec",
                "taskSpecs",
                "t0",
                "template",
                "metadata",
                "labels",
                "quotagroup.brainpp.cn/quotagroup",
            ),
            "wrong_gpu",
        ),
        (
            (
                "spec",
                "taskSpecs",
                "t0",
                "template",
                "metadata",
                "labels",
                "privatemachine.brainpp.cn/privatemachine",
            ),
            "no",
        ),
        (
            (
                "spec",
                "taskSpecs",
                "t0",
                "template",
                "metadata",
                "annotations",
                "mount.brainpp.cn/gpfs",
            ),
            _MOUNTS[0],
        ),
        (
            ("spec", "taskSpecs", "t0", "template", "spec", "restartPolicy"),
            "RestartJobOnFailure",
        ),
        (
            ("spec", "taskSpecs", "t0", "template", "spec", "terminationGracePeriodSeconds"),
            30,
        ),
        (
            ("spec", "taskSpecs", "t0", "template", "spec", "containers", 0, "image"),
            "registry.example/model@sha256:" + "a" * 64,
        ),
        (
            (
                "spec",
                "taskSpecs",
                "t0",
                "template",
                "spec",
                "containers",
                0,
                "resources",
                "requests",
                "cpu",
            ),
            "127",
        ),
        (
            (
                "spec",
                "taskSpecs",
                "t0",
                "template",
                "spec",
                "containers",
                0,
                "resources",
                "requests",
                "memory",
            ),
            "1499999Mi",
        ),
        (
            (
                "spec",
                "taskSpecs",
                "t0",
                "template",
                "spec",
                "containers",
                0,
                "resources",
                "requests",
                "nvidia.com/gpu",
            ),
            "6",
        ),
    ],
)
def test_scheduler_contract_mutations_fail_closed(path: tuple[object, ...], bad_value: object) -> None:
    raw = _raw_rjob()
    _set_path(raw, path, bad_value)
    with pytest.raises(ReceiptValidationError):
        validate_and_normalize_receipt(raw, _raw_replicas(), _expected())


def test_task_specs_and_container_sets_must_be_exact() -> None:
    raw = _raw_rjob()
    raw["spec"]["taskSpecs"]["extra"] = copy.deepcopy(raw["spec"]["taskSpecs"]["t0"])  # type: ignore[index]
    with pytest.raises(ReceiptValidationError, match="taskSpecs must contain exactly"):
        validate_and_normalize_receipt(raw, _raw_replicas(), _expected())

    raw = _raw_rjob()
    containers = raw["spec"]["taskSpecs"]["t0"]["template"]["spec"]["containers"]  # type: ignore[index]
    containers.append(copy.deepcopy(containers[0]))
    with pytest.raises(ReceiptValidationError, match="exactly one formal container"):
        validate_and_normalize_receipt(raw, _raw_replicas(), _expected())


def test_requests_must_equal_limits_and_have_no_unexpected_resources() -> None:
    raw = _raw_rjob()
    limits = raw["spec"]["taskSpecs"]["t0"]["template"]["spec"]["containers"][0]["resources"]["limits"]  # type: ignore[index]
    limits["cpu"] = "127"
    with pytest.raises(ReceiptValidationError, match="requests/limits cpu mismatch"):
        validate_and_normalize_receipt(raw, _raw_replicas(), _expected())

    raw = _raw_rjob()
    resources = raw["spec"]["taskSpecs"]["t0"]["template"]["spec"]["containers"][0]["resources"]  # type: ignore[index]
    resources["requests"]["rdma/mlnx_shared"] = "8"
    resources["limits"]["rdma/mlnx_shared"] = "8"
    with pytest.raises(ReceiptValidationError, match="exactly cpu, memory"):
        validate_and_normalize_receipt(raw, _raw_replicas(), _expected())


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("metadata", "annotations", "mount.brainpp.cn/gpfs"), _MOUNTS[0]),
        (("metadata", "labels", "quotagroup.brainpp.cn/quotagroup"), "wrong_gpu"),
        (("metadata", "labels", "privatemachine.brainpp.cn/privatemachine"), "no"),
        (("metadata", "labels", "rjob.brainpp.cn/zs-task-type"), "rjob-idle"),
        (("spec", "restartPolicy"), "RestartJobOnFailure"),
        (("spec", "terminationGracePeriodSeconds"), 30),
        (("spec", "containers", 0, "image"), "registry.example/model@sha256:" + "a" * 64),
        (("spec", "containers", 0, "resources", "requests", "cpu"), "127"),
        (("spec", "containers", 0, "resources", "limits", "cpu"), "127"),
        (("spec", "containers", 0, "resources", "requests", "memory"), "1499999Mi"),
        (("spec", "containers", 0, "resources", "limits", "memory"), "1499999Mi"),
        (("spec", "containers", 0, "resources", "requests", "nvidia.com/gpu"), "6"),
        (("spec", "containers", 0, "resources", "limits", "nvidia.com/gpu"), "6"),
        (("spec", "containers", 0, "command", 0), "/tmp/untrusted-worker-init"),
        (
            ("spec", "containers", 0, "command", 6),
            "\n".join(
                [
                    f"export COSMOS_SOURCE={_SNAPSHOT_ROOT}",
                    "export FORMAL_EXPERIMENT_ID=CWAM-E35-WRONG",
                    f"export FORMAL_RUN_NAME={_EXPERIMENT_ID}",
                    f"bash {_WRAPPER}",
                ]
            ),
        ),
    ],
)
def test_expanded_worker_contract_mutations_fail_closed(
    path: tuple[object, ...],
    bad_value: object,
) -> None:
    replicas = _raw_replicas()
    _set_path(replicas["items"][0], path, bad_value)
    with pytest.raises(ReceiptValidationError):
        validate_and_normalize_receipt(_raw_rjob(), replicas, _expected())


def test_expanded_worker_allows_only_audited_platform_injections() -> None:
    replicas = _raw_replicas()
    worker = replicas["items"][0]["spec"]["containers"][0]
    worker["resources"]["limits"]["example.com/unexpected"] = "1"
    with pytest.raises(ReceiptValidationError, match="unsupported extra resources"):
        validate_and_normalize_receipt(_raw_rjob(), replicas, _expected())

    replicas = _raw_replicas()
    worker = replicas["items"][0]["spec"]["containers"][0]
    worker["resources"]["requests"]["rdma/mlnx_shared"] = "8"
    with pytest.raises(ReceiptValidationError, match="requests must contain exactly"):
        validate_and_normalize_receipt(_raw_rjob(), replicas, _expected())

    replicas = _raw_replicas()
    worker = replicas["items"][0]["spec"]["containers"][0]
    worker["command"] = worker["command"][4:]
    receipt = validate_and_normalize_receipt(_raw_rjob(), replicas, _expected())
    assert receipt["replicas"][0]["worker_spec"]["worker_init_prefix"] == []


@pytest.mark.parametrize(
    "command",
    [
        [
            "bash",
            "-lc",
            "\n".join(
                [
                    f"export COSMOS_SOURCE={_SNAPSHOT_ROOT}",
                    "export FORMAL_EXPERIMENT_ID=CWAM-E35-WRONG",
                    f"export FORMAL_RUN_NAME={_EXPERIMENT_ID}",
                    f"bash {_WRAPPER}",
                ]
            ),
        ],
        [
            "bash",
            "-lc",
            "\n".join(
                [
                    f"export COSMOS_SOURCE={_SNAPSHOT_ROOT}",
                    f"export FORMAL_EXPERIMENT_ID={_EXPERIMENT_ID}",
                    f"export FORMAL_RUN_NAME={_EXPERIMENT_ID}",
                    "bash /tmp/unverified-wrapper.sh",
                ]
            ),
        ],
        ["bash", "-c", f"bash {_WRAPPER}"],
        ["bash", "-lc", f"{_formal_shell_script()}\necho injected"],
    ],
)
def test_command_must_bind_exact_snapshot_wrapper_and_experiment(command: list[str]) -> None:
    raw = _raw_rjob()
    raw["spec"]["taskSpecs"]["t0"]["template"]["spec"]["containers"][0]["command"] = command  # type: ignore[index]
    with pytest.raises(ReceiptValidationError):
        validate_and_normalize_receipt(raw, _raw_replicas(), _expected())


def test_integral_equivalent_kubernetes_quantities_are_normalized() -> None:
    raw = _raw_rjob()
    resources = raw["spec"]["taskSpecs"]["t0"]["template"]["spec"]["containers"][0]["resources"]  # type: ignore[index]
    resources["requests"]["cpu"] = resources["limits"]["cpu"] = "128000m"
    assert (
        validate_and_normalize_receipt(raw, _raw_replicas(), _expected())["spec"]["task"]["container"]["resources"][
            "cpu"
        ]
        == 128
    )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda rjob, replicas: replicas.update({"items": []}),
            "has not materialized",
        ),
        (
            lambda rjob, replicas: rjob["status"].pop("phase"),
            "no scheduler phase",
        ),
        (
            lambda rjob, replicas: rjob["status"].pop("taskStatuses"),
            "no taskStatuses",
        ),
        (
            lambda rjob, replicas: replicas["items"][0]["status"].pop("phase"),
            "no scheduler phase",
        ),
    ],
)
def test_incomplete_scheduler_materialization_is_retryable(mutator, message: str) -> None:
    raw_rjob = _raw_rjob()
    raw_replicas = _raw_replicas()
    mutator(raw_rjob, raw_replicas)
    with pytest.raises(ReceiptNotReady, match=message):
        validate_and_normalize_receipt(raw_rjob, raw_replicas, _expected())


def test_replica_identity_and_ownership_must_match() -> None:
    replicas = _raw_replicas()
    replicas["items"][0]["metadata"]["labels"]["rjob.brainpp.cn/rjob-name"] = "other-job"
    with pytest.raises(ReceiptValidationError, match="replica label"):
        validate_and_normalize_receipt(_raw_rjob(), replicas, _expected())

    replicas = _raw_replicas()
    second = copy.deepcopy(replicas["items"][0])
    second["metadata"]["name"] = f"{_SCHEDULER_NAME}-other"
    second["metadata"]["uid"] = "0ce21385-7c22-4016-acac-1e333132ec02"
    replicas["items"].append(second)
    with pytest.raises(ReceiptValidationError, match="exactly 1 scheduler replica"):
        validate_and_normalize_receipt(_raw_rjob(), replicas, _expected())


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("task_type", "idle"),
        ("priority", 8),
        ("backoff_limit", 2),
        ("task_name", "t1"),
        ("replicas", 2),
        ("private_machine", "no"),
        ("restart_policy", "never"),
        ("termination_grace_period_seconds", 30),
    ],
)
def test_expected_formal_constants_are_fail_closed(field: str, bad_value: object) -> None:
    expected = _expected()
    expected[field] = bad_value
    with pytest.raises(ReceiptValidationError, match=f"expected.{field}"):
        validate_expected_contract(expected)


def test_expected_schema_rejects_missing_extra_and_invalid_paths() -> None:
    expected = _expected()
    expected.pop("gpu")
    with pytest.raises(ReceiptValidationError, match="missing="):
        validate_expected_contract(expected)

    expected = _expected()
    expected["unknown"] = True
    with pytest.raises(ReceiptValidationError, match="extra="):
        validate_expected_contract(expected)

    expected = _expected()
    expected["wrapper_path"] = "/tmp/run.sh"
    with pytest.raises(ReceiptValidationError, match="formal wrapper inside snapshot_root"):
        validate_expected_contract(expected)


def test_expected_json_must_be_a_non_symlink_regular_file(tmp_path: Path) -> None:
    real = tmp_path / "expected-real.json"
    real.write_text(json.dumps(_expected()))
    link = tmp_path / "expected-link.json"
    link.symlink_to(real)
    with pytest.raises(ReceiptValidationError, match="non-symlink regular file"):
        load_expected_contract(link)


class _FakeApi:
    def __init__(self, raw_rjob: dict[str, object], raw_replicas: dict[str, object]):
        self.raw_rjob = raw_rjob
        self.raw_replicas = raw_replicas
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get_namespaced_custom_object(self, **kwargs):
        self.calls.append(("get", kwargs))
        return self.raw_rjob

    def list_namespaced_custom_object(self, **kwargs):
        self.calls.append(("list", kwargs))
        return self.raw_replicas


class _FakeClient:
    group = "rjob.brainpp.cn"
    version = "v1alpha1"
    namespace = "ailab-eailabagent"
    plural = "rjobs"

    def __init__(self, raw_rjob: dict[str, object] | None = None, raw_replicas: dict[str, object] | None = None):
        self.api = _FakeApi(raw_rjob or _raw_rjob(), raw_replicas or _raw_replicas())


def test_fetch_scheduler_objects_uses_exact_rjob_and_replica_selector() -> None:
    client = _FakeClient()
    raw_rjob, raw_replicas = fetch_scheduler_objects(client, _SCHEDULER_NAME)
    assert raw_rjob is client.api.raw_rjob
    assert raw_replicas is client.api.raw_replicas
    assert client.api.calls == [
        (
            "get",
            {
                "group": "rjob.brainpp.cn",
                "version": "v1alpha1",
                "namespace": "ailab-eailabagent",
                "plural": "rjobs",
                "name": _SCHEDULER_NAME,
                "_request_timeout": (5.0, 15.0),
            },
        ),
        (
            "list",
            {
                "group": "rjob.brainpp.cn",
                "version": "v1alpha1",
                "namespace": "ailab-eailabagent",
                "resource_version": "0",
                "plural": "replicas",
                "label_selector": f"rjob.brainpp.cn/rjob-name={_SCHEDULER_NAME}",
                "_request_timeout": (5.0, 15.0),
            },
        ),
    ]


def test_capture_retries_only_transient_or_not_ready_failures() -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    def fetcher(client, name):
        del client
        assert name == _SCHEDULER_NAME
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("temporary API outage")
        if len(calls) == 2:
            return _raw_rjob(), {"items": []}
        return _raw_rjob(), _raw_replicas()

    receipt = capture_with_retries(
        object(),
        _expected(),
        attempts=3,
        retry_seconds=0.25,
        sleeper=sleeps.append,
        fetcher=fetcher,
        captured_at_factory=lambda: "2026-07-28T00:00:02+00:00",
    )
    assert receipt["metadata"]["name"] == _SCHEDULER_NAME
    assert len(calls) == 3
    assert sleeps == [0.25, 0.25]

    validation_calls = 0

    def invalid_fetcher(client, name):
        nonlocal validation_calls
        del client, name
        validation_calls += 1
        raw = _raw_rjob()
        raw["spec"]["backoffLimit"] = 2
        return raw, _raw_replicas()

    with pytest.raises(ReceiptValidationError, match="backoffLimit"):
        capture_with_retries(object(), _expected(), attempts=5, fetcher=invalid_fetcher)
    assert validation_calls == 1


def test_capture_exhaustion_reports_no_receipt() -> None:
    with pytest.raises(ReceiptNotReady, match=r"after 2 attempt\(s\)"):
        capture_with_retries(
            object(),
            _expected(),
            attempts=2,
            retry_seconds=0,
            sleeper=lambda _: None,
            fetcher=lambda client, name: (_raw_rjob(), {"items": []}),
        )


def test_atomic_writer_publishes_read_only_json_and_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    payload = {"schema_version": 1, "status": "passed"}
    digest = write_json_atomic_exclusive(target, payload)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    assert target.read_bytes() == encoded
    assert digest == hashlib.sha256(encoded).hexdigest()
    assert stat.S_IMODE(target.stat().st_mode) == 0o444
    assert not list(tmp_path.glob(".receipt.json.*.tmp"))
    with pytest.raises(ReceiptValidationError, match="refusing to overwrite"):
        write_json_atomic_exclusive(target, {"different": True})
    assert target.read_bytes() == encoded


def test_atomic_writer_rejects_relative_and_symlink_parent_paths(tmp_path: Path) -> None:
    with pytest.raises(ReceiptValidationError, match="must be absolute"):
        write_json_atomic_exclusive(Path("receipt.json"), {})

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ReceiptValidationError, match="canonical non-symlink"):
        write_json_atomic_exclusive(linked_parent / "receipt.json", {})
    assert not (real_parent / "receipt.json").exists()


def test_run_capture_binds_cli_name_and_publishes_only_after_validation(tmp_path: Path) -> None:
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps(_expected()))
    output = tmp_path / "receipt.json"
    client = _FakeClient()
    receipt, digest = run_capture(
        scheduler_rjob_name=_SCHEDULER_NAME,
        expected_json=expected_path,
        output=output,
        attempts=1,
        retry_seconds=0,
        client_factory=lambda: client,
    )
    assert json.loads(output.read_text()) == receipt
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest

    called = False

    def forbidden_client_factory():
        nonlocal called
        called = True
        return client

    with pytest.raises(ReceiptValidationError, match="CLI --scheduler-rjob-name"):
        run_capture(
            scheduler_rjob_name="wrong-name",
            expected_json=expected_path,
            output=tmp_path / "wrong.json",
            attempts=1,
            retry_seconds=0,
            client_factory=forbidden_client_factory,
        )
    assert called is False
    assert not (tmp_path / "wrong.json").exists()


def test_failed_validation_never_creates_output(tmp_path: Path) -> None:
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps(_expected()))
    output = tmp_path / "receipt.json"
    invalid = _raw_rjob()
    invalid["spec"]["backoffLimit"] = 2
    with pytest.raises(ReceiptValidationError, match="backoffLimit"):
        run_capture(
            scheduler_rjob_name=_SCHEDULER_NAME,
            expected_json=expected_path,
            output=output,
            attempts=1,
            retry_seconds=0,
            client_factory=lambda: _FakeClient(raw_rjob=invalid),
        )
    assert not output.exists()
    assert not output.is_symlink()
