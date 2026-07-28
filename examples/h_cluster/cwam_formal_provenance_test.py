# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import examples.h_cluster.cwam_formal_provenance as provenance
from examples.h_cluster.cwam_formal_provenance import (
    _EXPECTED_CONFIG_PATH,
    _QWEN_PROCESSOR_FILES,
    ProvenanceError,
    _formal_rjob_contract,
    create_artifact_inventory,
    sha256_file,
    verify_artifact_inventory,
    verify_formal_contract,
    verify_snapshot,
    write_inventory,
)

_IMAGE = "registry.h.pjlab.org.cn/ailab/vulkan@sha256:59204dea15a88b9b444b4f20c2e54b6b92d41582ca718f35e9194335cc8615f7"
_COMMIT = "a" * 40
_WORKFLOW_SHA256 = "b" * 64
_STORAGE = {
    "output_root": "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/outputs/cwam-cosmos3-formal",
    "minimum_free_bytes": 5 * 1024**4,
    "estimated_required_bytes": 4 * 1024**4,
}
_RJOBS = {
    8: {
        "quota_group": "eailabagent_gpu",
        "cpu": 128,
        "memory_mib": 1_500_000,
        "negative_tag": "node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn",
        "task_type": "normal",
        "private_machine": "group",
        "priority": 9,
        "restart_policy": "never",
        "backoff_limit": 1,
    },
    6: {
        "quota_group": "eailabagent_gpu",
        "cpu": 96,
        "memory_mib": 1_125_000,
        "negative_tag": "node/gpu-lg-cmc-h-h200-0685.host.h.pjlab.org.cn",
        "task_type": "normal",
        "private_machine": "group",
        "priority": 9,
        "restart_policy": "never",
        "backoff_limit": 1,
    },
}
_PROTOCOLS = {
    8: {
        "experiment_family": "CWAM-E35",
        "world_size": 8,
        "per_rank_samples": 128,
        "grad_accum_iter": 2,
        "global_batch": 2_048,
        "learning_rate": 5.0e-5,
        "max_iter": 10_000,
        "warm_up_steps": 500,
        "cycle_length": 16_000,
        "validation_iter": 500,
        "max_val_iter": 8,
        "save_iter": 500,
        "train_sample_budget": 20_480_000,
    },
    6: {
        "experiment_family": "CWAM-E35A",
        "world_size": 6,
        "per_rank_samples": 128,
        "grad_accum_iter": 2,
        "global_batch": 1_536,
        "learning_rate": 3.75e-5,
        "max_iter": 13_340,
        "warm_up_steps": 667,
        "cycle_length": 21_344,
        "validation_iter": 667,
        "max_val_iter": 11,
        "save_iter": 667,
        "train_sample_budget": 20_490_240,
    },
}


def _write(path: Path, payload: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload.encode() if isinstance(payload, str) else payload)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    _write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _make_read_only_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            continue
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


class SnapshotVerificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.allowed_root = Path(self.temporary.name) / "run_snapshots"
        self.allowed_root.mkdir()

    def make_snapshot(self, name: str) -> tuple[Path, str]:
        root = self.allowed_root / name
        payloads = {
            "config/example.toml": b"batch_size = 128\n",
            "src/module.py": b"VALUE = 1\n",
        }
        for relative, payload in payloads.items():
            _write(root / relative, payload)
        manifest = {
            "schema_version": 1,
            "file_count": len(payloads),
            "files": [
                {"path": relative, "bytes": len(payload), "sha256": _sha256(payload)}
                for relative, payload in sorted(payloads.items())
            ],
        }
        _write_json(root / "SNAPSHOT_MANIFEST.json", manifest)
        manifest_sha256 = sha256_file(root / "SNAPSHOT_MANIFEST.json")
        _make_read_only_tree(root)
        return root, manifest_sha256

    def test_accepts_exact_read_only_snapshot(self) -> None:
        root, manifest_sha256 = self.make_snapshot("good")
        report = verify_snapshot(
            snapshot_root=root,
            allowed_root=self.allowed_root,
            expected_manifest_sha256=manifest_sha256,
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["file_count"], 2)

    def test_rejects_tamper_and_extra_file(self) -> None:
        root, manifest_sha256 = self.make_snapshot("tampered")
        target = root / "src/module.py"
        target.chmod(0o644)
        target.write_bytes(b"VALUE = 2\n")
        target.chmod(0o444)
        with self.assertRaisesRegex(ProvenanceError, "checksum mismatch"):
            verify_snapshot(
                snapshot_root=root,
                allowed_root=self.allowed_root,
                expected_manifest_sha256=manifest_sha256,
            )

        root, manifest_sha256 = self.make_snapshot("extra")
        root.chmod(0o755)
        _write(root / "extra.txt", "unexpected")
        (root / "extra.txt").chmod(0o444)
        root.chmod(0o555)
        with self.assertRaisesRegex(ProvenanceError, "file set mismatch"):
            verify_snapshot(
                snapshot_root=root,
                allowed_root=self.allowed_root,
                expected_manifest_sha256=manifest_sha256,
            )

    def test_rejects_symlink_special_file_and_path_escape(self) -> None:
        root, manifest_sha256 = self.make_snapshot("symlink")
        root.chmod(0o755)
        (root / "bad-link").symlink_to("src/module.py")
        root.chmod(0o555)
        with self.assertRaisesRegex(ProvenanceError, "symlink"):
            verify_snapshot(
                snapshot_root=root,
                allowed_root=self.allowed_root,
                expected_manifest_sha256=manifest_sha256,
            )

        root, manifest_sha256 = self.make_snapshot("fifo")
        root.chmod(0o755)
        os.mkfifo(root / "bad-fifo")
        root.chmod(0o555)
        with self.assertRaisesRegex(ProvenanceError, "special file"):
            verify_snapshot(
                snapshot_root=root,
                allowed_root=self.allowed_root,
                expected_manifest_sha256=manifest_sha256,
            )

        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        with self.assertRaisesRegex(ProvenanceError, "escapes allowed root"):
            verify_snapshot(
                snapshot_root=outside,
                allowed_root=self.allowed_root,
                expected_manifest_sha256=manifest_sha256,
            )


class ArtifactAndFormalContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base_checkpoint = self.root / "checkpoint"
        _write_json(self.base_checkpoint / "checkpoint.json", {"step": 0})
        _write(self.base_checkpoint / "model/.metadata", b"metadata")
        for index in range(6):
            _write(self.base_checkpoint / f"model/shard-{index}.distcp", bytes([index]) * 40)

        self.wan_vae = self.root / "Wan2.2_VAE.pth"
        _write(self.wan_vae, b"vae" * 30)

        self.qwen = self.root / "qwen"
        for index, relative in enumerate(_QWEN_PROCESSOR_FILES):
            _write(self.qwen / relative, f"qwen-{index}\n")

        self.environment = self.root / "environment"
        _write_json(self.environment / ".cwam_env_ready.json", {"status": "ready"})
        _write(self.environment / "pyvenv.cfg", "home = /usr/bin\n")
        _write(self.environment / "bin/python", b"python-launcher")
        site_packages = self.environment / "lib/python3.10/site-packages"
        _write(site_packages / "test.dist-info/METADATA", "Name: test\n")
        _write(site_packages / "test.dist-info/RECORD", "test\n")
        critical_files = (
            "nvidia/cuda_runtime/lib/libcudart.so.12",
            "torch/_Ctest.so",
            "torch/lib/libc10.so",
            "torch/lib/libc10_cuda.so",
            "torch/lib/libtorch_cpu.so",
            "torch/lib/libtorch_cuda.so",
            "torch/lib/libtorch_python.so",
            "torchcodec/libtorchcodec_core8.so",
            "av/_core_test.so",
            "pydantic_core/_pydantic_core_test.so",
        )
        for relative in critical_files:
            _write(site_packages / relative, f"native:{relative}\n")
        os.symlink("cuda_runtime", site_packages / "nvidia/cudart")

        self.source_root = self.root / "raw"
        _write(self.source_root / "videos/episode-000.mp4", b"a" * 80)
        _write(self.source_root / "videos/episode-001.mp4", b"short")

        self.dataset_root = self.root / "processed"
        dataset_payloads = {
            "data/actions.npy": b"actions" * 12,
            "splits/train.parquet": b"train-rows",
        }
        for relative, payload in dataset_payloads.items():
            _write(self.dataset_root / relative, payload)
        dataset_id = "fixture-dataset-v1"
        dataset_manifest = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "inputs": {"source_root": str(self.source_root.resolve())},
            "files": {
                relative: {"bytes": len(payload), "sha256": _sha256(payload)}
                for relative, payload in sorted(dataset_payloads.items())
            },
        }
        _write_json(self.dataset_root / "dataset_manifest.json", dataset_manifest)
        dataset_manifest_sha256 = sha256_file(self.dataset_root / "dataset_manifest.json")
        _write_json(
            self.dataset_root / "_SUCCESS.json",
            {
                "status": "complete",
                "dataset_id": dataset_id,
                "dataset_manifest": "dataset_manifest.json",
                "dataset_manifest_sha256": dataset_manifest_sha256,
            },
        )

        self.inventory_path = self.root / "artifact-inventory.json"
        inventory = create_artifact_inventory(
            base_checkpoint=self.base_checkpoint,
            wan_vae=self.wan_vae,
            qwen_tokenizer=self.qwen,
            environment=self.environment,
            dataset_root=self.dataset_root,
            source_root=self.source_root,
            large_file_threshold=16,
            sample_bytes=4,
        )
        self.inventory_sha256 = write_inventory(inventory, self.inventory_path)

    def verify_inventory(self) -> dict[str, object]:
        return verify_artifact_inventory(
            inventory_path=self.inventory_path,
            expected_inventory_sha256=self.inventory_sha256,
            base_checkpoint=self.base_checkpoint,
            wan_vae=self.wan_vae,
            qwen_tokenizer=self.qwen,
            environment=self.environment,
            dataset_root=self.dataset_root,
            source_root=self.source_root,
        )

    def test_inventory_accepts_exact_artifacts_and_records_sampling_boundary(self) -> None:
        report = self.verify_inventory()
        self.assertEqual(report["status"], "passed")
        self.assertGreater(report["verification_counts"]["size_and_sample_sha256"], 0)
        self.assertIn("unsampled bytes", report["runtime_large_file_boundary"])

    def test_processed_payload_is_always_fully_verified(self) -> None:
        target = self.dataset_root / "data/actions.npy"
        payload = bytearray(target.read_bytes())
        payload[len(payload) // 2] ^= 1
        target.write_bytes(payload)
        with self.assertRaisesRegex(ProvenanceError, "checksum mismatch"):
            self.verify_inventory()

    def test_raw_sample_tamper_and_exact_tree_addition_are_rejected(self) -> None:
        target = self.source_root / "videos/episode-000.mp4"
        original = target.read_bytes()
        payload = bytearray(original)
        payload[0] ^= 1
        target.write_bytes(payload)
        with self.assertRaisesRegex(ProvenanceError, "sampled checksum mismatch"):
            self.verify_inventory()

        target.write_bytes(original)
        _write(self.source_root / "videos/untracked.mp4", b"extra")
        with self.assertRaisesRegex(ProvenanceError, "file set mismatch"):
            self.verify_inventory()

    def test_processed_dataset_extra_file_is_rejected(self) -> None:
        _write(self.dataset_root / "unexpected.bin", b"extra")
        with self.assertRaisesRegex(ProvenanceError, "file set mismatch"):
            self.verify_inventory()

    def _make_snapshot_and_formal_manifest(
        self,
        *,
        status_value: str,
        world_size: int = 8,
        experiment_id: str | None = None,
    ) -> dict[str, object]:
        protocol = copy.deepcopy(_PROTOCOLS[world_size])
        if experiment_id is None:
            experiment_id = f"{protocol['experiment_family']}-TEST"
        allowed_root = self.root / "run_snapshots"
        snapshot_root = allowed_root / f"formal-{status_value}-{world_size}"
        config_payload = b"[job]\nexperiment = 'fixture'\n"
        _write(snapshot_root / _EXPECTED_CONFIG_PATH, config_payload)
        snapshot_manifest = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "source_dirty": False,
            "source_head": _COMMIT,
            "workflow_snapshot_sha256": _WORKFLOW_SHA256,
            "file_count": 1,
            "files": [
                {
                    "path": _EXPECTED_CONFIG_PATH,
                    "bytes": len(config_payload),
                    "sha256": _sha256(config_payload),
                }
            ],
        }
        _write_json(snapshot_root / "SNAPSHOT_MANIFEST.json", snapshot_manifest)
        snapshot_manifest_sha256 = sha256_file(snapshot_root / "SNAPSHOT_MANIFEST.json")
        _make_read_only_tree(snapshot_root)

        dataset_manifest_sha256 = sha256_file(self.dataset_root / "dataset_manifest.json")
        success_sha256 = sha256_file(self.dataset_root / "_SUCCESS.json")
        formal = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "status": status_value,
            "mode": "confirmatory",
            "protocol": protocol,
            "storage": copy.deepcopy(_STORAGE),
            "rjob": copy.deepcopy(_RJOBS[world_size]),
            "git": {
                "commit": _COMMIT,
                "dirty": False,
                "pushed": True,
                "snapshot_sha256": _WORKFLOW_SHA256,
                "source_manifest_sha256": snapshot_manifest_sha256,
            },
            "config": {"path": _EXPECTED_CONFIG_PATH, "sha256": _sha256(config_payload)},
            "data": {
                "root": str(self.dataset_root.resolve()),
                "source_root": str(self.source_root.resolve()),
                "version": "fixture-dataset-v1",
                "revision": f"manifest-sha256:{dataset_manifest_sha256}",
                "manifest_sha256": dataset_manifest_sha256,
                "success_sha256": success_sha256,
            },
            "environment": {"container_image": _IMAGE},
            "artifacts": {
                "inventory_path": str(self.inventory_path.resolve()),
                "inventory_sha256": self.inventory_sha256,
                "paths": {
                    "base_checkpoint": str(self.base_checkpoint.resolve()),
                    "wan_vae": str(self.wan_vae.resolve()),
                    "qwen_tokenizer": str(self.qwen.resolve()),
                    "environment": str(self.environment.resolve()),
                    "processed_dataset": str(self.dataset_root.resolve()),
                    "raw_video_source": str(self.source_root.resolve()),
                },
            },
            "command": "bash examples/h_cluster/run_cwam_cosmos3_nano_production_then_formal.sh",
            "hardware": f"single-node {world_size}x H200; 128 samples/rank",
        }
        formal_path = self.root / f"formal-{status_value}-{world_size}.json"
        _write_json(formal_path, formal)
        return {
            "experiment_id": experiment_id,
            "world_size": world_size,
            "snapshot_root": snapshot_root,
            "snapshot_manifest_sha256": snapshot_manifest_sha256,
            "formal_manifest_path": formal_path,
            "formal_manifest_sha256": sha256_file(formal_path),
        }

    def _verify_formal(
        self,
        fixture: dict[str, object],
        *,
        expected_experiment_mode: str = "confirmatory",
    ) -> dict[str, object]:
        return verify_formal_contract(
            snapshot_root=fixture["snapshot_root"],
            snapshot_manifest_sha256=fixture["snapshot_manifest_sha256"],
            formal_manifest_path=fixture["formal_manifest_path"],
            formal_manifest_sha256=fixture["formal_manifest_sha256"],
            experiment_id=fixture["experiment_id"],
            expected_commit=_COMMIT,
            expected_workflow_snapshot_sha256=_WORKFLOW_SHA256,
            nproc_per_node=fixture["world_size"],
            expected_container_image=_IMAGE,
            dataset_root=self.dataset_root,
            source_root=self.source_root,
            base_checkpoint=self.base_checkpoint,
            wan_vae=self.wan_vae,
            qwen_tokenizer=self.qwen,
            environment=self.environment,
            expected_experiment_mode=expected_experiment_mode,
        )

    def test_formal_contract_accepts_only_planned_or_queued(self) -> None:
        for world_size in (8, 6):
            for status_value in ("planned", "queued"):
                with self.subTest(world_size=world_size, status=status_value):
                    fixture = self._make_snapshot_and_formal_manifest(
                        status_value=status_value,
                        world_size=world_size,
                    )
                    report = self._verify_formal(fixture)
                    self.assertEqual(report["status"], "passed")
                    self.assertEqual(report["protocol"], _PROTOCOLS[world_size])

        fixture = self._make_snapshot_and_formal_manifest(status_value="running")
        with self.assertRaisesRegex(ProvenanceError, "not launchable"):
            self._verify_formal(fixture)

    def test_exploratory_mode_allows_unpushed_but_never_dirty_commit(
        self,
    ) -> None:
        fixture = self._make_snapshot_and_formal_manifest(status_value="planned")
        formal_path = fixture["formal_manifest_path"]
        formal = json.loads(formal_path.read_text())
        formal["mode"] = "exploratory"
        formal["git"]["pushed"] = False
        _write_json(formal_path, formal)
        fixture["formal_manifest_sha256"] = sha256_file(formal_path)

        report = self._verify_formal(
            fixture,
            expected_experiment_mode="exploratory",
        )
        self.assertEqual(report["experiment_mode"], "exploratory")
        self.assertIs(report["source_commit_pushed"], False)

        with self.assertRaisesRegex(ProvenanceError, "requested mode"):
            self._verify_formal(fixture)

        formal["git"]["dirty"] = True
        _write_json(formal_path, formal)
        fixture["formal_manifest_sha256"] = sha256_file(formal_path)
        with self.assertRaisesRegex(ProvenanceError, "clean commit"):
            self._verify_formal(
                fixture,
                expected_experiment_mode="exploratory",
            )

    def test_confirmatory_mode_still_requires_pushed_commit(self) -> None:
        fixture = self._make_snapshot_and_formal_manifest(status_value="planned")
        formal_path = fixture["formal_manifest_path"]
        formal = json.loads(formal_path.read_text())
        formal["git"]["pushed"] = False
        _write_json(formal_path, formal)
        fixture["formal_manifest_sha256"] = sha256_file(formal_path)

        with self.assertRaisesRegex(
            ProvenanceError,
            "confirmatory.*clean pushed commit",
        ):
            self._verify_formal(fixture)

    def test_formal_protocol_rejects_every_wrong_scalar(self) -> None:
        fixture = self._make_snapshot_and_formal_manifest(status_value="planned")
        formal_path = fixture["formal_manifest_path"]
        original = json.loads(formal_path.read_text())
        wrong_values = {
            "experiment_family": "CWAM-E35A",
            "world_size": 6,
            "per_rank_samples": 127,
            "grad_accum_iter": 1,
            "global_batch": 2_048.0,
            "learning_rate": 4.0e-5,
            "max_iter": 10_001,
            "warm_up_steps": 501,
            "cycle_length": 16_001,
            "validation_iter": 501,
            "max_val_iter": 9,
            "save_iter": 501,
            "train_sample_budget": 20_480_001,
        }
        self.assertEqual(set(wrong_values), set(original["protocol"]))
        for field, bad_value in wrong_values.items():
            with self.subTest(field=field):
                formal = copy.deepcopy(original)
                formal["protocol"][field] = bad_value
                _write_json(formal_path, formal)
                fixture["formal_manifest_sha256"] = sha256_file(formal_path)
                with self.assertRaisesRegex(ProvenanceError, rf"protocol {field} mismatch"):
                    self._verify_formal(fixture)

    def test_formal_protocol_rejects_key_drift_and_wrong_family_prefix(self) -> None:
        fixture = self._make_snapshot_and_formal_manifest(status_value="planned")
        formal_path = fixture["formal_manifest_path"]
        original = json.loads(formal_path.read_text())
        for mutation in ("missing", "extra"):
            with self.subTest(mutation=mutation):
                formal = copy.deepcopy(original)
                if mutation == "missing":
                    formal["protocol"].pop("train_sample_budget")
                else:
                    formal["protocol"]["unexpected"] = 1
                _write_json(formal_path, formal)
                fixture["formal_manifest_sha256"] = sha256_file(formal_path)
                with self.assertRaisesRegex(ProvenanceError, "protocol key mismatch"):
                    self._verify_formal(fixture)

        for world_size, wrong_experiment_id, expected_family in (
            (8, "CWAM-E35A-WRONG-TOPOLOGY", "CWAM-E35"),
            (6, "CWAM-E35-WRONG-TOPOLOGY", "CWAM-E35A"),
        ):
            with self.subTest(world_size=world_size, wrong_experiment_id=wrong_experiment_id):
                wrong_family = self._make_snapshot_and_formal_manifest(
                    status_value="queued",
                    world_size=world_size,
                    experiment_id=wrong_experiment_id,
                )
                with self.assertRaisesRegex(
                    ProvenanceError,
                    rf"requires an experiment ID prefixed by {expected_family}-",
                ):
                    self._verify_formal(wrong_family)

    def test_formal_storage_and_rjob_contracts_are_exact(self) -> None:
        fixture = self._make_snapshot_and_formal_manifest(status_value="planned")
        formal_path = fixture["formal_manifest_path"]
        original = json.loads(formal_path.read_text())
        mutations = (
            ("storage", "minimum_free_bytes", 1),
            ("storage", "estimated_required_bytes", 1),
            ("storage", "output_root", "/tmp/formal"),
            ("rjob", "cpu", 127),
            ("rjob", "memory_mib", 1),
            ("rjob", "quota_group", "other"),
            ("rjob", "backoff_limit", 2),
        )
        for section, field, bad_value in mutations:
            with self.subTest(section=section, field=field):
                formal = copy.deepcopy(original)
                formal[section][field] = bad_value
                _write_json(formal_path, formal)
                fixture["formal_manifest_sha256"] = sha256_file(formal_path)
                with self.assertRaisesRegex(ProvenanceError, rf"{section}\.{field} mismatch"):
                    self._verify_formal(fixture)

        for section in ("storage", "rjob"):
            with self.subTest(section=section):
                formal = copy.deepcopy(original)
                formal[section]["unexpected"] = True
                _write_json(formal_path, formal)
                fixture["formal_manifest_sha256"] = sha256_file(formal_path)
                with self.assertRaisesRegex(ProvenanceError, rf"{section} key mismatch"):
                    self._verify_formal(fixture)

    def test_formal_rjob_cpu_override_is_limited_to_eight_rank_continuations(self) -> None:
        self.assertEqual(_formal_rjob_contract(8)["cpu"], 128)
        self.assertEqual(_formal_rjob_contract(8, cpu_override=160)["cpu"], 160)
        self.assertEqual(_formal_rjob_contract(6)["cpu"], 96)
        for world_size, cpu in ((8, 96), (8, 192), (6, 128), (6, 160)):
            with self.subTest(world_size=world_size, cpu=cpu):
                with self.assertRaisesRegex(ProvenanceError, "does not allow rjob cpu"):
                    _formal_rjob_contract(world_size, cpu_override=cpu)

    def test_formal_contract_accepts_selected_160_cpu_eight_rank_job(self) -> None:
        fixture = self._make_snapshot_and_formal_manifest(status_value="planned")
        formal_path = fixture["formal_manifest_path"]
        formal = json.loads(formal_path.read_text())
        formal["rjob"]["cpu"] = 160
        formal["command"] = "bash examples/h_cluster/run_cwam_cosmos3_nano_formal_continuation.sh"
        _write_json(formal_path, formal)
        fixture["formal_manifest_sha256"] = sha256_file(formal_path)
        report = verify_formal_contract(
            snapshot_root=fixture["snapshot_root"],
            snapshot_manifest_sha256=fixture["snapshot_manifest_sha256"],
            formal_manifest_path=formal_path,
            formal_manifest_sha256=fixture["formal_manifest_sha256"],
            experiment_id=fixture["experiment_id"],
            expected_commit=_COMMIT,
            expected_workflow_snapshot_sha256=_WORKFLOW_SHA256,
            nproc_per_node=8,
            expected_container_image=_IMAGE,
            dataset_root=self.dataset_root,
            source_root=self.source_root,
            base_checkpoint=self.base_checkpoint,
            wan_vae=self.wan_vae,
            qwen_tokenizer=self.qwen,
            environment=self.environment,
            expected_rjob_cpu=160,
            expected_formal_wrapper="run_cwam_cosmos3_nano_formal_continuation.sh",
        )
        self.assertEqual(report["rjob"]["cpu"], 160)

    def test_cli_forwards_selected_cpu_only_to_formal_verification(self) -> None:
        argv = [
            "cwam_formal_provenance.py",
            "verify-formal-contract",
            "--snapshot-root",
            str(self.root),
            "--snapshot-manifest-sha256",
            "a" * 64,
            "--formal-manifest",
            str(self.root / "formal.json"),
            "--formal-manifest-sha256",
            "b" * 64,
            "--experiment-id",
            "CWAM-E35-CLI",
            "--expected-commit",
            "c" * 40,
            "--expected-workflow-snapshot-sha256",
            "d" * 64,
            "--nproc-per-node",
            "8",
            "--expected-container-image",
            _IMAGE,
            "--dataset-root",
            str(self.dataset_root),
            "--source-root",
            str(self.source_root),
            "--base-checkpoint",
            str(self.base_checkpoint),
            "--wan-vae",
            str(self.wan_vae),
            "--qwen-tokenizer",
            str(self.qwen),
            "--environment",
            str(self.environment),
            "--expected-rjob-cpu",
            "160",
            "--expected-formal-wrapper",
            "run_cwam_cosmos3_nano_formal_continuation.sh",
            "--expected-experiment-mode",
            "exploratory",
        ]
        with (
            mock.patch.object(provenance, "verify_formal_contract", return_value={"status": "passed"}) as verify,
            mock.patch.object(provenance, "_write_json_report"),
            mock.patch.object(provenance.sys, "argv", argv),
        ):
            provenance.main()
        self.assertEqual(verify.call_args.kwargs["expected_rjob_cpu"], 160)
        self.assertEqual(
            verify.call_args.kwargs["expected_formal_wrapper"],
            "run_cwam_cosmos3_nano_formal_continuation.sh",
        )
        self.assertEqual(
            verify.call_args.kwargs["expected_experiment_mode"],
            "exploratory",
        )

    def test_create_inventory_cli_remains_independent_of_formal_cpu_option(self) -> None:
        output = self.root / "cli-inventory.json"
        argv = [
            "cwam_formal_provenance.py",
            "create-artifact-inventory",
            "--base-checkpoint",
            str(self.base_checkpoint),
            "--wan-vae",
            str(self.wan_vae),
            "--qwen-tokenizer",
            str(self.qwen),
            "--environment",
            str(self.environment),
            "--dataset-root",
            str(self.dataset_root),
            "--source-root",
            str(self.source_root),
            "--output-json",
            str(output),
        ]
        with (
            mock.patch.object(provenance, "create_artifact_inventory", return_value={"schema_version": 1}) as create,
            mock.patch.object(provenance, "write_inventory", return_value="e" * 64),
            mock.patch.object(provenance.sys, "argv", argv),
        ):
            provenance.main()
        self.assertNotIn("expected_rjob_cpu", create.call_args.kwargs)

    def test_formal_contract_rejects_unbound_data_root(self) -> None:
        fixture = self._make_snapshot_and_formal_manifest(status_value="planned")
        formal_path = fixture["formal_manifest_path"]
        formal = json.loads(formal_path.read_text())
        formal["data"]["root"] = str(self.root / "other")
        _write_json(formal_path, formal)
        fixture["formal_manifest_sha256"] = sha256_file(formal_path)
        with self.assertRaisesRegex(ProvenanceError, "processed dataset root mismatch"):
            self._verify_formal(fixture)


class FormalShellContractTest(unittest.TestCase):
    def test_submit_and_runtime_scripts_preserve_formal_guards(self) -> None:
        source_root = Path(__file__).resolve().parents[2]
        submit = (source_root / "examples/h_cluster/submit_cwam_cosmos3_nano_formal_rjob.sh").read_text()
        runtime = (source_root / "examples/h_cluster/run_cwam_cosmos3_nano_production_then_formal.sh").read_text()

        self.assertIn(_IMAGE, submit)
        self.assertIn("--restart-policy never", submit)
        self.assertIn("--backoff_limit 1", submit)
        self.assertNotIn("--backoff_limit 0", submit)
        self.assertGreaterEqual(submit.count("printf -v "), 10)
        self.assertIn("'%q'", submit)
        self.assertIn('"${#RJOB_NAME}" -le 50', submit)
        self.assertIn('"$FORMAL_RUN_NAME" == "$FORMAL_EXPERIMENT_ID"', submit)
        self.assertIn('"status": "submission_unknown"', submit)
        self.assertIn('rjob get "$SCHEDULER_RJOB_NAME"', submit)
        self.assertNotIn("cleanup_unaccepted_reservation", submit)
        self.assertIn('payload.get("status") != "accepted"', runtime)
        self.assertIn("formal-storage-preflight.json", runtime)
        self.assertIn("--source-root", submit)
        self.assertIn("--source-root", runtime)
        self.assertIn("cwam_secure_paths.py", submit)
        self.assertIn("cwam_secure_paths.py", runtime)
        self.assertLess(
            submit.index("export PYTHONDONTWRITEBYTECODE=1"),
            submit.index('python3 "$COSMOS_SOURCE/examples/h_cluster/cwam_formal_provenance.py"'),
        )
        self.assertLess(
            runtime.index("export PYTHONDONTWRITEBYTECODE=1"),
            runtime.index('python3 "$COSMOS_SOURCE/examples/h_cluster/cwam_formal_provenance.py"'),
        )
        self.assertIn(
            "audited_worker_init_prefixes = [\n"
            "    [],\n"
            "    [\n"
            '        "/kubebrain/worker-init",\n'
            '        "run",\n'
            '        "--enable-pid-monitor",\n'
            '        "--",\n'
            "    ],\n"
            "    [\n"
            '        "/kubebrain/worker-init",\n'
            '        "run",\n'
            '        "--enable-pid-monitor",\n'
            '        "--metrics-addr=",\n'
            '        "--",\n'
            "    ],\n"
            "]",
            runtime,
        )
        self.assertIn("create-dir-exclusive", submit)
        self.assertIn("create-dir-exclusive", runtime)
        self.assertIn("path-security.json", runtime)
        self.assertIn("cwam_secure_tee.py", runtime)
        self.assertGreaterEqual(runtime.count("--file-mode 0444"), 3)
        self.assertIn("capture_cwam_formal_rjob_receipt.py", submit)
        self.assertGreaterEqual(submit.count("timeout --foreground"), 3)
        self.assertIn(
            "formal submit wrapper must execute from the verified COSMOS_SOURCE snapshot",
            submit,
        )
        self.assertIn(
            "formal runtime wrapper must execute from the verified COSMOS_SOURCE snapshot",
            runtime,
        )
        self.assertIn('"scheduler_rjob_name"', runtime)
        self.assertIn('"scheduler_rjob_uid"', runtime)
        self.assertIn("KUBEBRAIN_RJOB_UID", runtime)
        self.assertIn("time.monotonic() + 600", runtime)
        self.assertIn('"scheduler_receipt"', runtime)
        self.assertIn('"requests_equal_core_limits"', runtime)
        self.assertIn('"platform_injected_limit_keys"', runtime)
        self.assertNotIn('"requests_equal_limits"', runtime)
        self.assertIn("FORMAL_COMMAND_SHA256", submit)
        self.assertIn('"command_sha256"', runtime)
        self.assertNotIn('mkdir -p "$EVIDENCE_DIR"', runtime)
        self.assertIn("EXPECTED_SNAPSHOT_MANIFEST_SHA256", submit)
        self.assertIn("EXPECTED_SNAPSHOT_MANIFEST_SHA256", runtime)


if __name__ == "__main__":
    unittest.main()
