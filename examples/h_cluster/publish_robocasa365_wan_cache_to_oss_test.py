# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from examples.h_cluster import publish_robocasa365_wan_cache_to_oss as publisher


def test_remote_guard_accepts_only_personal_hdd2_prefix() -> None:
    publisher._validate_remote(
        "eailab-hdd2:eailab/chengjuntao/caches/robocasa365-wan22-triple480-fullpad/run-1"
    )
    for unsafe in (
        "eailab-hdd2:eailab/other-user/cache",
        "eailab-hdd2:fkp-migrate/chengjuntao/cache",
        "eailab-hdd2:eailab/chengjuntao/../other-user/cache",
        "eailab-hdd2:eailab/chengjuntao/",
    ):
        with pytest.raises(ValueError):
            publisher._validate_remote(unsafe)


def test_rclone_config_validation_checks_endpoint_without_emitting_credentials(tmp_path: Path) -> None:
    config = tmp_path / "rclone.conf"
    config.write_text(
        "\n".join(
            (
                "[eailab-hdd2]",
                "type = s3",
                "provider = Ceph",
                "endpoint = http://hdd2.h.pjlab.org.cn:8060",
                "acl = private",
                "access_key_id = test-access",
                "secret_access_key = test-secret",
                "",
            )
        )
    )
    config.chmod(0o600)
    publisher._validate_rclone_config(config)

    config.chmod(0o644)
    with pytest.raises(RuntimeError, match="owner-only"):
        publisher._validate_rclone_config(config)


def test_inventory_reuses_validated_shard_digest_and_hashes_metadata(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    shard = cache / "shards" / "shape-48x5x30x52" / "shard-000000.bin"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"latent-shard")
    manifest_path = cache / "manifest.json"
    manifest_path.write_bytes(b"{}\n")
    shard_sha = "a" * 64
    manifest = {
        "shards": [
            {
                "path": shard.relative_to(cache).as_posix(),
                "size_bytes": shard.stat().st_size,
                "sha256": shard_sha,
            }
        ]
    }

    inventory = publisher._build_inventory(cache, manifest)
    by_path = {entry["path"]: entry for entry in inventory["files"]}
    assert inventory["file_count"] == 2
    assert inventory["total_bytes"] == shard.stat().st_size + manifest_path.stat().st_size
    assert by_path[shard.relative_to(cache).as_posix()] == {
        "path": shard.relative_to(cache).as_posix(),
        "size_bytes": shard.stat().st_size,
        "sha256": shard_sha,
        "sha256_source": "validated_cache_manifest",
    }
    assert by_path["manifest.json"]["sha256"] == hashlib.sha256(b"{}\n").hexdigest()
    assert by_path["manifest.json"]["sha256_source"] == "publisher_full_read"


def test_fullpad_execution_scripts_pin_shape_eight_gpus_and_encrypted_handoff() -> None:
    root = Path(__file__).parent
    runner = (root / "build_and_publish_robocasa365_wan_fullpad_oss.sh").read_text()
    submitter = (root / "submit_robocasa365_wan_fullpad_oss_rjob.sh").read_text()
    provision = (root / "provision_robocasa365_oss_rclone_config.sh").read_text()
    generic = (root / "build_robocasa365_wan_latent_cache.sh").read_text()

    assert 'export ROBOCASA365_CACHE_STAGE="full_padding"' in runner
    assert "--gpu 8" in submitter
    assert "--store-host-nvme" in submitter
    assert "--enable-sshd" in submitter
    assert "shape=48x5x30x52" in submitter
    assert 'mkdir -p -- /nvme/cwam-robocasa365-wan-fullpad/$Q_RUN_ID' in submitter
    assert "/usr/bin/openssl cms -help" in submitter
    assert "OSS_CAPACITY_PREFLIGHT" in submitter
    assert "--s3-upload-concurrency" in submitter
    assert '"export PYTHONDONTWRITEBYTECODE=1"' in submitter
    assert '"export PYTHONNOUSERSITE=1"' in submitter
    assert "$Q_RUNTIME_PYTHON -B -c" in submitter
    worker_command = submitter[submitter.index("COMMAND_LINES=(") :]
    assert worker_command.index('"export PYTHONDONTWRITEBYTECODE=1"') < worker_command.index(
        '"$Q_RUNTIME_PYTHON -B -c'
    )
    assert "openssl-cms-enveloped-data-aes-256-cbc-der" in runner
    assert 'BUILD_LOG_FINAL="$WORK_ROOT_REAL/build.log"' in runner
    assert 'PUBLISH_LOG_FINAL="$WORK_ROOT_REAL/publish.log"' in runner
    assert "openssl-cms-enveloped-data-aes-256-cbc-der" in provision
    assert "secret_access_key" not in runner
    assert 'EXPECTED_LATENT_SHAPE="48,5,30,52"' in generic
    assert 'EXPECTED_IMAGE_SIZE="480,832,256,768"' in generic
    assert os.access(root / "build_and_publish_robocasa365_wan_fullpad_oss.sh", os.X_OK)
    assert os.access(root / "submit_robocasa365_wan_fullpad_oss_rjob.sh", os.X_OK)
