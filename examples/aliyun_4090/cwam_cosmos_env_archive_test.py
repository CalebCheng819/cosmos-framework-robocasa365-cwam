"""CPU tests for the complete Cosmos environment archive contract."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cwam_cosmos_env_archive as env_archive  # noqa: E402

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    source = tmp_path / "source"
    (source / "bin").mkdir(parents=True, mode=0o755)
    tool = source / "bin/tool"
    tool.write_bytes(b"complete-environment-tool\n")
    tool.chmod(0o555)
    (source / "bin/tool-link").symlink_to("tool")
    marker = source / env_archive.READY_MARKER
    marker.write_text(json.dumps({"schema_version": 1, "status": "ready"}) + "\n")
    marker.chmod(0o444)
    materialization = tmp_path / "materialization.json"
    materialization.write_text('{"status":"ready"}\n')
    materialization.chmod(0o444)
    publisher = tmp_path / "publisher.py"
    publisher.write_bytes(Path(env_archive.__file__).read_bytes())
    publisher.chmod(0o444)
    archive_parent = tmp_path / "archives"
    receipt_parent = tmp_path / "receipts"
    archive_parent.mkdir()
    receipt_parent.mkdir()
    return (
        source,
        archive_parent / "environment.tar",
        receipt_parent / "publication.json",
        materialization,
        publisher,
    )


def test_publish_and_verify_extract_round_trip(tmp_path: Path) -> None:
    source, archive, receipt, materialization, publisher = _fixture(tmp_path)
    published = env_archive.publish(
        source=source,
        archive=archive,
        receipt=receipt,
        materialization_receipt=materialization,
        publisher_source=publisher,
    )
    stage = tmp_path / "stage"
    stage.mkdir()
    verified = env_archive.verify_and_extract(
        archive=archive,
        receipt=receipt,
        expected_archive_sha256=published["archive_sha256"],
        expected_receipt_sha256=published["receipt_sha256"],
        expected_ready_marker_sha256=published["ready_marker_sha256"],
        stage=stage,
        verifier_source=publisher,
    )

    assert archive.stat().st_mode & 0o777 == 0o444
    assert receipt.stat().st_mode & 0o777 == 0o444
    assert verified["archive_sha256"] == verified["extracted_tar_sha256"]
    assert (stage / "bin/tool").read_bytes() == b"complete-environment-tool\n"
    assert (stage / "bin/tool-link").readlink() == Path("tool")


def test_verify_rejects_tampered_archive(tmp_path: Path) -> None:
    source, archive, receipt, materialization, publisher = _fixture(tmp_path)
    published = env_archive.publish(
        source=source,
        archive=archive,
        receipt=receipt,
        materialization_receipt=materialization,
        publisher_source=publisher,
    )
    archive.chmod(0o644)
    with archive.open("ab") as stream:
        stream.write(b"tampered")
    archive.chmod(0o444)
    stage = tmp_path / "stage"
    stage.mkdir()

    with pytest.raises(env_archive.EnvironmentArchiveError, match="binding mismatch"):
        env_archive.verify_and_extract(
            archive=archive,
            receipt=receipt,
            expected_archive_sha256=published["archive_sha256"],
            expected_receipt_sha256=published["receipt_sha256"],
            expected_ready_marker_sha256=published["ready_marker_sha256"],
            stage=stage,
            verifier_source=publisher,
        )


def test_archive_member_validation_rejects_parent_escape(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar"
    payload = b"escape"
    with tarfile.open(archive, "w") as bundle:
        member = tarfile.TarInfo("../escape")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))

    with pytest.raises(env_archive.EnvironmentArchiveError, match="unsafe archive member"):
        env_archive._validate_archive_members(archive)


def test_receipt_hash_binds_publisher_source(tmp_path: Path) -> None:
    source, archive, receipt, materialization, publisher = _fixture(tmp_path)
    published = env_archive.publish(
        source=source,
        archive=archive,
        receipt=receipt,
        materialization_receipt=materialization,
        publisher_source=publisher,
    )
    value = json.loads(receipt.read_text())

    assert value["publisher"]["sha256"] == hashlib.sha256(publisher.read_bytes()).hexdigest()
    assert value["archive"]["sha256"] == published["archive_sha256"]
