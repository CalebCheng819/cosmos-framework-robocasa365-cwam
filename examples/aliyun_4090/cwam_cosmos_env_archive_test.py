"""CPU tests for the complete Cosmos environment archive contract."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

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
    marker.write_text(
        json.dumps(
            {
                **env_archive._REQUIRED_READY_MARKER_FIELDS,
                "source_head": "test-source-commit",
                "packages": env_archive._REQUIRED_READY_MARKER_PACKAGES,
            },
            sort_keys=True,
        )
        + "\n"
    )
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


def test_metadata_verification_extracts_without_new_hash_identity(tmp_path: Path) -> None:
    source, archive, receipt, materialization, publisher = _fixture(tmp_path)
    published = env_archive.publish(
        source=source,
        archive=archive,
        receipt=receipt,
        materialization_receipt=materialization,
        publisher_source=publisher,
    )
    receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
    stage = tmp_path / "metadata-stage"
    stage.mkdir()

    verified = env_archive.verify_metadata_and_extract(
        archive=archive,
        receipt=receipt,
        expected_publication_archive=archive,
        expected_archive_bytes=archive.stat().st_size,
        expected_receipt_bytes=receipt.stat().st_size,
        expected_ready_marker_bytes=receipt_value["ready_marker"]["size_bytes"],
        expected_regular_files=published["inventory"]["regular_files"],
        expected_directories=published["inventory"]["directories_including_root"],
        expected_symlinks=published["inventory"]["symlinks"],
        expected_regular_file_bytes=published["inventory"]["regular_file_bytes"],
        expected_source_commit="test-source-commit",
        stage=stage,
    )

    assert verified["verification_mode"] == "metadata_no_hash"
    assert not any("sha" in key for key in verified)
    assert (stage / "bin/tool").read_bytes() == b"complete-environment-tool\n"


def test_metadata_verification_rejects_semantic_receipt_drift(tmp_path: Path) -> None:
    source, archive, receipt, materialization, publisher = _fixture(tmp_path)
    published = env_archive.publish(
        source=source,
        archive=archive,
        receipt=receipt,
        materialization_receipt=materialization,
        publisher_source=publisher,
    )
    receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
    receipt.chmod(0o644)
    content = receipt.read_text(encoding="utf-8").replace(
        '"status": "published"', '"status": "publisheD"'
    )
    receipt.write_text(content, encoding="utf-8")
    receipt.chmod(0o444)
    stage = tmp_path / "metadata-stage"
    stage.mkdir()

    with pytest.raises(env_archive.EnvironmentArchiveError, match="header is invalid"):
        env_archive.verify_metadata_and_extract(
            archive=archive,
            receipt=receipt,
            expected_publication_archive=archive,
            expected_archive_bytes=archive.stat().st_size,
            expected_receipt_bytes=receipt.stat().st_size,
            expected_ready_marker_bytes=receipt_value["ready_marker"]["size_bytes"],
            expected_regular_files=published["inventory"]["regular_files"],
            expected_directories=published["inventory"]["directories_including_root"],
            expected_symlinks=published["inventory"]["symlinks"],
            expected_regular_file_bytes=published["inventory"]["regular_file_bytes"],
            expected_source_commit="test-source-commit",
            stage=stage,
        )


def test_tar_stream_is_deterministic_for_pax_paths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    long_name = "environment-entry-" + "x" * 120
    (source / long_name).write_bytes(b"pax payload\n")

    first = env_archive._tar_stream_sha256(source)
    second = env_archive._tar_stream_sha256(source)

    assert first == second
    assert (
        "--pax-option=exthdr.name=%d/PaxHeaders/%f,delete=atime,delete=ctime"
        in env_archive._tar_command(source, "-")
    )


def test_sha256_tolerates_ctime_only_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "cpfs-publication.tar"
    target.write_bytes(b"immutable archive\n")
    metadata = target.lstat()
    original_lstat = Path.lstat
    calls = 0

    def changing_lstat(path: Path) -> object:
        nonlocal calls
        value = original_lstat(path)
        if path != target:
            return value
        calls += 1
        return SimpleNamespace(
            st_dev=value.st_dev,
            st_ino=value.st_ino,
            st_mode=value.st_mode,
            st_size=value.st_size,
            st_mtime_ns=value.st_mtime_ns,
            st_ctime_ns=metadata.st_ctime_ns + calls,
        )

    monkeypatch.setattr(Path, "lstat", changing_lstat)

    size, digest = env_archive._sha256(target)

    assert size == len(b"immutable archive\n")
    assert digest == hashlib.sha256(b"immutable archive\n").hexdigest()


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
