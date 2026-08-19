#!/usr/bin/env python3
"""Extract and overlay a DCP parameter subtree without loading the full source.

This utility supports the B5 VOM publication path: extract the trained
``strong_action_expert`` EMA subtree, then replace the same subtree in the
already-published B4 serving checkpoint. Validation uses DCP metadata and an
exact tensor readback; it intentionally does not compute file digests.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pickle
import shutil
import stat
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata


class OverlayError(RuntimeError):
    """Raised when a checkpoint fails a publication gate."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _require_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    metadata = resolved.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise OverlayError(f"{label} must be a non-symlink directory: {resolved}")
    return resolved


def _new_path(path: Path, label: str) -> Path:
    lexical = Path(os.path.abspath(path.expanduser()))
    parent = _require_directory(lexical.parent, f"{label} parent")
    result = parent / lexical.name
    if result.exists() or result.is_symlink():
        raise OverlayError(f"{label} already exists: {result}")
    return result


def _load_metadata(model: Path, label: str) -> Metadata:
    path = model / ".metadata"
    try:
        with path.open("rb") as stream:
            value = pickle.load(stream)  # noqa: S301 - trusted research checkpoint.
    except Exception as error:
        raise OverlayError(f"cannot read {label} DCP metadata {path}: {error}") from error
    if not isinstance(value, Metadata):
        raise OverlayError(f"unexpected {label} metadata type: {type(value)!r}")
    return value


def _tensor_specs(metadata: Metadata, label: str) -> dict[str, TensorStorageMetadata]:
    result: dict[str, TensorStorageMetadata] = {}
    for key, value in metadata.state_dict_metadata.items():
        if not isinstance(value, TensorStorageMetadata):
            raise OverlayError(f"{label} contains a non-tensor entry: {key}")
        result[key] = value
    if not result:
        raise OverlayError(f"{label} has no tensors")
    return result


def _logical_bytes(specs: dict[str, TensorStorageMetadata]) -> int:
    return sum(int(value.size.numel()) * value.properties.dtype.itemsize for value in specs.values())


def _allocate(specs: dict[str, TensorStorageMetadata]) -> dict[str, torch.Tensor]:
    return {
        key: torch.empty(tuple(value.size), dtype=value.properties.dtype, device="cpu")
        for key, value in specs.items()
    }


def _validate_bf16(specs: dict[str, TensorStorageMetadata], label: str) -> None:
    wrong = sorted(key for key, value in specs.items() if value.properties.dtype != torch.bfloat16)
    if wrong:
        raise OverlayError(f"{label} contains non-BF16 tensors: {wrong[:10]}")


def _validate_expected(actual: int, expected: int | None, label: str) -> None:
    if expected is not None and actual != expected:
        raise OverlayError(f"{label} mismatch: expected={expected}, actual={actual}")


def _validate_contract(
    specs: dict[str, TensorStorageMetadata], contract_model: Path | None
) -> None:
    if contract_model is None:
        return
    contract = _tensor_specs(_load_metadata(contract_model, "contract"), "contract")
    if set(specs) != set(contract):
        missing = sorted(set(contract) - set(specs))
        extra = sorted(set(specs) - set(contract))
        raise OverlayError(f"contract key mismatch: missing={missing[:10]}, extra={extra[:10]}")
    for key, value in specs.items():
        expected = contract[key]
        if value.size != expected.size or value.properties.dtype != expected.properties.dtype:
            raise OverlayError(
                f"contract tensor mismatch for {key}: "
                f"actual=({tuple(value.size)}, {value.properties.dtype}), "
                f"expected=({tuple(expected.size)}, {expected.properties.dtype})"
            )


def _inventory(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        records.append({"relative_path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size})
    return records


def _write_json(path: Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")


def _seal(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _save_and_verify(
    state: dict[str, torch.Tensor],
    target: Path,
    receipt: dict[str, Any],
    source_config: Path | None,
    seal: bool,
) -> dict[str, Any]:
    stage = target.with_name(f".{target.name}.incomplete-{uuid.uuid4().hex}")
    stage.mkdir(mode=0o755)
    try:
        model = stage / "model"
        dcp.save(state, checkpoint_id=model)
        output_specs = _tensor_specs(_load_metadata(model, "output"), "output")
        if set(output_specs) != set(state):
            raise OverlayError("saved checkpoint key set differs from the input state")
        readback = _allocate(output_specs)
        dcp.load(readback, checkpoint_id=model)
        for key, tensor in state.items():
            if not torch.equal(tensor, readback[key]):
                raise OverlayError(f"exact tensor readback failed: {key}")
        if source_config is not None:
            source_config = source_config.expanduser().resolve(strict=True)
            shutil.copy2(source_config, stage / "config.yaml")
        receipt.update(
            {
                "status": "succeeded",
                "completed_at": _utc_now(),
                "output_key_count": len(state),
                "output_logical_bytes": sum(tensor.numel() * tensor.element_size() for tensor in state.values()),
                "exact_readback_key_count": len(readback),
                "files": _inventory(stage),
            }
        )
        _write_json(stage / "dcp-prefix-overlay-receipt.json", receipt)
        if seal:
            _seal(stage)
        os.replace(stage, target)
        return receipt
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def extract(args: argparse.Namespace) -> dict[str, Any]:
    source_model = _require_directory(args.source_model, "source model")
    target = _new_path(args.target_artifact, "target artifact")
    source_specs = _tensor_specs(_load_metadata(source_model, "source"), "source")
    selected_specs = {key: value for key, value in source_specs.items() if key.startswith(args.prefix)}
    _validate_expected(len(selected_specs), args.expected_key_count, "selected key count")
    _validate_expected(_logical_bytes(selected_specs), args.expected_logical_bytes, "selected logical bytes")
    _validate_bf16(selected_specs, "selected subtree")
    state = _allocate(selected_specs)
    dcp.load(state, checkpoint_id=source_model)
    return _save_and_verify(
        state,
        target,
        {
            "kind": "cwam_dcp_prefix_extract",
            "started_at": _utc_now(),
            "source_model": str(source_model),
            "selected_prefix": args.prefix,
            "source_revision": args.source_revision,
            "source_run_id": args.source_run_id,
        },
        args.source_config,
        args.seal,
    )


def overlay(args: argparse.Namespace) -> dict[str, Any]:
    base_model = _require_directory(args.base_model, "base model")
    delta_model = _require_directory(args.delta_model, "delta model")
    target = _new_path(args.target_artifact, "target artifact")
    base_specs = _tensor_specs(_load_metadata(base_model, "base"), "base")
    delta_specs = _tensor_specs(_load_metadata(delta_model, "delta"), "delta")
    if any(not key.startswith(args.prefix) for key in delta_specs):
        raise OverlayError("delta checkpoint contains keys outside the replacement prefix")
    kept_specs = {key: value for key, value in base_specs.items() if not key.startswith(args.prefix)}
    if set(kept_specs) & set(delta_specs):
        raise OverlayError("base and delta key sets overlap after prefix removal")
    output_specs = {**kept_specs, **delta_specs}
    _validate_expected(len(base_specs), args.expected_base_key_count, "base key count")
    _validate_expected(len(kept_specs), args.expected_kept_key_count, "kept base key count")
    _validate_expected(len(delta_specs), args.expected_delta_key_count, "delta key count")
    _validate_expected(len(output_specs), args.expected_output_key_count, "output key count")
    _validate_expected(_logical_bytes(output_specs), args.expected_logical_bytes, "output logical bytes")
    _validate_bf16(output_specs, "output checkpoint")
    _validate_contract(output_specs, args.contract_model)

    state = _allocate(kept_specs)
    dcp.load(state, checkpoint_id=base_model)
    delta_state = _allocate(delta_specs)
    dcp.load(delta_state, checkpoint_id=delta_model)
    state.update(delta_state)
    return _save_and_verify(
        state,
        target,
        {
            "kind": "cwam_dcp_prefix_overlay",
            "started_at": _utc_now(),
            "base_model": str(base_model),
            "delta_model": str(delta_model),
            "replacement_prefix": args.prefix,
            "base_key_count": len(base_specs),
            "kept_base_key_count": len(kept_specs),
            "delta_key_count": len(delta_specs),
            "contract_model": str(args.contract_model) if args.contract_model else None,
            "source_revision": args.source_revision,
            "source_run_id": args.source_run_id,
        },
        args.source_config,
        args.seal,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("extract", "overlay"):
        command = subparsers.add_parser(name)
        command.add_argument("--prefix", required=True)
        command.add_argument("--target-artifact", type=Path, required=True)
        command.add_argument("--source-config", type=Path)
        command.add_argument("--source-revision", required=True)
        command.add_argument("--source-run-id", required=True)
        command.add_argument("--expected-logical-bytes", type=int)
        command.add_argument("--seal", action=argparse.BooleanOptionalAction, default=True)
    extract_parser = subparsers.choices["extract"]
    extract_parser.add_argument("--source-model", type=Path, required=True)
    extract_parser.add_argument("--expected-key-count", type=int)
    overlay_parser = subparsers.choices["overlay"]
    overlay_parser.add_argument("--base-model", type=Path, required=True)
    overlay_parser.add_argument("--delta-model", type=Path, required=True)
    overlay_parser.add_argument("--contract-model", type=Path)
    overlay_parser.add_argument("--expected-base-key-count", type=int)
    overlay_parser.add_argument("--expected-kept-key-count", type=int)
    overlay_parser.add_argument("--expected-delta-key-count", type=int)
    overlay_parser.add_argument("--expected-output-key-count", type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = extract(args) if args.command == "extract" else overlay(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
