# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Strict fixed replay banks for routed-C interventions."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


_SCHEMA_VERSION = 2
_INTERVENTIONS = ("shuffle", "cross_episode")


def _trace_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(paths):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported action-relevance trace payload: {path}")
        metadata = payload.get("metadata")
        layers = payload.get("layers")
        if not isinstance(metadata, list) or not isinstance(layers, list) or not layers:
            raise ValueError(f"Trace has no aligned metadata/layers: {path}")
        batch_size = len(metadata)
        layer_tensors: dict[str, torch.Tensor] = {}
        for layer in layers:
            if not isinstance(layer, dict) or not isinstance(layer.get("module"), str):
                raise ValueError(f"Trace layer record is malformed: {path}")
            routed = layer.get("routed_c")
            if not isinstance(routed, torch.Tensor) or routed.ndim != 3 or routed.shape[0] != batch_size:
                raise ValueError(f"Trace routed C is not [B,Q,D]: {path}")
            if layer.get("intervention_mode") != "normal":
                raise ValueError(f"Replay bank source must use normal C: {path}")
            layer_tensors[layer["module"]] = routed.detach().to(dtype=torch.float16, device="cpu")
        for sample_index, item in enumerate(metadata):
            if not isinstance(item, dict) or not isinstance(item.get("evidence"), dict):
                raise ValueError(f"Trace metadata has no evidence object: {path}")
            evidence = item["evidence"]
            episode_key = evidence.get("episode_key")
            task_id = evidence.get("task_id")
            control_step = evidence.get("control_step")
            if not isinstance(episode_key, str) or not episode_key:
                raise ValueError(f"Trace evidence has no episode_key: {path}")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"Trace evidence has no task_id: {path}")
            if type(control_step) is not int or control_step < 0:
                raise ValueError(f"Trace evidence has invalid control_step: {path}")
            records.append(
                {
                    "episode_key": episode_key,
                    "task_id": task_id,
                    "control_step": control_step,
                    "layers": {
                        name: tensor[sample_index : sample_index + 1]
                        for name, tensor in layer_tensors.items()
                    },
                }
            )
    if not records:
        raise ValueError("No action-relevance traces were provided")
    return records


def _assert_compatible(target: dict[str, Any], donor: dict[str, Any], *, condition: str) -> None:
    target_shapes = {name: tuple(value.shape) for name, value in target["layers"].items()}
    donor_shapes = {name: tuple(value.shape) for name, value in donor["layers"].items()}
    if target_shapes != donor_shapes:
        raise ValueError(
            f"{condition} routed-C shape drift: "
            f"target={target['episode_key']}, donor={donor['episode_key']}"
        )


def _cyclic_donors(group: list[dict[str, Any]], *, seed: int) -> dict[str, dict[str, Any]]:
    ordered = sorted(group, key=lambda item: item["episode_key"])
    if len(ordered) < 2:
        raise ValueError("A strict intervention group requires at least two eligible episodes")
    offset = 1 + (seed % (len(ordered) - 1))
    return {
        target["episode_key"]: ordered[(target_index + offset) % len(ordered)]
        for target_index, target in enumerate(ordered)
    }


def build_fixed_intervention_bank(
    trace_paths: Iterable[Path],
    *,
    seed: int,
) -> dict[str, Any]:
    """Build strict same-task shuffle and different-task cross-episode donors."""
    records = _trace_records(trace_paths)
    seen_targets: set[tuple[str, int]] = set()
    for record in records:
        target_key = (record["episode_key"], record["control_step"])
        if target_key in seen_targets:
            raise ValueError(f"Duplicate replay target in traces: {target_key}")
        seen_targets.add(target_key)

    by_task_step: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task_step[(record["task_id"], record["control_step"])].append(record)
        by_step[record["control_step"]].append(record)

    shuffle_donors: dict[tuple[str, int], dict[str, Any]] = {}
    for (task_id, control_step), group in sorted(by_task_step.items()):
        try:
            donors = _cyclic_donors(group, seed=seed)
        except ValueError as exc:
            raise ValueError(
                "shuffle requires at least two episodes for every task/control-step "
                f"group, got task={task_id!r}, step={control_step}"
            ) from exc
        for target in group:
            shuffle_donors[(target["episode_key"], control_step)] = donors[target["episode_key"]]

    cross_donors: dict[tuple[str, int], dict[str, Any]] = {}
    for control_step, group in sorted(by_step.items()):
        ordered = sorted(group, key=lambda item: (item["task_id"], item["episode_key"]))
        for target_index, target in enumerate(ordered):
            eligible = [
                donor
                for donor in ordered
                if donor["episode_key"] != target["episode_key"]
                and donor["task_id"] != target["task_id"]
            ]
            if not eligible:
                raise ValueError(
                    "cross_episode requires a different-task, different-episode donor for "
                    f"target={target['episode_key']!r}, step={control_step}"
                )
            donor = eligible[(target_index + seed) % len(eligible)]
            cross_donors[(target["episode_key"], control_step)] = donor

    entries: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for target in records:
        key = (target["episode_key"], target["control_step"])
        conditions: dict[str, dict[str, Any]] = {}
        for condition, donors in (("shuffle", shuffle_donors), ("cross_episode", cross_donors)):
            donor = donors[key]
            _assert_compatible(target, donor, condition=condition)
            conditions[condition] = {
                "donor_episode_key": donor["episode_key"],
                "donor_task_id": donor["task_id"],
                "layers": donor["layers"],
            }
        entries[target["episode_key"]][target["control_step"]] = {
            "task_id": target["task_id"],
            "conditions": conditions,
        }

    return {
        "schema_version": _SCHEMA_VERSION,
        "pairing": {
            "shuffle": "same_task_same_control_step_different_episode",
            "cross_episode": "different_task_same_control_step_different_episode",
        },
        "seed": int(seed),
        "entries": dict(entries),
        "target_count": len(records),
    }


class FixedInterventionReplayBank:
    """Validated read-only access to exact routed-C intervention donors."""

    def __init__(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict) or payload.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("Unsupported routed-C replay bank schema")
        if payload.get("pairing") != {
            "shuffle": "same_task_same_control_step_different_episode",
            "cross_episode": "different_task_same_control_step_different_episode",
        }:
            raise ValueError("Routed-C replay bank has an unsupported pairing contract")
        entries = payload.get("entries")
        if not isinstance(entries, dict) or not entries:
            raise ValueError("Routed-C replay bank has no entries")
        self.payload = payload

    @classmethod
    def load(cls, path: Path) -> "FixedInterventionReplayBank":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls(payload)

    def routed_c(
        self,
        *,
        condition: str,
        episode_key: str,
        task_id: str,
        control_step: int,
    ) -> tuple[dict[str, torch.Tensor], str, str]:
        if condition not in _INTERVENTIONS:
            raise ValueError(f"Replay condition must be one of {_INTERVENTIONS}, got {condition!r}")
        try:
            target = self.payload["entries"][episode_key][control_step]
        except KeyError as exc:
            raise KeyError(
                f"Replay bank has no exact target episode={episode_key!r}, step={control_step}"
            ) from exc
        if target.get("task_id") != task_id:
            raise ValueError(
                "Replay bank task drift: "
                f"target={episode_key!r}, expected={target.get('task_id')!r}, actual={task_id!r}"
            )
        item = target.get("conditions", {}).get(condition)
        if not isinstance(item, dict):
            raise ValueError(f"Replay bank target has no {condition} donor")
        donor_episode_key = item.get("donor_episode_key")
        donor_task_id = item.get("donor_task_id")
        layers = item.get("layers")
        if not isinstance(donor_episode_key, str) or donor_episode_key == episode_key:
            raise ValueError("Replay bank donor must name a different episode")
        if condition == "shuffle" and donor_task_id != task_id:
            raise ValueError("shuffle donor must have the same task")
        if condition == "cross_episode" and donor_task_id == task_id:
            raise ValueError("cross_episode donor must have a different task")
        if not isinstance(layers, dict) or not layers:
            raise ValueError("Replay bank target has no layer tensors")
        if not all(isinstance(name, str) and isinstance(value, torch.Tensor) for name, value in layers.items()):
            raise ValueError("Replay bank layer payload is malformed")
        return layers, donor_episode_key, donor_task_id
