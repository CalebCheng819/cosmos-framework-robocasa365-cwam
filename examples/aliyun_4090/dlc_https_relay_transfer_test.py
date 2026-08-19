#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("dlc_https_relay_transfer.py")
SPEC = importlib.util.spec_from_file_location("dlc_https_relay_transfer", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class TransferLayoutTest(unittest.TestCase):
    def test_parts_cover_each_object_exactly(self) -> None:
        parts = module.all_parts({"model/a": 10, "config.yaml": 3}, 4)
        grouped = {name: [] for name in ("model/a", "config.yaml")}
        for part in parts:
            grouped[part.object_name].append(part)
        self.assertEqual([(part.start, part.end) for part in grouped["model/a"]], [(0, 3), (4, 7), (8, 9)])
        self.assertEqual(sum(part.size for part in grouped["model/a"]), 10)
        self.assertEqual(sum(part.size for part in grouped["config.yaml"]), 3)

    def test_merge_object_reuses_exact_parts(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            destination = root / "published"
            parts_root = root / "parts"
            parts = module.all_parts({"model/a": 10}, 4)
            payload = b"0123456789"
            for part in parts:
                path = module.part_path(parts_root, part)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload[part.start : part.end + 1])
            module.merge_object(destination, parts_root, "model/a", len(payload), parts)
            self.assertEqual((destination / "model/a").read_bytes(), payload)

    def test_relay_url_quotes_each_component(self) -> None:
        self.assertEqual(
            module.relay_url("https://example.test/", "checkpoint/model file"),
            "https://example.test/checkpoint/model%20file",
        )


if __name__ == "__main__":
    unittest.main()
