import tempfile
import unittest
from pathlib import Path, PurePosixPath

import relay_chunk_transfer as module


class RelayChunkTransferTest(unittest.TestCase):
    def test_manifest_and_chunk_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.tsv"
            manifest.write_text(
                "remote:a\texperiment/a\t10\t/tmp/output-a\n"
                "remote:b\texperiment/b\t8\t/tmp/output-b\n",
                encoding="utf-8",
            )
            objects = module.read_manifest(manifest)
        chunks = module.split_chunks(objects, 4)
        self.assertEqual([(item.offset, item.count) for item in chunks], [(0, 4), (4, 4), (8, 2), (0, 4), (4, 4)])
        self.assertEqual(chunks[2].relay_path, PurePosixPath("experiment/a.parts/000002"))

    def test_manifest_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.tsv"
            manifest.write_text("remote:a\t../a\t10\t/tmp/output\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid"):
                module.read_manifest(manifest)

    def test_parse_head_size_uses_last_response(self):
        payload = (
            b"HTTP/1.1 200 Connection established\r\n\r\n"
            b"HTTP/1.1 200 OK\r\nContent-Length: 123\r\nAccept-Ranges: bytes\r\n\r\n"
        )
        self.assertEqual(module._parse_head_size(payload), (200, 123))

    def test_url_quotes_each_path_component(self):
        self.assertEqual(
            module._url("https://relay.example/", PurePosixPath("a b/c+d")),
            "https://relay.example/a%20b/c%2Bd",
        )


if __name__ == "__main__":
    unittest.main()
