from __future__ import annotations

import http.client
import threading
from contextlib import contextmanager
from pathlib import Path

from tools.cwam_transfer_relay import RelayServer


TOKEN = "test-token-that-is-at-least-thirty-two-characters"


@contextmanager
def running_relay(root: Path):
    server = RelayServer(("127.0.0.1", 0), root=root, token=TOKEN)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request(port: int, method: str, path: str, *, body: bytes | None = None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    payload = response.read()
    result = response.status, dict(response.getheaders()), payload
    connection.close()
    return result


def test_authenticated_put_head_get_and_range(tmp_path: Path):
    auth = {"Authorization": f"Bearer {TOKEN}"}
    payload = b"0123456789"
    with running_relay(tmp_path) as port:
        status, _, _ = request(port, "PUT", "/nested/model.bin", body=payload)
        assert status == 401

        status, _, _ = request(port, "PUT", "/nested/model.bin", body=payload, headers=auth)
        assert status == 201
        assert (tmp_path / "nested/model.bin").read_bytes() == payload

        status, headers, body = request(port, "HEAD", "/nested/model.bin", headers=auth)
        assert status == 200
        assert headers["Content-Length"] == str(len(payload))
        assert body == b""

        status, headers, body = request(
            port,
            "GET",
            "/nested/model.bin",
            headers={**auth, "Range": "bytes=3-6"},
        )
        assert status == 206
        assert headers["Content-Range"] == "bytes 3-6/10"
        assert body == b"3456"


def test_rejects_parent_traversal(tmp_path: Path):
    with running_relay(tmp_path) as port:
        status, _, _ = request(
            port,
            "PUT",
            "/%2e%2e/escape.bin",
            body=b"not-written",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
    assert status == 400
    assert not (tmp_path.parent / "escape.bin").exists()
