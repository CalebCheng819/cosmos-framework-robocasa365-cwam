#!/usr/bin/env python3
"""Short-lived authenticated HTTPS relay for large experiment artifacts."""

from __future__ import annotations

import argparse
import hmac
import os
import ssl
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit


COPY_CHUNK_BYTES = 8 * 1024 * 1024


class RelayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "CWAMTransferRelay/1"

    @property
    def relay(self) -> "RelayServer":
        return self.server  # type: ignore[return-value]

    def _authorized(self) -> bool:
        expected = f"Bearer {self.relay.token}"
        provided = self.headers.get("Authorization", "")
        return hmac.compare_digest(provided, expected)

    def _reject_unauthorized(self) -> None:
        self.send_response(HTTPStatus.UNAUTHORIZED)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def _relative_path(self) -> PurePosixPath | None:
        raw = unquote(urlsplit(self.path).path)
        relative = PurePosixPath(raw.lstrip("/"))
        if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            return None
        return relative

    def _target(self) -> Path | None:
        relative = self._relative_path()
        if relative is None:
            return None
        target = self.relay.root.joinpath(*relative.parts)
        try:
            target.resolve().relative_to(self.relay.root)
        except ValueError:
            return None
        return target

    def _empty_response(self, status: HTTPStatus) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _prepare(self) -> Path | None:
        if not self._authorized():
            self._reject_unauthorized()
            return None
        target = self._target()
        if target is None:
            self._empty_response(HTTPStatus.BAD_REQUEST)
            return None
        return target

    def do_PUT(self) -> None:  # noqa: N802
        target = self._prepare()
        if target is None:
            return
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length) if raw_length is not None else -1
        except ValueError:
            length = -1
        if length < 0:
            self._empty_response(HTTPStatus.LENGTH_REQUIRED)
            return

        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{target.name}.uploading-{os.getpid()}-{self.thread_id}")
        remaining = length
        try:
            with partial.open("wb") as output:
                while remaining:
                    block = self.rfile.read(min(COPY_CHUNK_BYTES, remaining))
                    if not block:
                        raise ConnectionError("request body ended before Content-Length")
                    output.write(block)
                    remaining -= len(block)
                output.flush()
                os.fsync(output.fileno())
            os.replace(partial, target)
        except Exception:
            partial.unlink(missing_ok=True)
            raise

        self.send_response(HTTPStatus.CREATED)
        self.send_header("Content-Length", "0")
        self.end_headers()

    @property
    def thread_id(self) -> int:
        import threading

        return threading.get_ident()

    def do_HEAD(self) -> None:  # noqa: N802
        self._send_file(include_body=False)

    def do_GET(self) -> None:  # noqa: N802
        self._send_file(include_body=True)

    def _send_file(self, *, include_body: bool) -> None:
        target = self._prepare()
        if target is None:
            return
        if not target.is_file():
            self._empty_response(HTTPStatus.NOT_FOUND)
            return

        size = target.stat().st_size
        start = 0
        end = size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            try:
                unit, value = range_header.split("=", 1)
                first, last = value.split("-", 1)
                if unit != "bytes" or not first or "," in value:
                    raise ValueError
                start = int(first)
                end = int(last) if last else size - 1
                if start < 0 or end < start or end >= size:
                    raise ValueError
            except ValueError:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT

        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not include_body:
            return

        with target.open("rb") as source:
            source.seek(start)
            remaining = length
            while remaining:
                block = source.read(min(COPY_CHUNK_BYTES, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)

    def log_message(self, format_string: str, *args: object) -> None:
        client = self.client_address[0]
        print(f"{self.log_date_time_string()} client={client} {format_string % args}", flush=True)


class RelayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], *, root: Path, token: str):
        super().__init__(address, RelayHandler)
        self.root = root.resolve()
        self.token = token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cert", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("CWAM_RELAY_TOKEN", "")
    if len(token) < 32:
        raise SystemExit("CWAM_RELAY_TOKEN must contain at least 32 characters")
    args.root.mkdir(parents=True, exist_ok=True)
    server = RelayServer((args.bind, args.port), root=args.root, token=token)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(args.cert, args.key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    print(f"relay_ready bind={args.bind}:{args.port} root={args.root.resolve()}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
