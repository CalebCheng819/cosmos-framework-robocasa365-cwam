#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Expose several loopback TCP ports that forward to one fixed upstream."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import signal


async def _copy(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(1024 * 1024):
            writer.write(data)
            await writer.drain()
    finally:
        with contextlib.suppress(Exception):
            writer.write_eof()


async def _serve(args: argparse.Namespace) -> None:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        local = writer.get_extra_info("sockname")
        print(json.dumps({"event": "connection", "local": local, "peer": peer}), flush=True)
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(args.upstream_host, args.upstream_port)
            await asyncio.gather(
                _copy(reader, upstream_writer),
                _copy(upstream_reader, writer),
            )
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            if "upstream_writer" in locals():
                upstream_writer.close()
                with contextlib.suppress(Exception):
                    await upstream_writer.wait_closed()

    servers = [
        await asyncio.start_server(handle, host=args.listen_host, port=port, reuse_address=True)
        for port in args.listen_ports
    ]
    print(
        json.dumps(
            {
                "event": "ready",
                "listen_host": args.listen_host,
                "listen_ports": args.listen_ports,
                "upstream_host": args.upstream_host,
                "upstream_port": args.upstream_port,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signal_number, stop.set)
    await stop.wait()
    for server in servers:
        server.close()
    await asyncio.gather(*(server.wait_closed() for server in servers))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", dest="listen_ports", action="append", type=int, required=True)
    parser.add_argument("--upstream-host", required=True)
    parser.add_argument("--upstream-port", type=int, required=True)
    return parser


def main() -> int:
    asyncio.run(_serve(_parser().parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
