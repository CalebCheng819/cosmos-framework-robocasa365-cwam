#!/usr/bin/env python3
"""Run the PAI DLC client with an active CredentialsURI profile.

The short-lived credential is written only to a mode-0600 file in tmpfs and
removed after the DLC client exits. Credential values are never logged.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

PROFILE = Path("/root/.aliyun/config.json")


def _active_credentials_uri() -> str:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    current = profile.get("current")
    entry = next(
        (item for item in profile.get("profiles", []) if item.get("name") == current),
        None,
    )
    if not entry or entry.get("mode") != "CredentialsURI":
        raise SystemExit("active Alibaba Cloud profile is not CredentialsURI")
    uri = entry.get("credentials_uri")
    if not uri:
        raise SystemExit("active Alibaba Cloud profile has no credentials_uri")
    return str(uri)


def _fetch_credential(uri: str) -> dict[str, str]:
    request = urllib.request.Request(uri, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.load(response)
    fields = {
        "AccessKeyId": payload.get("AccessKeyId", ""),
        "AccessKeySecret": payload.get("AccessKeySecret", ""),
        "SecurityToken": payload.get("SecurityToken", ""),
    }
    missing = [name for name, value in fields.items() if not value]
    if missing:
        raise SystemExit(f"CredentialsURI response is missing fields: {missing}")
    return fields


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: dlc_credentials_uri_exec.py <dlc command...>")
    command = sys.argv[1:]
    forbidden = {"-a", "--access_id", "-k", "--access_key", "-S", "--security_token"}
    if any(argument.split("=", 1)[0] in forbidden for argument in command):
        raise SystemExit("credential-bearing DLC flags are forbidden")

    credential = _fetch_credential(_active_credentials_uri())
    lines = [
        'endpoint = "pai-dlc.cn-beijing.aliyuncs.com"',
        'region = "cn-beijing"',
        'protocol = "https"',
        "",
        "[user]",
        f"access_id = {json.dumps(credential['AccessKeyId'])}",
        f"access_key = {json.dumps(credential['AccessKeySecret'])}",
        f"security_token = {json.dumps(credential['SecurityToken'])}",
        "",
    ]

    config_path: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(prefix="dlc-credentials-uri-", suffix=".toml", dir="/dev/shm")
        config_path = Path(raw_path)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
            handle.flush()
            os.fsync(handle.fileno())
        completed = subprocess.run([command[0], "-c", str(config_path), *command[1:]], check=False)
        return completed.returncode
    finally:
        if config_path is not None:
            config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
