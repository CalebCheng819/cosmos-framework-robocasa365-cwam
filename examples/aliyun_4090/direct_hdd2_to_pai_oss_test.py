"""Static contract tests for the direct H-to-PAI OSS transfer helper."""

from pathlib import Path


SCRIPT = Path(__file__).with_name("direct_hdd2_to_pai_oss.sh")


def test_temporary_remote_does_not_require_bucket_management_permissions() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert '"no_check_bucket": "true"' in script
    assert '"session_token": credential["SecurityToken"]' in script
