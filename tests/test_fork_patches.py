"""Tests for the fork patches: TELEGRAM_ALLOW_PUBLIC_CHANNELS, TELEGRAM_MAX_DIALOGS,
TELEGRAM_ENV_FILE."""

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pytest
from telethon.tl.types import Channel, Chat, ChatPhotoEmpty, User, Username

import main
from telegram_mcp import runtime
from telegram_mcp.runtime import (
    get_dialogs_capped,
    is_chat_allowed,
    is_chat_allowlist_enabled,
    is_public_broadcast_channel,
    max_dialogs_limit,
    validate_id,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clean_policy_env(monkeypatch):
    for key in (
        "TELEGRAM_ALLOWED_CHAT_IDS",
        "TELEGRAM_ALLOW_PUBLIC_CHANNELS",
        "TELEGRAM_MAX_DIALOGS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(runtime, "ALLOWED_CHAT_IDS", None)
    if hasattr(main, "ALLOWED_CHAT_IDS"):
        monkeypatch.setattr(main, "ALLOWED_CHAT_IDS", None)


def _channel(cid: int, *, broadcast: bool, username=None, usernames=None) -> Channel:
    return Channel(
        id=cid,
        title=f"chat {cid}",
        photo=ChatPhotoEmpty(),
        date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        broadcast=broadcast,
        megagroup=not broadcast,
        username=username,
        usernames=usernames,
        access_hash=1,
    )


def _user(uid: int, username=None) -> User:
    return User(id=uid, username=username, access_hash=1)


def _basic_group(cid: int) -> Chat:
    return Chat(
        id=cid,
        title="group",
        photo=ChatPhotoEmpty(),
        participants_count=2,
        date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        version=1,
    )


# --- TELEGRAM_ALLOW_PUBLIC_CHANNELS -------------------------------------------------------


def test_public_mode_off_keeps_upstream_behavior():
    assert is_chat_allowlist_enabled() is False
    assert is_chat_allowed(123, _user(123)) is True


def test_public_mode_enables_allowlist_without_ids(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOW_PUBLIC_CHANNELS", "1")
    assert is_chat_allowlist_enabled() is True


def test_public_mode_allows_only_public_broadcast_channels(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOW_PUBLIC_CHANNELS", "true")
    public_channel = _channel(10, broadcast=True, username="news")
    collectible_channel = _channel(
        11, broadcast=True, usernames=[Username(username="alias", active=True)]
    )
    private_channel = _channel(12, broadcast=True)
    public_group = _channel(13, broadcast=False, username="chatgroup")

    assert is_chat_allowed(-10010, public_channel) is True
    assert is_chat_allowed(-10011, collectible_channel) is True
    assert is_chat_allowed(-10012, private_channel) is False
    assert is_chat_allowed(-10013, public_group) is False
    assert is_chat_allowed(55, _user(55, username="someone")) is False
    assert is_chat_allowed(-77, _basic_group(77)) is False
    # Without an entity a bare ID cannot prove it is a public channel: fail closed.
    assert is_chat_allowed(-10010) is False


def test_inactive_collectible_username_is_not_public():
    channel = _channel(14, broadcast=True, usernames=[Username(username="old", active=False)])
    assert is_public_broadcast_channel(channel) is False


def test_public_mode_still_honors_explicit_ids(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOW_PUBLIC_CHANNELS", "1")
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "-77, 55")
    assert is_chat_allowed(-77, _basic_group(77)) is True
    assert is_chat_allowed(55) is True
    assert is_chat_allowed(56, _user(56)) is False


@pytest.mark.asyncio
async def test_validate_id_resolves_numeric_ids_in_public_mode(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOW_PUBLIC_CHANNELS", "1")
    entities = {
        -10010: _channel(10, broadcast=True, username="news"),
        -10013: _channel(13, broadcast=False, username="chatgroup"),
        55: _user(55, username="someone"),
    }

    async def fake_resolve(identifier, client):
        return entities[identifier]

    monkeypatch.setattr(runtime, "get_client", lambda account=None: object())
    monkeypatch.setattr(runtime, "resolve_entity", fake_resolve)

    @validate_id("chat_id")
    async def read(chat_id: Union[int, str], account: Optional[str] = None):
        return f"read:{chat_id}"

    assert await read(chat_id=-10010) == "read:-10010"
    for denied in (-10013, 55):
        result = await read(chat_id=denied)
        assert "restricted by privacy policy" in str(result)
        assert "TELEGRAM_ALLOW_PUBLIC_CHANNELS" in str(result)


# --- TELEGRAM_MAX_DIALOGS -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 300),
        ("", 300),
        ("50", 50),
        ("0", None),
        ("all", None),
        ("none", None),
        ("-5", None),
        ("x", 300),
    ],
)
def test_max_dialogs_limit(monkeypatch, raw, expected):
    if raw is not None:
        monkeypatch.setenv("TELEGRAM_MAX_DIALOGS", raw)
    assert max_dialogs_limit() == expected


@pytest.mark.asyncio
async def test_get_dialogs_capped_passes_limit(monkeypatch):
    calls = []

    class FakeClient:
        async def get_dialogs(self, **kwargs):
            calls.append(kwargs)
            return []

    monkeypatch.setenv("TELEGRAM_MAX_DIALOGS", "25")
    await get_dialogs_capped(FakeClient(), archived=False)
    monkeypatch.setenv("TELEGRAM_MAX_DIALOGS", "0")
    await get_dialogs_capped(FakeClient())
    assert calls == [{"limit": 25, "archived": False}, {"limit": None}]


def test_no_uncapped_full_dialog_scans_remain():
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in REPO_ROOT.joinpath("telegram_mcp").rglob("*.py")
        if ".get_dialogs()" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


# --- TELEGRAM_ENV_FILE --------------------------------------------------------------------


def _import_runtime(env: dict, cwd: Path) -> subprocess.CompletedProcess:
    code = (
        "import os; from telegram_mcp import runtime as r; "
        "print(r.TELEGRAM_API_ID, os.environ.get('TELEGRAM_MAX_DIALOGS'))"
    )
    return subprocess.run(
        [sys.executable, "-c", code], cwd=cwd, env=env, capture_output=True, text=True, timeout=120
    )


def _base_env() -> dict:
    keep = {"PATH", "HOME", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP"}
    env = {k: v for k, v in os.environ.items() if k in keep}
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env


def test_env_file_supplies_settings(tmp_path):
    env_file = tmp_path / "telegram.env"
    env_file.write_text(
        "TELEGRAM_API_ID=4242\nTELEGRAM_API_HASH=abc\nTELEGRAM_SESSION_NAME=:memory:\n"
        "TELEGRAM_MAX_DIALOGS=77\n",
        encoding="utf-8",
    )
    env = _base_env() | {"TELEGRAM_ENV_FILE": str(env_file)}
    result = _import_runtime(env, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-2:] == ["4242", "77"]


def test_process_env_wins_over_env_file(tmp_path):
    env_file = tmp_path / "telegram.env"
    env_file.write_text(
        "TELEGRAM_API_ID=4242\nTELEGRAM_API_HASH=abc\nTELEGRAM_SESSION_NAME=:memory:\n",
        encoding="utf-8",
    )
    env = _base_env() | {"TELEGRAM_ENV_FILE": str(env_file), "TELEGRAM_API_ID": "1"}
    result = _import_runtime(env, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-2] == "1"


def test_missing_env_file_fails_loudly(tmp_path):
    env = _base_env() | {"TELEGRAM_ENV_FILE": str(tmp_path / "missing.env")}
    result = _import_runtime(env, tmp_path)
    assert result.returncode == 1
    assert "TELEGRAM_ENV_FILE does not exist" in result.stderr
