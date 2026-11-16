"""Shared pytest setup for import-time Telegram configuration."""

import os

import pytest

os.environ.setdefault("TELEGRAM_API_ID", "12345")
os.environ.setdefault("TELEGRAM_API_HASH", "dummy_hash")
os.environ.setdefault("TELEGRAM_SESSION_NAME", "test_session")


@pytest.fixture
def transcript_cache_dir(tmp_path, monkeypatch):
    """Isolated, per-test SQLite transcript cache directory."""
    d = tmp_path / "transcripts"
    monkeypatch.setenv("TELEGRAM_TRANSCRIPT_CACHE_DIR", str(d))
    return d


@pytest.fixture(autouse=True)
def _clear_expected_username(monkeypatch):
    """Keep a developer's TELEGRAM_EXPECTED_USERNAME* (.env) away from fake clients."""
    for key in list(os.environ):
        if key.startswith("TELEGRAM_EXPECTED_USERNAME"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def isolated_mcp_auth(monkeypatch):
    """Let a test switch OAuth on for the shared FastMCP instance, undone afterwards."""
    from mcp.types import CallToolRequest

    from telegram_mcp import runtime

    server = runtime.mcp
    settings = server.settings
    for name in ("auth", "host", "port", "streamable_http_path", "transport_security"):
        monkeypatch.setattr(settings, name, getattr(settings, name))
    monkeypatch.setattr(server, "_auth_server_provider", server._auth_server_provider)
    monkeypatch.setattr(server, "_token_verifier", server._token_verifier)
    monkeypatch.setattr(server, "_custom_starlette_routes", list(server._custom_starlette_routes))
    handlers = server._mcp_server.request_handlers
    monkeypatch.setitem(handlers, CallToolRequest, handlers[CallToolRequest])
    for name in (
        "MCP_AUTH",
        "MCP_AUTH_SECRET",
        "MCP_AUTH_ISSUER_URL",
        "MCP_ACCESS_CODES",
        "MCP_ACCESS_TOKEN_TTL",
        "MCP_REFRESH_TOKEN_TTL",
        "MCP_AUTH_ALLOW_LOCALHOST",
        "MCP_HTTP_PATH",
        "MCP_ALLOWED_HOSTS",
        "MCP_ALLOWED_ORIGINS",
    ):
        monkeypatch.delenv(name, raising=False)
    return server
