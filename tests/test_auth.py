"""Unit tests for the MCP_AUTH=1 OAuth provider (telegram_mcp/auth.py)."""

import asyncio
import importlib.util
import re
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from mcp.server.auth.provider import (
    AuthorizationParams,
    AuthorizeError,
    RegistrationError,
    TokenError,
)
from mcp.shared.auth import OAuthClientInformationFull
from pydantic import AnyUrl

from telegram_mcp import auth, runner

# Test-only values; nothing here is a real credential.
SECRET = b"unit-test-secret-" + b"x" * 32
ISSUER = "https://tg.example"
RESOURCE = ISSUER + "/mcp"
CODE_A = "abcd-efgh-jkmn-pqrs"
CODE_B = "tuvw-xyz2-3456-789a"
REDIRECT = "https://claude.ai/api/mcp/auth_callback"


def _config(**codes):
    codes = codes or {"fargona_sh": CODE_A, "toshkent": CODE_B}
    return auth.AuthConfig(
        secret=SECRET,
        issuer_url=ISSUER,
        access_codes={name: auth.hash_access_code(code) for name, code in codes.items()},
    )


class Clock:
    def __init__(self, now=1_800_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def _provider(config=None, clock=None, resource=RESOURCE):
    return auth.AccessCodeOAuthProvider(
        config or _config(), resource, clock=clock or Clock(), failure_delay=0
    )


async def _registered_client(provider):
    info = OAuthClientInformationFull(
        redirect_uris=[AnyUrl(REDIRECT)], token_endpoint_auth_method="none"
    )
    await provider.register_client(info)
    return await provider.get_client(info.client_id)


async def _tokens(provider, advisor="fargona_sh"):
    client = await _registered_client(provider)
    return client, provider._issue_tokens(client.client_id, advisor, [auth.SCOPE])


def _flip_last_char(text):
    return text[:-1] + ("A" if text[-1] != "A" else "B")


# ----- access codes -------------------------------------------------------


def test_generated_code_is_easy_to_type():
    code = auth.generate_access_code()
    assert re.fullmatch(r"[a-z2-9]{4}(-[a-z2-9]{4}){3}", code)
    assert not set(code) & set("01ilo")


def test_code_hash_ignores_case_spaces_and_dashes():
    assert auth.hash_access_code("ABCD efgh-JKMN  pqrs") == auth.hash_access_code(CODE_A)
    assert auth.hash_access_code(CODE_A) != auth.hash_access_code(CODE_B)


def test_match_access_code_finds_the_advisor():
    codes = _config().access_codes
    assert auth.match_access_code("ABCD-EFGH-JKMN-PQRS", codes) == "fargona_sh"
    assert auth.match_access_code(CODE_B, codes) == "toshkent"
    assert auth.match_access_code("abcd-efgh-jkmn-pqrt", codes) is None
    assert auth.match_access_code("", codes) is None


def test_parse_access_codes_rejects_bad_entries_without_echoing_hashes():
    digest = auth.hash_access_code(CODE_A)
    assert auth.parse_access_codes(f" a_1:{digest.upper()} ,, ") == {"a_1": digest}
    for raw in (f"a:{digest[:-1]}", f"bad name:{digest}", digest, f"a:{digest},a:{'0' * 64}"):
        with pytest.raises(auth.AuthConfigError) as exc:
            auth.parse_access_codes(raw)
        assert digest[:20] not in str(exc.value)
    with pytest.raises(auth.AuthConfigError, match="reuses"):
        auth.parse_access_codes(f"a:{digest},b:{digest}")


def test_make_access_code_script_prints_code_and_matching_entry(capsys):
    path = Path(__file__).resolve().parent.parent / "tools" / "make_access_code.py"
    spec = importlib.util.spec_from_file_location("make_access_code", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    assert script.main(["fargona_sh"]) == 0
    out = capsys.readouterr().out
    code = re.search(r"^\s+([a-z2-9]{4}(?:-[a-z2-9]{4}){3})$", out, re.M).group(1)
    entry = re.search(r"^\s+(fargona_sh:[0-9a-f]{64})$", out, re.M).group(1)
    assert entry == f"fargona_sh:{auth.hash_access_code(code)}"
    assert out.count(code) == 1


# ----- configuration --------------------------------------------------------


def test_auth_only_for_http_transports_with_flag():
    assert auth.auth_enabled("http", {"MCP_AUTH": "1"})
    assert auth.auth_enabled("sse", {"MCP_AUTH": "true"})
    assert not auth.auth_enabled("stdio", {"MCP_AUTH": "1"})
    assert not auth.auth_enabled("http", {})
    assert not auth.auth_enabled("http", {"MCP_AUTH": "0"})


def test_config_lists_every_missing_setting():
    with pytest.raises(auth.AuthConfigError) as exc:
        auth.load_auth_config({"MCP_AUTH": "1"})
    message = str(exc.value)
    for name in ("MCP_AUTH_SECRET", "MCP_AUTH_ISSUER_URL", "MCP_ACCESS_CODES"):
        assert name in message


@pytest.mark.parametrize(
    "override,fragment",
    [
        ({"MCP_AUTH_SECRET": "short"}, "too short"),
        ({"MCP_AUTH_ISSUER_URL": "http://tg.example"}, "https://"),
        ({"MCP_AUTH_ISSUER_URL": "https://tg.example/mcp"}, "no path"),
        ({"MCP_ACCESS_TOKEN_TTL": "-5"}, "MCP_ACCESS_TOKEN_TTL"),
        ({"MCP_ACCESS_CODES": ""}, "MCP_ACCESS_CODES"),
    ],
)
def test_config_rejects_invalid_values(override, fragment):
    env = {
        "MCP_AUTH_SECRET": SECRET.decode(),
        "MCP_AUTH_ISSUER_URL": ISSUER,
        "MCP_ACCESS_CODES": f"a:{auth.hash_access_code(CODE_A)}",
        **override,
    }
    with pytest.raises(auth.AuthConfigError, match=re.escape(fragment)) as exc:
        auth.load_auth_config(env)
    assert SECRET.decode() not in str(exc.value)


def test_config_accepts_complete_settings():
    config = auth.load_auth_config(
        {
            "MCP_AUTH_SECRET": SECRET.decode(),
            "MCP_AUTH_ISSUER_URL": ISSUER + "/",
            "MCP_ACCESS_CODES": f"a:{auth.hash_access_code(CODE_A)}",
            "MCP_ACCESS_TOKEN_TTL": "600",
        }
    )
    assert config.issuer_url == ISSUER
    assert config.access_token_ttl == 600
    assert config.refresh_token_ttl == auth.DEFAULT_REFRESH_TOKEN_TTL
    assert not config.allow_localhost


# ----- startup wiring --------------------------------------------------------


def test_startup_fails_on_missing_auth_config(isolated_mcp_auth, monkeypatch):
    monkeypatch.setenv("MCP_TRANSPORT", "http")
    monkeypatch.setenv("MCP_AUTH", "1")
    monkeypatch.setenv("MCP_AUTH_SECRET", "too-short-secret-value")
    with pytest.raises(SystemExit) as exc:
        runner._validate_auth_config()
    assert "MCP_AUTH_ISSUER_URL" in str(exc.value)
    assert "too-short-secret-value" not in str(exc.value)


def test_startup_ignores_auth_flag_for_stdio(isolated_mcp_auth, monkeypatch):
    monkeypatch.setenv("MCP_TRANSPORT", "stdio")
    monkeypatch.setenv("MCP_AUTH", "1")
    runner._validate_auth_config()  # no SystemExit


def test_stdio_mode_is_unaffected_by_auth_flag(isolated_mcp_auth, monkeypatch):
    server = isolated_mcp_auth
    monkeypatch.setenv("MCP_AUTH", "1")
    stdio = AsyncMock()
    monkeypatch.setattr(server, "run_stdio_async", stdio)
    handlers_before = server._mcp_server.request_handlers.copy()
    routes_before = list(server._custom_starlette_routes)

    asyncio.run(runner._serve("stdio"))

    stdio.assert_awaited_once()
    assert server.settings.auth is None
    assert server._token_verifier is None
    assert server._mcp_server.request_handlers == handlers_before
    assert server._custom_starlette_routes == routes_before


def test_http_without_auth_flag_stays_open(isolated_mcp_auth, monkeypatch):
    server = isolated_mcp_auth
    monkeypatch.setattr(server, "run_streamable_http_async", AsyncMock())
    asyncio.run(runner._serve("http"))
    assert server.settings.auth is None


def test_serve_installs_auth_on_secret_path(isolated_mcp_auth, monkeypatch):
    server = isolated_mcp_auth
    monkeypatch.setenv("MCP_AUTH", "1")
    monkeypatch.setenv("MCP_AUTH_SECRET", SECRET.decode())
    monkeypatch.setenv("MCP_AUTH_ISSUER_URL", ISSUER)
    monkeypatch.setenv("MCP_ACCESS_CODES", f"a:{auth.hash_access_code(CODE_A)}")
    monkeypatch.setenv("MCP_HTTP_PATH", "/abc123/mcp/")
    start = AsyncMock()
    monkeypatch.setattr(server, "run_streamable_http_async", start)

    asyncio.run(runner._serve("http"))

    start.assert_awaited_once()
    assert str(server.settings.auth.resource_server_url) == ISSUER + "/abc123/mcp"
    assert server._auth_server_provider.resource_url == ISSUER + "/abc123/mcp"


# ----- clients and redirect URIs -------------------------------------------


@pytest.mark.parametrize(
    "uri,local,ok",
    [
        (REDIRECT, False, True),
        ("https://claude.com/api/mcp/auth_callback", False, True),
        ("https://claude.ai.evil.example/cb", False, False),
        ("https://evil.example/cb?x=https://claude.ai", False, False),
        ("http://claude.ai/api/mcp/auth_callback", False, False),
        ("https://claude.ai:8443/cb", False, False),
        ("https://user@claude.ai/cb", False, False),
        ("http://localhost:6274/oauth/callback", False, False),
        ("http://localhost:6274/oauth/callback", True, True),
        ("http://127.0.0.1:9000/cb", True, True),
        ("http://192.168.1.5/cb", True, False),
    ],
)
def test_redirect_uri_allowlist(uri, local, ok):
    assert auth.redirect_uri_allowed(uri, local) is ok


@pytest.mark.asyncio
async def test_registration_is_stateless_and_survives_restart():
    client = await _registered_client(_provider())
    again = await _provider().get_client(client.client_id)
    assert again is not None
    assert [str(u) for u in again.redirect_uris] == [REDIRECT]
    assert again.client_secret is None


@pytest.mark.asyncio
async def test_confidential_client_secret_is_derived_not_stored():
    provider = _provider()
    info = OAuthClientInformationFull(
        redirect_uris=[AnyUrl(REDIRECT)], token_endpoint_auth_method="client_secret_post"
    )
    info.client_secret = "random-from-sdk"
    await provider.register_client(info)
    assert info.client_secret != "random-from-sdk"
    assert (await _provider().get_client(info.client_id)).client_secret == info.client_secret


@pytest.mark.asyncio
async def test_registration_rejects_redirect_outside_allowlist():
    info = OAuthClientInformationFull(
        redirect_uris=[AnyUrl(REDIRECT), AnyUrl("https://evil.example/cb")],
        token_endpoint_auth_method="none",
    )
    with pytest.raises(RegistrationError) as exc:
        await _provider().register_client(info)
    assert exc.value.error == "invalid_redirect_uri"


@pytest.mark.asyncio
async def test_tampered_or_foreign_client_id_is_unknown():
    client = await _registered_client(_provider())
    body, mac = client.client_id.split(".")
    assert await _provider().get_client(f"{_flip_last_char(body)}.{mac}") is None
    other = auth.AuthConfig(secret=b"y" * 40, issuer_url=ISSUER, access_codes={})
    assert await _provider(other).get_client(client.client_id) is None


@pytest.mark.asyncio
async def test_authorize_rejects_foreign_resource():
    provider = _provider()
    client = await _registered_client(provider)
    params = AuthorizationParams(
        state="s",
        scopes=None,
        code_challenge="c" * 43,
        redirect_uri=AnyUrl(REDIRECT),
        redirect_uri_provided_explicitly=True,
        resource="https://other.example/mcp",
    )
    with pytest.raises(AuthorizeError):
        await provider.authorize(client, params)
    params.resource = RESOURCE
    assert (await provider.authorize(client, params)).startswith(ISSUER + "/login?req=")


# ----- tokens -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_access_token_round_trip_and_survives_restart():
    provider = _provider()
    _, tokens = await _tokens(provider)
    for verifier in (provider, _provider()):
        info = await verifier.load_access_token(tokens.access_token)
        assert info.subject == "fargona_sh"
        assert info.scopes == [auth.SCOPE]
        assert info.resource == RESOURCE


@pytest.mark.asyncio
async def test_revoked_advisor_is_rejected_on_next_check():
    _, tokens = await _tokens(_provider())
    without = _provider(_config(toshkent=CODE_B))
    assert await without.load_access_token(tokens.access_token) is None
    client = await _registered_client(without)
    assert await without.load_refresh_token(client, tokens.refresh_token) is None


@pytest.mark.asyncio
async def test_reissued_code_revokes_old_tokens():
    _, tokens = await _tokens(_provider())
    reissued = _provider(_config(fargona_sh="mnpq-rstu-vwxy-z234", toshkent=CODE_B))
    assert await reissued.load_access_token(tokens.access_token) is None


@pytest.mark.asyncio
async def test_expired_access_token_is_rejected():
    clock = Clock()
    provider = _provider(clock=clock)
    _, tokens = await _tokens(provider)
    clock.now += auth.DEFAULT_ACCESS_TOKEN_TTL + 1
    assert await provider.load_access_token(tokens.access_token) is None


@pytest.mark.asyncio
async def test_tampered_and_misused_tokens_are_rejected():
    provider = _provider()
    client, tokens = await _tokens(provider)
    body, mac = tokens.access_token.split(".")
    assert await provider.load_access_token(f"{_flip_last_char(body)}.{mac}") is None
    assert await provider.load_access_token(f"{body}.{_flip_last_char(mac)}") is None
    assert await provider.load_access_token("not-a-token") is None
    # A refresh token is not an access token, and vice versa.
    assert await provider.load_access_token(tokens.refresh_token) is None
    assert await provider.load_refresh_token(client, tokens.access_token) is None
    # Another MCP endpoint (audience) does not accept it.
    elsewhere = _provider(resource=ISSUER + "/other")
    assert await elsewhere.load_access_token(tokens.access_token) is None
    # Another secret does not accept it.
    rotated = auth.AuthConfig(
        secret=b"z" * 40, issuer_url=ISSUER, access_codes=_config().access_codes
    )
    assert await _provider(rotated).load_access_token(tokens.access_token) is None


@pytest.mark.asyncio
async def test_refresh_token_rotates_once():
    provider = _provider()
    client, tokens = await _tokens(provider)
    refresh = await provider.load_refresh_token(client, tokens.refresh_token)
    new = await provider.exchange_refresh_token(client, refresh, [auth.SCOPE])
    assert await provider.load_access_token(new.access_token) is not None
    assert await provider.load_refresh_token(client, tokens.refresh_token) is None
    with pytest.raises(TokenError):
        await provider.exchange_refresh_token(client, refresh, [auth.SCOPE])


@pytest.mark.asyncio
async def test_revoke_token():
    provider = _provider()
    _, tokens = await _tokens(provider)
    info = await provider.load_access_token(tokens.access_token)
    await provider.revoke_token(info)
    assert await provider.load_access_token(tokens.access_token) is None


# ----- rate limiting ----------------------------------------------------------


def test_rate_limit_per_ip_and_window():
    clock = Clock(0.0)
    limiter = auth.FailureLimiter(per_ip=3, total=100, window=60, clock=clock)
    for _ in range(3):
        assert not limiter.blocked("1.1.1.1")
        limiter.record_failure("1.1.1.1")
    assert limiter.blocked("1.1.1.1")
    assert not limiter.blocked("2.2.2.2")
    clock.now = 61.0
    assert not limiter.blocked("1.1.1.1")


def test_rate_limit_global_cap():
    limiter = auth.FailureLimiter(per_ip=100, total=5, window=60, clock=Clock(0.0))
    for i in range(5):
        limiter.record_failure(f"10.0.0.{i}")
    assert limiter.blocked("10.9.9.9")
