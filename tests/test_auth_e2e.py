"""End-to-end OAuth flow against the real ASGI app (in-process, no network).

metadata discovery -> register -> authorize -> login page + access code ->
token -> MCP initialize / tools/list with the bearer token. The Telegram client
is never reached: these MCP methods do not touch it, and tools/call targets a
name that does not exist.
"""

import base64
import hashlib
import json
import logging
import re
from urllib.parse import parse_qs, urlsplit

import pytest
from mcp.server.transport_security import TransportSecuritySettings
from starlette.testclient import TestClient

import telegram_mcp.tools  # noqa: F401 - registers the MCP tools
from telegram_mcp import auth
from telegram_mcp.transport.http import create_streamable_http_app

# Test-only values; nothing here is a real credential.
SECRET = b"e2e-test-secret-" + b"q" * 32
ISSUER = "https://tg.example"
MCP_PATH = "/s3cr3t/mcp"
RESOURCE = ISSUER + MCP_PATH
CODE = "abcd-efgh-jkmn-pqrs"
WRONG_CODE = "zzzz-zzzz-zzzz-zzzz"
REDIRECT = "https://claude.ai/api/mcp/auth_callback"
VERIFIER = "e2e-verifier-" + "v" * 43
CHALLENGE = base64.urlsafe_b64encode(hashlib.sha256(VERIFIER.encode()).digest()).rstrip(b"=")

MCP_HEADERS = {
    "MCP-Protocol-Version": "2025-11-25",
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-11-25",
        "capabilities": {},
        "clientInfo": {"name": "e2e", "version": "1"},
    },
}
TOOLS_LIST = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}


@pytest.fixture
def oauth(isolated_mcp_auth):
    server = isolated_mcp_auth
    server.settings.streamable_http_path = MCP_PATH
    server.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=["tg.example"], allowed_origins=[]
    )
    config = auth.AuthConfig(
        secret=SECRET,
        issuer_url=ISSUER,
        access_codes={"fargona_sh": auth.hash_access_code(CODE)},
    )
    provider = auth.install_auth(server, config, "http")
    provider.failure_delay = 0
    with TestClient(create_streamable_http_app(server), base_url=ISSUER) as client:
        yield client, provider


def _rpc(response):
    """Parse a JSON or single-event SSE JSON-RPC response."""
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        lines = response.text.splitlines()
        data = [line[5:].strip() for line in lines if line.startswith("data:")]
        return json.loads(data[-1])
    return response.json()


def _register(client):
    response = client.post(
        "/register",
        json={
            "redirect_uris": [REDIRECT],
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "client_name": "Claude",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["client_id"]


def _open_login(client, client_id, state="state-123"):
    response = client.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": REDIRECT,
            "code_challenge": CHALLENGE.decode(),
            "code_challenge_method": "S256",
            "state": state,
            "scope": auth.SCOPE,
            "resource": RESOURCE,
        },
        follow_redirects=False,
    )
    assert response.status_code == 302, response.text
    login_url = response.headers["location"]
    assert login_url.startswith(f"{ISSUER}/login?req=")
    page = client.get(login_url)
    assert page.status_code == 200
    req = re.search(r'name="req" value="([^"]+)"', page.text).group(1)
    csrf = re.search(r'name="csrf" value="([^"]+)"', page.text).group(1)
    return page, req, csrf


def _submit(client, req, csrf, code):
    return client.post(
        "/login", data={"req": req, "csrf": csrf, "code": code}, follow_redirects=False
    )


def _authorization_code(client, client_id, state="state-123"):
    _, req, csrf = _open_login(client, client_id, state)
    response = _submit(client, req, csrf, CODE.upper().replace("-", " "))
    assert response.status_code == 302, response.text
    target = urlsplit(response.headers["location"])
    assert f"{target.scheme}://{target.netloc}{target.path}" == REDIRECT
    query = parse_qs(target.query)
    assert query["state"] == [state]
    return query["code"][0]


def _exchange(client, client_id, code, verifier=VERIFIER):
    return client.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT,
            "client_id": client_id,
            "code_verifier": verifier,
        },
    )


def _mcp(client, payload, token=None):
    headers = dict(MCP_HEADERS)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return client.post(MCP_PATH, headers=headers, json=payload)


def test_full_oauth_flow_then_mcp_calls(oauth, caplog):
    client, _ = oauth

    # 1. Unauthenticated MCP request: 401 pointing at protected-resource metadata.
    denied = _mcp(client, INITIALIZE)
    assert denied.status_code == 401
    challenge = denied.headers["www-authenticate"]
    prm_url = f"{ISSUER}/.well-known/oauth-protected-resource{MCP_PATH}"
    assert challenge.startswith("Bearer ")
    assert f'resource_metadata="{prm_url}"' in challenge

    # 2. Discovery: protected resource -> authorization server metadata.
    prm = client.get(prm_url).json()
    assert prm["resource"] == RESOURCE
    assert prm["authorization_servers"][0].rstrip("/") == ISSUER
    metadata = client.get(f"{ISSUER}/.well-known/oauth-authorization-server").json()
    assert metadata["registration_endpoint"] == f"{ISSUER}/register"
    assert metadata["code_challenge_methods_supported"] == ["S256"]

    # 3-5. Register, log in with the access code, exchange the code.
    client_id = _register(client)
    page, _, _ = _open_login(client, client_id)
    assert "<title>Telegram monitoring — kirish</title>" in page.text
    assert '<label for="code">Kirish kodi</label>' in page.text
    assert "<script" not in page.text and "<link" not in page.text
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]

    tokens = _exchange(client, client_id, _authorization_code(client, client_id))
    assert tokens.status_code == 200, tokens.text
    body = tokens.json()
    assert body["token_type"] == "Bearer" and body["expires_in"] == 3600
    access = body["access_token"]

    # 6. MCP with the bearer token.
    init = _mcp(client, INITIALIZE, access)
    assert init.status_code == 200, init.text
    assert _rpc(init)["result"]["serverInfo"]["name"] == "telegram"

    listed = _mcp(client, TOOLS_LIST, access)
    assert listed.status_code == 200
    assert len(_rpc(listed)["result"]["tools"]) > 10

    # 7. Audit line: advisor + tool name, never arguments.
    caplog.set_level(logging.INFO, logger="telegram_mcp.access")
    auth.access_logger.addHandler(caplog.handler)
    try:
        call = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "no_such_tool", "arguments": {"text": "private-message-body"}},
        }
        assert _mcp(client, call, access).status_code == 200
    finally:
        auth.access_logger.removeHandler(caplog.handler)
    lines = [r.getMessage() for r in caplog.records if r.name == "telegram_mcp.access"]
    assert "tool_call advisor=fargona_sh tool=no_such_tool" in lines
    assert not any("private-message-body" in line or access in line for line in lines)

    # 8. Same call without a token, or with a garbage token: 401.
    assert _mcp(client, TOOLS_LIST).status_code == 401
    assert _mcp(client, TOOLS_LIST, access[:-2] + "xx").status_code == 401


def test_refresh_token_flow_rotates(oauth):
    client, _ = oauth
    client_id = _register(client)
    first = _exchange(client, client_id, _authorization_code(client, client_id)).json()
    refresh = {
        "grant_type": "refresh_token",
        "refresh_token": first["refresh_token"],
        "client_id": client_id,
    }
    second = client.post("/token", data=refresh)
    assert second.status_code == 200, second.text
    assert _mcp(client, TOOLS_LIST, second.json()["access_token"]).status_code == 200
    replay = client.post("/token", data=refresh)
    assert replay.status_code == 400
    assert replay.json()["error"] == "invalid_grant"


def test_pkce_mismatch_is_rejected(oauth):
    client, _ = oauth
    client_id = _register(client)
    code = _authorization_code(client, client_id)
    response = _exchange(client, client_id, code, verifier="w" * 50)
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_grant"


def test_authorization_code_is_single_use(oauth):
    client, _ = oauth
    client_id = _register(client)
    code = _authorization_code(client, client_id)
    assert _exchange(client, client_id, code).status_code == 200
    reused = _exchange(client, client_id, code)
    assert reused.status_code == 400
    assert reused.json()["error"] == "invalid_grant"


def test_registration_rejects_foreign_redirect(oauth):
    client, _ = oauth
    response = client.post(
        "/register",
        json={"redirect_uris": ["https://evil.example/cb"], "token_endpoint_auth_method": "none"},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_redirect_uri"


def test_wrong_code_then_rate_limit(oauth):
    client, provider = oauth
    client_id = _register(client)
    _, req, csrf = _open_login(client, client_id)

    wrong = _submit(client, req, csrf, WRONG_CODE)
    assert wrong.status_code == 401
    assert "Kod noto&#x27;g&#x27;ri" in wrong.text
    for _ in range(provider.limiter.per_ip - 1):
        _, req, csrf = _open_login(client, client_id)
        assert _submit(client, req, csrf, WRONG_CODE).status_code == 401

    # Blocked now, even with the right code.
    _, req, csrf = _open_login(client, client_id)
    assert _submit(client, req, csrf, CODE).status_code == 429


def test_login_requires_csrf_token(oauth):
    client, _ = oauth
    client_id = _register(client)
    _, req, _ = _open_login(client, client_id)
    assert _submit(client, req, "forged", CODE).status_code == 400
    _, req, _ = _open_login(client, client_id)
    assert _submit(client, req, "ünïcödé", CODE).status_code == 400
    _, req, csrf = _open_login(client, client_id)
    client.cookies.clear()  # cross-site POST: the SameSite=Strict cookie is not sent
    assert _submit(client, req, csrf, CODE).status_code == 400


def test_login_rejects_tampered_request(oauth):
    client, _ = oauth
    client_id = _register(client)
    _, req, csrf = _open_login(client, client_id)
    body, mac = req.split(".")
    tampered = f"{body[:-1]}{'A' if body[-1] != 'A' else 'B'}.{mac}"
    response = _submit(client, tampered, csrf, CODE)
    assert response.status_code == 400
    assert 'name="code"' not in response.text
