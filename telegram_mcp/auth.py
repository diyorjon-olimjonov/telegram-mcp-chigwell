"""OAuth 2.1 login for the HTTP transports, keyed on per-advisor access codes.

Claude custom connectors authenticate only through the MCP authorization spec
(OAuth 2.1 + PKCE, usually with Dynamic Client Registration). This module plugs
an authorization-server provider into the MCP SDK's own OAuth machinery
(``/authorize``, ``/token``, ``/register``, ``/revoke``, both ``.well-known``
metadata documents and the bearer-token middleware). The only extra endpoint is
the login page, where an advisor types the personal access code the operator
issued with ``tools/make_access_code.py``.

Everything that must survive a restart is stateless and signed with
``MCP_AUTH_SECRET`` (HMAC-SHA256): registered clients (the client_id *is* the
signed registration), access tokens and refresh tokens. Only short-lived things
live in memory: authorization codes (5 minutes, single use), the login rate
limiter, and the "already used" marks of rotated or revoked tokens.

Revocation:
- remove (or re-issue) an advisor's entry in ``MCP_ACCESS_CODES`` and redeploy:
  every token of that advisor fails its next check;
- change ``MCP_AUTH_SECRET``: every client, token and refresh token is void.

Never log codes, tokens or the secret. The access log records only the advisor
name and the tool name.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import hashlib
import hmac
import html
import json
import logging
import os
import re
import secrets
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlsplit

from pydantic import AnyUrl
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    RefreshToken,
    RegistrationError,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

SCOPE = "telegram"
LOGIN_PATH = "/login"
# Public OAuth endpoints (served by the SDK, plus the login page). The HTTP
# protocol-negotiation middleware lets these through untouched when auth is on.
OAUTH_PATHS = frozenset({"/authorize", "/token", "/register", "/revoke", LOGIN_PATH})
OAUTH_PATH_PREFIXES = (
    "/.well-known/oauth-authorization-server",
    "/.well-known/oauth-protected-resource",
)

AUTH_CODE_TTL = 300  # seconds an authorization code stays redeemable
LOGIN_REQUEST_TTL = 600  # seconds the login page stays usable after /authorize
DEFAULT_ACCESS_TOKEN_TTL = 3600
DEFAULT_REFRESH_TOKEN_TTL = 30 * 24 * 3600
MIN_SECRET_BYTES = 32

# Access codes: easy to read aloud and type, no look-alikes (0/o, 1/l/i).
CODE_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"
CODE_GROUPS = 4
CODE_GROUP_SIZE = 4

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TRUE = {"1", "true", "yes", "on"}
_CLIENT_AUTH_METHODS = ("none", "client_secret_post", "client_secret_basic")
_ALLOWED_REDIRECT_HOSTS = ("claude.ai", "claude.com")
_LOCAL_REDIRECT_HOSTS = ("localhost", "127.0.0.1")
_CSRF_COOKIE = "tgmcp_login"

access_logger = logging.getLogger("telegram_mcp.access")


class AuthConfigError(ValueError):
    """MCP_AUTH=1 but the auth configuration is missing or invalid."""


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AuthConfig:
    secret: bytes
    issuer_url: str  # public https base URL, no trailing slash
    access_codes: Mapping[str, str]  # advisor name -> sha256 hex of the code
    access_token_ttl: int = DEFAULT_ACCESS_TOKEN_TTL
    refresh_token_ttl: int = DEFAULT_REFRESH_TOKEN_TTL
    allow_localhost: bool = False


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in _TRUE


def auth_enabled(transport: str, env: Mapping[str, str] = os.environ) -> bool:
    """OAuth is on only for the HTTP transports and only with MCP_AUTH=1."""
    return transport in ("http", "sse") and _truthy(env.get("MCP_AUTH"))


def parse_access_codes(raw: str) -> dict[str, str]:
    """Parse ``name:sha256hex,name:sha256hex``. Errors never echo the hashes."""
    codes: dict[str, str] = {}
    for index, entry in enumerate((e.strip() for e in raw.split(",")), start=1):
        if not entry:
            continue
        name, sep, digest = entry.partition(":")
        name, digest = name.strip(), digest.strip().lower()
        if not sep or not _NAME_RE.match(name):
            raise AuthConfigError(
                f"MCP_ACCESS_CODES entry #{index}: expected 'name:sha256hex' with a name of "
                "letters, digits, '_', '-' or '.'"
            )
        if not _SHA256_RE.match(digest):
            raise AuthConfigError(
                f"MCP_ACCESS_CODES entry #{index} ({name}): the hash must be 64 hex characters "
                "(use tools/make_access_code.py)"
            )
        if name in codes:
            raise AuthConfigError(f"MCP_ACCESS_CODES: advisor '{name}' is listed twice")
        if digest in codes.values():
            raise AuthConfigError(
                f"MCP_ACCESS_CODES entry #{index} ({name}) reuses another advisor's code"
            )
        codes[name] = digest
    return codes


def _parse_ttl(env: Mapping[str, str], name: str, default: int, problems: list[str]) -> int:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        problems.append(f"{name} must be a positive number of seconds")
        return default
    return value


def _parse_issuer(raw: str, problems: list[str]) -> str:
    issuer = raw.strip().rstrip("/")
    if not issuer:
        problems.append(
            "MCP_AUTH_ISSUER_URL is not set (the public https base URL, "
            "e.g. https://your-app.up.railway.app)"
        )
        return ""
    parts = urlsplit(issuer)
    local = parts.hostname in _LOCAL_REDIRECT_HOSTS
    if parts.scheme != "https" and not (parts.scheme == "http" and local):
        problems.append("MCP_AUTH_ISSUER_URL must start with https://")
    elif not parts.hostname or parts.path or parts.query or parts.fragment:
        problems.append(
            "MCP_AUTH_ISSUER_URL must be a bare base URL (scheme and host only, no path)"
        )
    return issuer


def load_auth_config(env: Mapping[str, str] = os.environ) -> AuthConfig:
    """Read and validate the auth settings; raise AuthConfigError listing every problem."""
    problems: list[str] = []

    secret = (env.get("MCP_AUTH_SECRET") or "").strip().encode("utf-8")
    if len(secret) < MIN_SECRET_BYTES:
        state = "is not set" if not secret else "is too short"
        problems.append(
            f"MCP_AUTH_SECRET {state} (need at least {MIN_SECRET_BYTES} bytes; "
            "generate one with `openssl rand -hex 32`)"
        )

    issuer = _parse_issuer(env.get("MCP_AUTH_ISSUER_URL") or "", problems)

    codes: dict[str, str] = {}
    try:
        codes = parse_access_codes(env.get("MCP_ACCESS_CODES") or "")
    except AuthConfigError as exc:
        problems.append(str(exc))
    else:
        if not codes:
            problems.append(
                "MCP_ACCESS_CODES is empty (add name:sha256hex entries from "
                "tools/make_access_code.py)"
            )

    access_ttl = _parse_ttl(env, "MCP_ACCESS_TOKEN_TTL", DEFAULT_ACCESS_TOKEN_TTL, problems)
    refresh_ttl = _parse_ttl(env, "MCP_REFRESH_TOKEN_TTL", DEFAULT_REFRESH_TOKEN_TTL, problems)

    if problems:
        raise AuthConfigError(
            "MCP_AUTH=1 but the OAuth configuration is incomplete:\n  - " + "\n  - ".join(problems)
        )
    return AuthConfig(
        secret=secret,
        issuer_url=issuer,
        access_codes=codes,
        access_token_ttl=access_ttl,
        refresh_token_ttl=refresh_ttl,
        allow_localhost=_truthy(env.get("MCP_AUTH_ALLOW_LOCALHOST")),
    )


# --------------------------------------------------------------------------
# Access codes
# --------------------------------------------------------------------------


def generate_access_code() -> str:
    """Return a random code like ``k7mq-x3ta-9dwe-hp2r`` (~79 bits of entropy)."""
    groups = (
        "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_GROUP_SIZE))
        for _ in range(CODE_GROUPS)
    )
    return "-".join(groups)


def normalize_access_code(code: str) -> str:
    """Ignore case, spaces and dashes, so ``K7MQ X3TA…`` matches ``k7mq-x3ta-…``."""
    return re.sub(r"[\s\-]+", "", code).lower()


def hash_access_code(code: str) -> str:
    return hashlib.sha256(normalize_access_code(code).encode("utf-8")).hexdigest()


def match_access_code(code: str, access_codes: Mapping[str, str]) -> Optional[str]:
    """Return the advisor whose code this is, comparing every entry in constant time."""
    digest = hash_access_code(code)
    found = None
    for name, expected in access_codes.items():
        if hmac.compare_digest(digest, expected):
            found = name
    return found


# --------------------------------------------------------------------------
# Signed blobs (client IDs, login requests, tokens)
# --------------------------------------------------------------------------


def _b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64d(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


class Signer:
    """HMAC-SHA256 signed JSON. ``purpose`` keeps one kind of blob from passing as another."""

    def __init__(self, secret: bytes, clock: Callable[[], float] = time.time):
        self._key = hashlib.sha256(b"telegram-mcp-oauth-v1\x00" + secret).digest()
        self._clock = clock

    def _mac(self, purpose: str, body: str) -> bytes:
        message = f"{purpose}.{body}".encode("ascii")
        return hmac.new(self._key, message, hashlib.sha256).digest()

    def mac_hex(self, *parts: str) -> str:
        message = "\x00".join(parts).encode("utf-8")
        return hmac.new(self._key, message, hashlib.sha256).hexdigest()

    def sign(self, purpose: str, payload: dict[str, Any]) -> str:
        raw = json.dumps(
            payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
        body = _b64e(raw)
        return f"{body}.{_b64e(self._mac(purpose, body))}"

    def unsign(self, purpose: str, blob: Any, *, check_expiry: bool = True) -> Optional[dict]:
        if not isinstance(blob, str) or blob.count(".") != 1 or len(blob) > 8192:
            return None
        body, mac = blob.split(".")
        try:
            valid = hmac.compare_digest(_b64d(mac), self._mac(purpose, body))
            payload = json.loads(_b64d(body)) if valid else None
        except (ValueError, UnicodeError):
            return None
        if not isinstance(payload, dict):
            return None
        expires = payload.get("exp")
        if check_expiry and (not isinstance(expires, int) or expires <= self._clock()):
            return None
        return payload


# --------------------------------------------------------------------------
# Login rate limiting
# --------------------------------------------------------------------------


class FailureLimiter:
    """Sliding-window cap on failed logins, per client IP and in total."""

    def __init__(
        self,
        per_ip: int = 5,
        total: int = 30,
        window: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.per_ip = per_ip
        self.total = total
        self.window = window
        self._clock = clock
        self._by_ip: dict[str, collections.deque] = {}
        self._all: collections.deque = collections.deque()

    def _prune(self, events: collections.deque, now: float) -> None:
        while events and events[0] <= now - self.window:
            events.popleft()

    def blocked(self, ip: str) -> bool:
        now = self._clock()
        self._prune(self._all, now)
        events = self._by_ip.get(ip)
        if events is not None:
            self._prune(events, now)
            if not events:
                del self._by_ip[ip]
        return len(self._all) >= self.total or len(self._by_ip.get(ip, ())) >= self.per_ip

    def record_failure(self, ip: str) -> None:
        now = self._clock()
        self._all.append(now)
        self._by_ip.setdefault(ip, collections.deque()).append(now)
        if len(self._by_ip) > 10_000:  # bound memory under a spray of fake IPs
            stale = [k for k, v in self._by_ip.items() if not v or v[-1] <= now - self.window]
            for key in stale:
                del self._by_ip[key]


def client_ip(request: Request) -> str:
    """The caller's IP. Behind Railway's proxy that is the last X-Forwarded-For hop."""
    forwarded = request.headers.get("x-forwarded-for", "")
    hops = [hop.strip() for hop in forwarded.split(",") if hop.strip()]
    if hops:
        return hops[-1]
    return request.client.host if request.client else "unknown"


def redirect_uri_allowed(uri: str, allow_localhost: bool) -> bool:
    """Only Claude's callback hosts, plus http://localhost for local testing when enabled."""
    try:
        parts = urlsplit(str(uri))
        port = parts.port
    except ValueError:
        return False
    if parts.fragment or parts.username or parts.password:
        return False
    host = parts.hostname
    if parts.scheme == "https" and host in _ALLOWED_REDIRECT_HOSTS and port in (None, 443):
        return True
    return allow_localhost and parts.scheme == "http" and host in _LOCAL_REDIRECT_HOSTS


# --------------------------------------------------------------------------
# Login page
# --------------------------------------------------------------------------

TEXT_TITLE = "Telegram monitoring — kirish"
TEXT_FIELD = "Kirish kodi"
TEXT_BUTTON = "Kirish"
TEXT_WRONG_CODE = "Kod noto'g'ri"
TEXT_TOO_MANY = "Urinishlar juda ko'p. Bir daqiqadan keyin qayta urinib ko'ring."
TEXT_EXPIRED = "Sahifa eskirgan. Claude'da qaytadan ulaning (Connect)."
TEXT_RETRY = "Sahifa yangilandi. Kodni qaytadan kiriting."

_PAGE = """<!doctype html>
<html lang="uz">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>{title}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 0; padding: 16px; background: #f6f7f9;
  color: #111; }}
main {{ max-width: 360px; margin: 10vh auto; background: #fff; padding: 24px;
  border: 1px solid #d0d4da; border-radius: 8px; }}
h1 {{ font-size: 1.25rem; margin: 0 0 16px; }}
label {{ display: block; font-weight: 600; margin-bottom: 6px; }}
input {{ width: 100%; box-sizing: border-box; font: inherit; font-size: 1.1rem;
  padding: 10px; border: 1px solid #6b7280; border-radius: 6px; letter-spacing: .05em; }}
button {{ margin-top: 16px; width: 100%; font: inherit; font-weight: 600; padding: 10px;
  border: 0; border-radius: 6px; background: #1d4ed8; color: #fff; cursor: pointer; }}
input:focus, button:focus {{ outline: 3px solid #93c5fd; outline-offset: 2px; }}
.error {{ color: #b91c1c; background: #fef2f2; border: 1px solid #fca5a5; border-radius: 6px;
  padding: 8px 10px; margin: 0 0 16px; }}
@media (prefers-color-scheme: dark) {{
  body {{ background: #111827; color: #f3f4f6; }}
  main {{ background: #1f2937; border-color: #374151; }}
  input {{ background: #111827; color: #f3f4f6; border-color: #9ca3af; }}
  .error {{ color: #fecaca; background: #450a0a; border-color: #b91c1c; }}
}}
</style>
</head>
<body>
<main>
<h1>{title}</h1>
{error}{form}
</main>
</body>
</html>
"""

_FORM = """<form method="post" action="{action}">
<input type="hidden" name="req" value="{req}">
<input type="hidden" name="csrf" value="{csrf}">
<label for="code">{field}</label>
<input id="code" name="code" type="text" required autofocus autocomplete="one-time-code"
  autocapitalize="none" autocorrect="off" spellcheck="false" maxlength="64"{invalid}>
<button type="submit">{button}</button>
</form>
"""


def _page_headers(redirect_origin: Optional[str]) -> dict[str, str]:
    # form-action also governs the redirect that follows the POST, so the
    # client's callback origin has to be allowed there.
    form_action = "'self'" + (f" {redirect_origin}" if redirect_origin else "")
    return {
        "Content-Security-Policy": (
            "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; "
            f"frame-ancestors 'none'; form-action {form_action}"
        ),
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    }


# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------


class AccessCodeOAuthProvider:
    """OAuth authorization server + resource-server token check for one MCP endpoint."""

    def __init__(
        self,
        config: AuthConfig,
        resource_url: str,
        *,
        clock: Callable[[], float] = time.time,
        failure_delay: float = 1.0,
        limiter: Optional[FailureLimiter] = None,
    ):
        self.config = config
        self.resource_url = resource_url
        self.failure_delay = failure_delay
        self.limiter = limiter or FailureLimiter()
        self._clock = clock
        self._signer = Signer(config.secret, clock)
        self._auth_codes: dict[str, AuthorizationCode] = {}
        self._spent: dict[str, int] = {}  # revoked / rotated token id -> its expiry
        self._secure_cookie = config.issuer_url.startswith("https://")

    # ----- helpers -----------------------------------------------------

    def _now(self) -> int:
        return int(self._clock())

    def _client_secret(self, client_id: str) -> str:
        return self._signer.mac_hex("client-secret", client_id)

    def _code_fingerprint(self, name: str) -> Optional[str]:
        """Binds tokens to the advisor's current code: re-issuing a code revokes old tokens."""
        digest = self.config.access_codes.get(name)
        return self._signer.mac_hex("code", name, digest)[:16] if digest else None

    def _advisor_active(self, payload: dict) -> bool:
        name = payload.get("sub")
        fingerprint = self._code_fingerprint(name) if isinstance(name, str) else None
        return fingerprint is not None and hmac.compare_digest(
            fingerprint, str(payload.get("kf", ""))
        )

    def _mark_spent(self, token_id: str, expires: int) -> None:
        now = self._now()
        for key in [k for k, exp in self._spent.items() if exp <= now]:
            del self._spent[key]
        self._spent[token_id] = expires

    def _resource_ok(self, resource: Optional[str]) -> bool:
        return resource is None or resource.rstrip("/") == self.resource_url.rstrip("/")

    def _issue_tokens(self, client_id: str, advisor: str, scopes: list[str]) -> OAuthToken:
        now = self._now()
        common = {
            "sub": advisor,
            "cid": client_id,
            "sc": scopes,
            "iat": now,
            "kf": self._code_fingerprint(advisor),
        }
        access = dict(
            common,
            exp=now + self.config.access_token_ttl,
            jti=secrets.token_urlsafe(9),
            aud=self.resource_url,
        )
        refresh = dict(
            common, exp=now + self.config.refresh_token_ttl, jti=secrets.token_urlsafe(9)
        )
        return OAuthToken(
            access_token=self._signer.sign("access", access),
            token_type="Bearer",
            expires_in=self.config.access_token_ttl,
            scope=" ".join(scopes),
            refresh_token=self._signer.sign("refresh", refresh),
        )

    # ----- clients (stateless dynamic registration) --------------------

    async def get_client(self, client_id: str) -> Optional[OAuthClientInformationFull]:
        payload = self._signer.unsign("client", client_id, check_expiry=False)
        if not payload:
            return None
        uris = payload.get("r")
        method = payload.get("m")
        if not isinstance(uris, list) or not uris or method not in _CLIENT_AUTH_METHODS:
            return None
        if not all(redirect_uri_allowed(u, self.config.allow_localhost) for u in uris):
            return None
        return OAuthClientInformationFull(
            client_id=client_id,
            client_secret=None if method == "none" else self._client_secret(client_id),
            client_id_issued_at=payload.get("iat"),
            redirect_uris=[AnyUrl(u) for u in uris],
            token_endpoint_auth_method=method,
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            scope=SCOPE,
        )

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        uris = [str(u) for u in client_info.redirect_uris or []]
        allowed = all(redirect_uri_allowed(u, self.config.allow_localhost) for u in uris)
        if not uris or not allowed:
            raise RegistrationError(
                "invalid_redirect_uri",
                "Only Claude's callback URLs (https://claude.ai/..., https://claude.com/...) "
                "may be registered",
            )
        method = client_info.token_endpoint_auth_method
        if method not in _CLIENT_AUTH_METHODS:
            raise RegistrationError(
                "invalid_client_metadata",
                f"token_endpoint_auth_method {method} is not supported",
            )
        # The SDK handler made up a random client_id/secret and returns this object
        # as the registration response. Replace them with a signed, self-contained
        # registration so the client survives server restarts without storage.
        client_id = self._signer.sign("client", {"r": uris, "m": method, "iat": self._now()})
        client_info.client_id = client_id
        client_info.client_secret = None if method == "none" else self._client_secret(client_id)
        client_info.client_secret_expires_at = None
        client_info.scope = SCOPE

    # ----- authorization -----------------------------------------------

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        if not redirect_uri_allowed(str(params.redirect_uri), self.config.allow_localhost):
            raise AuthorizeError("invalid_request", "redirect_uri is not allowed")
        if not self._resource_ok(params.resource):
            raise AuthorizeError("invalid_request", "resource does not match this server")
        scopes = params.scopes or [SCOPE]
        if any(scope != SCOPE for scope in scopes):
            raise AuthorizeError("invalid_scope", f"only the '{SCOPE}' scope exists")
        pending = {
            "c": client.client_id,
            "r": str(params.redirect_uri),
            "x": params.redirect_uri_provided_explicitly,
            "cc": params.code_challenge,
            "s": params.state,
            "sc": scopes,
            "res": params.resource,
            "exp": self._now() + LOGIN_REQUEST_TTL,
        }
        req = self._signer.sign("login", pending)
        return f"{self.config.issuer_url}{LOGIN_PATH}?req={req}"

    def _csrf_for(self, nonce: str, req: str) -> str:
        return self._signer.mac_hex("csrf", nonce, req)

    def _render(
        self,
        req: Optional[str],
        pending: Optional[dict],
        error: Optional[str] = None,
        status: int = 200,
    ) -> Response:
        error_html = f'<p class="error" role="alert">{html.escape(error)}</p>\n' if error else ""
        redirect_origin = None
        form_html = ""
        nonce = None
        if req and pending:
            redirect = urlsplit(pending["r"])
            redirect_origin = f"{redirect.scheme}://{redirect.netloc}"
            nonce = secrets.token_urlsafe(16)
            form_html = _FORM.format(
                action=html.escape(LOGIN_PATH),
                req=html.escape(req),
                csrf=html.escape(self._csrf_for(nonce, req)),
                field=html.escape(TEXT_FIELD),
                button=html.escape(TEXT_BUTTON),
                invalid=' aria-invalid="true"' if error else "",
            )
        body = _PAGE.format(title=html.escape(TEXT_TITLE), error=error_html, form=form_html)
        response = HTMLResponse(body, status_code=status, headers=_page_headers(redirect_origin))
        if nonce:
            response.set_cookie(
                _CSRF_COOKIE,
                nonce,
                max_age=LOGIN_REQUEST_TTL,
                path=LOGIN_PATH,
                secure=self._secure_cookie,
                httponly=True,
                samesite="strict",
            )
        return response

    async def handle_login(self, request: Request) -> Response:
        if request.method == "GET":
            req = request.query_params.get("req", "")
            pending = self._signer.unsign("login", req)
            if not pending:
                return self._render(None, None, TEXT_EXPIRED, status=400)
            return self._render(req, pending)
        return await self._handle_login_post(request)

    async def _handle_login_post(self, request: Request) -> Response:
        form = await request.form()
        req = str(form.get("req") or "")
        pending = self._signer.unsign("login", req)
        if not pending:
            return self._render(None, None, TEXT_EXPIRED, status=400)

        nonce = request.cookies.get(_CSRF_COOKIE, "")
        csrf = str(form.get("csrf") or "")
        expected = self._csrf_for(nonce, req).encode()
        # Compare bytes: compare_digest raises on non-ASCII str input.
        if not nonce or not hmac.compare_digest(csrf.encode("utf-8"), expected):
            return self._render(req, pending, TEXT_RETRY, status=400)

        ip = client_ip(request)
        if self.limiter.blocked(ip):
            access_logger.warning("login blocked (rate limit) ip=%s", ip)
            return self._render(req, pending, TEXT_TOO_MANY, status=429)

        advisor = match_access_code(str(form.get("code") or "")[:64], self.config.access_codes)
        if advisor is None:
            self.limiter.record_failure(ip)
            access_logger.warning("login failed ip=%s", ip)
            if self.failure_delay:
                await asyncio.sleep(self.failure_delay)
            return self._render(req, pending, TEXT_WRONG_CODE, status=401)

        return await self._complete_login(pending, advisor, ip)

    async def _complete_login(self, pending: dict, advisor: str, ip: str) -> Response:
        client = await self.get_client(pending["c"])
        if client is None or not redirect_uri_allowed(pending["r"], self.config.allow_localhost):
            return self._render(None, None, TEXT_EXPIRED, status=400)
        now = self._clock()
        for key in [k for k, c in self._auth_codes.items() if c.expires_at <= now]:
            del self._auth_codes[key]
        code = secrets.token_urlsafe(32)
        self._auth_codes[code] = AuthorizationCode(
            code=code,
            scopes=pending["sc"],
            expires_at=now + AUTH_CODE_TTL,
            client_id=pending["c"],
            code_challenge=pending["cc"],
            redirect_uri=AnyUrl(pending["r"]),
            redirect_uri_provided_explicitly=bool(pending["x"]),
            resource=pending.get("res"),
            subject=advisor,
        )
        access_logger.info("login ok advisor=%s ip=%s", advisor, ip)
        target = construct_redirect_uri(pending["r"], code=code, state=pending.get("s"))
        response = RedirectResponse(target, status_code=302, headers={"Cache-Control": "no-store"})
        response.delete_cookie(_CSRF_COOKIE, path=LOGIN_PATH)
        return response

    # ----- tokens ------------------------------------------------------

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> Optional[AuthorizationCode]:
        code = self._auth_codes.get(authorization_code)
        if code is None or code.client_id != client.client_id:
            return None
        return code

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        # pop() makes the code single-use even if two requests race.
        if self._auth_codes.pop(authorization_code.code, None) is None:
            raise TokenError("invalid_grant", "authorization code was already used")
        advisor = authorization_code.subject or ""
        if advisor not in self.config.access_codes:
            raise TokenError("invalid_grant", "access has been revoked")
        return self._issue_tokens(client.client_id or "", advisor, authorization_code.scopes)

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> Optional[RefreshToken]:
        payload = self._signer.unsign("refresh", refresh_token)
        if (
            not payload
            or payload.get("cid") != client.client_id
            or payload.get("jti") in self._spent
            or not self._advisor_active(payload)
        ):
            return None
        return RefreshToken(
            token=refresh_token,
            client_id=payload["cid"],
            scopes=list(payload.get("sc") or []),
            expires_at=payload["exp"],
            subject=payload["sub"],
            resource=self.resource_url,
        )

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        payload = self._signer.unsign("refresh", refresh_token.token)
        if not payload or payload.get("jti") in self._spent:
            raise TokenError("invalid_grant", "refresh token was already used")
        self._mark_spent(payload["jti"], payload["exp"])  # rotation: one use only
        return self._issue_tokens(client.client_id or "", payload["sub"], scopes)

    async def load_access_token(self, token: str) -> Optional[AccessToken]:
        payload = self._signer.unsign("access", token)
        if (
            not payload
            or payload.get("jti") in self._spent
            or payload.get("aud") != self.resource_url
            or not self._advisor_active(payload)
        ):
            return None
        return AccessToken(
            token=token,
            client_id=payload["cid"],
            scopes=list(payload.get("sc") or []),
            expires_at=payload["exp"],
            resource=payload["aud"],
            subject=payload["sub"],
        )

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        for purpose in ("access", "refresh"):
            payload = self._signer.unsign(purpose, token.token, check_expiry=False)
            if payload and isinstance(payload.get("jti"), str):
                self._mark_spent(payload["jti"], int(payload.get("exp") or 0))


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------

_SAFE_TOOL_NAME = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")


def _configure_access_logger() -> None:
    if access_logger.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter("%(asctime)sZ access %(message)s", "%Y-%m-%dT%H:%M:%S")
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    access_logger.addHandler(handler)
    access_logger.setLevel(logging.INFO)
    access_logger.propagate = False


def install_audit_hook(server: Any) -> None:
    """Log one line per tool call: advisor and tool name only, never arguments or results."""
    from mcp.types import CallToolRequest

    handlers = server._mcp_server.request_handlers
    original = handlers[CallToolRequest]
    if getattr(original, "_telegram_mcp_audit", False):
        return

    async def audited(req: Any) -> Any:
        token = get_access_token()
        advisor = token.subject if token and token.subject else "-"
        name = getattr(getattr(req, "params", None), "name", "")
        tool = name if isinstance(name, str) and _SAFE_TOOL_NAME.match(name) else "<invalid>"
        access_logger.info("tool_call advisor=%s tool=%s", advisor, tool)
        return await original(req)

    audited._telegram_mcp_audit = True  # type: ignore[attr-defined]
    handlers[CallToolRequest] = audited


def install_auth(server: Any, config: AuthConfig, transport: str) -> AccessCodeOAuthProvider:
    """Turn on OAuth for ``server`` (a FastMCP instance) before its HTTP app is built.

    FastMCP reads ``settings.auth`` and the provider when ``streamable_http_app()`` /
    ``sse_app()`` runs, so this must happen after MCP_HTTP_PATH has been applied
    and before the server starts. The resource URL (and with it the protected-resource
    metadata URL) follows whatever endpoint path is in effect.
    """
    from mcp.server.auth.provider import ProviderTokenVerifier
    from mcp.server.auth.settings import (
        AuthSettings,
        ClientRegistrationOptions,
        RevocationOptions,
    )

    settings = server.settings
    path = settings.streamable_http_path if transport == "http" else settings.sse_path
    resource_url = config.issuer_url + path
    provider = AccessCodeOAuthProvider(config, resource_url)

    settings.auth = AuthSettings(
        issuer_url=config.issuer_url,
        resource_server_url=resource_url,
        # The provider checks each token's audience itself (aud == resource_url).
        validate_token_resource=False,
        required_scopes=[SCOPE],
        client_registration_options=ClientRegistrationOptions(
            enabled=True, valid_scopes=[SCOPE], default_scopes=[SCOPE]
        ),
        revocation_options=RevocationOptions(enabled=True),
    )
    server._auth_server_provider = provider
    server._token_verifier = ProviderTokenVerifier(provider)
    server.custom_route(LOGIN_PATH, methods=["GET", "POST"], include_in_schema=False)(
        provider.handle_login
    )
    _configure_access_logger()
    install_audit_hook(server)
    print(
        f"OAuth login enabled for {len(config.access_codes)} advisor(s); "
        f"issuer {config.issuer_url}",
        file=sys.stderr,
    )
    return provider


def is_oauth_path(path: str) -> bool:
    return path in OAUTH_PATHS or path.startswith(OAUTH_PATH_PREFIXES)
