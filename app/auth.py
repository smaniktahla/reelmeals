"""
Authentication module — local username/password + optional OIDC (Authentik, etc.)
"""

import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Optional

import httpx
from fastapi import Cookie, HTTPException, Request, Response

# ── Config ─────────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))

DATA_DIR = "/app/data"
GLOBAL_FILE = os.path.join(DATA_DIR, "global.json")


def _load_global() -> dict:
    try:
        if os.path.exists(GLOBAL_FILE):
            with open(GLOBAL_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _get_cfg(key: str, env_key: str, default: str = "") -> str:
    """Read from global.json first, fall back to .env, then default."""
    g = _load_global()
    val = g.get(key, "")
    if val:
        return val
    return os.getenv(env_key, default)


def _get_cfg_bool(key: str, env_key: str, default: bool = False) -> bool:
    g = _load_global()
    val = g.get(key)
    if val is not None:
        return bool(val)
    return os.getenv(env_key, str(default)).lower() == "true"


def get_auth_local() -> bool:
    return _get_cfg_bool("auth_local", "AUTH_LOCAL", True)

def get_auth_oidc() -> bool:
    return _get_cfg_bool("auth_oidc", "AUTH_OIDC", False)

def get_oidc_issuer() -> str:
    return _get_cfg("oidc_issuer", "OIDC_ISSUER").rstrip("/")

def get_oidc_client_id() -> str:
    return _get_cfg("oidc_client_id", "OIDC_CLIENT_ID")

def get_oidc_client_secret() -> str:
    return _get_cfg("oidc_client_secret", "OIDC_CLIENT_SECRET")

def get_oidc_redirect_uri() -> str:
    return _get_cfg("oidc_redirect_uri", "OIDC_REDIRECT_URI")

def get_oidc_scopes() -> str:
    return _get_cfg("oidc_scopes", "OIDC_SCOPES", "openid profile email")

def get_oidc_provider_name() -> str:
    return _get_cfg("oidc_provider_name", "OIDC_PROVIDER_NAME", "SSO")

# In-memory session store: token → {user_id, created_at, expires_at}
sessions: dict[str, dict] = {}

SESSION_MAX_AGE = 60 * 60 * 24 * 30  # 30 days


# ── Password hashing (bcrypt-like using pbkdf2 — no extra deps) ────────────────
def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000)
    return f"{salt}:{dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, dk_hex = stored.split(":", 1)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000)
        return hmac.compare_digest(dk.hex(), dk_hex)
    except Exception:
        return False


# ── Session management ─────────────────────────────────────────────────────────
def create_session(user_id: str) -> str:
    token = secrets.token_urlsafe(48)
    sessions[token] = {
        "user_id":    user_id,
        "created_at": time.time(),
        "expires_at": time.time() + SESSION_MAX_AGE,
    }
    return token


def get_session_user(token: str) -> Optional[str]:
    sess = sessions.get(token)
    if not sess:
        return None
    if time.time() > sess["expires_at"]:
        sessions.pop(token, None)
        return None
    return sess["user_id"]


def destroy_session(token: str):
    sessions.pop(token, None)


def set_session_cookie(response: Response, token: str):
    response.set_cookie(
        key="reelmeals_session",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=SESSION_MAX_AGE,
        path="/",
    )


def clear_session_cookie(response: Response):
    response.delete_cookie(key="reelmeals_session", path="/")


# ── Dependency: get current user from request ──────────────────────────────────
def get_current_user_id(request: Request) -> Optional[str]:
    token = request.cookies.get("reelmeals_session")
    if not token:
        return None
    return get_session_user(token)


def require_user(request: Request) -> str:
    user_id = get_current_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user_id


# ── OIDC helpers ───────────────────────────────────────────────────────────────
_oidc_config_cache: dict = {}


async def _get_oidc_config() -> dict:
    if _oidc_config_cache:
        return _oidc_config_cache
    issuer = get_oidc_issuer()
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{issuer}/.well-known/openid-configuration")
        resp.raise_for_status()
        _oidc_config_cache.update(resp.json())
    return _oidc_config_cache


async def get_oidc_authorize_url(state: str) -> str:
    cfg = await _get_oidc_config()
    params = {
        "client_id":     get_oidc_client_id(),
        "response_type": "code",
        "scope":         get_oidc_scopes(),
        "redirect_uri":  get_oidc_redirect_uri(),
        "state":         state,
    }
    from urllib.parse import urlencode
    return f"{cfg['authorization_endpoint']}?{urlencode(params)}"


async def exchange_oidc_code(code: str) -> dict:
    """Exchange authorization code for tokens, return userinfo dict."""
    cfg = await _get_oidc_config()
    async with httpx.AsyncClient(timeout=10) as client:
        token_resp = await client.post(
            cfg["token_endpoint"],
            data={
                "grant_type":    "authorization_code",
                "client_id":     get_oidc_client_id(),
                "client_secret": get_oidc_client_secret(),
                "code":          code,
                "redirect_uri":  get_oidc_redirect_uri(),
            },
        )
        token_resp.raise_for_status()
        tokens = token_resp.json()

        userinfo_resp = await client.get(
            cfg["userinfo_endpoint"],
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        userinfo_resp.raise_for_status()
        return userinfo_resp.json()


def get_auth_config() -> dict:
    """Return auth configuration for the frontend."""
    return {
        "local_enabled": get_auth_local(),
        "oidc_enabled":  get_auth_oidc(),
        "oidc_name":     get_oidc_provider_name(),
    }


def clear_oidc_cache():
    """Clear cached OIDC discovery config (call after settings change)."""
    _oidc_config_cache.clear()
