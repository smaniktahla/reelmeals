"""
User management — profiles, settings, and data directory management.
"""

import json
import os
import secrets
import time
from typing import Optional

from auth import hash_password, verify_password

DATA_DIR = "/app/data"
USERS_DIR = os.path.join(DATA_DIR, "users")
GLOBAL_FILE = os.path.join(DATA_DIR, "global.json")


def _ensure_dirs():
    os.makedirs(USERS_DIR, exist_ok=True)


def _user_dir(user_id: str) -> str:
    return os.path.join(USERS_DIR, user_id)


def _profile_path(user_id: str) -> str:
    return os.path.join(_user_dir(user_id), "profile.json")


def _settings_path(user_id: str) -> str:
    return os.path.join(_user_dir(user_id), "settings.json")


def _cookies_path(user_id: str) -> str:
    return os.path.join(_user_dir(user_id), "cookies.txt")


def _recipes_dir(user_id: str) -> str:
    return os.path.join(_user_dir(user_id), "recipes")


# ── Global settings ────────────────────────────────────────────────────────────
def load_global() -> dict:
    try:
        if os.path.exists(GLOBAL_FILE):
            with open(GLOBAL_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {"admins": [], "setup_complete": False}


def save_global(data: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(GLOBAL_FILE, "w") as f:
        json.dump(data, f, indent=2)


def is_setup_complete() -> bool:
    return load_global().get("setup_complete", False)


def is_admin(user_id: str) -> bool:
    return user_id in load_global().get("admins", [])


# ── User CRUD ──────────────────────────────────────────────────────────────────
def create_user(username: str, password: Optional[str] = None,
                email: str = "", display_name: str = "",
                oidc_sub: str = "", oidc_issuer: str = "") -> str:
    """Create a new user. Returns user_id."""
    _ensure_dirs()
    user_id = username.lower().replace(" ", "_")

    user_dir = _user_dir(user_id)
    if os.path.exists(user_dir):
        raise ValueError(f"User '{user_id}' already exists")

    os.makedirs(user_dir, exist_ok=True)
    os.makedirs(_recipes_dir(user_id), exist_ok=True)

    profile = {
        "user_id":      user_id,
        "username":     username,
        "display_name": display_name or username,
        "email":        email,
        "password_hash": hash_password(password) if password else "",
        "oidc_sub":     oidc_sub,
        "oidc_issuer":  oidc_issuer,
        "created_at":   time.time(),
    }
    with open(_profile_path(user_id), "w") as f:
        json.dump(profile, f, indent=2)

    # Default settings
    settings = {
        "llm_provider":    "anthropic",
        "anthropic_api_key": "",
        "openai_api_key":  "",
        "openai_model":    "gpt-4o",
        "tandoor_url":     "",
        "tandoor_token":   "",
        "mealie_url":      "",
        "mealie_token":    "",
        "cache_max":       50,
    }
    with open(_settings_path(user_id), "w") as f:
        json.dump(settings, f, indent=2)

    # Empty recipe index
    with open(os.path.join(_recipes_dir(user_id), "index.json"), "w") as f:
        json.dump([], f)

    # Mark first user as admin
    g = load_global()
    if not g.get("admins"):
        g["admins"] = [user_id]
        g["setup_complete"] = True
        save_global(g)
        print(f"[users] First user '{user_id}' set as admin")

    print(f"[users] Created user '{user_id}'")
    return user_id


def get_user(user_id: str) -> Optional[dict]:
    path = _profile_path(user_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def find_user_by_oidc(sub: str, issuer: str) -> Optional[dict]:
    """Find a user by OIDC subject + issuer."""
    _ensure_dirs()
    for uid in list_user_ids():
        user = get_user(uid)
        if user and user.get("oidc_sub") == sub and user.get("oidc_issuer") == issuer:
            return user
    return None


def authenticate_local(username: str, password: str) -> Optional[str]:
    """Verify username/password, return user_id or None."""
    user_id = username.lower().replace(" ", "_")
    user = get_user(user_id)
    if not user or not user.get("password_hash"):
        return None
    if verify_password(password, user["password_hash"]):
        return user_id
    return None


def list_user_ids() -> list[str]:
    _ensure_dirs()
    try:
        return [d for d in os.listdir(USERS_DIR)
                if os.path.isdir(os.path.join(USERS_DIR, d))]
    except Exception:
        return []


# ── Per-user settings ──────────────────────────────────────────────────────────
def get_settings(user_id: str) -> dict:
    path = _settings_path(user_id)
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {"cache_max": 50}


def update_settings(user_id: str, updates: dict) -> dict:
    settings = get_settings(user_id)
    # Only allow known keys
    allowed = {
        "llm_provider", "anthropic_api_key", "openai_api_key", "openai_model",
        "tandoor_url", "tandoor_token", "mealie_url", "mealie_token", "cache_max",
    }
    for k, v in updates.items():
        if k in allowed:
            settings[k] = v
    with open(_settings_path(user_id), "w") as f:
        json.dump(settings, f, indent=2)
    return settings


# ── Per-user cookies ───────────────────────────────────────────────────────────
def get_cookies_path(user_id: str) -> str:
    """Return path to user's cookies.txt (may not exist)."""
    return _cookies_path(user_id)


def save_cookies(user_id: str, content: str):
    path = _cookies_path(user_id)
    with open(path, "w") as f:
        f.write(content)
    print(f"[users] Saved cookies for '{user_id}' ({len(content)} bytes)")


def get_cookies(user_id: str) -> str:
    path = _cookies_path(user_id)
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return ""


# ── Migration: import v3 cache into a user's recipe library ───────────────────
def migrate_v3_cache(user_id: str, cache_file: str = "/app/data/cache.json"):
    """Import recipes from v3 global cache.json into user's library."""
    if not os.path.exists(cache_file):
        return 0

    try:
        with open(cache_file) as f:
            old_cache = json.load(f)
    except Exception as e:
        print(f"[migrate] Failed to read v3 cache: {e}")
        return 0

    from recipes import save_recipe
    count = 0
    for url_hash, entry in old_cache.items():
        recipe = entry.get("recipe")
        if not recipe:
            continue
        try:
            save_recipe(
                user_id,
                recipe,
                source_url=entry.get("url", ""),
                thumbnail_b64=entry.get("thumbnail_b64"),
            )
            count += 1
        except Exception as e:
            print(f"[migrate] Skipped recipe: {e}")

    if count:
        # Rename old cache to avoid re-migration
        os.rename(cache_file, cache_file + ".migrated")
        print(f"[migrate] Imported {count} recipes from v3 cache into '{user_id}'")
    return count
