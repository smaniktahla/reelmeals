"""
Recipe library — persistent JSON storage per user.
No eviction. Each recipe is a standalone JSON file.
"""

import base64
import datetime
import hashlib
import json
import os
import re
import time
from typing import Optional

DATA_DIR = "/app/data"


def _recipes_dir(user_id: str) -> str:
    return os.path.join(DATA_DIR, "users", user_id, "recipes")


def _index_path(user_id: str) -> str:
    return os.path.join(_recipes_dir(user_id), "index.json")


def _recipe_path(user_id: str, slug: str) -> str:
    return os.path.join(_recipes_dir(user_id), f"{slug}.json")


def _make_slug(name: str) -> str:
    """Generate a URL-safe slug from a recipe name."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9 -]", "", s)
    s = re.sub(r" +", "-", s)
    s = s[:60].rstrip("-")
    return s or "recipe"


def _unique_slug(user_id: str, base_slug: str) -> str:
    """Ensure slug is unique within user's library."""
    slug = base_slug
    counter = 1
    while os.path.exists(_recipe_path(user_id, slug)):
        slug = f"{base_slug}-{counter}"
        counter += 1
    return slug


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.strip().lower().encode()).hexdigest()[:16]


# ── Index management ───────────────────────────────────────────────────────────
def _load_index(user_id: str) -> list[dict]:
    path = _index_path(user_id)
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_index(user_id: str, index: list[dict]):
    os.makedirs(_recipes_dir(user_id), exist_ok=True)
    with open(_index_path(user_id), "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


# ── CRUD ───────────────────────────────────────────────────────────────────────
def save_recipe(user_id: str, recipe: dict, source_url: str = "",
                thumbnail_b64: Optional[str] = None,
                thumbnail_bytes: Optional[bytes] = None) -> str:
    """Save a recipe to the user's library. Returns the slug."""
    os.makedirs(_recipes_dir(user_id), exist_ok=True)

    name = recipe.get("name", "Untitled Recipe")
    base_slug = _make_slug(name)

    # Check if we already have this URL
    if source_url:
        existing = find_by_url(user_id, source_url)
        if existing:
            slug = existing["slug"]
            # Update existing
            entry = load_recipe(user_id, slug)
            if entry:
                entry["recipe"] = recipe
                entry["updated_at"] = datetime.datetime.utcnow().isoformat()
                if thumbnail_b64:
                    entry["thumbnail_b64"] = thumbnail_b64
                elif thumbnail_bytes:
                    entry["thumbnail_b64"] = base64.b64encode(thumbnail_bytes).decode()
                with open(_recipe_path(user_id, slug), "w") as f:
                    json.dump(entry, f, indent=2, ensure_ascii=False)
                # Update index
                _update_index_entry(user_id, slug, name)
                return slug

    slug = _unique_slug(user_id, base_slug)

    # Build thumbnail b64
    if thumbnail_bytes and not thumbnail_b64:
        thumbnail_b64 = base64.b64encode(thumbnail_bytes).decode()

    entry = {
        "slug":          slug,
        "source_url":    source_url,
        "url_hash":      _url_hash(source_url) if source_url else "",
        "recipe":        recipe,
        "thumbnail_b64": thumbnail_b64,
        "created_at":    datetime.datetime.utcnow().isoformat(),
        "updated_at":    datetime.datetime.utcnow().isoformat(),
    }

    with open(_recipe_path(user_id, slug), "w") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)

    # Append to index
    index = _load_index(user_id)
    index.append({
        "slug":       slug,
        "name":       name,
        "source_url": source_url,
        "url_hash":   entry["url_hash"],
        "created_at": entry["created_at"],
    })
    _save_index(user_id, index)

    print(f"[recipes] Saved '{name}' as {slug} for user {user_id}")
    return slug


def _update_index_entry(user_id: str, slug: str, name: str):
    index = _load_index(user_id)
    for item in index:
        if item["slug"] == slug:
            item["name"] = name
            break
    _save_index(user_id, index)


def load_recipe(user_id: str, slug: str) -> Optional[dict]:
    path = _recipe_path(user_id, slug)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def delete_recipe(user_id: str, slug: str) -> bool:
    path = _recipe_path(user_id, slug)
    if not os.path.exists(path):
        return False
    os.remove(path)
    index = _load_index(user_id)
    index = [i for i in index if i["slug"] != slug]
    _save_index(user_id, index)
    print(f"[recipes] Deleted {slug} for user {user_id}")
    return True


def list_recipes(user_id: str) -> list[dict]:
    """Return index entries (lightweight — no full recipe data)."""
    return _load_index(user_id)


def find_by_url(user_id: str, url: str) -> Optional[dict]:
    """Find a recipe in the index by source URL hash."""
    url_hash = _url_hash(url)
    for item in _load_index(user_id):
        if item.get("url_hash") == url_hash:
            return item
    return None


def get_recipe_count(user_id: str) -> int:
    return len(_load_index(user_id))


def search_recipes(user_id: str, query: str) -> list[dict]:
    """Simple name-based search."""
    q = query.lower().strip()
    if not q:
        return list_recipes(user_id)
    return [item for item in _load_index(user_id)
            if q in item.get("name", "").lower()]
