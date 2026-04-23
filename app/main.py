"""
ReelMeals v4 — multi-user recipe extractor with persistent JSON library.
"""

import base64
import io
import json
import os
import secrets
import uuid
import zipfile

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import auth
import integrations
import pipeline
import recipes
import users
from text_import import router as text_import_router

APP_VERSION = "4.0.0"

app = FastAPI(title="ReelMeals")
app.include_router(text_import_router)


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    pipeline.init_whisper()
    # Migrate v3 cache if first admin user exists and cache.json is present
    g = users.load_global()
    if g.get("admins") and os.path.exists("/app/data/cache.json"):
        users.migrate_v3_cache(g["admins"][0])


# ── Request models ─────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    email: str = ""
    display_name: str = ""

class ExtractRequest(BaseModel):
    url: str

class PushRequest(BaseModel):
    slug: str

class SettingsUpdate(BaseModel):
    llm_provider: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openai_model: str | None = None
    tandoor_url: str | None = None
    tandoor_token: str | None = None
    mealie_url: str | None = None
    mealie_token: str | None = None
    cache_max: int | None = None

class CookiesUpdate(BaseModel):
    content: str


# ── Auth routes ────────────────────────────────────────────────────────────────
@app.get("/api/auth/status")
async def auth_status(request: Request):
    """Return current auth state for the frontend."""
    user_id = auth.get_current_user_id(request)
    setup = users.is_setup_complete()
    cfg = auth.get_auth_config()
    if user_id:
        user = users.get_user(user_id)
        return {
            "authenticated": True,
            "user_id":       user_id,
            "display_name":  user.get("display_name", user_id) if user else user_id,
            "is_admin":      users.is_admin(user_id),
            "setup_complete": setup,
            **cfg,
        }
    return {"authenticated": False, "setup_complete": setup, **cfg}


@app.post("/api/auth/register")
async def register(req: RegisterRequest, response: Response):
    if not auth.get_auth_local():
        raise HTTPException(400, "Local registration is disabled")
    # Only allow registration if no users yet, or if setup is complete (open registration)
    if users.is_setup_complete() and not os.getenv("ALLOW_REGISTRATION", "true").lower() == "true":
        raise HTTPException(403, "Registration is closed")
    try:
        user_id = users.create_user(
            username=req.username,
            password=req.password,
            email=req.email,
            display_name=req.display_name,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    token = auth.create_session(user_id)
    auth.set_session_cookie(response, token)
    return {"success": True, "user_id": user_id}


@app.post("/api/auth/login")
async def login(req: LoginRequest, response: Response):
    if not auth.get_auth_local():
        raise HTTPException(400, "Local login is disabled")
    user_id = users.authenticate_local(req.username, req.password)
    if not user_id:
        raise HTTPException(401, "Invalid username or password")
    token = auth.create_session(user_id)
    auth.set_session_cookie(response, token)
    user = users.get_user(user_id)
    return {
        "success": True,
        "user_id": user_id,
        "display_name": user.get("display_name", user_id) if user else user_id,
    }


@app.post("/api/auth/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get("reelmeals_session")
    if token:
        auth.destroy_session(token)
    auth.clear_session_cookie(response)
    return {"success": True}


@app.get("/api/auth/oidc/login")
async def oidc_login():
    if not auth.get_auth_oidc():
        raise HTTPException(400, "OIDC is not enabled")
    state = secrets.token_urlsafe(32)
    # Store state in a simple in-memory dict (good enough for single-instance)
    auth.sessions[f"oidc_state:{state}"] = {"created_at": __import__("time").time()}
    url = await auth.get_oidc_authorize_url(state)
    return RedirectResponse(url)


@app.get("/api/auth/oidc/callback")
async def oidc_callback(request: Request, response: Response):
    code  = request.query_params.get("code")
    state = request.query_params.get("state")

    if not code or not state:
        raise HTTPException(400, "Missing code or state")

    state_key = f"oidc_state:{state}"
    if state_key not in auth.sessions:
        raise HTTPException(400, "Invalid or expired state")
    auth.sessions.pop(state_key, None)

    userinfo = await auth.exchange_oidc_code(code)
    sub    = userinfo.get("sub", "")
    email  = userinfo.get("email", "")
    name   = userinfo.get("preferred_username") or userinfo.get("name") or email.split("@")[0]

    # Find or create user
    user = users.find_user_by_oidc(sub, auth.get_oidc_issuer())
    if user:
        user_id = user["user_id"]
    else:
        try:
            user_id = users.create_user(
                username=name,
                email=email,
                display_name=userinfo.get("name", name),
                oidc_sub=sub,
                oidc_issuer=auth.get_oidc_issuer(),
            )
        except ValueError:
            # User with same name exists — link by appending suffix
            user_id = users.create_user(
                username=f"{name}_{sub[:6]}",
                email=email,
                display_name=userinfo.get("name", name),
                oidc_sub=sub,
                oidc_issuer=auth.get_oidc_issuer(),
            )

    token = auth.create_session(user_id)
    resp = RedirectResponse("/", status_code=302)
    auth.set_session_cookie(resp, token)
    return resp


# ── Config / settings routes ───────────────────────────────────────────────────
@app.get("/api/config")
async def get_config(request: Request):
    user_id = auth.get_current_user_id(request)
    if not user_id:
        return {"authenticated": False}
    settings = users.get_settings(user_id)
    return {
        "authenticated":   True,
        "tandoor_enabled": bool(settings.get("tandoor_url") and settings.get("tandoor_token")),
        "mealie_enabled":  bool(settings.get("mealie_url") and settings.get("mealie_token")),
        "llm_provider":    settings.get("llm_provider", "anthropic"),
        "llm_configured":  bool(
            (settings.get("llm_provider") == "anthropic" and settings.get("anthropic_api_key"))
            or (settings.get("llm_provider") == "openai" and settings.get("openai_api_key"))
        ),
        "version":         APP_VERSION,
    }


@app.get("/api/settings")
async def get_settings(request: Request):
    user_id = auth.require_user(request)
    settings = users.get_settings(user_id)
    # Mask API keys for display
    masked = dict(settings)
    for key in ("anthropic_api_key", "openai_api_key", "tandoor_token", "mealie_token"):
        val = masked.get(key, "")
        if val and len(val) > 8:
            masked[key] = val[:4] + "…" + val[-4:]
        elif val:
            masked[key] = "••••"
    masked["version"] = APP_VERSION
    masked["has_anthropic_key"] = bool(settings.get("anthropic_api_key"))
    masked["has_openai_key"]    = bool(settings.get("openai_api_key"))
    masked["has_tandoor_token"] = bool(settings.get("tandoor_token"))
    masked["has_mealie_token"]  = bool(settings.get("mealie_token"))
    return masked


@app.post("/api/settings")
async def update_settings(req: SettingsUpdate, request: Request):
    user_id = auth.require_user(request)
    updates = {k: v for k, v in req.dict().items() if v is not None}
    # Don't overwrite keys with masked values
    for key in ("anthropic_api_key", "openai_api_key", "tandoor_token", "mealie_token"):
        if key in updates and "…" in str(updates[key]):
            del updates[key]
    settings = users.update_settings(user_id, updates)
    return {"success": True}


# ── Cookies routes ─────────────────────────────────────────────────────────────
@app.get("/api/cookies")
async def get_cookies(request: Request):
    user_id = auth.require_user(request)
    content = users.get_cookies(user_id)
    has_cookies = bool(content.strip())
    line_count = len([l for l in content.strip().split("\n") if l and not l.startswith("#")]) if has_cookies else 0
    return {"has_cookies": has_cookies, "line_count": line_count, "content": content}


@app.post("/api/cookies")
async def save_cookies(req: CookiesUpdate, request: Request):
    user_id = auth.require_user(request)
    users.save_cookies(user_id, req.content)
    return {"success": True}


# ── Recipe library routes ──────────────────────────────────────────────────────
@app.get("/api/recipes")
async def list_recipes(request: Request, q: str = ""):
    user_id = auth.require_user(request)
    if q:
        items = recipes.search_recipes(user_id, q)
    else:
        items = recipes.list_recipes(user_id)
    return {"recipes": items, "total": len(items)}


class TextRecipeCreate(BaseModel):
    title: str = ""
    name: str = ""
    description: str = ""
    servings: int | None = None
    prep_time: int | None = None
    cook_time: int | None = None
    total_time: int | None = None
    source: str = ""
    ingredients: list = []
    steps: list = []
    tags: list = []


@app.post("/api/recipes")
async def create_recipe(req: TextRecipeCreate, request: Request):
    user_id = auth.require_user(request)
    # Normalize: text_import uses "title", pipeline uses "name"
    recipe_dict = req.model_dump()
    if not recipe_dict.get("name") and recipe_dict.get("title"):
        recipe_dict["name"] = recipe_dict["title"]
    slug = recipes.save_recipe(user_id, recipe_dict, source_url="")
    return {"slug": slug, "id": slug}


@app.get("/api/recipes/{slug}")
async def get_recipe(slug: str, request: Request):
    user_id = auth.require_user(request)
    entry = recipes.load_recipe(user_id, slug)
    if not entry:
        raise HTTPException(404, "Recipe not found")
    return entry


@app.delete("/api/recipes/{slug}")
async def delete_recipe(slug: str, request: Request):
    user_id = auth.require_user(request)
    ok = recipes.delete_recipe(user_id, slug)
    return {"success": ok}


@app.get("/api/recipes/{slug}/thumbnail")
async def get_recipe_thumbnail(slug: str, request: Request):
    user_id = auth.require_user(request)
    entry = recipes.load_recipe(user_id, slug)
    if not entry or not entry.get("thumbnail_b64"):
        return StreamingResponse(io.BytesIO(b""), media_type="image/jpeg", status_code=404)
    img = base64.b64decode(entry["thumbnail_b64"])
    return StreamingResponse(io.BytesIO(img), media_type="image/jpeg")


# ── Extract route ──────────────────────────────────────────────────────────────
@app.post("/api/extract")
async def extract(req: ExtractRequest, request: Request, background_tasks: BackgroundTasks):
    user_id = auth.require_user(request)
    settings = users.get_settings(user_id)

    # Check LLM is configured
    provider = settings.get("llm_provider", "anthropic")
    if provider == "anthropic" and not settings.get("anthropic_api_key"):
        return {"error": "Anthropic API key not configured. Go to Settings."}
    if provider == "openai" and not settings.get("openai_api_key"):
        return {"error": "OpenAI API key not configured. Go to Settings."}

    # Check if already in library
    existing = recipes.find_by_url(user_id, req.url)
    if existing:
        slug = existing["slug"]
        entry = recipes.load_recipe(user_id, slug)
        if entry:
            job_id = str(uuid.uuid4())
            thumbnail_bytes = None
            if entry.get("thumbnail_b64"):
                thumbnail_bytes = base64.b64decode(entry["thumbnail_b64"])
            pipeline.jobs[job_id] = {
                "status":    "done",
                "step":      "Complete! (from library)",
                "progress":  100,
                "recipe":    entry["recipe"],
                "thumbnail": thumbnail_bytes,
                "error":     None,
                "cached":    True,
                "slug":      slug,
                "user_id":   user_id,
            }
            return {"job_id": job_id, "from_cache": True, "slug": slug}

    job_id = str(uuid.uuid4())
    pipeline.jobs[job_id] = {
        "status":    "running",
        "step":      "Starting…",
        "progress":  0,
        "recipe":    None,
        "thumbnail": None,
        "error":     None,
        "cached":    False,
        "user_id":   user_id,
        "source_url": req.url,
    }

    cookies_path = users.get_cookies_path(user_id)
    background_tasks.add_task(
        _run_and_save, job_id, req.url, user_id, cookies_path, settings
    )
    return {"job_id": job_id, "from_cache": False}


async def _run_and_save(job_id: str, url: str, user_id: str,
                        cookies_path: str, llm_settings: dict):
    """Run pipeline then save to user's library."""
    await pipeline.run_pipeline(job_id, url, cookies_path, llm_settings)
    job = pipeline.jobs.get(job_id)
    if job and job.get("status") == "done" and job.get("recipe"):
        slug = recipes.save_recipe(
            user_id,
            job["recipe"],
            source_url=url,
            thumbnail_bytes=job.get("thumbnail"),
        )
        job["slug"] = slug


# ── Status / thumbnail ─────────────────────────────────────────────────────────
@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    job = pipeline.jobs.get(job_id)
    if not job:
        return {"error": "Job not found"}
    return {k: v for k, v in job.items() if k not in ("thumbnail", "user_id")}


@app.get("/api/thumbnail/{job_id}")
async def get_thumbnail(job_id: str):
    job = pipeline.jobs.get(job_id)
    if not job or not job.get("thumbnail"):
        return StreamingResponse(io.BytesIO(b""), media_type="image/jpeg", status_code=404)
    return StreamingResponse(io.BytesIO(job["thumbnail"]), media_type="image/jpeg")

# ── Push routes ────────────────────────────────────────────────────────────────
@app.post("/api/push-to-tandoor")
async def push_tandoor(req: PushRequest, request: Request):
    user_id = auth.require_user(request)
    settings = users.get_settings(user_id)
    entry = recipes.load_recipe(user_id, req.slug)
    if not entry:
        return {"success": False, "error": "Recipe not found"}
    thumbnail = base64.b64decode(entry["thumbnail_b64"]) if entry.get("thumbnail_b64") else None
    return await integrations.push_to_tandoor(
        entry["recipe"], thumbnail,
        settings.get("tandoor_url", ""), settings.get("tandoor_token", ""),
    )


@app.post("/api/push-to-mealie")
async def push_mealie(req: PushRequest, request: Request):
    user_id = auth.require_user(request)
    settings = users.get_settings(user_id)
    entry = recipes.load_recipe(user_id, req.slug)
    if not entry:
        return {"success": False, "error": "Recipe not found"}
    thumbnail = base64.b64decode(entry["thumbnail_b64"]) if entry.get("thumbnail_b64") else None
    return await integrations.push_to_mealie(
        entry["recipe"], thumbnail,
        settings.get("mealie_url", ""), settings.get("mealie_token", ""),
    )


# ── Download ZIP ───────────────────────────────────────────────────────────────
@app.get("/api/download/{slug}")
async def download_recipe(slug: str, request: Request):
    user_id = auth.require_user(request)
    entry = recipes.load_recipe(user_id, slug)
    if not entry:
        return {"error": "Recipe not found"}

    recipe = entry["recipe"]
    export = {
        "version": "1.0",
        "recipe":  recipe,
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("recipe.json", json.dumps(export, indent=2, ensure_ascii=False))
        if entry.get("thumbnail_b64"):
            zf.writestr("thumbnail.jpg", base64.b64decode(entry["thumbnail_b64"]))
    buf.seek(0)

    safe = recipe.get("name", "recipe").replace(" ", "_").replace("/", "-")[:50]
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{safe}.zip"'},
    )


# ── Admin routes ───────────────────────────────────────────────────────────────
def _require_admin(request: Request) -> str:
    user_id = auth.require_user(request)
    if not users.is_admin(user_id):
        raise HTTPException(403, "Admin access required")
    return user_id


@app.get("/api/admin/users")
async def admin_list_users(request: Request):
    _require_admin(request)
    result = []
    for uid in users.list_user_ids():
        user = users.get_user(uid)
        if not user:
            continue
        count = recipes.get_recipe_count(uid)
        result.append({
            "user_id":      uid,
            "username":     user.get("username", uid),
            "display_name": user.get("display_name", uid),
            "email":        user.get("email", ""),
            "is_admin":     users.is_admin(uid),
            "recipe_count": count,
            "created_at":   user.get("created_at"),
        })
    return {"users": result}


@app.delete("/api/admin/users/{user_id}")
async def admin_delete_user(user_id: str, request: Request):
    admin_id = _require_admin(request)
    if user_id == admin_id:
        raise HTTPException(400, "Cannot delete yourself")
    user = users.get_user(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    # Remove user directory
    import shutil
    user_dir = os.path.join("/app/data/users", user_id)
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)
    # Remove from admins if applicable
    g = users.load_global()
    if user_id in g.get("admins", []):
        g["admins"].remove(user_id)
        users.save_global(g)
    return {"success": True, "deleted": user_id}


@app.post("/api/admin/users/{user_id}/toggle-admin")
async def admin_toggle_admin(user_id: str, request: Request):
    admin_id = _require_admin(request)
    user = users.get_user(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    g = users.load_global()
    admins = g.get("admins", [])
    if user_id in admins:
        if user_id == admin_id:
            raise HTTPException(400, "Cannot remove your own admin status")
        admins.remove(user_id)
    else:
        admins.append(user_id)
    g["admins"] = admins
    users.save_global(g)
    return {"success": True, "is_admin": user_id in admins}


@app.post("/api/admin/users/{user_id}/reset-password")
async def admin_reset_password(user_id: str, body: dict, request: Request):
    _require_admin(request)
    new_password = body.get("password", "")
    if not new_password or len(new_password) < 4:
        raise HTTPException(400, "Password must be at least 4 characters")
    user = users.get_user(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    user["password_hash"] = auth.hash_password(new_password)
    with open(users._profile_path(user_id), "w") as f:
        json.dump(user, f, indent=2)
    return {"success": True}


@app.post("/api/admin/create-user")
async def admin_create_user(req: RegisterRequest, request: Request):
    _require_admin(request)
    try:
        user_id = users.create_user(
            username=req.username,
            password=req.password,
            email=req.email,
            display_name=req.display_name,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"success": True, "user_id": user_id}


@app.get("/api/admin/auth-config")
async def admin_get_auth_config(request: Request):
    _require_admin(request)
    g = users.load_global()
    # Return current effective values (global.json overrides .env)
    return {
        "auth_local":         auth.get_auth_local(),
        "auth_oidc":          auth.get_auth_oidc(),
        "oidc_issuer":        auth.get_oidc_issuer(),
        "oidc_client_id":     auth.get_oidc_client_id(),
        "oidc_client_secret": _mask(auth.get_oidc_client_secret()),
        "oidc_redirect_uri":  auth.get_oidc_redirect_uri(),
        "oidc_provider_name": auth.get_oidc_provider_name(),
        "oidc_scopes":        auth.get_oidc_scopes(),
        "has_oidc_secret":    bool(auth.get_oidc_client_secret()),
        "allow_registration": os.getenv("ALLOW_REGISTRATION", "true").lower() == "true"
                              or g.get("allow_registration", True),
    }


def _mask(val: str) -> str:
    if not val:
        return ""
    if len(val) > 8:
        return val[:4] + "…" + val[-4:]
    return "••••"


@app.post("/api/admin/auth-config")
async def admin_update_auth_config(body: dict, request: Request):
    _require_admin(request)
    g = users.load_global()

    allowed_keys = {
        "auth_local", "auth_oidc", "oidc_issuer", "oidc_client_id",
        "oidc_client_secret", "oidc_redirect_uri", "oidc_provider_name",
        "oidc_scopes", "allow_registration",
    }
    for k, v in body.items():
        if k in allowed_keys:
            # Don't overwrite secret with masked value
            if k == "oidc_client_secret" and isinstance(v, str) and "…" in v:
                continue
            g[k] = v

    users.save_global(g)
    # Clear OIDC discovery cache so new issuer takes effect
    auth.clear_oidc_cache()
    return {"success": True}


# ── Static files (must be last) ───────────────────────────────────────────────
app.mount("/", StaticFiles(directory="/app/static", html=True), name="static")
