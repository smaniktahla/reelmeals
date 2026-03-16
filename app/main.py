import asyncio
import datetime
import hashlib
import io
import json
import os
import subprocess
import tempfile
import uuid
import zipfile

import anthropic
import httpx
import yt_dlp
from faster_whisper import WhisperModel
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Recipe Extractor")

# ── Config from environment ────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE     = os.getenv("WHISPER_DEVICE", "cpu")
TANDOOR_URL        = os.getenv("TANDOOR_URL", "").rstrip("/")
TANDOOR_TOKEN      = os.getenv("TANDOOR_TOKEN", "")
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
COOKIES_PATH       = "/app/cookies.txt"

# ── Cache config ───────────────────────────────────────────────────────────────
CACHE_MAX  = 3
CACHE_FILE = "/app/data/cache.json"

# ── In-memory stores ───────────────────────────────────────────────────────────
jobs:  dict[str, dict] = {}
cache: dict[str, dict] = {}

whisper_model: WhisperModel | None = None


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global whisper_model
    compute = "float16" if WHISPER_DEVICE == "cuda" else "int8"
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=compute)
    print(f"[whisper] Loaded model={WHISPER_MODEL_SIZE} device={WHISPER_DEVICE} compute={compute}")
    _load_settings()
    _load_cache()


# ── Cache helpers ──────────────────────────────────────────────────────────────
def _url_hash(url: str) -> str:
    return hashlib.sha256(url.strip().lower().encode()).hexdigest()[:16]


def _load_cache():
    global cache
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE) as f:
                cache = json.load(f)
            print(f"[cache] Loaded {len(cache)} cached recipes from disk")
    except Exception as e:
        print(f"[cache] Failed to load cache: {e}")
        cache = {}


def _save_cache():
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[cache] Failed to save cache: {e}")


def _load_settings():
    global CACHE_MAX
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE) as f:
                s = json.load(f)
            CACHE_MAX = int(s.get("cache_max", CACHE_MAX))
            print(f"[settings] cache_max={CACHE_MAX}")
    except Exception as e:
        print(f"[settings] Failed to load settings: {e}")


def _save_settings():
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, "w") as f:
            json.dump({"cache_max": CACHE_MAX}, f)
    except Exception as e:
        print(f"[settings] Failed to save settings: {e}")


def _add_to_cache(url: str, job_id: str, recipe: dict):
    url_hash = _url_hash(url)
    cache.pop(url_hash, None)
    while len(cache) >= CACHE_MAX:
        oldest = next(iter(cache))
        print(f"[cache] Evicting: {cache[oldest].get('name', oldest)}")
        del cache[oldest]
    cache[url_hash] = {
        "url":       url,
        "job_id":    job_id,
        "recipe":    recipe,
        "name":      recipe.get("name", "Unknown"),
        "cached_at": datetime.datetime.utcnow().isoformat(),
    }
    _save_cache()
    print(f"[cache] Stored: {recipe.get('name')} (hash={url_hash})")


def _get_from_cache(url: str) -> dict | None:
    return cache.get(_url_hash(url))


# ── Request/response models ────────────────────────────────────────────────────
class ExtractRequest(BaseModel):
    url: str

class TandoorPushRequest(BaseModel):
    job_id: str


# ── API endpoints ──────────────────────────────────────────────────────────────
@app.post("/api/extract")
async def extract(req: ExtractRequest, background_tasks: BackgroundTasks):
    cached = _get_from_cache(req.url)
    if cached:
        print(f"[cache] Hit: {req.url[:60]}")
        job_id = cached["job_id"]
        jobs[job_id] = {
            "status":    "done",
            "step":      "Complete! (loaded from cache)",
            "progress":  100,
            "recipe":    cached["recipe"],
            "thumbnail": cached.get("thumbnail"),
            "error":     None,
            "cached":    True,
            "cached_at": cached.get("cached_at"),
        }
        return {"job_id": job_id, "from_cache": True}

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status":    "running",
        "step":      "Starting…",
        "progress":  0,
        "recipe":    None,
        "thumbnail": None,
        "error":     None,
        "cached":    False,
    }
    background_tasks.add_task(run_pipeline, job_id, req.url)
    return {"job_id": job_id, "from_cache": False}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {"error": "Job not found"}
    # Don't send the full base64 thumbnail in status polls — too much data
    return {k: v for k, v in job.items() if k != "thumbnail"}


@app.get("/api/thumbnail/{job_id}")
async def get_thumbnail(job_id: str):
    """Return the thumbnail as a JPEG image."""
    job = jobs.get(job_id)
    if not job or not job.get("thumbnail"):
        return StreamingResponse(io.BytesIO(b""), media_type="image/jpeg", status_code=404)
    img_bytes = job["thumbnail"]
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")


@app.get("/api/cache")
async def list_cache():
    return [
        {
            "url_hash":  k,
            "name":      v.get("name"),
            "url":       v.get("url"),
            "job_id":    v.get("job_id"),
            "cached_at": v.get("cached_at"),
        }
        for k, v in cache.items()
    ]


@app.delete("/api/cache/{url_hash}")
async def delete_cache_entry(url_hash: str):
    if url_hash in cache:
        name = cache[url_hash].get("name")
        del cache[url_hash]
        _save_cache()
        return {"success": True, "deleted": name}
    return {"success": False, "error": "Not found"}


@app.delete("/api/cache")
async def clear_cache():
    count = len(cache)
    cache.clear()
    _save_cache()
    return {"success": True, "cleared": count}


@app.get("/api/settings")
async def get_settings():
    return {"cache_max": CACHE_MAX}


@app.post("/api/settings")
async def update_settings(body: dict):
    global CACHE_MAX
    if "cache_max" in body:
        val = int(body["cache_max"])
        if val < 1 or val > 50:
            return {"success": False, "error": "cache_max must be between 1 and 50"}
        CACHE_MAX = val
        _save_settings()
        print(f"[settings] cache_max updated to {CACHE_MAX}")
    return {"success": True, "cache_max": CACHE_MAX}


@app.post("/api/push-to-tandoor")
async def push_to_tandoor(req: TandoorPushRequest):
    job = jobs.get(req.job_id)
    if not job or not job.get("recipe"):
        return {"success": False, "error": "Job not found or recipe not ready"}

    recipe  = job["recipe"]
    headers = {"Authorization": f"Bearer {TANDOOR_TOKEN}"}

    ingredients = _build_ingredients(recipe)
    steps       = _build_steps(recipe, ingredients)

    payload = {
        "name":         recipe["name"],
        "description":  recipe.get("description", ""),
        "servings":     recipe.get("servings", 4),
        "working_time": recipe.get("prepTime", 0),
        "waiting_time": recipe.get("cookTime", 0),
        "keywords":     [{"name": k} for k in recipe.get("keywords", [])],
        "steps":        steps,
        "private":      False,
        "source_url":   "",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        # Step 1 — create recipe
        resp = await client.post(
            f"{TANDOOR_URL}/api/recipe/",
            json=payload,
            headers={**headers, "Content-Type": "application/json"},
        )

        if resp.status_code not in (200, 201):
            return {"success": False, "error": resp.text}

        data = resp.json()
        rid  = data.get("id")

        # Step 2 — upload thumbnail if available
        if rid and job.get("thumbnail"):
            try:
                img_bytes = job["thumbnail"]
                files = {"image": ("thumbnail.jpg", img_bytes, "image/jpeg")}
                await client.put(
                    f"{TANDOOR_URL}/api/recipe/{rid}/image/",
                    files=files,
                    headers=headers,
                )
                print(f"[tandoor] Uploaded thumbnail for recipe {rid}")
            except Exception as e:
                print(f"[tandoor] Thumbnail upload failed (non-fatal): {e}")

    return {"success": True, "recipe_id": rid, "url": f"{TANDOOR_URL}/view/recipe/{rid}"}


@app.get("/api/download/{job_id}")
async def download_recipe(job_id: str):
    job = jobs.get(job_id)
    if not job or not job.get("recipe"):
        return {"error": "Job not found or recipe not ready"}

    recipe      = job["recipe"]
    ingredients = _build_ingredients(recipe)
    steps       = _build_steps(recipe, ingredients)

    export = {
        "version": "1.0",
        "recipe": {
            "name":         recipe["name"],
            "description":  recipe.get("description", ""),
            "servings":     recipe.get("servings", 4),
            "working_time": recipe.get("prepTime", 0),
            "waiting_time": recipe.get("cookTime", 0),
            "keywords":     [{"name": k} for k in recipe.get("keywords", [])],
            "steps":        steps,
        },
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("recipe.json", json.dumps(export, indent=2, ensure_ascii=False))
        if job.get("thumbnail"):
            zf.writestr("thumbnail.jpg", job["thumbnail"])
    buf.seek(0)

    safe = recipe["name"].replace(" ", "_").replace("/", "-")[:50]
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{safe}.zip"'},
    )


# ── Pipeline ───────────────────────────────────────────────────────────────────
async def run_pipeline(job_id: str, url: str):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1 — Download video (not just audio)
            _update(job_id, "Downloading video…", 10)
            video_path = await _download_video(url, tmpdir)

            # 2 — Extract thumbnail
            _update(job_id, "Extracting thumbnail…", 30)
            thumbnail = await _extract_thumbnail(video_path, tmpdir)
            if thumbnail:
                jobs[job_id]["thumbnail"] = thumbnail
                print(f"[pipeline] Thumbnail extracted ({len(thumbnail)} bytes)")

            # 3 — Extract audio
            _update(job_id, "Extracting audio…", 45)
            audio_path = await _extract_audio(video_path, tmpdir)

            # 4 — Transcribe
            _update(job_id, "Transcribing audio…", 55)
            loop = asyncio.get_event_loop()
            transcript = await loop.run_in_executor(None, _transcribe, audio_path)

            # 5 — Parse recipe
            _update(job_id, "Extracting recipe with AI…", 80)
            recipe = await _parse_recipe(transcript)

            # 6 — Done
            jobs[job_id].update({
                "status":   "done",
                "step":     "Complete!",
                "progress": 100,
                "recipe":   recipe,
                "cached":   False,
            })
            _add_to_cache(url, job_id, recipe)

    except Exception as exc:
        jobs[job_id].update({"status": "error", "error": str(exc), "step": "Failed"})
        print(f"[pipeline] Error for {job_id}: {exc}")


def _update(job_id: str, step: str, progress: int):
    jobs[job_id]["step"]     = step
    jobs[job_id]["progress"] = progress


async def _download_video(url: str, tmpdir: str) -> str:
    """Download best video+audio merged file."""
    out_tmpl = os.path.join(tmpdir, "video.%(ext)s")
    opts = {
        "format":      "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl":     out_tmpl,
        "quiet":       True,
        "no_warnings": True,
        # Merge to mp4 so ffmpeg gets a reliable container
        "merge_output_format": "mp4",
    }
    if os.path.exists(COOKIES_PATH) and os.path.getsize(COOKIES_PATH) > 0:
        opts["cookiefile"] = COOKIES_PATH

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: _ydl_download(opts, url))

    # Find the downloaded file
    for fname in os.listdir(tmpdir):
        if fname.startswith("video."):
            return os.path.join(tmpdir, fname)
    raise RuntimeError("yt-dlp did not produce a video file")


async def _extract_thumbnail(video_path: str, tmpdir: str) -> bytes | None:
    """Grab a single frame at 5 seconds (or 10% into duration, whichever is less)."""
    try:
        thumb_path = os.path.join(tmpdir, "thumbnail.jpg")

        # Get video duration first
        probe = await asyncio.get_event_loop().run_in_executor(None, lambda: subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True
        ))
        duration = float(probe.stdout.strip() or "60")
        seek_secs = min(5.0, duration * 0.1)

        result = await asyncio.get_event_loop().run_in_executor(None, lambda: subprocess.run(
            ["ffmpeg", "-ss", str(seek_secs), "-i", video_path,
             "-frames:v", "1", "-q:v", "2", "-y", thumb_path],
            capture_output=True
        ))

        if result.returncode == 0 and os.path.exists(thumb_path):
            with open(thumb_path, "rb") as f:
                return f.read()
    except Exception as e:
        print(f"[thumbnail] Extraction failed (non-fatal): {e}")
    return None


async def _extract_audio(video_path: str, tmpdir: str) -> str:
    """Extract audio track from video to mp3."""
    audio_path = os.path.join(tmpdir, "audio.mp3")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: subprocess.run(
        ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", "-y", audio_path],
        capture_output=True
    ))
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr.decode()}")
    return audio_path


def _ydl_download(opts: dict, url: str):
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])


def _transcribe(audio_path: str) -> str:
    segments, _ = whisper_model.transcribe(audio_path, beam_size=5, language="en")
    return " ".join(s.text.strip() for s in segments)


async def _parse_recipe(transcript: str) -> dict:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = f"""You are a precise recipe extraction assistant. Given a transcript from a cooking video, extract the complete recipe and return it as valid JSON only — no markdown fences, no preamble, no explanation.

Return this exact structure:
{{
  "name": "Recipe Name",
  "description": "One or two sentences about the dish.",
  "servings": 4,
  "prepTime": 15,
  "cookTime": 30,
  "keywords": ["tag1", "tag2"],
  "ingredients": [
    {{"amount": 2.0, "unit": "cups", "food": "all-purpose flour", "note": ""}},
    {{"amount": 1.0, "unit": "tsp",  "food": "kosher salt",        "note": ""}},
    {{"amount": 0.0, "unit": "",     "food": "olive oil",           "note": "to taste"}}
  ],
  "steps": [
    {{"text": "Preheat oven to 375°F (190°C).", "time": 0}},
    {{"text": "Mix dry ingredients in a large bowl.", "time": 5}}
  ]
}}

Rules:
- amount: use 0.0 if not mentioned
- unit: empty string if not applicable
- step time: estimated minutes (0 if instantaneous)
- Include ALL ingredients and ALL steps mentioned

Transcript:
{transcript}"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ── Tandoor helpers ────────────────────────────────────────────────────────────
def _build_ingredients(recipe: dict) -> list:
    return [
        {
            "food":      {"name": ing["food"]},
            "unit":      {"name": ing["unit"]} if ing.get("unit") else None,
            "amount":    ing.get("amount", 0),
            "note":      ing.get("note", ""),
            "order":     0,
            "is_header": False,
        }
        for ing in recipe.get("ingredients", [])
    ]


def _build_steps(recipe: dict, ingredients: list) -> list:
    return [
        {
            "name":        "",
            "instruction": step["text"],
            "ingredients": ingredients if i == 0 else [],
            "time":        step.get("time", 0),
            "order":       i,
            "step_recipe": None,
        }
        for i, step in enumerate(recipe.get("steps", []))
    ]


# ── Static files ───────────────────────────────────────────────────────────────
app.mount("/", StaticFiles(directory="/app/static", html=True), name="static")
