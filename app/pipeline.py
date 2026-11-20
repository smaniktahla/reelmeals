"""
Pipeline — download video → extract thumbnail → transcribe → AI parse recipe.
"""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile

import httpx

import anthropic
import yt_dlp
from faster_whisper import WhisperModel

# ── Whisper setup ──────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE     = os.getenv("WHISPER_DEVICE", "cpu")

# ── Local LLM setup ─────────────────────────────────────────────────────────────
# Points at AI1's dedicated llama-server (real 32K context), not Ollama's
# default instance on AI2 — that one silently caps total context at 4096
# tokens regardless of the max_tokens requested, which isn't enough for a
# reasoning model to both "think" through a full transcript and emit the
# recipe JSON.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://10.10.10.105:8081/v1")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen35-4b")

whisper_model: WhisperModel | None = None


def init_whisper():
    global whisper_model
    compute = "float16" if WHISPER_DEVICE == "cuda" else "int8"
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=compute)
    print(f"[whisper] Loaded model={WHISPER_MODEL_SIZE} device={WHISPER_DEVICE} compute={compute}")


# ── Job store (in-memory, keyed by job_id) ─────────────────────────────────────
jobs: dict[str, dict] = {}


def _update(job_id: str, step: str, progress: int):
    if job_id in jobs:
        jobs[job_id]["step"]     = step
        jobs[job_id]["progress"] = progress


# ── Full pipeline ──────────────────────────────────────────────────────────────
async def run_pipeline(job_id: str, url: str, cookies_path: str = "",
                       llm_settings: dict | None = None):
    """Run the full extraction pipeline. llm_settings = user's LLM config.

    Source material = video transcript + post caption + top comments. Any of
    them may carry the recipe, so a failed video download no longer kills the
    job as long as the caption/comments (metadata) could still be fetched.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            _update(job_id, "Downloading video…", 10)
            video_path, info = await _fetch_media(url, tmpdir, cookies_path)

            transcript = ""
            if video_path:
                _update(job_id, "Extracting thumbnail…", 30)
                thumbnail = await _extract_thumbnail(video_path, tmpdir)
                if thumbnail:
                    jobs[job_id]["thumbnail"] = thumbnail

                _update(job_id, "Extracting audio…", 45)
                try:
                    audio_path = await _extract_audio(video_path, tmpdir)
                    _update(job_id, "Transcribing audio…", 55)
                    loop = asyncio.get_event_loop()
                    transcript = await loop.run_in_executor(None, _transcribe, audio_path)
                except RuntimeError as e:
                    # No audio track etc. — the caption/comments may still have it
                    print(f"[pipeline] Audio step skipped: {e}")
            elif info.get("thumbnail"):
                jobs[job_id]["thumbnail"] = await _fetch_url_bytes(info["thumbnail"])

            source_text = build_source_text(info, transcript)
            _update(job_id, "Extracting recipe with AI…", 80)
            recipe = await _parse_recipe(source_text, llm_settings or {})

            jobs[job_id].update({
                "status":   "done",
                "step":     "Complete!",
                "progress": 100,
                "recipe":   recipe,
                "cached":   False,
            })

    except Exception as exc:
        jobs[job_id].update({"status": "error", "error": _friendly_error(url, cookies_path, exc),
                             "step": "Failed"})
        print(f"[pipeline] Error for {job_id}: {exc}")


# ── Error hints ────────────────────────────────────────────────────────────────
def _cookie_names(cookies_path: str, domain_part: str) -> set[str]:
    names: set[str] = set()
    try:
        with open(cookies_path) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 7 and domain_part in parts[0]:
                    names.add(parts[5])
    except OSError:
        pass
    return names


def _friendly_error(url: str, cookies_path: str, exc: Exception) -> str:
    msg = str(exc)
    if "instagram.com" in url and ("login required" in msg or "not available" in msg):
        if "sessionid" not in _cookie_names(cookies_path, "instagram.com"):
            return ("Instagram needs a logged-in session. The cookies saved in Settings have no "
                    "Instagram 'sessionid' cookie — export cookies.txt again while logged in to "
                    "instagram.com (cookies from a logged-out browser or from facebook.com alone "
                    "won't work) and paste it into Settings. Original error: " + msg)
    return msg


# ── Caption / comments → text ──────────────────────────────────────────────────
MAX_COMMENTS = 40
MAX_COMMENT_CHARS = 6000


def build_source_text(info: dict, transcript: str) -> str:
    """Combine caption, comments and transcript into one labelled blob."""
    parts = []
    title = (info.get("title") or "").strip()
    desc = (info.get("description") or "").strip()
    if desc or title:
        # IG/TikTok titles are usually just a truncated caption — skip if redundant
        head = desc if desc.startswith(title[:40]) or not title else f"{title}\n{desc}"
        parts.append("=== POST CAPTION / DESCRIPTION ===\n" + head)

    comments = info.get("comments") or []
    if comments:
        # Creator's own comments first (recipes are often pinned/posted there),
        # then the most-liked. Keep the long ones: recipes are long.
        def rank(c):
            return (not c.get("author_is_uploader"), -(c.get("like_count") or 0), -len(c.get("text") or ""))
        lines, used = [], 0
        for c in sorted(comments, key=rank)[:MAX_COMMENTS]:
            text = (c.get("text") or "").strip()
            if not text or used + len(text) > MAX_COMMENT_CHARS:
                continue
            who = "CREATOR" if c.get("author_is_uploader") else (c.get("author") or "viewer")
            lines.append(f"[{who}] {text}")
            used += len(text)
        if lines:
            parts.append("=== COMMENTS ===\n" + "\n---\n".join(lines))

    if transcript and transcript.strip():
        parts.append("=== SPOKEN TRANSCRIPT ===\n" + transcript.strip())
    return "\n\n".join(parts)


async def _fetch_url_bytes(url: str) -> bytes | None:
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            r = await client.get(url)
            if r.status_code == 200:
                return r.content
    except Exception as e:
        print(f"[thumbnail] fetch failed (non-fatal): {e}")
    return None


# ── Video download + metadata ──────────────────────────────────────────────────
async def _fetch_media(url: str, tmpdir: str, cookies_path: str = "") -> tuple[str | None, dict]:
    """Return (video_path | None, info). info carries caption + comments.

    Tries the full download (with comments). If that fails, falls back to a
    metadata-only fetch so a recipe in the caption/comments can still be used.
    Raises the original download error if neither yields anything.
    """
    out_tmpl = os.path.join(tmpdir, "video.%(ext)s")
    opts = {
        "format":              "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl":             out_tmpl,
        "quiet":               True,
        "no_warnings":         True,
        "merge_output_format": "mp4",
        "getcomments":         True,
        # YouTube: max_comments,max_parents,max_replies,max_replies_per_thread
        "extractor_args":      {"youtube": {"max_comments": [f"{MAX_COMMENTS}", f"{MAX_COMMENTS}", "0", "0"]}},
    }
    # yt-dlp rewrites its cookiefile on exit — work on a copy so the user's
    # saved cookies are never clobbered.
    if cookies_path and os.path.exists(cookies_path) and os.path.getsize(cookies_path) > 0:
        tmp_cookies = os.path.join(tmpdir, "cookies.txt")
        shutil.copyfile(cookies_path, tmp_cookies)
        opts["cookiefile"] = tmp_cookies

    loop = asyncio.get_event_loop()
    try:
        info = await loop.run_in_executor(None, lambda: _ydl_extract(opts, url, download=True))
    except Exception as dl_err:
        print(f"[pipeline] Download failed, trying metadata only: {dl_err}")
        meta_opts = {k: v for k, v in opts.items() if k != "format"}
        meta_opts["ignore_no_formats_error"] = True
        try:
            info = await loop.run_in_executor(None, lambda: _ydl_extract(meta_opts, url, download=False))
        except Exception:
            raise dl_err
        if not (info.get("description") or info.get("comments")):
            raise dl_err
        return None, info

    for fname in os.listdir(tmpdir):
        if fname.startswith("video."):
            return os.path.join(tmpdir, fname), info
    raise RuntimeError("yt-dlp did not produce a video file")


def _ydl_extract(opts: dict, url: str, download: bool) -> dict:
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.sanitize_info(ydl.extract_info(url, download=download)) or {}


# ── Thumbnail ──────────────────────────────────────────────────────────────────
async def _extract_thumbnail(video_path: str, tmpdir: str) -> bytes | None:
    try:
        thumb_path = os.path.join(tmpdir, "thumbnail.jpg")
        loop = asyncio.get_event_loop()
        probe = await loop.run_in_executor(None, lambda: subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True
        ))
        duration  = float(probe.stdout.strip() or "60")
        seek_secs = min(5.0, duration * 0.1)
        result = await loop.run_in_executor(None, lambda: subprocess.run(
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


# ── Audio extraction ───────────────────────────────────────────────────────────
async def _extract_audio(video_path: str, tmpdir: str) -> str:
    audio_path = os.path.join(tmpdir, "audio.mp3")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: subprocess.run(
        ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", "-y", audio_path],
        capture_output=True
    ))
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr.decode()}")
    return audio_path


# ── Transcription ──────────────────────────────────────────────────────────────
def _transcribe(audio_path: str) -> str:
    segments, _ = whisper_model.transcribe(audio_path, beam_size=5, language="en")
    text = " ".join(s.text.strip() for s in segments)
    print(f"[whisper] Transcript length: {len(text)} chars")
    if len(text) < 20:
        print(f"[whisper] WARNING: Very short transcript: '{text}'")
    return text


# ── LLM recipe parsing ────────────────────────────────────────────────────────
RECIPE_PROMPT = """You are a precise recipe extraction assistant. Given source material from a cooking video post — any of: the post caption, viewer/creator comments, and a spoken-audio transcript — extract the complete recipe and return it as valid JSON only — no markdown fences, no preamble, no explanation.

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
- The recipe may be written in the caption or a comment (often by the CREATOR) rather than spoken. Prefer exact written quantities over approximate spoken ones, and merge the sources into one recipe
- Ignore social-media noise (calls to action, "comment X for the recipe", hashtags, unrelated chatter)

Source material:
{transcript}"""


async def _parse_recipe(transcript: str, llm_settings: dict) -> dict:
    provider = llm_settings.get("llm_provider", "anthropic")
    if provider == "openai":
        return await _parse_openai(transcript, llm_settings)
    if provider == "local":
        return await _parse_local(transcript, llm_settings)
    return await _parse_anthropic(transcript, llm_settings)


async def _parse_anthropic(transcript: str, llm_settings: dict) -> dict:
    api_key = llm_settings.get("anthropic_api_key", "")
    if not api_key:
        raise RuntimeError("Anthropic API key not configured. Go to Settings to add it.")
    if not transcript or len(transcript.strip()) < 10:
        raise RuntimeError(f"Transcript is empty or too short ({len(transcript)} chars). The video may have no speech audio and no recipe in its caption/comments, or cookies may be needed for this site.")
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": RECIPE_PROMPT.format(transcript=transcript)}],
    )
    raw = message.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    print(f"[llm] Response length: {len(raw)} chars, starts with: {raw[:80]!r}")
    if not raw:
        raise RuntimeError("LLM returned an empty response. The transcript may not contain a recipe.")
    return json.loads(raw)


async def _parse_openai(transcript: str, llm_settings: dict) -> dict:
    api_key = llm_settings.get("openai_api_key", "")
    model   = llm_settings.get("openai_model", "gpt-4o")
    if not api_key:
        raise RuntimeError("OpenAI API key not configured. Go to Settings to add it.")
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise RuntimeError("openai package not installed.")
    client = AsyncOpenAI(api_key=api_key)
    response = await client.chat.completions.create(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": RECIPE_PROMPT.format(transcript=transcript)}],
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


async def _parse_local(transcript: str, llm_settings: dict) -> dict:
    if not transcript or len(transcript.strip()) < 10:
        raise RuntimeError(f"Transcript is empty or too short ({len(transcript)} chars). The video may have no speech audio and no recipe in its caption/comments, or cookies may be needed for this site.")
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise RuntimeError("openai package not installed.")
    client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    response = await client.chat.completions.create(
        model=OLLAMA_MODEL,
        # Reasoning models spend a large chunk of this "thinking" before
        # writing the answer — a full video transcript needs ~15-20k tokens
        # of headroom, not the 2k that's fine for a short text snippet.
        max_tokens=20000,
        messages=[{"role": "user", "content": RECIPE_PROMPT.format(transcript=transcript)}],
    )
    raw = (response.choices[0].message.content or "").strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    print(f"[llm] Response length: {len(raw)} chars, starts with: {raw[:80]!r}")
    if not raw:
        raise RuntimeError("LLM returned an empty response. The transcript may not contain a recipe.")
    return json.loads(raw)
