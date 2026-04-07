"""
Pipeline — download video → extract thumbnail → transcribe → AI parse recipe.
"""

import asyncio
import json
import os
import subprocess
import tempfile

import anthropic
import yt_dlp
from faster_whisper import WhisperModel

# ── Whisper setup ──────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE     = os.getenv("WHISPER_DEVICE", "cpu")

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
    """Run the full extraction pipeline. llm_settings = user's LLM config."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            _update(job_id, "Downloading video…", 10)
            video_path = await _download_video(url, tmpdir, cookies_path)

            _update(job_id, "Extracting thumbnail…", 30)
            thumbnail = await _extract_thumbnail(video_path, tmpdir)
            if thumbnail:
                jobs[job_id]["thumbnail"] = thumbnail

            _update(job_id, "Extracting audio…", 45)
            audio_path = await _extract_audio(video_path, tmpdir)

            _update(job_id, "Transcribing audio…", 55)
            loop = asyncio.get_event_loop()
            transcript = await loop.run_in_executor(None, _transcribe, audio_path)

            _update(job_id, "Extracting recipe with AI…", 80)
            recipe = await _parse_recipe(transcript, llm_settings or {})

            jobs[job_id].update({
                "status":   "done",
                "step":     "Complete!",
                "progress": 100,
                "recipe":   recipe,
                "cached":   False,
            })

    except Exception as exc:
        jobs[job_id].update({"status": "error", "error": str(exc), "step": "Failed"})
        print(f"[pipeline] Error for {job_id}: {exc}")


# ── Video download ─────────────────────────────────────────────────────────────
async def _download_video(url: str, tmpdir: str, cookies_path: str = "") -> str:
    out_tmpl = os.path.join(tmpdir, "video.%(ext)s")
    opts = {
        "format":              "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl":             out_tmpl,
        "quiet":               True,
        "no_warnings":         True,
        "merge_output_format": "mp4",
    }
    if cookies_path and os.path.exists(cookies_path) and os.path.getsize(cookies_path) > 0:
        opts["cookiefile"] = cookies_path

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: _ydl_download(opts, url))

    for fname in os.listdir(tmpdir):
        if fname.startswith("video."):
            return os.path.join(tmpdir, fname)
    raise RuntimeError("yt-dlp did not produce a video file")


def _ydl_download(opts: dict, url: str):
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])


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
    return " ".join(s.text.strip() for s in segments)


# ── LLM recipe parsing ────────────────────────────────────────────────────────
RECIPE_PROMPT = """You are a precise recipe extraction assistant. Given a transcript from a cooking video, extract the complete recipe and return it as valid JSON only — no markdown fences, no preamble, no explanation.

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


async def _parse_recipe(transcript: str, llm_settings: dict) -> dict:
    provider = llm_settings.get("llm_provider", "anthropic")
    if provider == "openai":
        return await _parse_openai(transcript, llm_settings)
    return await _parse_anthropic(transcript, llm_settings)


async def _parse_anthropic(transcript: str, llm_settings: dict) -> dict:
    api_key = llm_settings.get("anthropic_api_key", "")
    if not api_key:
        raise RuntimeError("Anthropic API key not configured. Go to Settings to add it.")
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": RECIPE_PROMPT.format(transcript=transcript)}],
    )
    raw = message.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
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
