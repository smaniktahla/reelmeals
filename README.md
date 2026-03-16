# Recipe Extractor

Extract structured recipes from social media cooking videos and push them directly to [Tandoor](https://tandoor.dev/).

Paste a Facebook, Instagram, TikTok, YouTube, or any yt-dlp-supported URL — the app downloads the video, transcribes the audio with Whisper, parses the recipe with Claude AI, grabs a thumbnail, and optionally pushes everything to your Tandoor instance.

![Recipe Extractor UI](https://i.imgur.com/placeholder.png)

## Features

- **Any social media video** — Facebook, Instagram, TikTok, YouTube, and anything yt-dlp supports
- **Local transcription** — faster-whisper runs entirely on your own hardware (CPU or GPU)
- **AI recipe parsing** — Claude extracts ingredients, steps, times, servings, and keywords
- **Thumbnail extraction** — ffmpeg grabs a frame at 25% into the video
- **Tandoor integration** — push recipe + thumbnail directly, or download as a ZIP
- **URL cache** — recently extracted recipes reload instantly without re-processing
- **Configurable cache size** — adjustable via the Settings panel in the UI

## Requirements

- Docker + Docker Compose
- [Anthropic API key](https://console.anthropic.com)
- Tandoor instance (optional — can also export as ZIP)

## Setup

```bash
git clone https://github.com/yourusername/recipe-extractor.git
cd recipe-extractor
cp .env.example .env
nano .env           # fill in your API key and Tandoor details
touch cookies.txt   # empty placeholder — replace with real cookies for FB/IG
docker compose up -d --build
```

Access the UI at **http://localhost:8090**

## Facebook & Instagram cookies

yt-dlp needs browser cookies to download FB/IG videos:

1. Install the [cookies.txt extension](https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp) for your browser
2. Log into Facebook and/or Instagram
3. Export cookies and save as `cookies.txt` in the project root
4. Restart the container: `docker restart recipe-extractor-v2`

A single `cookies.txt` file can contain cookies for multiple sites.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `small` | Whisper model size (tiny/base/small/medium) |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `TANDOOR_URL` | — | Your Tandoor instance URL |
| `TANDOOR_TOKEN` | — | Tandoor API token (Settings → API Tokens) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |

## Whisper model sizes

| Model | Size | ~Speed (CPU) | ~Speed (GPU) |
|---|---|---|---|
| tiny | 75 MB | 15–30 sec | instant |
| base | 145 MB | 30–60 sec | ~5 sec |
| small | 460 MB | 1–3 min ✓ | ~10 sec |
| medium | 1.5 GB | 3–6 min | ~20 sec |

## Updating yt-dlp

Social media sites change frequently. Update yt-dlp without rebuilding:

```bash
docker compose exec recipe-extractor-v2 pip install -U yt-dlp
docker restart recipe-extractor-v2
```

## Project structure

```
recipe-extractor/
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── .gitignore
├── cookies.txt          ← your exported browser cookies (not committed)
└── app/
    ├── main.py          ← FastAPI backend
    ├── requirements.txt
    └── static/
        └── index.html   ← Web UI
```

## Stack

- **[FastAPI](https://fastapi.tiangolo.com/)** — backend API
- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** — local speech-to-text
- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)** — video download
- **[Anthropic Claude](https://www.anthropic.com/)** — recipe parsing
- **[ffmpeg](https://ffmpeg.org/)** — thumbnail extraction
- **[Tandoor](https://tandoor.dev/)** — recipe manager integration
