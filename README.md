# ReelMeals

Extract structured recipes from social media cooking videos. Multi-user, self-hosted, no database — just JSON files.

Paste a Facebook, Instagram, TikTok, YouTube, or any yt-dlp-supported URL — the app downloads the video, transcribes the audio with Whisper, parses the recipe with AI, and saves it to your personal recipe library. Optionally push to [Tandoor](https://tandoor.dev/) or [Mealie](https://mealie.io/).

## What's new in v4

- **Multi-user** — each user gets their own recipe library, API keys, and settings
- **Persistent recipe library** — no more cache eviction, all recipes saved permanently as JSON
- **Browse & search** — visual recipe library with thumbnails and search
- **Settings in the UI** — API keys, Tandoor/Mealie config, and cookies all managed from the browser
- **OIDC support** — optional SSO via Authentik, Keycloak, or any OpenID Connect provider
- **Local auth** — simple username/password with first-user-is-admin

## Quick Start

```bash
git clone https://github.com/smaniktahla/reelmeals.git
cd reelmeals
cp .env.example .env
nano .env        # set SECRET_KEY and whisper config
docker compose up -d --build
```

Open **http://localhost:8091** — create your admin account, then configure your API keys in Settings.

## Authentication

ReelMeals supports two auth methods (can be used together):

| Method | Config | Use case |
|--------|--------|----------|
| Local | `AUTH_LOCAL=true` (default) | Username/password, first user is admin |
| OIDC | `AUTH_OIDC=true` + issuer config | SSO via Authentik, Keycloak, etc. |

### OIDC Setup (Authentik example)

1. In Authentik, create an OAuth2/OIDC provider for ReelMeals
2. Set the redirect URI to `http://<your-host>:8091/api/auth/oidc/callback`
3. Add the env vars to `.env`:
   ```
   AUTH_OIDC=true
   OIDC_ISSUER=https://auth.example.com/application/o/reelmeals
   OIDC_CLIENT_ID=<client-id>
   OIDC_CLIENT_SECRET=<client-secret>
   OIDC_REDIRECT_URI=http://<your-host>:8091/api/auth/oidc/callback
   OIDC_PROVIDER_NAME=Authentik
   ```

## Configuration

Only infrastructure settings live in `.env`. Everything else is per-user in the web UI:

| `.env` variable | Default | Description |
|-----------------|---------|-------------|
| `WHISPER_MODEL` | `small` | Whisper model size (tiny/base/small/medium) |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `SECRET_KEY` | — | Session signing key (generate a random string) |
| `AUTH_LOCAL` | `true` | Enable local username/password auth |
| `AUTH_OIDC` | `false` | Enable OIDC SSO |
| `ALLOW_REGISTRATION` | `true` | Allow new users to register |

**Per-user settings** (configured in the web UI):
- Anthropic / OpenAI API keys and model selection
- Tandoor URL and token
- Mealie URL and token
- Browser cookies for Facebook/Instagram (paste in Settings)

## Data Storage

All data lives in `/app/data` (Docker volume `reelmeals-data`):

```
/app/data/
├── global.json              # Admin list, setup state
└── users/
    └── {username}/
        ├── profile.json     # User profile
        ├── settings.json    # API keys, integrations
        ├── cookies.txt      # Browser cookies
        └── recipes/
            ├── index.json   # Recipe index for fast listing
            └── {slug}.json  # Individual recipe files
```

No database. Back up the volume and you have everything.

## Upgrading from v3

On first run, v4 automatically imports your v3 `cache.json` recipes into the first (admin) user's library. The old cache file is renamed to `cache.json.migrated`.

## Stack

- **[FastAPI](https://fastapi.tiangolo.com/)** — backend API
- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** — local speech-to-text
- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)** — video download
- **[Anthropic Claude](https://www.anthropic.com/)** or **[OpenAI](https://openai.com/)** — recipe parsing
- **[Tandoor](https://tandoor.dev/)** / **[Mealie](https://mealie.io/)** — optional recipe manager sync
