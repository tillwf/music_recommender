import os
import shutil

# Spotify
SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI = os.environ.get("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8080/callback")
SPOTIFY_AVAILABLE = bool(SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET and shutil.which("spotify-mcp"))

# OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Langfuse
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Raindrop
RAINDROP_WRITE_KEY = os.environ.get("RAINDROP_WRITE_KEY", "")

# Datadog
DD_SERVICE = os.environ.get("DD_SERVICE", "music-recommender")

# App
DATABASE_PATH = os.environ.get("DATABASE_PATH", "music_recommender.db")
