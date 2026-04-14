import asyncio
import json
import logging
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI

logger = logging.getLogger(__name__)

_server_params = StdioServerParameters(
    command="spotify-mcp",
    args=[],
    env={
        **os.environ,
        "SPOTIFY_CLIENT_ID": SPOTIFY_CLIENT_ID,
        "SPOTIFY_CLIENT_SECRET": SPOTIFY_CLIENT_SECRET,
        "SPOTIFY_REDIRECT_URI": SPOTIFY_REDIRECT_URI,
    },
)


async def _call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """Open a connection to the Spotify MCP server, call a tool, and return the result."""
    async with stdio_client(_server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            for block in result.content:
                if hasattr(block, "text"):
                    try:
                        return json.loads(block.text)
                    except json.JSONDecodeError:
                        return {"raw": block.text}
            return {}


def _run(coro):
    """Run an async coroutine from sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


def search_tracks(query: str, limit: int = 5) -> list[dict]:
    """Search Spotify for tracks matching the query.

    Returns a list of dicts with keys: track_id, artist_name, track_name, embed_url.
    Raises RuntimeError if the MCP server is not available.
    """
    try:
        # Tool is named "SpotifySearch" in varunneal/spotify-mcp
        raw = _run(_call_mcp_tool("SpotifySearch", {"query": query, "qtype": "track", "limit": limit}))
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to Spotify MCP server: {e}. "
            "Make sure spotify-mcp is installed (pip install spotify-mcp) and Spotify credentials are set."
        ) from e

    tracks = []
    # Server returns {"tracks": [{"name": ..., "id": ..., "artist": "..."}, ...]}
    items = raw.get("tracks", [])
    for item in items:
        track_id = item.get("id", "")
        # Artist can be a string ("artist") or a list ("artists")
        artist_name = item.get("artist", "")
        if not artist_name:
            artists = item.get("artists", [])
            artist_name = artists[0] if isinstance(artists, list) and artists else "Unknown"
            if isinstance(artist_name, dict):
                artist_name = artist_name.get("name", "Unknown")
        track_name = item.get("name", "")
        tracks.append({
            "track_id": track_id,
            "artist_name": artist_name,
            "track_name": track_name,
            "embed_url": build_embed_url(track_id),
        })
    return tracks


def get_track_info(track_id: str) -> dict:
    """Get details for a specific Spotify track."""
    try:
        # Tool is named "SpotifyGetInfo" in varunneal/spotify-mcp
        return _run(_call_mcp_tool("SpotifyGetInfo", {"item_id": track_id, "qtype": "track"}))
    except Exception as e:
        raise RuntimeError(f"Failed to get track info from Spotify MCP server: {e}") from e


def build_embed_url(track_id: str) -> str:
    return f"https://open.spotify.com/embed/track/{track_id}"
