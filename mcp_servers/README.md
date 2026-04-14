# Spotify MCP Server Setup

This app uses [varunneal/spotify-mcp](https://github.com/varunneal/spotify-mcp) to interact with Spotify.

## Prerequisites

1. **Spotify Developer Account**: Go to https://developer.spotify.com/dashboard
2. **Create an App**: Click "Create App", give it a name
3. **Set Redirect URI**: Add `http://127.0.0.1:8080/callback` as a redirect URI
4. **Copy Credentials**: Note the Client ID and Client Secret

## Install the MCP Server

```bash
pip install spotify-mcp
```

## Configuration

Set these in your `.env` file:

```
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8080/callback
```

The first time you run the app, the MCP server will trigger Spotify's OAuth flow in your browser.
