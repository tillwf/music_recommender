# Music Recommender

A Streamlit app that recommends songs you don't know yet using an LLM-powered strategist/recommender chain. Rate each song (known/unknown, liked/disliked) and the system adapts its strategy to help you discover new music.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in your API keys
```

## Run

```bash
./run.sh
```

This starts the Streamlit app with Datadog LLM Observability tracing enabled. `DD_SERVICE` from `.env` is used as the `ml_app` name for both `ddtrace-run` and the LLM Obs SDK.

## REST API

Start the API server:

```bash
./run_api.sh
```

### Endpoints

**`POST /recommend`** — all fields optional.

```bash
# First recommendation (no context)
curl -s -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{}' | jq

# Recommend after a vote (fresh message history)
curl -s -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "vote": {
      "artist_name": "Queen",
      "track_name":  "Bohemian Rhapsody",
      "known":  true,
      "liked":  true
    }
  }' | jq

# Reproduce from a Datadog LLM span:
#   strategist_messages  → span "strategist" task  → input.messages
#   recommender_messages → span "recommender" task → input.messages
#   vote                 → the vote to generate the next recommendation for
curl -s -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "strategist_messages": [
      {"role": "system",    "content": "..."},
      {"role": "user",      "content": "Song: Queen - Bohemian Rhapsody. User knew it, liked it."},
      {"role": "assistant", "content": "{\"strategy\": \"go deeper into rock classics\"}"}
    ],
    "recommender_messages": [
      {"role": "system",    "content": "..."},
      {"role": "user",      "content": "..."},
      {"role": "assistant", "content": "{\"artist\": \"Led Zeppelin\", \"track\": \"Kashmir\"}"}
    ],
    "vote": {
      "artist_name": "Led Zeppelin",
      "track_name":  "Kashmir",
      "known":  false,
      "liked":  true
    }
  }' | jq
```

## Required API keys

- **Spotify** - for track search and embeds
- **OpenAI** - for the recommendation LLM calls
- **Datadog** - for LLM observability (`DD_SERVICE` is used as the `ml_app` name in LLM Obs)
- **Raindrop** - for AI analytics and user feedback signals
