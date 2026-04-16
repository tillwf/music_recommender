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

This starts the Streamlit app with Datadog LLM Observability tracing enabled.

## Required API keys

- **Spotify** - for track search and embeds
- **OpenAI** - for the recommendation LLM calls
- **Datadog** - for LLM observability (`DD_SERVICE` is used as the `ml_app` name in LLM Obs)
- **Raindrop** - for AI analytics and user feedback signals
