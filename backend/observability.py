from ddtrace.llmobs import LLMObs
from langfuse.openai import OpenAI

from config import OPENAI_API_KEY


def init_observability():
    """Initialize Datadog LLM Observability SDK.

    Called once at app startup. Works alongside ddtrace-run which handles
    APM and auto-instrumentation. This adds manual span control, evaluations,
    and session tracking.
    """
    LLMObs.enable(
        ml_app="music-recommender",
        agentless_enabled=True,
    )


def get_openai_client():
    """Return an OpenAI client wrapped by Langfuse.

    Langfuse captures prompts/completions/tokens at the client level.
    Datadog (ddtrace-run) auto-instruments at the library level.
    Both coexist: no extra code needed beyond this wrapper.
    """
    return OpenAI(api_key=OPENAI_API_KEY)


def submit_recommendation_evaluation(span_context, known, liked):
    """Submit a discovery evaluation for a recommendation.

    Scoring:
    - unknown + liked = 1.0 (the goal!)
    - known + liked = 0.5 (good taste match, but not a discovery)
    - unknown + disliked = 0.25 (at least it was new)
    - known + disliked = 0.0 (bad recommendation)
    """
    if not known and liked:
        score = 1.0
        assessment = "pass"
        reasoning = "User discovered a new song they liked"
    elif known and liked:
        score = 0.5
        assessment = "pass"
        reasoning = "User already knew the song but liked it"
    elif not known and not liked:
        score = 0.25
        assessment = "fail"
        reasoning = "User didn't know the song and didn't like it"
    else:
        score = 0.0
        assessment = "fail"
        reasoning = "User already knew the song and didn't like it"

    LLMObs.submit_evaluation(
        span=span_context,
        label="discovery",
        metric_type="score",
        value=score,
        ml_app="music-recommender",
        assessment=assessment,
        reasoning=reasoning,
    )


def submit_session_evaluation(span_context, discovery_rate, total_songs):
    """Submit an end-of-session evaluation with discovery rate."""
    LLMObs.submit_evaluation(
        span=span_context,
        label="session_discovery_rate",
        metric_type="score",
        value=discovery_rate,
        ml_app="music-recommender",
        assessment="pass" if discovery_rate > 0.3 else "fail",
        reasoning=f"User discovered new liked songs {discovery_rate:.0%} of the time ({total_songs} songs total)",
    )
