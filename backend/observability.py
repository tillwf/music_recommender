from ddtrace.llmobs import LLMObs
from langfuse.openai import OpenAI
import raindrop.analytics as raindrop
from raindrop.analytics import Instruments

from config import OPENAI_API_KEY, RAINDROP_WRITE_KEY


def init_observability():
    """Initialize Datadog LLM Observability SDK and Raindrop AI.

    Called once at app startup. Works alongside ddtrace-run which handles
    APM and auto-instrumentation. This adds manual span control, evaluations,
    and session tracking.
    """
    LLMObs.enable(
        ml_app="music-recommender",
        agentless_enabled=True,
    )
    if RAINDROP_WRITE_KEY:
        raindrop.init(RAINDROP_WRITE_KEY, tracing_enabled=True, instruments={Instruments.OPENAI})


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


def track_raindrop_ai(user_id, event, model, input_text, output_text, convo_id, properties=None):
    """Log a semantic AI event to Raindrop. Returns the event_id for signal linking."""
    if not RAINDROP_WRITE_KEY:
        return None
    return raindrop.track_ai(
        user_id=user_id,
        event=event,
        model=model,
        input=input_text,
        output=output_text,
        convo_id=convo_id,
        properties=properties or {},
    )


def track_raindrop_signal(event_id, name, signal_type, sentiment, comment=None):
    """Submit a user feedback signal to Raindrop, linked to a specific AI event."""
    if not RAINDROP_WRITE_KEY:
        return
    raindrop.track_signal(
        event_id=event_id,
        name=name,
        signal_type=signal_type,
        sentiment=sentiment,
        comment=comment,
    )


def shutdown_observability():
    """Flush and shut down observability SDKs."""
    if RAINDROP_WRITE_KEY:
        raindrop.shutdown()
