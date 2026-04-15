import json
import logging
import random

from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import task, tool, workflow

from backend.observability import get_openai_client, track_raindrop_ai
from backend.database import get_session_history, get_session_summary_data
from backend.spotify_client import search_tracks
from config import OPENAI_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

STRATEGIST_SYSTEM = """You are a music discovery strategist. Your job is to decide WHAT APPROACH to use for the next song recommendation, NOT to pick a specific song.

Your ultimate goal: help the user discover songs they DON'T KNOW and WILL LIKE.

You have access to the user's session history. Each entry shows a song and two axes:
- known/didn't know: whether the user had heard the song before
- liked/didn't like: whether the user enjoyed it
- Some entries include free-text feedback from the user explaining why they feel that way. Pay close attention to this — it reveals taste nuances the buttons can't capture.

You will also receive a "Session insights" block with computed stats. USE THESE to adapt your strategy:
- If the user knows most songs → they have broad music knowledge, STOP suggesting well-known hits. Go for deep cuts, B-sides, obscure artists, international tracks, or niche genres.
- If the user doesn't know most songs → they may be casual listeners, stick to more recognizable tracks but vary the genre.
- If the user likes most songs → their taste is broad, try to narrow down what they DON'T like.
- If the user dislikes most songs → you're in the wrong territory, pivot hard to something completely different.

CRITICAL: If the "known" rate is above 60%, you MUST pick something obscure. No more mainstream hits, no greatest hits, no top-40 songs. Think: album deep cuts, indie releases, foreign language tracks, remix versions, lesser-known artists in liked genres.

Based on patterns in the history, choose a creative strategy for the next pick. Examples:
- "User knows everything mainstream → find an obscure jazz fusion track from the 70s"
- "Go deeper into an artist they liked"
- "Pick something totally random to surprise them"
- "They seem to like 80s rock, let's try 80s post-punk they probably don't know"
- "They know all the classics, let's try an obscure deep cut"
- "They liked covers, let's try the original version of something"
- "They mentioned loving bass lines, find something bass-heavy they won't know"
- "Try a track from a completely different country/language"

Be creative and varied. Don't repeat the same strategy twice in a row.

Respond ONLY with valid JSON:
{"strategy": "short strategy description for the recommender", "explanation": "human-friendly explanation shown to the user, be fun and conversational"}"""

RECOMMENDER_SYSTEM = """You are a music recommendation engine. Pick ONE specific song that matches the given strategy.

The ultimate goal is to help the user discover songs they DON'T KNOW and WILL LIKE. Pay attention to any free-text feedback in the history — it reveals what specifically the user enjoys or dislikes about songs.

You have a tool available: spotify_search. You MUST call it to verify the song exists on Spotify before making your final recommendation.

Rules:
- Pick a real, existing song
- Do NOT repeat any song from the session history
- Do NOT repeat any artist already in the session history
- Match the strategy as closely as possible
- Always call spotify_search to verify availability

After calling the tool, respond with valid JSON: {"artist": "...", "track": "..."}"""

FIRST_SONG_SYSTEM = """Pick a well-known, popular song that most people would recognize. Choose something fun and upbeat that works as a conversation starter about music taste.

You have a tool available: spotify_search. Call it to verify the song exists on Spotify.

After calling the tool, respond with valid JSON: {"artist": "...", "track": "..."}"""

SUMMARIZER_SYSTEM = """You are a music taste analyst. Based on a user's listening session, create a fun summary of their taste and award them a creative badge.

The badge should be a short, catchy title (2-3 words) like:
- "Music Explorer" (varied taste across genres)
- "Country Digger" (clearly loves country)
- "Classical Maestro" (gravitates toward classical)
- "Indie Archaeologist" (loves discovering unknown tracks)
- "Pop Connoisseur" (knows and loves pop hits)
- "Genre Rebel" (dislikes mainstream, likes obscure)
- "Nostalgia Seeker" (prefers older well-known tracks)

Be creative with the badge. The summary should be 2-3 sentences, conversational and fun.

Respond ONLY with valid JSON:
{"badge_name": "...", "badge_description": "one-line badge meaning", "summary": "2-3 sentence taste summary"}"""

# ---------------------------------------------------------------------------
# OpenAI tool definition for Spotify search
# ---------------------------------------------------------------------------

SPOTIFY_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "spotify_search",
        "description": "Search Spotify for a track. Returns the top result with track_id, artist_name, track_name.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query, e.g. 'Miles Davis So What'",
                }
            },
            "required": ["query"],
        },
    },
}

# ---------------------------------------------------------------------------
# Prompt templates for DD tracking
# ---------------------------------------------------------------------------

STRATEGIST_PROMPT_TEMPLATE = {
    "id": "strategist",
    "version": "2.0.0",
    "template": STRATEGIST_SYSTEM,
}

RECOMMENDER_PROMPT_TEMPLATE = {
    "id": "recommender",
    "version": "2.0.0",
    "template": RECOMMENDER_SYSTEM,
}

SUMMARIZER_PROMPT_TEMPLATE = {
    "id": "summarizer",
    "version": "1.0.0",
    "template": SUMMARIZER_SYSTEM,
}

FIRST_SONG_PROMPT_TEMPLATE = {
    "id": "first-song",
    "version": "1.0.0",
    "template": FIRST_SONG_SYSTEM,
}


# ---------------------------------------------------------------------------
# Tool execution helper
# ---------------------------------------------------------------------------

@tool(name="spotify_search")
def _execute_spotify_search(query):
    """Execute a Spotify search and return results. Traced as a DD tool span."""
    results = search_tracks(query, limit=1)
    LLMObs.annotate(
        input_data=query,
        output_data=json.dumps(results),
    )
    return results


def _handle_tool_calls(response, messages):
    """Process tool calls from the LLM response, execute them, and return updated messages + results."""
    assistant_msg = response.choices[0].message
    messages.append(assistant_msg.model_dump())

    tool_results = []
    for tool_call in assistant_msg.tool_calls:
        if tool_call.function.name == "spotify_search":
            args = json.loads(tool_call.function.arguments)
            results = _execute_spotify_search(args["query"])
            result_str = json.dumps(results)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_str,
            })
            tool_results.extend(results)

    return messages, tool_results


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

def _compute_session_insights(session_id):
    """Compute stats from the session history to help the strategist adapt."""
    history = get_session_history(session_id)
    if not history:
        return ""

    total = len(history)
    known_count = sum(1 for v in history if v["known"])
    liked_count = sum(1 for v in history if v["liked"])
    unknown_liked = sum(1 for v in history if not v["known"] and v["liked"])

    known_rate = known_count / total
    liked_rate = liked_count / total

    lines = [
        f"Session insights ({total} songs so far):",
        f"  - Known rate: {known_rate:.0%} ({known_count}/{total} songs the user already knew)",
        f"  - Liked rate: {liked_rate:.0%} ({liked_count}/{total} songs the user liked)",
        f"  - Discoveries so far: {unknown_liked} (songs they didn't know and liked)",
    ]

    if known_rate > 0.6:
        lines.append("  ⚠ HIGH KNOWN RATE — user has broad music knowledge. STOP suggesting mainstream hits. Go obscure.")
    if known_rate == 1.0:
        lines.append("  ⚠ User has known EVERY song so far. You MUST go significantly more obscure.")
    if liked_rate < 0.3:
        lines.append("  ⚠ LOW LIKE RATE — current direction isn't working. Pivot to something very different.")

    return "\n".join(lines)


def _llm_call_with_history(messages, temperature=0.7, tools=None):
    """Make an LLM call with full message history. Returns (response, raw_text, parsed_json)."""
    client = get_openai_client()
    kwargs = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools

    response = client.chat.completions.create(**kwargs)
    msg = response.choices[0].message

    if msg.tool_calls:
        return response, None, None

    raw = msg.content
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON: %s", raw)
        parsed = {}
    return response, raw, parsed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_conversation():
    """Return initial conversation message lists."""
    return (
        [{"role": "system", "content": STRATEGIST_SYSTEM}],
        [{"role": "system", "content": RECOMMENDER_SYSTEM}],
    )


_FIRST_SONG_HINTS = [
    "Pick a fun pop song from the 2010s.",
    "Pick a classic rock anthem from the 70s or 80s.",
    "Pick a feel-good R&B or soul track.",
    "Pick a catchy hip-hop or rap song.",
    "Pick an upbeat electronic or dance track.",
    "Pick a well-known indie rock song.",
    "Pick a famous Latin pop or reggaeton hit.",
    "Pick a beloved funk or disco track.",
    "Pick a sing-along pop-punk or alternative hit.",
    "Pick a crowd-pleasing song from the 90s.",
]


def get_first_song(session_id):
    """Pick a popular starting song. Returns (track_dict, recommender_messages)."""
    hint = random.choice(_FIRST_SONG_HINTS)
    messages = [
        {"role": "system", "content": FIRST_SONG_SYSTEM},
        {"role": "user", "content": f"Pick a song to start the session. {hint}"},
    ]

    with LLMObs.workflow(name="first_song", session_id=session_id) as span:
        with LLMObs.annotation_context(prompt=FIRST_SONG_PROMPT_TEMPLATE):
            response, raw, parsed = _llm_call_with_history(
                messages, temperature=0.9, tools=[SPOTIFY_SEARCH_TOOL],
            )

        # Handle tool call if the LLM wants to search
        spotify_results = []
        if response.choices[0].message.tool_calls:
            messages, spotify_results = _handle_tool_calls(response, messages)
            # Call LLM again to get final answer
            response, raw, parsed = _llm_call_with_history(messages, temperature=0.9)

        if spotify_results:
            track = spotify_results[0]
        else:
            query = f"{parsed.get('artist', '')} {parsed.get('track', '')}".strip() or "Bohemian Rhapsody Queen"
            results = _execute_spotify_search(query)
            track = results[0] if results else {
                "track_id": "", "artist_name": parsed.get("artist", "Unknown"),
                "track_name": parsed.get("track", "Unknown"), "embed_url": "",
            }

        track["llm_prompt"] = "Pick a song to start the session."
        track["llm_response"] = raw
        track["llm_model"] = OPENAI_MODEL

        LLMObs.annotate(
            span=span,
            input_data="Start session — pick a popular song",
            output_data=f"{track['artist_name']} - {track['track_name']}",
            metadata={"model": OPENAI_MODEL},
            tags={"session_id": session_id},
        )

        track["raindrop_event_id"] = track_raindrop_ai(
            user_id=session_id,
            event="first_song",
            model=OPENAI_MODEL,
            input_text="Start session — pick a popular song",
            output_text=f"{track['artist_name']} - {track['track_name']}",
            convo_id=session_id,
            properties={"hint": hint},
        )

    return track


def get_next_song(session_id, strategist_messages, recommender_messages, vote_result):
    """Run strategist + recommender chain with conversation memory.

    Args:
        session_id: Current session UUID
        strategist_messages: Full strategist conversation history
        recommender_messages: Full recommender conversation history
        vote_result: dict with keys: artist_name, track_name, known, liked, feedback

    Returns:
        (track_dict, strategy_explanation, updated_strategist_messages, updated_recommender_messages, span_context)
    """
    # Build vote feedback message with session insights
    known_str = "knew" if vote_result["known"] else "didn't know"
    liked_str = "liked" if vote_result["liked"] else "didn't like"
    feedback_line = ""
    if vote_result.get("feedback"):
        feedback_line = f" Feedback: \"{vote_result['feedback']}\""

    insights = _compute_session_insights(session_id)
    vote_msg = f"Song: {vote_result['artist_name']} - {vote_result['track_name']}. User {known_str} it, {liked_str} it.{feedback_line}"
    if insights:
        vote_msg += f"\n\n{insights}"

    with LLMObs.workflow(name="recommendation_cycle", session_id=session_id) as wf_span:
        # ---- Chain 1: Strategist ----
        with LLMObs.task(name="strategist") as strat_span:
            strategist_messages.append({"role": "user", "content": vote_msg})

            with LLMObs.annotation_context(prompt=STRATEGIST_PROMPT_TEMPLATE):
                _, strat_raw, strat_parsed = _llm_call_with_history(
                    strategist_messages, temperature=1.0,
                )

            strategy = strat_parsed.get("strategy", "pick something popular")
            explanation = strat_parsed.get("explanation", strategy)

            strategist_messages.append({"role": "assistant", "content": strat_raw})

            LLMObs.annotate(
                span=strat_span,
                input_data=vote_msg,
                output_data=strat_raw,
                metadata={"model": OPENAI_MODEL, "temperature": 1.0},
            )

        # ---- Chain 2: Recommender (with tool calling) ----
        with LLMObs.task(name="recommender") as rec_span:
            # Build list of previously played artists for the recommender
            history = get_session_history(session_id)
            played_artists = list({v["artist_name"] for v in history})
            rec_user_msg = f"Result: user {known_str} it, {liked_str} it.{feedback_line}\n\nNew strategy: {strategy}\n\nPreviously played artists (do NOT repeat): {', '.join(played_artists)}"

            recommender_messages.append({"role": "user", "content": rec_user_msg})

            with LLMObs.annotation_context(prompt=RECOMMENDER_PROMPT_TEMPLATE):
                response, rec_raw, rec_parsed = _llm_call_with_history(
                    recommender_messages, temperature=0.7, tools=[SPOTIFY_SEARCH_TOOL],
                )

            # Handle tool calls
            spotify_results = []
            if response.choices[0].message.tool_calls:
                recommender_messages, spotify_results = _handle_tool_calls(response, recommender_messages)
                # Call LLM again to get final JSON answer
                response, rec_raw, rec_parsed = _llm_call_with_history(
                    recommender_messages, temperature=0.7,
                )

            recommender_messages.append({"role": "assistant", "content": rec_raw})

            if spotify_results:
                track = spotify_results[0]
            else:
                query = f"{rec_parsed.get('artist', '')} {rec_parsed.get('track', '')}".strip() or "popular song"
                results = _execute_spotify_search(query)
                track = results[0] if results else {
                    "track_id": "", "artist_name": rec_parsed.get("artist", "Unknown"),
                    "track_name": rec_parsed.get("track", "Unknown"), "embed_url": "",
                }

            LLMObs.annotate(
                span=rec_span,
                input_data=rec_user_msg,
                output_data=rec_raw,
                metadata={"model": OPENAI_MODEL, "temperature": 0.7, "strategy": strategy},
            )

        # Annotate the workflow span
        LLMObs.annotate(
            span=wf_span,
            input_data=vote_msg,
            output_data=f"{track['artist_name']} - {track['track_name']}",
            metadata={"strategy": strategy, "model": OPENAI_MODEL},
            tags={
                "session_id": session_id,
                "has_feedback": "true" if vote_result.get("feedback") else "false",
            },
        )
        span_context = LLMObs.export_span(span=wf_span)

        track["raindrop_event_id"] = track_raindrop_ai(
            user_id=session_id,
            event="recommendation_cycle",
            model=OPENAI_MODEL,
            input_text=vote_msg,
            output_text=f"{track['artist_name']} - {track['track_name']}",
            convo_id=session_id,
            properties={
                "strategy": strategy,
                "has_feedback": bool(vote_result.get("feedback")),
            },
        )

    track["strategy_text"] = explanation
    track["llm_prompt"] = rec_user_msg
    track["llm_response"] = rec_raw
    track["llm_model"] = OPENAI_MODEL

    return track, explanation, strategist_messages, recommender_messages, span_context


def get_session_summary(session_id):
    """Run the summarizer chain. Returns (summary_dict, span_context)."""
    summary_data = get_session_summary_data(session_id)
    history = get_session_history(session_id)

    messages = [
        {"role": "system", "content": SUMMARIZER_SYSTEM},
        {"role": "user", "content": f"Session listening history:\n{summary_data}"},
    ]

    with LLMObs.workflow(name="session_summary", session_id=session_id) as span:
        with LLMObs.annotation_context(prompt=SUMMARIZER_PROMPT_TEMPLATE):
            _, raw, parsed = _llm_call_with_history(messages, temperature=0.8)

        # Compute discovery rate
        total = len(history)
        discoveries = sum(1 for v in history if not v["known"] and v["liked"])
        discovery_rate = discoveries / total if total > 0 else 0.0

        LLMObs.annotate(
            span=span,
            input_data=summary_data,
            output_data=raw,
            metadata={"model": OPENAI_MODEL, "total_songs": total, "discoveries": discoveries},
            metrics={"discovery_rate": discovery_rate},
            tags={"session_id": session_id},
        )
        span_context = LLMObs.export_span(span=span)

        track_raindrop_ai(
            user_id=session_id,
            event="session_summary",
            model=OPENAI_MODEL,
            input_text=summary_data,
            output_text=raw,
            convo_id=session_id,
            properties={
                "total_songs": total,
                "discoveries": discoveries,
                "discovery_rate": discovery_rate,
            },
        )

    return {
        "badge_name": parsed.get("badge_name", "Music Listener"),
        "badge_description": parsed.get("badge_description", "You listened to some songs!"),
        "summary": parsed.get("summary", "Thanks for listening!"),
        "discovery_rate": discovery_rate,
        "total_songs": total,
    }, span_context
