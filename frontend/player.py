import streamlit as st

from backend.database import add_vote
from backend.recommender import get_next_song, get_session_summary
from backend.observability import submit_recommendation_evaluation


def _on_vote(known: bool, liked: bool):
    track = st.session_state.current_track
    session_id = st.session_state.session_id
    feedback = st.session_state.get("feedback_text", "").strip() or None

    add_vote(
        session_id=session_id,
        track_id=track["track_id"],
        artist=track["artist_name"],
        track=track["track_name"],
        known=known,
        liked=liked,
        feedback=feedback,
        strategy_text=track.get("strategy_text"),
        llm_prompt=track.get("llm_prompt"),
        llm_response=track.get("llm_response"),
        llm_model=track.get("llm_model"),
    )

    vote_result = {
        "artist_name": track["artist_name"],
        "track_name": track["track_name"],
        "known": known,
        "liked": liked,
        "feedback": feedback,
    }

    next_track, explanation, strat_msgs, rec_msgs, span_context = get_next_song(
        session_id=session_id,
        strategist_messages=st.session_state.strategist_messages,
        recommender_messages=st.session_state.recommender_messages,
        vote_result=vote_result,
    )

    # Submit discovery evaluation to Datadog
    submit_recommendation_evaluation(span_context, known, liked)

    st.session_state.current_track = next_track
    st.session_state.strategy_text = explanation
    st.session_state.strategist_messages = strat_msgs
    st.session_state.recommender_messages = rec_msgs
    st.session_state.feedback_text = ""


def _on_end_session():
    session_id = st.session_state.session_id
    summary, span_context = get_session_summary(session_id)

    # Submit session-level evaluation to Datadog
    from backend.observability import submit_session_evaluation
    submit_session_evaluation(span_context, summary["discovery_rate"], summary["total_songs"])

    st.session_state.summary = summary
    st.session_state.session_ended = True


def render():
    track = st.session_state.get("current_track")
    if not track:
        return

    st.subheader(f"{track['artist_name']} - {track['track_name']}")

    if track.get("embed_url"):
        st.markdown(
            f'<iframe src="{track["embed_url"]}" width="100%" height="152" frameborder="0"'
            ' allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"'
            ' style="border-radius:12px"></iframe>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("No Spotify embed available for this track.")

    st.text_input("Any thoughts on this song? (optional)", key="feedback_text")

    st.markdown("#### What do you think?")

    col1, col2 = st.columns(2)
    with col1:
        st.button("I know it, I like it",
                   key="vote_known_liked",
                   on_click=_on_vote, args=(True, True),
                   use_container_width=True)
        st.button("I don't know it, I like it",
                   key="vote_unknown_liked",
                   on_click=_on_vote, args=(False, True),
                   use_container_width=True)

    with col2:
        st.button("I know it, I don't like it",
                   key="vote_known_disliked",
                   on_click=_on_vote, args=(True, False),
                   use_container_width=True)
        st.button("I don't know it, I don't like it",
                   key="vote_unknown_disliked",
                   on_click=_on_vote, args=(False, False),
                   use_container_width=True)

    st.divider()
    st.button("End Session", key="end_session", on_click=_on_end_session, type="secondary")
