import uuid

import streamlit as st

from backend.recommender import get_first_song, init_conversation


def _on_new_session():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_ended = False
    st.session_state.summary = None
    st.session_state.strategy_text = None
    strat_msgs, rec_msgs = init_conversation()
    st.session_state.strategist_messages = strat_msgs
    st.session_state.recommender_messages = rec_msgs
    st.session_state.current_track = get_first_song(st.session_state.session_id)


def render():
    summary = st.session_state.get("summary", {})

    st.markdown("---")
    st.markdown(f"# {summary.get('badge_name', 'Music Listener')}")
    st.caption(summary.get("badge_description", ""))
    st.markdown("---")
    st.markdown(summary.get("summary", "Thanks for listening!"))

    discovery_rate = summary.get("discovery_rate", 0)
    total_songs = summary.get("total_songs", 0)
    if total_songs > 0:
        st.metric("Discovery Rate", f"{discovery_rate:.0%}", help="Songs you didn't know and liked")

    st.markdown("---")

    st.button("Start New Session", key="new_session", on_click=_on_new_session, type="primary")
