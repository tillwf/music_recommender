import uuid

import streamlit as st

from backend.database import init_db
from backend.observability import init_observability
from backend.recommender import get_first_song, init_conversation
from frontend import player, strategy, summary

st.set_page_config(page_title="Music Recommender", layout="centered")
st.title("Music Recommender")

init_db()
init_observability()

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_ended = False
    st.session_state.summary = None
    st.session_state.strategy_text = None
    st.session_state.current_track = None
    strat_msgs, rec_msgs = init_conversation()
    st.session_state.strategist_messages = strat_msgs
    st.session_state.recommender_messages = rec_msgs

# Load first song if needed
if st.session_state.current_track is None and not st.session_state.session_ended:
    with st.spinner("Picking a song to start..."):
        st.session_state.current_track = get_first_song(st.session_state.session_id)

# Route to the right view
if st.session_state.session_ended:
    summary.render()
else:
    strategy.render()
    player.render()
