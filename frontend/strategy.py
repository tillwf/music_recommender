import streamlit as st


def render():
    strategy_text = st.session_state.get("strategy_text")
    if strategy_text:
        st.info(strategy_text)
