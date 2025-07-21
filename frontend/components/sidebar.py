import streamlit as st
from constants import *


class Sidebar:
    def __init__(self):
        pass

    def render(self):
        with st.sidebar:
            self._render_header()
            self._render_new_session_button()
            self._render_session_info()

    def _render_header(self):
        st.header(SIDEBAR_HEADER)

    def _render_new_session_button(self):
        if st.button(NEW_SESSION_BUTTON, use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    def _render_session_info(self):
        st.subheader(SESSION_INFO_HEADER)
        message_count = len(st.session_state.get("messages", []))
        st.metric(MESSAGES_METRIC, message_count)
