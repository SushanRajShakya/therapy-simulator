import streamlit as st
from constants import *


class ChatInterface:
    def __init__(self):
        self._initialize_session()

    def _initialize_session(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def render(self):
        self._render_header()
        self._display_messages()
        self._handle_input()

    def _render_header(self):
        st.title(TITLE)

    def _display_messages(self):
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])

    def _handle_input(self):
        if user_input := st.chat_input(CHAT_INPUT_PLACEHOLDER):
            self._add_user_message(user_input)
            self._add_bot_response(user_input)
            st.rerun()

    def _add_user_message(self, content):
        st.session_state.messages.append({"role": "user", "content": content})

    def _add_bot_response(self, user_input):
        bot_response = DEFAULT_BOT_RESPONSE.format(user_input)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
