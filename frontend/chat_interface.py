import requests
import streamlit as st

from constants import *


class ChatInterface:
    def __init__(self):
        self.api_url = "http://localhost:8000"
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
            # Add user message and display it immediately
            self._add_user_message(user_input)
            st.rerun()

        # Check if we need to process a bot response
        if hasattr(st.session_state, "pending_user_input"):
            user_input = st.session_state.pending_user_input
            delattr(st.session_state, "pending_user_input")

            # Show typing indicator or processing message
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    self._add_bot_response(user_input)

    def _add_user_message(self, content):
        st.session_state.messages.append({"role": "user", "content": content})
        # Set flag to process bot response on next rerun
        st.session_state.pending_user_input = content

    def _add_bot_response(self, user_input):
        try:
            # Call FastAPI backend
            response = requests.post(
                f"{self.api_url}/chat",
                json={
                    "message": user_input,
                    "session_id": st.session_state.get("session_id", "default"),
                },
                timeout=30,
            )

            if response.status_code == 200:
                bot_response = response.json()["response"]
            else:
                bot_response = "Sorry, I'm having trouble connecting right now."

        except requests.exceptions.RequestException:
            bot_response = "Connection error. Please check if the server is running."

        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        st.rerun()
