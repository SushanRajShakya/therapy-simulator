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

            # Show typing indicator while getting response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Get the response from API first
                    bot_response = self._get_bot_response(user_input)

                # Now stream the response with typing effect
                self._stream_response(bot_response)

    def _add_user_message(self, content):
        st.session_state.messages.append({"role": "user", "content": content})
        # Set flag to process bot response on next rerun
        st.session_state.pending_user_input = content

    def _get_bot_response(self, user_input):
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
                return response.json()["response"]
            else:
                return "Sorry, I'm having trouble connecting right now."

        except requests.exceptions.RequestException:
            return "Connection error. Please check if the server is running."

    def _stream_response(self, bot_response):
        import time

        # Create a placeholder for the streaming response
        message_placeholder = st.empty()
        full_response = ""

        # Stream the response character by character
        for char in bot_response:
            full_response += char
            message_placeholder.markdown(full_response + "▌")  # Add cursor effect
            time.sleep(
                0.005
            )  # Adjust speed of typing (0.005 seconds per character - faster typing)

        # Remove cursor and show final message
        message_placeholder.markdown(full_response)

        # Add the complete response to session state
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
