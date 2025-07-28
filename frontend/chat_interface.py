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
        if "session_id" not in st.session_state:
            import uuid

            st.session_state.session_id = str(uuid.uuid4())

    def render(self):
        self._render_header()
        self._display_messages()
        self._handle_input()
        self._render_session_controls()

    def _render_header(self):
        st.title(TITLE)
        st.caption(
            "💬 Have a conversation with your CBT therapist. Click 'End Session' when you're ready for a conclusion."
        )

    def _display_messages(self):
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                if message.get("is_conclusion", False):
                    # Style the conclusion differently
                    st.chat_message("assistant").markdown(
                        f"🎯 **Session Conclusion:**\n\n{message['content']}"
                    )
                else:
                    st.chat_message("assistant").write(message["content"])

    def _render_session_controls(self):
        # Only show controls if session is not ended
        if not st.session_state.get("session_ended", False):
            st.markdown("---")
            col1, col2 = st.columns([3, 1])

            with col2:
                if st.button(
                    "🏁 End Session",
                    type="secondary",
                    help="Get a therapeutic conclusion and end the session",
                ):
                    st.session_state.end_session_requested = True
                    st.rerun()
        else:
            st.markdown("---")
            st.success(
                "✅ Session completed. Thank you for using the therapy simulator!"
            )
            if st.button("🔄 Start New Session", type="primary"):
                # Clear session and restart
                st.session_state.clear()
                st.rerun()

    def _handle_input(self):
        # Don't allow input if session is ended
        if st.session_state.get("session_ended", False):
            return

        # Handle end session request
        if st.session_state.get("end_session_requested", False):
            delattr(st.session_state, "end_session_requested")

            # Show loading message
            with st.chat_message("assistant"):
                with st.spinner("Preparing your session conclusion..."):
                    # Get conclusion from API
                    conclusion = self._get_session_conclusion()

                # Display conclusion
                st.markdown(f"🎯 **Session Conclusion:**\n\n{conclusion}")

            # Add conclusion to messages and mark session as ended
            st.session_state.messages.append(
                {"role": "assistant", "content": conclusion, "is_conclusion": True}
            )
            st.session_state.session_ended = True
            st.rerun()
            return

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

            # Add the response to messages and rerun to display it properly
            st.session_state.messages.append(
                {"role": "assistant", "content": bot_response}
            )
            st.rerun()

    def _add_user_message(self, content):
        st.session_state.messages.append({"role": "user", "content": content})
        # Set flag to process bot response on next rerun
        st.session_state.pending_user_input = content

    def _get_session_conclusion(self):
        """Get session conclusion from the API"""
        try:
            response = requests.post(
                f"{self.api_url}/chat",
                json={
                    "message": "Please provide a session conclusion.",
                    "session_id": st.session_state.session_id,
                    "end_session": True,
                },
                timeout=60,
            )

            if response.status_code == 200:
                return response.json()["response"]
            else:
                return "Thank you for sharing so openly today. Your willingness to explore your thoughts and feelings shows real courage. Continue to be patient and kind with yourself as you work through these challenges."

        except requests.exceptions.RequestException:
            return "Thank you for our conversation today. Take care of yourself and remember that seeking help is a sign of strength."

    def _get_bot_response(self, user_input):
        try:
            # Call FastAPI backend
            response = requests.post(
                f"{self.api_url}/chat",
                json={
                    "message": user_input,
                    "session_id": st.session_state.session_id,
                },
                timeout=120,
            )

            if response.status_code == 200:
                return response.json()["response"]
            else:
                return "Sorry, I'm having trouble connecting right now."

        except requests.exceptions.RequestException:
            return "Connection error. Please check if the server is running."

    def _stream_response(self, bot_response):
        import time

        # Use Streamlit's built-in streaming function with proper timing
        def response_generator():
            for char in bot_response:
                yield char
                time.sleep(0.005)  # Small delay to create streaming effect

        # Stream the response using st.write_stream
        st.write_stream(response_generator())
