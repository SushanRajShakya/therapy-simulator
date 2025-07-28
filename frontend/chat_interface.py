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
        # Always render header first to keep it at top
        self._render_header()
        
        # Chat content
        self._display_messages()
        self._handle_input()
        self._render_session_completion_message()

    def _render_header(self):
        # Create a prominent header that stays at top
        st.markdown("### Therapy Session Simulator")
        st.info(
            "💬 Have a conversation with your CBT therapist. The bot will naturally recognize when you're ready to end the session, or you can use the manual controls in the sidebar."
        )
        st.markdown("---")  # Add separator after header

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

    def _render_session_completion_message(self):
        # Only show completion message if session is ended
        if st.session_state.get("session_ended", False):
            st.markdown("---")
            st.success(
                "✅ Session completed. Thank you for using the therapy simulator!"
            )

    def _handle_input(self):
        # Don't allow input if session is ended
        if st.session_state.get("session_ended", False):
            return

        # Handle end session request
        if st.session_state.get("end_session_requested", False):
            delattr(st.session_state, "end_session_requested")

            # Check if there's therapeutic content before ending
            if not self._has_therapeutic_content():
                st.error("⚠️ Cannot end session without meaningful therapeutic conversation. Please share your thoughts or concerns first.")
                return

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

            # Check if this was a natural session conclusion
            if st.session_state.get("is_natural_conclusion", False):
                # Add the response as a conclusion to messages
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": bot_response,
                        "is_conclusion": True,
                    }
                )
                delattr(st.session_state, "is_natural_conclusion")
            else:
                # Add the response to messages normally
                st.session_state.messages.append(
                    {"role": "assistant", "content": bot_response}
                )

            st.rerun()

    def _add_user_message(self, content):
        st.session_state.messages.append({"role": "user", "content": content})
        # Set flag to process bot response on next rerun
        st.session_state.pending_user_input = content

    def _has_therapeutic_content(self):
        """Check if the session has meaningful therapeutic content"""
        messages = st.session_state.get("messages", [])
        
        # Need at least one user message and one assistant response for therapeutic content
        if len(messages) < 2:
            return False
            
        # Check if there are any user messages that seem therapeutic
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        
        # Simple heuristic: if user has sent more than just greetings/procedural messages
        # or if there are multiple exchanges, assume therapeutic content exists
        if len(user_messages) >= 2:
            return True
            
        # Check if the single user message seems substantive (not just greeting)
        if len(user_messages) == 1:
            user_msg = user_messages[0]["content"].lower().strip()
            greeting_patterns = [
                "hi", "hello", "hey", "good morning", "good afternoon", 
                "nice to meet", "should we start", "how are you", "doctor"
            ]
            # If message is longer than 20 chars and doesn't match greeting patterns, consider it therapeutic
            if len(user_msg) > 20 and not any(pattern in user_msg for pattern in greeting_patterns):
                return True
                
        return False

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
                response_data = response.json()

                # Check if the session was naturally ended by the bot
                if response_data.get("is_session_ended", False):
                    st.session_state.session_ended = True
                    # Mark this as a conclusion for special formatting
                    st.session_state.is_natural_conclusion = True

                return response_data["response"]
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
