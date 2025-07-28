import streamlit as st

from constants import *


class Sidebar:
    def __init__(self):
        pass

    def render(self):
        with st.sidebar:
            self._render_header()
            self._render_session_controls()
            self._render_session_info()

    def _render_header(self):
        st.header(SIDEBAR_HEADER)

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
                "hi",
                "hello",
                "hey",
                "good morning",
                "good afternoon",
                "nice to meet",
                "should we start",
                "how are you",
                "doctor",
            ]
            # If message is longer than 20 chars and doesn't match greeting patterns, consider it therapeutic
            if len(user_msg) > 20 and not any(
                pattern in user_msg for pattern in greeting_patterns
            ):
                return True

        return False

    def _render_session_controls(self):
        # Check if session is active
        session_active = not st.session_state.get("session_ended", False)

        if session_active:
            # Active session controls
            st.markdown("**Current Session**")

            # Check if there's therapeutic content to end
            has_therapeutic_content = self._has_therapeutic_content()

            # End session button
            with st.expander("End Session", expanded=False):
                if has_therapeutic_content:
                    st.caption(
                        "The bot can naturally detect when you want to end the session, or you can manually end it here."
                    )
                    if st.button(
                        "End Session Manually",
                        type="secondary",
                        help="Manually end the session and get a therapeutic conclusion",
                        use_container_width=True,
                        key="end_session_sidebar",
                    ):
                        st.session_state.end_session_requested = True
                        st.rerun()
                else:
                    st.caption(
                        "⚠️ Start a conversation with meaningful content before ending the session. The bot needs therapeutic context to provide a proper conclusion."
                    )
                    st.button(
                        "End Session Manually",
                        type="secondary",
                        help="Cannot end session without therapeutic content",
                        use_container_width=True,
                        disabled=True,
                        key="end_session_sidebar_disabled",
                    )

            st.markdown("---")

            # New session button (for starting fresh)
            if st.button(
                "🆕 Start New Session", use_container_width=True, type="primary"
            ):
                st.session_state.clear()
                st.rerun()

        else:
            # Session ended - show restart option
            st.markdown("**Session Ended**")
            if st.button(
                "🔄 Start New Session", use_container_width=True, type="primary"
            ):
                st.session_state.clear()
                st.rerun()

    def _render_session_info(self):
        st.markdown("---")
        st.subheader(SESSION_INFO_HEADER)
        message_count = len(st.session_state.get("messages", []))
        st.metric(MESSAGES_METRIC, message_count)
