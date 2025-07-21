import streamlit as st
from streamlit_chat import message

# Header
st.title("Therapy Session Simulator")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# Chat input
if user_input := st.chat_input("Type your message here:"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Add chat bot response to chat
    bot_response = f"I hear you saying: {user_input}. Tell me more about that."
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

    # Rerun to update the display
    st.rerun()
