import streamlit as st
from constants import *
from config import *
from components.sidebar import Sidebar
from components.chat_interface import ChatInterface
from styles.theme import *

# Page config
st.set_page_config(
    page_title=TITLE,
    layout=LAYOUT,
    initial_sidebar_state=SIDEBAR_STATE,
)


def main():
    # Apply theme
    apply_theme()

    # Initialize components
    sidebar = Sidebar()
    chat_interface = ChatInterface()

    # Render components
    sidebar.render()
    chat_interface.render()


main()
