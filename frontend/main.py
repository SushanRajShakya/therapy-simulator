import streamlit as st
from constants import *
from config import *
from sidebar import Sidebar
from chat_interface import ChatInterface

# Page config
st.set_page_config(
    page_title=TITLE,
    layout=LAYOUT,
    initial_sidebar_state=SIDEBAR_STATE,
)


def main():
    # Initialize components
    sidebar = Sidebar()
    chat_interface = ChatInterface()

    # Render components
    sidebar.render()
    chat_interface.render()


main()
