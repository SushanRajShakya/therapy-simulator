import streamlit as st


def apply_theme():
    # Custom theme styling
    st.markdown(
        """
        <style>
        :root {
            --primary-color: #1987bc;
        }

        .stButton > button[kind="primary"] {
            background-color: var(--primary-color);
            border-color: var(--primary-color);
        }

        .stButton > button[kind="primary"]:hover {
            background-color: #2299d2;
            border-color: #2299d2;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
