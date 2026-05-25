# ui/dotfield_bg.py
"""Embed React Bits DotField as a full-page Streamlit background."""

import os
import streamlit as st
import streamlit.components.v1 as components

_BG_HTML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "static",
    "dotfield_background.html",
)


def render_dotfield_background():
    """
    Renders DotField behind all Streamlit content.
    Uses static HTML (same logic as frontend/DotField/DotField.jsx).
    """
    with open(_BG_HTML_PATH, encoding="utf-8") as f:
        bg_html = f.read()

    st_layer_css = """
    <style>
    /* Full-viewport background iframe */
    div[data-testid="stHtml"] {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        z-index: 0 !important;
        pointer-events: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stHtml"] iframe {
        width: 100vw !important;
        height: 100vh !important;
        border: none !important;
        pointer-events: none !important;
    }
    [data-testid="stAppViewContainer"] {
        background: transparent !important;
    }
    [data-testid="stHeader"] {
        background: rgba(15, 23, 42, 0.55) !important;
        backdrop-filter: blur(6px);
    }
    section.main {
        position: relative !important;
        z-index: 1 !important;
    }
    section.main .block-container {
        background: rgba(15, 23, 42, 0.78) !important;
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(99, 102, 241, 0.15);
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """

    st.markdown(st_layer_css, unsafe_allow_html=True)
    components.html(bg_html, height=0, scrolling=False)
