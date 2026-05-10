"""Lightweight password gate for the Streamlit app.

Reads the expected password from st.secrets["app_password"] (Streamlit Cloud)
or APP_PASSWORD env var (local). Uses a constant-time comparison to avoid
timing attacks. The auth flag lives in st.session_state so it persists across
reruns within a session.
"""
from __future__ import annotations

import hmac
import os

import streamlit as st


def _expected_password() -> str | None:
    try:
        return st.secrets.get("app_password") or os.getenv("APP_PASSWORD")
    except (FileNotFoundError, KeyError):
        return os.getenv("APP_PASSWORD")


def require_password() -> None:
    """Block the app until the correct password is entered. Call once at the top of main()."""
    if st.session_state.get("auth_ok"):
        return

    expected = _expected_password()
    if not expected:
        st.error(
            "APP_PASSWORD is not configured. Set `app_password` in "
            "`.streamlit/secrets.toml` (or the APP_PASSWORD env var)."
        )
        st.stop()

    st.title("📈 AI Portfolio Advisor")
    st.caption("Sign in to continue")
    with st.form("login_form", clear_on_submit=False):
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
        if submitted:
            if hmac.compare_digest(pw, expected):
                st.session_state["auth_ok"] = True
                st.rerun()
            else:
                st.error("Wrong password.")
    st.stop()
