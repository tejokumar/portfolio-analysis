"""Two-tier password gate for the Streamlit app.

- OWNER_PASSWORD → sees everything (including the personal Dashboard tab).
- APP_PASSWORD   → guest access; same as owner minus the Dashboard.

Either secret can be missing; if only one is set, only that tier exists.
Constant-time comparison avoids timing attacks. The role lives in
st.session_state so it persists across reruns within a session.
"""
from __future__ import annotations

import hmac
import os

import streamlit as st


def _secret(name: str) -> str | None:
    try:
        v = st.secrets.get(name.lower()) or os.getenv(name)
    except (FileNotFoundError, KeyError):
        v = os.getenv(name)
    return v or None


def _owner_pw() -> str | None:
    return _secret("OWNER_PASSWORD")


def _guest_pw() -> str | None:
    return _secret("APP_PASSWORD")


def is_owner() -> bool:
    return st.session_state.get("role") == "owner"


def is_authed() -> bool:
    return bool(st.session_state.get("role"))


def require_password() -> None:
    """Block the app until a recognized password is entered."""
    if is_authed():
        return

    owner_pw = _owner_pw()
    guest_pw = _guest_pw()

    if not owner_pw and not guest_pw:
        st.error(
            "No passwords configured. Set `OWNER_PASSWORD` and/or `APP_PASSWORD` "
            "in `.streamlit/secrets.toml` (or env vars). The owner password "
            "unlocks the Dashboard; the app password is guest-tier."
        )
        st.stop()

    st.title("📈 AI Portfolio Advisor")
    st.caption("Sign in to continue")
    with st.form("login_form", clear_on_submit=False):
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
        if submitted:
            if owner_pw and hmac.compare_digest(pw, owner_pw):
                st.session_state["role"] = "owner"
                st.rerun()
            elif guest_pw and hmac.compare_digest(pw, guest_pw):
                st.session_state["role"] = "guest"
                st.rerun()
            else:
                st.error("Wrong password.")
    st.stop()
