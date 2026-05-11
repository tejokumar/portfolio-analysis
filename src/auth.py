"""Two-tier password gate with persistent session via signed cookie.

- OWNER_PASSWORD → full access including Dashboard.
- APP_PASSWORD   → guest access; Dashboard hidden.

After successful login, a signed token is written to a 30-day cookie so iOS
PWAs (which iOS evicts from memory frequently) don't have to re-prompt every
relaunch. Token is HMAC-SHA256 signed with the owner password as the key —
no plaintext password lives in the cookie.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from datetime import datetime, timedelta, timezone

import streamlit as st

try:
    import extra_streamlit_components as stx
    _COOKIES_AVAILABLE = True
except ImportError:
    _COOKIES_AVAILABLE = False

COOKIE_NAME = "pa_auth"
COOKIE_TTL_DAYS = 30
_VALID_ROLES = {"owner", "guest"}


def _secret(name: str) -> str | None:
    try:
        v = st.secrets.get(name.lower()) or os.getenv(name)
    except Exception:  # noqa: BLE001
        v = os.getenv(name)
    return v or None


def _owner_pw() -> str | None:
    return _secret("OWNER_PASSWORD")


def _guest_pw() -> str | None:
    return _secret("APP_PASSWORD")


def _signing_key() -> bytes:
    """Derive an HMAC key from the owner password (stable across deploys
    as long as the owner password doesn't change)."""
    seed = _owner_pw() or _guest_pw() or "fallback-signing-key-set-OWNER_PASSWORD"
    return hashlib.sha256(seed.encode()).digest()


def _make_token(role: str) -> str:
    expires_at = int(time.time()) + COOKIE_TTL_DAYS * 86400
    payload = f"{role}|{expires_at}"
    sig = hmac.new(_signing_key(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}|{sig}"


def _verify_token(token: str | None) -> str | None:
    if not token:
        return None
    try:
        role, expires_at, sig = token.split("|", 2)
    except (ValueError, AttributeError):
        return None
    expected_sig = hmac.new(
        _signing_key(),
        f"{role}|{expires_at}".encode(),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        return None
    try:
        if int(expires_at) < int(time.time()):
            return None
    except ValueError:
        return None
    if role not in _VALID_ROLES:
        return None
    return role


def _cookie_manager():
    """CookieManager is a Streamlit widget, so it can't live inside cached
    functions. Instantiate fresh each call — the `key` arg ensures the
    underlying iframe component is the same DOM element across reruns."""
    if not _COOKIES_AVAILABLE:
        return None
    try:
        return stx.CookieManager(key="pa_cookie_manager")
    except Exception:  # noqa: BLE001
        return None


def is_owner() -> bool:
    return st.session_state.get("role") == "owner"


def is_authed() -> bool:
    return bool(st.session_state.get("role"))


def require_password() -> None:
    """Block the app until a recognized password is entered. Persists across
    PWA relaunches via a 30-day signed cookie."""
    if is_authed():
        return

    # Try cookie first — instant re-auth for returning users.
    cm = _cookie_manager()
    if cm is not None:
        try:
            token = cm.get(COOKIE_NAME)
        except Exception:  # noqa: BLE001 — cookie component may not be ready
            token = None
        role = _verify_token(token)
        if role:
            st.session_state["role"] = role
            return

    owner_pw = _owner_pw()
    guest_pw = _guest_pw()

    if not owner_pw and not guest_pw:
        st.error(
            "No passwords configured. Set `OWNER_PASSWORD` and/or `APP_PASSWORD` "
            "in `.streamlit/secrets.toml` (or env vars)."
        )
        st.stop()

    st.title("📈 AI Portfolio Advisor")
    st.caption("Sign in to continue · session stays active for 30 days")
    with st.form("login_form", clear_on_submit=False):
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
        if submitted:
            role = None
            if owner_pw and hmac.compare_digest(pw, owner_pw):
                role = "owner"
            elif guest_pw and hmac.compare_digest(pw, guest_pw):
                role = "guest"

            if role:
                st.session_state["role"] = role
                if cm is not None:
                    try:
                        expires = datetime.now(timezone.utc) + timedelta(days=COOKIE_TTL_DAYS)
                        cm.set(COOKIE_NAME, _make_token(role), expires_at=expires)
                    except Exception:  # noqa: BLE001
                        pass  # cookie set failed — session_state auth still works
                st.rerun()
            else:
                st.error("Wrong password.")
    st.stop()


def sign_out() -> None:
    """Clear both session-state and the persistent cookie."""
    st.session_state.pop("role", None)
    cm = _cookie_manager()
    if cm is not None:
        try:
            cm.delete(COOKIE_NAME)
        except Exception:  # noqa: BLE001
            pass
