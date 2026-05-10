"""Market-aware clock for cache scheduling.

All schedules are anchored to America/New_York and skip weekends. A "bucket key"
is a string that changes exactly when fresh data should be fetched — feed it as
an argument to st.cache_data and the fetcher reruns whenever the bucket rolls.

Schedules:
  - holdings:       once per evening (18:00 ET).
  - fmp / grok:     once at 09:15 ET pre-market on weekdays.
  - news:           every 15 minutes during market hours, plus pre-market 09:15.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
DISPLAY_TZ = ZoneInfo("America/Los_Angeles")  # PST/PDT — markets remain ET-based.
DISPLAY_LABEL = "PT"

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
PREMARKET_REFRESH = time(9, 15)
EVENING_REFRESH = time(18, 0)
PRE_BRIEFING_REFRESH = time(9, 15)   # 15 min before open
POST_BRIEFING_REFRESH = time(17, 0)  # 1 hr after close


def now_et() -> datetime:
    return datetime.now(ET)


def to_display(dt_: datetime) -> datetime:
    """Convert any tz-aware datetime to the display timezone (Pacific)."""
    return dt_.astimezone(DISPLAY_TZ)


def now_display() -> datetime:
    return to_display(now_et())


def fmt_dt(dt_: datetime, with_date: bool = True) -> str:
    """Format a datetime in the display timezone, e.g. '2026-05-10 06:15 PT'."""
    local = to_display(dt_)
    if with_date:
        return f"{local:%Y-%m-%d %H:%M} {DISPLAY_LABEL}"
    return f"{local:%H:%M} {DISPLAY_LABEL}"


def is_weekday(d: datetime) -> bool:
    return d.weekday() < 5


def _previous_weekday(d: datetime) -> datetime:
    d = d - timedelta(days=1)
    while not is_weekday(d):
        d -= timedelta(days=1)
    return d


def _next_weekday(d: datetime) -> datetime:
    d = d + timedelta(days=1)
    while not is_weekday(d):
        d += timedelta(days=1)
    return d


def _at(dt_: datetime, t: time) -> datetime:
    return dt_.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)


# ---- News: every 15 min during market hours; pre-market refresh at 9:15 ----

def news_bucket(n: datetime | None = None) -> str:
    n = n or now_et()
    if not is_weekday(n):
        return _previous_weekday(n).strftime("%Y%m%d-postclose")
    t = n.time()
    if t < PREMARKET_REFRESH:
        return _previous_weekday(n).strftime("%Y%m%d-postclose")
    if PREMARKET_REFRESH <= t < MARKET_OPEN:
        return n.strftime("%Y%m%d-premkt")
    if MARKET_OPEN <= t < MARKET_CLOSE:
        slot_min = (t.minute // 15) * 15
        return n.strftime(f"%Y%m%d-{t.hour:02d}{slot_min:02d}")
    return n.strftime("%Y%m%d-postclose")


def next_news_refresh(n: datetime | None = None) -> datetime:
    n = n or now_et()
    if not is_weekday(n):
        return _at(_next_weekday(n), PREMARKET_REFRESH)
    t = n.time()
    if t < PREMARKET_REFRESH:
        return _at(n, PREMARKET_REFRESH)
    if PREMARKET_REFRESH <= t < MARKET_OPEN:
        return _at(n, MARKET_OPEN)
    if MARKET_OPEN <= t < MARKET_CLOSE:
        next_min = ((n.minute // 15) + 1) * 15
        if next_min >= 60:
            return (n.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
        return n.replace(minute=next_min, second=0, microsecond=0)
    return _at(_next_weekday(n), PREMARKET_REFRESH)


# ---- FMP / Grok: once per weekday at 9:15 ET pre-market ----

def daily_premarket_bucket(n: datetime | None = None) -> str:
    n = n or now_et()
    if is_weekday(n) and n.time() >= PREMARKET_REFRESH:
        return n.strftime("%Y%m%d-premkt")
    return _previous_weekday(n).strftime("%Y%m%d-premkt")


def next_premarket_refresh(n: datetime | None = None) -> datetime:
    n = n or now_et()
    today_915 = _at(n, PREMARKET_REFRESH)
    if is_weekday(n) and n < today_915:
        return today_915
    return _at(_next_weekday(n), PREMARKET_REFRESH)


# ---- Holdings: once per evening at 18:00 ET ----

def holdings_bucket(n: datetime | None = None) -> str:
    n = n or now_et()
    if n.time() >= EVENING_REFRESH:
        return n.strftime("%Y%m%d-evening")
    return (n - timedelta(days=1)).strftime("%Y%m%d-evening")


def next_holdings_refresh(n: datetime | None = None) -> datetime:
    n = n or now_et()
    today_18 = _at(n, EVENING_REFRESH)
    if n < today_18:
        return today_18
    return today_18 + timedelta(days=1)


# ---- Pre-market briefing: 09:15 ET on weekdays ----

def pre_briefing_bucket(n: datetime | None = None) -> str:
    n = n or now_et()
    if is_weekday(n) and n.time() >= PRE_BRIEFING_REFRESH:
        return n.strftime("%Y%m%d-pre")
    return _previous_weekday(n).strftime("%Y%m%d-pre")


def next_pre_briefing(n: datetime | None = None) -> datetime:
    n = n or now_et()
    today_915 = _at(n, PRE_BRIEFING_REFRESH)
    if is_weekday(n) and n < today_915:
        return today_915
    return _at(_next_weekday(n), PRE_BRIEFING_REFRESH)


# ---- Post-market briefing: 17:00 ET on weekdays (1 hr after close) ----

def post_briefing_bucket(n: datetime | None = None) -> str:
    n = n or now_et()
    if is_weekday(n) and n.time() >= POST_BRIEFING_REFRESH:
        return n.strftime("%Y%m%d-post")
    return _previous_weekday(n).strftime("%Y%m%d-post")


def next_post_briefing(n: datetime | None = None) -> datetime:
    n = n or now_et()
    today_17 = _at(n, POST_BRIEFING_REFRESH)
    if is_weekday(n) and n < today_17:
        return today_17
    return _at(_next_weekday(n), POST_BRIEFING_REFRESH)


# ---- Active polling window (X Chatter + FlowGod) ----
# Both are scoped to weekday 09:15 ET pre-market through 17:00 ET (1 hr after
# close). Outside that window, the bucket key freezes to the last in-window
# slot so cached data persists and no Grok calls fire.

ACTIVE_WINDOW_START = PRE_BRIEFING_REFRESH   # 09:15 ET
ACTIVE_WINDOW_END = POST_BRIEFING_REFRESH    # 17:00 ET


def in_active_window(n: datetime | None = None) -> bool:
    n = n or now_et()
    return is_weekday(n) and ACTIVE_WINDOW_START <= n.time() < ACTIVE_WINDOW_END


def _last_active_dt(n: datetime) -> datetime:
    """Most recent in-window moment ≤ n. Used to anchor off-hours bucket keys
    to the final in-window slot so the cache stays warm."""
    if in_active_window(n):
        return n
    if is_weekday(n) and n.time() >= ACTIVE_WINDOW_END:
        # Today's window has already closed — anchor to its final minute.
        return _at(n, ACTIVE_WINDOW_END) - timedelta(minutes=1)
    # Before today's window opens, or weekend: roll back to previous weekday's close.
    prev = _previous_weekday(n)
    return _at(prev, ACTIVE_WINDOW_END) - timedelta(minutes=1)


def _next_active_open(n: datetime) -> datetime:
    """When does the next in-window period start?"""
    if is_weekday(n) and n.time() < ACTIVE_WINDOW_START:
        return _at(n, ACTIVE_WINDOW_START)
    return _at(_next_weekday(n), ACTIVE_WINDOW_START)


def _slot_bucket(dt_: datetime, slot_size_min: int, suffix: str) -> str:
    slot = (dt_.minute // slot_size_min) * slot_size_min
    return dt_.strftime(f"%Y%m%d-{dt_.hour:02d}{slot:02d}-{suffix}")


def _next_slot_dt(n: datetime, slot_size_min: int) -> datetime:
    next_min = ((n.minute // slot_size_min) + 1) * slot_size_min
    if next_min < 60:
        return n.replace(minute=next_min, second=0, microsecond=0)
    return n.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)


# ---- X Hot Chatter: every 15 minutes during active window ----

def chatter_bucket(n: datetime | None = None) -> str:
    n = n or now_et()
    eff = n if in_active_window(n) else _last_active_dt(n)
    return _slot_bucket(eff, 15, "chat")


def next_chatter_refresh(n: datetime | None = None) -> datetime:
    n = n or now_et()
    if in_active_window(n):
        return _next_slot_dt(n, 15)
    return _next_active_open(n)


# ---- FlowGod: every 10 minutes during active window ----

def flowgod_bucket(n: datetime | None = None) -> str:
    n = n or now_et()
    eff = n if in_active_window(n) else _last_active_dt(n)
    return _slot_bucket(eff, 10, "flow")


def next_flowgod_refresh(n: datetime | None = None) -> datetime:
    n = n or now_et()
    if in_active_window(n):
        return _next_slot_dt(n, 10)
    return _next_active_open(n)


# ---- Display helpers ----

def fmt_age(then: datetime, now: datetime | None = None) -> str:
    """Human-readable age like '5 min ago' or '2 hr ago'."""
    now = now or now_et()
    delta = now - then
    secs = int(delta.total_seconds())
    if secs < 0:
        secs = 0
    if secs < 60:
        return f"{secs} sec ago"
    mins = secs // 60
    if mins < 60:
        return f"{mins} min ago"
    hrs = mins // 60
    if hrs < 24:
        return f"{hrs} hr ago"
    days = hrs // 24
    return f"{days} day{'s' if days != 1 else ''} ago"


def fmt_until(future: datetime, now: datetime | None = None) -> str:
    """Human-readable countdown like 'in 12 min' or 'in 3 hr'."""
    now = now or now_et()
    delta = future - now
    secs = int(delta.total_seconds())
    if secs <= 0:
        return "any moment"
    mins = secs // 60
    if mins < 60:
        return f"in {mins} min"
    hrs = mins // 60
    if hrs < 24:
        return f"in {hrs} hr"
    days = hrs // 24
    return f"in {days} day{'s' if days != 1 else ''}"
