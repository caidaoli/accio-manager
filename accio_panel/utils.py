from __future__ import annotations

import random
import time
from datetime import datetime


def new_utdid() -> str:
    timestamp = int(time.time() * 1000)
    random_hex = "".join(random.choice("0123456789abcdef") for _ in range(16))
    return f"utd-{timestamp}-{random_hex}"


def mask_token(token: str, prefix: int = 10, suffix: int = 6) -> str:
    if len(token) <= prefix + suffix:
        return token
    return f"{token[:prefix]}...{token[-suffix:]}"


def format_timestamp(timestamp: int | None) -> str:
    if not timestamp:
        return "未知"
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def format_countdown_hours(seconds: int | float | None) -> str:
    if seconds in (None, "", 0):
        return "-"
    try:
        total = float(seconds)
    except (TypeError, ValueError):
        return "-"
    if total <= 0:
        return "即将重置"
    hours = round(total / 3600, 1)
    return f"{hours} 小时后"
