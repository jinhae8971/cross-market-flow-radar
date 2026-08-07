"""로컬통화 → USD 환산.

원칙: 원천 수집은 로컬통화로 저장하고, 환산은 신호 계층 직전에 한다.
국경 간 배분에서는 '환율 변동 자체가 신호'이므로 두 축을 분리해야
"주가 유입인가 / 통화 되돌림인가"를 구분할 수 있다.
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache

import yfinance as yf

PAIR = {"KRW": "KRW=X", "JPY": "JPY=X", "EUR": "EURUSD=X", "GBP": "GBPUSD=X"}
INVERTED = {"KRW", "JPY"}  # USD/XXX 로 호가되므로 나눗셈


@lru_cache(maxsize=64)
def _rate(ccy: str, day_iso: str) -> float:
    if ccy == "USD":
        return 1.0
    sym = PAIR.get(ccy)
    if not sym:
        raise KeyError(f"환율 미지원 통화: {ccy}")
    day = dt.date.fromisoformat(day_iso)
    hist = yf.Ticker(sym).history(
        start=day - dt.timedelta(days=7), end=day + dt.timedelta(days=1)
    )
    if hist.empty:
        raise RuntimeError(f"{sym} 환율 조회 실패 ({day_iso})")
    return float(hist["Close"].iloc[-1])


def to_usd(amount: float, ccy: str, day: dt.date) -> float:
    r = _rate(ccy, day.isoformat())
    return amount / r if ccy in INVERTED else amount * r
