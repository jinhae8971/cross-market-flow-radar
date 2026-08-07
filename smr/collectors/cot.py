"""CFTC COT — 투기적 포지션 (유럽·미국·일본 공통).

Non-commercial(투기 계정)의 순포지션 '변화'를 흐름으로 환산한다.
주간(화요일 기준, 금요일 공표)이므로 lag_days=3, confidence는 중간.

유럽에 특히 중요한 이유: 유럽은 한국·일본과 달리 거래소 차원의
투자자별 매매동향 공시가 없다. EUR 선물 투기 포지션이 역외 자금의
유럽 익스포저를 가장 빠르게 드러내는 공개 지표다.
"""
from __future__ import annotations

import datetime as dt

from functools import lru_cache

import requests
import yfinance as yf

from ..schema import FlowRecord
from ..fx import to_usd

ENDPOINT = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"

# 시장 → (COT 계약명, 계약승수, 기준가 심볼, 승수 표시통화)
#   notional = Δ순계약수 × 승수 × 기준가  (그 뒤 USD 환산)
CONTRACTS = {
    "EU": ("EURO FX - CHICAGO MERCANTILE EXCHANGE", 125_000, "EURUSD=X", "USD"),
    "JP": ("NIKKEI STOCK AVERAGE YEN DENOM - CHICAGO MERCANTILE EXCHANGE", 500, "^N225", "JPY"),
    "US": ("S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE", 50, "^GSPC", "USD"),
}


def _series(name: str, limit: int = 12) -> list[dict]:
    r = requests.get(
        ENDPOINT,
        params={
            "$limit": limit,
            "$order": "report_date_as_yyyy_mm_dd DESC",
            "$where": f"market_and_exchange_names = '{name}'",
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _net(row: dict) -> float:
    long_ = float(row.get("noncomm_positions_long_all", 0) or 0)
    short = float(row.get("noncomm_positions_short_all", 0) or 0)
    return long_ - short


@lru_cache(maxsize=8)
def _spot(symbol: str) -> float:
    hist = yf.Ticker(symbol).history(period="5d")
    if hist.empty:
        raise RuntimeError(f"{symbol} 기준가 조회 실패")
    return float(hist["Close"].iloc[-1])


def collect(weeks: int = 8) -> list[FlowRecord]:
    out: list[FlowRecord] = []
    for market, (name, mult, px_sym, ccy) in CONTRACTS.items():
        try:
            rows = _series(name, limit=weeks + 1)
        except Exception as exc:
            print(f"[cot] {market} 조회 실패: {exc}")
            continue
        rows = sorted(rows, key=lambda r: r["report_date_as_yyyy_mm_dd"])
        for prev, cur in zip(rows, rows[1:]):
            day = dt.datetime.fromisoformat(
                cur["report_date_as_yyyy_mm_dd"].replace("Z", "")
            ).date()
            delta = _net(cur) - _net(prev)
            try:
                px = _spot(px_sym)
            except Exception as exc:
                print(f"[cot] {market} 기준가 실패: {exc}")
                continue
            notional = to_usd(delta * mult * px, ccy, day)
            out.append(
                FlowRecord(
                    ts=day,
                    market=market,
                    actor="spec",
                    instrument=name.split(" - ")[0],
                    net_flow_usd=round(notional, 2),
                    lag_days=3,
                    confidence=0.6,
                    source="cftc_cot",
                )
            )
    return out
