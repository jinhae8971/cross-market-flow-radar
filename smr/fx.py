"""로컬통화 → USD 환산.

원칙: 원천 수집은 로컬통화로 저장하고, 환산은 신호 계층 직전에 한다.
국경 간 배분에서는 '환율 변동 자체가 신호'이므로 두 축을 분리해야
"주가 유입인가 / 통화 되돌림인가"를 구분할 수 있다.

원천 (2026-09-23 교체):
  1순위 ECB 기준환율(Frankfurter) — 날짜가 명시되고, 구간 1회 요청으로 수십 일을
        받는다. 러너에서 정상 동작 확인.
  2순위 yfinance — 러너 IP가 간헐적으로 429를 맞는다(2026-09-17 전면 차단 이력).
환율이 없는 날(주말·ECB 휴일)은 7일 이내 직전 기준일 값을 쓴다.
"""
from __future__ import annotations

import bisect
import datetime as dt

import requests

FRANKFURTER = "https://api.frankfurter.app/{start}..{end}"
CURRENCIES = ("KRW", "JPY", "EUR", "GBP")
PAIR = {"KRW": "KRW=X", "JPY": "JPY=X", "EUR": "EURUSD=X", "GBP": "GBPUSD=X"}
MAX_LOOKBACK_DAYS = 7

# 통화 → 정렬된 [(날짜, USD 1달러당 통화 수량)]
_CACHE: dict[str, list[tuple[dt.date, float]]] = {}
# 실제로 조회를 마친 구간. 캐시에 '가까운 과거 값'이 있다는 이유로 조회를 건너뛰면
# 조회 순서에 따라 같은 날짜가 다른 환율로 환산된다(재실행 때마다 값이 바뀐다).
_COVERED: list[tuple[dt.date, dt.date]] = []


def _covered(day: dt.date) -> bool:
    return any(a <= day <= b for a, b in _COVERED)


def _merge(ccy: str, pairs: list[tuple[dt.date, float]]) -> None:
    cur = dict(_CACHE.get(ccy, []))
    cur.update(pairs)
    _CACHE[ccy] = sorted(cur.items())


def prefetch(start: dt.date, end: dt.date) -> bool:
    """구간 환율을 한 번에 받아 캐시한다. 실패해도 예외를 던지지 않는다."""
    try:
        r = requests.get(
            FRANKFURTER.format(start=(start - dt.timedelta(days=MAX_LOOKBACK_DAYS)).isoformat(),
                               end=end.isoformat()),
            params={"from": "USD", "to": ",".join(CURRENCIES)}, timeout=20)
        r.raise_for_status()
        rates = r.json().get("rates", {})
    except Exception as exc:  # noqa: BLE001
        print(f"[fx] ECB 환율 조회 실패 — yfinance로 대체: {exc}")
        return False
    for ccy in CURRENCIES:
        _merge(ccy, [(dt.date.fromisoformat(d), float(v[ccy]))
                     for d, v in rates.items() if ccy in v])
    _COVERED.append((start, end))
    return True


def _lookup(ccy: str, day: dt.date) -> float | None:
    rows = _CACHE.get(ccy) or []
    i = bisect.bisect_right(rows, (day, float("inf"))) - 1
    if i >= 0 and (day - rows[i][0]).days <= MAX_LOOKBACK_DAYS:
        return rows[i][1]
    return None


def _yahoo(ccy: str, day: dt.date) -> float:
    import yfinance as yf

    hist = yf.Ticker(PAIR[ccy]).history(start=day - dt.timedelta(days=MAX_LOOKBACK_DAYS),
                                        end=day + dt.timedelta(days=1))
    if hist.empty:
        raise RuntimeError(f"{PAIR[ccy]} 환율 조회 실패 ({day})")
    px = float(hist["Close"].iloc[-1])
    # KRW=X·JPY=X는 USD당 통화, EURUSD=X·GBPUSD=X는 통화당 USD로 호가된다.
    return px if ccy in ("KRW", "JPY") else 1.0 / px


def per_usd(ccy: str, day: dt.date) -> float:
    """USD 1달러당 통화 수량."""
    if ccy == "USD":
        return 1.0
    if ccy not in PAIR:
        raise KeyError(f"환율 미지원 통화: {ccy}")
    if not _covered(day):
        prefetch(day, day)
    rate = _lookup(ccy, day) if _covered(day) else None
    if rate is None:
        rate = _yahoo(ccy, day)
        _merge(ccy, [(day, rate)])
    return rate


def to_usd(amount: float, ccy: str, day: dt.date) -> float:
    return amount / per_usd(ccy, day)
