"""국가 ETF 추정 유출입 — 4개 시장 공통 백본.

왜 이게 백본인가:
  KRX/JPX/유럽 규제기관의 원천 데이터는 형식·지연·접근성이 제각각이고
  클라우드 IP를 차단하는 곳도 있다. 반면 미국 상장 국가 ETF의 AUM은
  4개 시장 모두에 대해 동일한 방식·동일한 지연(T+0)으로 얻을 수 있다.
  따라서 ETF 흐름을 '항상 도는 최소 신뢰 축'으로 두고,
  각국 원천 데이터는 해상도를 높이는 enrichment로 붙인다.

추정식 (업계 표준):
    flow_t = AUM_t - AUM_{t-1} * (1 + r_t)
  = 가격 상승분으로 설명되지 않는 자산 증가분 = 순설정(창출/환매)

한계: AUM 스냅샷 기반이라 소급 수집이 불가하다. 하루라도 놓치면 그날 흐름은
      영구 결측이므로 워크플로우 실패 알림이 필수다(collect.yml에 구성).
"""
from __future__ import annotations

import datetime as dt
import json
import os
from typing import Sequence

import pandas as pd
import yfinance as yf

from ..schema import FlowRecord

# 시장별 대표 ETF. 역외 투자자의 국가 배분을 대리한다.
UNIVERSE: dict[str, tuple[str, ...]] = {
    "KR": ("EWY", "FLKR"),
    "JP": ("EWJ", "DXJ", "BBJP"),
    "EU": ("VGK", "EZU", "FEZ", "EWG", "EWU", "EWQ"),
    "US": ("SPY", "QQQ", "IWM", "DIA"),
}

AUM_CACHE = "data/aum_snapshots.json"


def _snapshot(tickers: Sequence[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for sym in tickers:
        try:
            info = yf.Ticker(sym).info
        except Exception as exc:  # 개별 종목 실패가 전체를 죽이지 않게
            print(f"[etf_flow] {sym} info 실패: {exc}")
            continue
        aum = info.get("totalAssets")
        px = info.get("previousClose") or info.get("navPrice")
        if not aum or not px:
            print(f"[etf_flow] {sym} AUM/가격 결측 — 스킵")
            continue
        out[sym] = {"aum": float(aum), "px": float(px)}
    return out


def _load_cache(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def collect(as_of: dt.date | None = None, cache_path: str = AUM_CACHE) -> list[FlowRecord]:
    as_of = as_of or dt.date.today()
    cache = _load_cache(cache_path)
    prev = cache.get("latest", {})
    prev_date = cache.get("date")

    records: list[FlowRecord] = []
    fresh: dict[str, dict] = {}

    for market, tickers in UNIVERSE.items():
        snap = _snapshot(tickers)
        fresh.update(snap)
        if not prev:
            continue
        for sym, cur in snap.items():
            old = prev.get(sym)
            if not old or not old.get("px"):
                continue
            ret = cur["px"] / old["px"] - 1.0
            flow = cur["aum"] - old["aum"] * (1.0 + ret)
            # 일 AUM의 0.02% 미만은 반올림 노이즈로 간주
            if abs(flow) < cur["aum"] * 2e-4:
                flow = 0.0
            records.append(
                FlowRecord(
                    ts=as_of,
                    market=market,
                    actor="foreign",
                    instrument=sym,
                    net_flow_usd=round(flow, 2),
                    lag_days=0,
                    confidence=0.75,  # 추정식 기반이므로 원천 공시보다 낮게
                    source="etf_aum_delta",
                )
            )

    if fresh:
        _save_cache(cache_path, {"date": as_of.isoformat(), "latest": fresh,
                                 "prev_date": prev_date})
    return records


def load_price_history(period: str = "6mo") -> pd.DataFrame:
    """확증 레이어용 국가 ETF 종가. 흐름과 가격이 같은 방향인지 검증한다."""
    syms = [s for t in UNIVERSE.values() for s in t]
    df = yf.download(syms, period=period, progress=False, auto_adjust=True)["Close"]
    return df.dropna(how="all")


def backfill(period: str = "1y") -> list[FlowRecord]:
    """과거 구간 시딩 — 자금흐름 승수(CMF 계열) 기반 대용치.

    AUM 스냅샷 방식은 소급이 불가하므로, 시스템 가동 이전 구간은
    일중 종가 위치와 거래대금으로 매집/분산을 추정해 채운다.
        MFM = ((C-L) - (H-C)) / (H-L)      종가가 고가에 가까울수록 +1
        MFV = MFM × 거래대금                 = 그날의 매집 강도(달러)
    확정치가 아니므로 confidence=0.5. 이후 실측 AUM 흐름이 같은 키를
    덮어쓰면 자동으로 신뢰도가 올라간다(FlowStore.upsert가 keep='last').
    """
    syms = [s for t in UNIVERSE.values() for s in t]
    raw = yf.download(syms, period=period, progress=False,
                      auto_adjust=False, group_by="ticker")
    sym_to_market = {s: m for m, t in UNIVERSE.items() for s in t}

    records: list[FlowRecord] = []
    for sym in syms:
        try:
            d = raw[sym].dropna()
        except KeyError:
            continue
        if d.empty:
            continue
        rng = (d["High"] - d["Low"]).replace(0, pd.NA)
        mfm = (((d["Close"] - d["Low"]) - (d["High"] - d["Close"])) / rng).fillna(0.0)
        mfv = mfm * d["Close"] * d["Volume"]
        for ts, val in mfv.items():
            if pd.isna(val):
                continue
            records.append(
                FlowRecord(
                    ts=ts.date(),
                    market=sym_to_market[sym],
                    actor="foreign",
                    instrument=sym,
                    net_flow_usd=round(float(val), 2),
                    lag_days=0,
                    confidence=0.5,
                    source="etf_moneyflow_proxy",
                )
            )
    return records
