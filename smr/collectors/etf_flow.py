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

SYM_TO_MARKET: dict[str, str] = {s: m for m, t in UNIVERSE.items() for s in t}


def _snapshot(tickers: Sequence[str]) -> dict[str, dict]:
    """티커별 AUM 스냅샷. 가격은 여기서 받지 않는다(세션 정합성 때문)."""
    out: dict[str, dict] = {}
    for sym in tickers:
        try:
            info = yf.Ticker(sym).info
        except Exception as exc:  # 개별 종목 실패가 전체를 죽이지 않게
            print(f"[etf_flow] {sym} info 실패: {exc}")
            continue
        aum = info.get("totalAssets")
        if not aum:
            print(f"[etf_flow] {sym} AUM 결측 — 스킵")
            continue
        out[sym] = {"aum": float(aum)}
    return out


def _closes(tickers: Sequence[str], period: str = "10d") -> pd.DataFrame:
    """최근 종가. 인덱스가 '실제 세션일'이므로 날짜 라벨의 단일 진실 원천이다.

    이전 구현은 info['previousClose']를 썼는데, 이 값에는 날짜가 없다.
    그래서 as_of를 dt.date.today()로 붙였고, 워크플로우가 TZ=Asia/Seoul로
    돌기 때문에 미국 마감 직후(=KST 다음 날 새벽) 실행분이 하루 뒤 날짜로
    기록됐다. 토요일자 레코드가 생긴 원인이다.
    """
    # auto_adjust=False 의도적 — 배당락일에는 AUM도 실제로 줄어든다.
    # 배당 조정 가격을 쓰면 그 감소분이 '유출'로 잘못 잡힌다.
    df = yf.download(list(tickers), period=period, progress=False,
                     auto_adjust=False)["Close"]
    if isinstance(df, pd.Series):  # 단일 티커
        df = df.to_frame(name=list(tickers)[0])
    return df.dropna(how="all")


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


def collect(as_of: dt.date | None = None, cache_path: str = AUM_CACHE,
            closes: pd.DataFrame | None = None) -> list[FlowRecord]:
    """직전 스냅샷 대비 순설정을 추정한다.

    새 세션이 없으면 빈 리스트를 반환한다(캐시도 건드리지 않는다).
    "새 세션 없음"과 "흐름이 0이었음"은 전혀 다른 사실이고,
    후자로 기록하면 z-score의 분모와 로테이션 분모를 동시에 오염시킨다.
    """
    syms = [s for t in UNIVERSE.values() for s in t]
    if closes is None:
        closes = _closes(syms)
    if closes.empty or len(closes.index) < 2:
        print("[etf_flow] 종가 이력 부족 — 스킵")
        return []

    session = closes.index[-1].date()          # 방금 마감된 실제 세션
    if as_of is not None:                      # 테스트/수동 재수집용 오버라이드
        session = as_of

    cache = _load_cache(cache_path)
    prev = cache.get("latest", {})
    prev_session = cache.get("session")

    if prev_session == session.isoformat():
        print(f"[etf_flow] {session} 세션은 이미 수집됨 — 스킵")
        return []

    if prev and prev_session is None:
        # 레거시 캐시(세션 라벨 없음)는 어느 세션 기준인지 알 수 없다.
        # 이 상태에서 delta를 계산하면 이미 반영된 AUM을 한 번 더 빼면서
        # 가격수익률의 부호만 뒤집힌 유령 흐름이 만들어진다.
        # 따라서 기준점만 새 포맷으로 다시 심고 이번 회차는 건너뛴다.
        print("[etf_flow] 레거시 캐시 감지 — 기준점만 재설정하고 스킵")
        snap = _snapshot(syms)
        if snap:
            _save_cache(cache_path, {"session": session.isoformat(),
                                     "latest": snap, "prev_session": None})
        return []

    snap = _snapshot(syms)
    if not snap:
        print("[etf_flow] AUM 스냅샷 전량 실패 — 캐시 보존")
        return []

    records: list[FlowRecord] = []
    if prev:
        for sym, cur in snap.items():
            old = prev.get(sym)
            if not old or not old.get("aum"):
                continue
            try:
                c1 = float(closes[sym].iloc[-1])
                c0 = float(closes[sym].iloc[-2])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            if not c0:
                continue
            ret = c1 / c0 - 1.0
            flow = cur["aum"] - old["aum"] * (1.0 + ret)
            # 일 AUM의 0.02% 미만은 반올림 노이즈로 간주
            if abs(flow) < cur["aum"] * 2e-4:
                flow = 0.0
            records.append(
                FlowRecord(
                    ts=session,
                    market=SYM_TO_MARKET[sym],
                    actor="foreign",
                    instrument=sym,
                    net_flow_usd=round(flow, 2),
                    lag_days=0,
                    confidence=0.75,  # 추정식 기반이므로 원천 공시보다 낮게
                    source="etf_aum_delta",
                )
            )

    _save_cache(cache_path, {"session": session.isoformat(), "latest": snap,
                             "prev_session": prev_session})
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
