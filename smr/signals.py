"""신호 엔진 — 강도 / 지속성 / 확증의 3단 구성.

단발 급등을 걸러내는 것이 이 계층의 존재 이유다.
자금흐름 데이터는 만기·리밸런싱 하루에 평상시 수십 배가 찍히므로
z-score만 쓰면 알림의 절반이 캘린더 이벤트가 된다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── 핵심 계열 정의 (v2, 2026-09-23) ─────────────────────────────────────────
# 시장마다 '한 가지 측정 방식'만 합산한다. v1은 (날짜·시장·주체)마다 신뢰도가
# 가장 높은 소스를 골랐는데, 그러면 원천 하나가 빠진 날 다른 척도의 값이 끼어든다
# (예: 한국 외국인 순매수 $0.5B 대신 EWY 설정 $0.05B). 계열 안에서 척도가 바뀌면
# z-score가 의미를 잃는다. 그래서 시장별 핵심 소스를 고정하고, 빠진 날은
# 다른 척도로 메우지 않고 '관측 없음'으로 둔다.
#   한국      KOSPI 외국인 순매수 (KRX 원천 > 다음 금융)
#   일·유·미   미국 상장 국가 ETF 순설정 (발행사 공시 발행좌수 x NAV)
# 국내 주체(기관·연기금·개인)와 COT 투기 포지션은 맥락 지표로만 쓴다 —
# 한 시장의 주체별 순매수는 상계되어 합이 0이고, COT는 주간·명목 단위라
# 일별 계열에 더하면 화요일마다 척도가 튄다.
CORE_SOURCES: dict[str, tuple[str, ...]] = {
    "KR": ("krx", "daum_kr"),
    "JP": ("etf_flow",),
    "EU": ("etf_flow",),
    "US": ("etf_flow",),
}
SOURCE_RANK = {"krx": 3, "daum_kr": 2, "etf_flow": 2}
CORE_ACTOR = "foreign"
# 시장 내 순자산 기준 커버리지가 이보다 낮은 날은 계열에서 뺀다. 구성 종목이
# 날마다 달라지면 합계의 분산이 바뀌어 z-score가 흔들리기 때문이다.
MIN_COVERAGE = 0.8


def core_rows(df: pd.DataFrame, universe: dict | None = None) -> pd.DataFrame:
    """시장별 핵심 소스의 행만 남기고, 같은 (날짜·시장·종목)은 상위 소스 1개만."""
    if df.empty:
        return df
    d = df[df["actor"] == CORE_ACTOR].copy()
    keep = pd.Series(False, index=d.index)
    for market, sources in CORE_SOURCES.items():
        m = (d["market"] == market) & d["source"].isin(sources)
        if universe and market in universe:
            m &= d["instrument"].isin(universe[market])
        keep |= m
    d = d[keep]
    if d.empty:
        return d
    d["_rank"] = d["source"].map(SOURCE_RANK).fillna(0)
    d = (d.sort_values("_rank", ascending=False)
          .drop_duplicates(subset=["ts", "market", "instrument"], keep="first")
          .drop(columns="_rank"))
    return d.sort_values(["ts", "market", "instrument"]).reset_index(drop=True)


def aggregate(df: pd.DataFrame, freq: str = "D", weights: dict | None = None,
              universe: dict | None = None,
              min_coverage: float = MIN_COVERAGE) -> pd.DataFrame:
    """시장 x 기간 순유입(USD, 가중치 없음) + 순자산 기준 커버리지.

    v1은 신뢰도를 곱했는데, 한 계열 안에서 소스가 바뀌면 같은 달러가 0.75배·0.9배로
    다르게 기록돼 척도가 흔들린다. 신뢰도는 소스 선택과 발행 게이트에만 쓴다.
    """
    cols = ["ts", "market", "net_flow_usd", "coverage"]
    d = core_rows(df, universe)
    if d.empty:
        return pd.DataFrame(columns=cols)
    d = d.copy()
    d["ts"] = pd.to_datetime(d["ts"])
    if universe:
        w = weights or {}
        d["_w"] = d["instrument"].map(lambda s: w.get(s, 1.0))
        total = {m: sum(w.get(s, 1.0) for s in syms) for m, syms in universe.items()}
        d["_cov"] = d["_w"] / d["market"].map(lambda m: total.get(m, float("nan")))
    else:
        d["_cov"] = float("nan")
    g = (d.groupby(["market", pd.Grouper(key="ts", freq=freq)])
          .agg(net_flow_usd=("net_flow_usd", "sum"), coverage=("_cov", "sum"))
          .reset_index())
    # 종목 구성이 정의되지 않은 시장(한국 KOSPI 등)은 관측 존재 = 완전 커버리지.
    g.loc[g["market"].map(lambda m: not universe or m not in universe), "coverage"] = 1.0
    g = g[g["coverage"].fillna(1.0) >= min_coverage - 1e-9]
    return g[cols[1:2] + cols[0:1] + cols[2:]].sort_values(["market", "ts"]).reset_index(drop=True)


def zscore(s: pd.Series, window: int = 20, min_periods: int = 8) -> pd.Series:
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std(ddof=0)
    return (s - mu) / sd.replace(0, np.nan)


def cusum(s: pd.Series, drift_k: float = 0.5, window: int = 60) -> pd.Series:
    """정규화 CUSUM. 누적 편차가 임계를 넘으면 체제전환 후보.

    반환값의 절대값이 클수록 '방향이 바뀐 뒤 되돌아오지 않고 있다'는 뜻.
    """
    mu = s.rolling(window, min_periods=10).mean()
    sd = s.rolling(window, min_periods=10).std(ddof=0).replace(0, np.nan)
    norm = ((s - mu) / sd).fillna(0.0)

    pos, neg = 0.0, 0.0
    out = []
    for x in norm:
        pos = max(0.0, pos + x - drift_k)
        neg = min(0.0, neg + x + drift_k)
        out.append(pos if pos >= abs(neg) else neg)
    return pd.Series(out, index=s.index)


def persistence(s: pd.Series, window: int = 20) -> pd.Series:
    """'평소 대비' 초과 유입/유출이 연속된 기간 수.

    원계열 부호로 세면 안 된다 — 미국처럼 상시 순유입인 시장은 run이 늘 최대치가
    되어 지속성 트리거가 무력화된다. 이동평균 대비 편차의 부호로 센다.
    """
    dev = s - s.rolling(window, min_periods=3).mean()
    sign = np.sign(dev.fillna(0.0))
    run, out = 0, []
    prev = 0.0
    for v in sign:
        run = run + 1 if v == prev and v != 0 else (1 if v != 0 else 0)
        prev = v
        out.append(int(run * v))
    return pd.Series(out, index=s.index)


def build(df: pd.DataFrame, freq: str = "D", weights: dict | None = None,
          universe: dict | None = None) -> pd.DataFrame:
    agg = aggregate(df, freq, weights=weights, universe=universe)
    frames = []
    for market, grp in agg.groupby("market"):
        g = grp.sort_values("ts").copy()
        g["z20"] = zscore(g["net_flow_usd"], 20)
        g["z60"] = zscore(g["net_flow_usd"], 60, min_periods=20)
        g["cusum"] = cusum(g["net_flow_usd"])
        g["run"] = persistence(g["net_flow_usd"])  # 편차 기준
        g["cum20"] = g["net_flow_usd"].rolling(20, min_periods=1).sum()
        frames.append(g)
    return pd.concat(frames, ignore_index=True) if frames else agg


def alerts(sig: pd.DataFrame, z_thr: float = 1.8, cusum_thr: float = 3.0,
           run_thr: int = 3, as_of=None) -> list[dict]:
    """3개 조건 중 2개 이상 충족 시에만 발화 — 확증 원칙.

    as_of를 주면 그날 관측된 시장만 판정한다. 원천이 멈춘 시장의 마지막 행으로
    발화하면 며칠 묵은 신호가 오늘 신호처럼 나간다.
    """
    out = []
    if sig.empty:
        return out
    for market, grp in sig.groupby("market"):
        g = grp.sort_values("ts")
        row = g.iloc[-1]
        if as_of is not None and row["ts"].date() != as_of:
            continue
        prev_cusum = abs(g["cusum"].iloc[-2]) if len(g) >= 2 else 0.0
        checks = {
            "강도": abs(row.get("z20", 0) or 0) >= z_thr,
            # 단발 급등은 CUSUM을 한 번에 임계 위로 밀어올린다. 직전 기간에도
            # 누적이 살아 있었을 때만 '체제전환'으로 인정한다.
            "체제전환": (abs(row.get("cusum", 0) or 0) >= cusum_thr
                     and prev_cusum >= cusum_thr * 0.6),
            "지속성": abs(row.get("run", 0) or 0) >= run_thr,
        }
        hit = [k for k, v in checks.items() if v]
        if len(hit) >= 2:
            out.append({
                "market": market,
                "ts": row["ts"].date().isoformat(),
                "flow_usd": float(row["net_flow_usd"]),
                "z20": float(row.get("z20") or 0),
                "cusum": float(row.get("cusum") or 0),
                "run": int(row.get("run") or 0),
                "triggers": hit,
                "direction": "유입" if row["net_flow_usd"] > 0 else "유출",
            })
    return sorted(out, key=lambda x: -abs(x["z20"]))
