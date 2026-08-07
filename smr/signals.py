"""신호 엔진 — 강도 / 지속성 / 확증의 3단 구성.

단발 급등을 걸러내는 것이 이 계층의 존재 이유다.
자금흐름 데이터는 만기·리밸런싱 하루에 평상시 수십 배가 찍히므로
z-score만 쓰면 알림의 절반이 캘린더 이벤트가 된다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """시장 × 기간 순유입(USD). confidence를 가중치로 쓴다.

    추정치(ETF 0.75)와 원천 공시(KRX 0.95)를 단순 합산하면
    노이즈가 큰 계열이 신호를 지배한다.
    """
    if df.empty:
        return pd.DataFrame(columns=["ts", "market", "net_flow_usd"])
    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"])
    d["weighted"] = d["net_flow_usd"] * d["confidence"]
    g = (
        d.groupby(["market", pd.Grouper(key="ts", freq=freq)])["weighted"]
        .sum()
        .reset_index()
        .rename(columns={"weighted": "net_flow_usd"})
    )
    return g.sort_values(["market", "ts"]).reset_index(drop=True)


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


def build(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    agg = aggregate(df, freq)
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
           run_thr: int = 3) -> list[dict]:
    """3개 조건 중 2개 이상 충족 시에만 발화 — 확증 원칙."""
    out = []
    for market, grp in sig.groupby("market"):
        g = grp.sort_values("ts")
        row = g.iloc[-1]
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
