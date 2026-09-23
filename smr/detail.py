"""시장별 상세 — 대시보드가 '요약의 근거'를 보여주기 위한 계층.

텔레그램은 결론만, 대시보드는 그 결론이 나온 이유를 담는다.
  · 어떤 종목이 그 흐름을 만들었나 (핵심 계열 기여도)
  · 같은 시장의 다른 주체·파생 포지션은 어땠나 (맥락 지표 — 합산하지 않음)
  · 어떤 원천이고 기준일 커버리지는 얼마인가
  · 신호값이 임계치 대비 어디쯤인가
"""
from __future__ import annotations

import pandas as pd

from . import signals

THRESHOLDS = {"z20": 1.8, "cusum": 3.0, "run": 3}

CONTEXT_LABEL = {
    ("KR", "institution"): "기관 순매수(KOSPI)",
    ("KR", "pension"): "연기금 순매수(KOSPI)",
    ("KR", "retail"): "개인 순매수(KOSPI)",
    ("KR", "etf"): "EWY 순설정(미국 상장)",
    ("*", "spec"): "투기적 선물 포지션 변화(COT)",
}
CORE_LABEL = {"KR": "외국인 순매수(KOSPI)", "JP": "ETF 순설정", "EU": "ETF 순설정",
              "US": "ETF 순설정"}


def _b(x: float) -> float:
    return round(float(x) / 1e9, 3)


def build(df: pd.DataFrame, sig: pd.DataFrame, universe: dict | None = None,
          coverage: dict | None = None, as_of=None) -> dict:
    """as_of를 주면 그날 기준으로 자른다 — 브리프의 시장별 수치가 같은 날을 가리키게.

    한국은 장 마감이 미국 공시보다 하루 빨라 최신 행이 기준일보다 앞설 수 있다.
    최신 행을 그대로 쓰면 한 줄은 오늘, 나머지는 어제 값인 브리프가 된다.
    """
    if df.empty or sig.empty:
        return {}
    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"])
    core = signals.core_rows(d, universe)
    cut_ts = pd.Timestamp(as_of) if as_of is not None else None
    out: dict[str, dict] = {}

    for market, g in sig.groupby("market"):
        g = g.sort_values("ts")
        if cut_ts is not None and (g["ts"] <= cut_ts).any():
            g = g[g["ts"] <= cut_ts]
        row = g.iloc[-1]
        sessions = list(g["ts"])

        def last(n: int, frame: pd.DataFrame) -> pd.DataFrame:
            cut = sessions[-n] if len(sessions) >= n else sessions[0]
            return frame[frame["ts"] >= cut]

        c = core[(core["market"] == market) & (core["ts"] <= row["ts"])]
        contrib = (last(5, c).groupby("instrument")["net_flow_usd"].sum()
                   .sort_values(key=lambda s: s.abs(), ascending=False).head(6))

        # 맥락 지표 — 같은 시장의 다른 주체·원천. 합계에 넣지 않는다.
        raw = d[(d["market"] == market) & (d["ts"] <= row["ts"])]
        ctx = []
        if market == "KR":
            kr = last(20, raw[raw["source"].isin(signals.CORE_SOURCES["KR"])])
            for actor in ("institution", "pension", "retail"):
                v = kr[kr["actor"] == actor]["net_flow_usd"].sum()
                if len(kr[kr["actor"] == actor]):
                    ctx.append({"name": CONTEXT_LABEL[("KR", actor)], "flow": _b(v)})
            ewy = last(20, raw[(raw["source"] == "etf_flow") & (raw["instrument"] == "EWY")])
            if len(ewy):
                ctx.append({"name": CONTEXT_LABEL[("KR", "etf")],
                            "flow": _b(ewy["net_flow_usd"].sum())})
        cot = raw[raw["actor"] == "spec"].sort_values("ts").tail(4)
        if len(cot):
            ctx.append({"name": CONTEXT_LABEL[("*", "spec")] + " · 4주",
                        "flow": _b(cot["net_flow_usd"].sum())})

        src = last(20, c).groupby("source").agg(rows=("net_flow_usd", "size"),
                                                conf=("confidence", "mean"))
        cum = {f"d{n}": _b(g["net_flow_usd"].tail(n).sum()) for n in (5, 20, 60)}
        n_obs = int(g["net_flow_usd"].notna().sum())

        out[market] = {
            "label": CORE_LABEL.get(market, ""),
            "ts": row["ts"].date().isoformat(),
            "latest": _b(row["net_flow_usd"]),
            "cum": cum,
            "observations": n_obs,
            "signal": {
                "z20": None if pd.isna(row.get("z20")) else round(float(row["z20"]), 2),
                "z60": None if pd.isna(row.get("z60")) else round(float(row["z60"]), 2),
                "cusum": round(float(row.get("cusum") or 0), 2),
                "run": int(row.get("run") or 0),
                "thresholds": THRESHOLDS,
            },
            "contributors": [{"name": k, "flow": _b(v)} for k, v in contrib.items()],
            "actors": ctx,
            "sources": [{"name": k, "rows": int(r["rows"]),
                         "confidence": round(float(r["conf"]), 2)}
                        for k, r in src.iterrows()],
            "coverage": (coverage or {}).get(market),
        }
    return out
