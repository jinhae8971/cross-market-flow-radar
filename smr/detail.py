"""시장별 상세 — 대시보드가 '요약의 근거'를 보여주기 위한 계층.

텔레그램은 결론만, 대시보드는 그 결론이 나온 이유를 담는다.
따라서 여기서는 알림 한 줄을 되짚을 수 있는 최소 단위를 만든다.
  · 어떤 종목이 그 흐름을 만들었나 (기여도)
  · 어떤 주체인가 (actor mix)
  · 얼마나 믿을 수 있는 데이터인가 (source/confidence mix)
  · 신호값이 임계치 대비 어디쯤인가
"""
from __future__ import annotations

import pandas as pd

THRESHOLDS = {"z20": 1.8, "cusum": 3.0, "run": 3}


def _b(x: float) -> float:
    return round(float(x) / 1e9, 3)


def build(df: pd.DataFrame, sig: pd.DataFrame) -> dict:
    if df.empty or sig.empty:
        return {}

    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"])
    last_ts = d["ts"].max()
    out: dict[str, dict] = {}

    for market, g in sig.groupby("market"):
        g = g.sort_values("ts")
        row = g.iloc[-1]
        raw = d[d["market"] == market]

        def window(days: int) -> pd.DataFrame:
            return raw[raw["ts"] > last_ts - pd.Timedelta(days=days)]

        # 종목 기여도 — 최근 5거래일 누적, 절대값 상위
        w5 = window(7)
        contrib = (
            (w5.assign(w=w5["net_flow_usd"] * w5["confidence"])
               .groupby("instrument")["w"].sum())
            .sort_values(key=lambda s: s.abs(), ascending=False)
            .head(6)
        )

        w20 = window(28)
        actors = (
            w20.assign(w=w20["net_flow_usd"] * w20["confidence"])
               .groupby("actor")["w"].sum().sort_values(key=lambda s: s.abs(),
                                                        ascending=False)
        )
        srcs = (
            w20.groupby("source")
               .agg(rows=("net_flow_usd", "size"), conf=("confidence", "mean"))
               .sort_values("rows", ascending=False)
        )

        cum = {f"d{n}": _b(g["net_flow_usd"].tail(n).sum()) for n in (5, 20, 60)}

        out[market] = {
            "latest": _b(row["net_flow_usd"]),
            "cum": cum,
            "signal": {
                "z20": None if pd.isna(row.get("z20")) else round(float(row["z20"]), 2),
                "z60": None if pd.isna(row.get("z60")) else round(float(row["z60"]), 2),
                "cusum": round(float(row.get("cusum") or 0), 2),
                "run": int(row.get("run") or 0),
                "thresholds": THRESHOLDS,
            },
            "contributors": [{"name": k, "flow": _b(v)} for k, v in contrib.items()],
            "actors": [{"name": k, "flow": _b(v)} for k, v in actors.items()],
            "sources": [
                {"name": k, "rows": int(r["rows"]), "confidence": round(float(r["conf"]), 2)}
                for k, r in srcs.iterrows()
            ],
        }
    return out
