"""로테이션 매트릭스 — 절대금액이 아니라 share로 본다.

절대 USD로 4개 시장을 비교하면 미국이 항상 압도해 비교가 무의미하다.
각 시장의 자기 규모로 정규화한 뒤 share를 내야
"돈이 어디서 빠져 어디로 갔는가"가 드러난다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def share(sig: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """시장별 유입을 자기 과거 변동성으로 표준화한 뒤 점유율로 환산."""
    if sig.empty:
        return sig
    piv = sig.pivot_table(index="ts", columns="market",
                          values="net_flow_usd", aggfunc="sum").fillna(0.0)
    scale = piv.abs().rolling(window, min_periods=5).mean().replace(0, np.nan)
    norm = (piv / scale).fillna(0.0)
    pos = norm.clip(lower=0.0)
    total = pos.sum(axis=1).replace(0, np.nan)
    out = (pos.div(total, axis=0) * 100).astype("float64")
    return out.fillna(0.0).round(1)


def matrix(sig: pd.DataFrame, lookback: int = 5) -> dict:
    """최근 lookback 구간의 share 변화 = 로테이션 방향."""
    sh = share(sig)
    if len(sh) < lookback + 1:
        return {"ready": False, "rows": []}
    cur, prev = sh.iloc[-1], sh.iloc[-(lookback + 1)]
    rows = []
    for m in sh.columns:
        share_now = float(cur[m])
        delta = round(float(cur[m] - prev[m]), 1)
        # 수준(share)과 모멘텀(delta)을 함께 본다. 수준만 보면 이미 끝난 흐름을,
        # 변화만 보면 꾸준히 커지는 흐름을 놓친다(자기 평균 정규화 때문에
        # 선형 증가 구간에서는 delta가 오히려 음수로 나온다).
        rows.append({"market": m, "share": share_now, "delta": delta,
                     "score": round(share_now + delta, 1)})
    rows.sort(key=lambda r: -r["score"])
    return {"ready": True, "rows": rows,
            "from": rows[-1]["market"], "to": rows[0]["market"]}
