"""로테이션 매트릭스 — 절대금액이 아니라 상대 배분으로 본다.

절대 USD로 4개 시장을 비교하면 미국이 항상 압도해 비교가 무의미하다.
각 시장의 자기 규모로 정규화한 뒤 배분율을 내야
"돈이 어디서 빠져 어디로 갔는가"가 드러난다.

배분율 산식이 바뀐 이유(2026-08-11):
    이전 구현은 양(+)의 유입만 남기고(clip) 그 합으로 나눴다.
    4개 시장이 모두 순유출인 날에는 분모가 0이 되어 전 시장 배분율이
    0%로 붕괴하고, delta는 5거래일 전 값과의 차이라 그대로 남아
    "0% ▼56.1" 같은 모순된 표시가 나왔다. 더 나쁜 것은 score가 전부
    동률이 되어 정렬이 알파벳 순으로 고정되고, 그 결과 from/to가
    데이터가 아니라 컬럼 순서를 반영했다는 점이다.

    그래서 softmax 배분으로 교체한다. 모든 시장이 유출인 날에도
    "덜 빠진 곳 / 더 빠진 곳"이라는 상대 정보는 살아 있고, 분모가
    0이 될 수 없으므로 붕괴 구간이 사라진다. 배분율은 절대 유입액의
    점유율이 아니라 상대 선호도이며, 대시보드 문구도 그에 맞춘다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 단일일 노이즈를 죽이기 위한 평활 창. 5거래일 = 로테이션 판단의 최소 단위.
SMOOTH = 5
# softmax 온도. 낮을수록 승자독식, 높을수록 균등. 1.0은 정규화된 1σ 차이가
# 배분율 약 2.7배 차이로 나타나는 수준.
TEMPERATURE = 1.0
# 배분율 격차가 이보다 작으면 방향을 단정하지 않는다(%p).
MIN_SPREAD = 2.0


def share(sig: pd.DataFrame, window: int = 20,
          smooth: int = SMOOTH, temperature: float = TEMPERATURE) -> pd.DataFrame:
    """시장별 유입을 자기 과거 변동성으로 표준화한 뒤 상대 배분율로 환산."""
    if sig.empty:
        return pd.DataFrame()
    piv = sig.pivot_table(index="ts", columns="market",
                          values="net_flow_usd", aggfunc="sum")
    if piv.empty:
        return pd.DataFrame()

    # 결측은 0이 아니라 '관측 없음'이다. 0으로 메우면 그 시장이 그날
    # 실제로 흐름이 없었다는 잘못된 관측을 만들어낸다.
    scale = piv.abs().rolling(window, min_periods=5).mean().replace(0, np.nan)
    norm = (piv / scale).rolling(smooth, min_periods=1).mean()

    # 한 시장이라도 정규화가 불가능한 날은 배분율 자체를 내지 않는다.
    valid = norm.notna().all(axis=1)

    shifted = norm.sub(norm.max(axis=1), axis=0) / float(temperature)  # 오버플로 방지
    exp = np.exp(shifted)
    out = exp.div(exp.sum(axis=1), axis=0) * 100.0
    out = out.where(valid)
    return out.round(1)


def matrix(sig: pd.DataFrame, lookback: int = 5) -> dict:
    """최근 lookback 구간의 배분율 변화 = 로테이션 방향."""
    sh = share(sig)
    if sh.empty:
        return {"ready": False, "rows": [], "reason": "데이터 축적 중"}

    sh = sh.dropna(how="any")
    if len(sh) < lookback + 1:
        return {"ready": False, "rows": [], "reason": "데이터 축적 중"}

    cur, prev = sh.iloc[-1], sh.iloc[-(lookback + 1)]
    rows = []
    for m in sh.columns:
        share_now = float(cur[m])
        delta = round(float(cur[m] - prev[m]), 1)
        # 수준(share)과 모멘텀(delta)을 함께 본다. 수준만 보면 이미 끝난 흐름을,
        # 변화만 보면 꾸준히 커지는 흐름을 놓친다.
        rows.append({"market": m, "share": round(share_now, 1), "delta": delta,
                     "score": round(share_now + delta, 1)})
    rows.sort(key=lambda r: -r["score"])

    out = {"ready": True, "rows": rows, "as_of": sh.index[-1].date().isoformat()}

    # 격차가 의미 없을 만큼 작으면 방향을 만들어내지 않는다.
    # (동률일 때 정렬 안정성 때문에 알파벳 순이 방향으로 둔갑하던 버그의 정면 차단)
    spread = rows[0]["score"] - rows[-1]["score"]
    if spread < MIN_SPREAD:
        out.update({"from": None, "to": None, "reason": "시장 간 격차 미미"})
    else:
        out.update({"from": rows[-1]["market"], "to": rows[0]["market"]})
    return out
