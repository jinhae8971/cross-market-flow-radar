"""저장소 자가 복구 — 과거 결함이 남긴 오염 레코드를 걷어낸다.

대상: '새 세션이 없었는데 흐름 0으로 기록된 날'.
2026-08-11 이전 수집기는 새 세션 여부를 판별하지 못해 미국 휴장 구간
(주말·월요일 아침 실행)에도 delta 0 레코드를 실제 관측치로 저장했다.
이 레코드는 z-score의 표준편차를 줄여 신호를 부풀리고, 로테이션 분모를
0으로 만든다. 판별 기준은 단순하다 — 해당 날짜의 추정 흐름이 종목 하나도
빠짐없이 정확히 0이면 그것은 관측이 아니라 미갱신이다.
"""
from __future__ import annotations

import pandas as pd

DEAD_SOURCE = "etf_aum_delta"


def dead_sessions(df: pd.DataFrame) -> list[pd.Timestamp]:
    """전 종목 흐름이 정확히 0인 날짜 목록."""
    if df.empty:
        return []
    d = df[df["source"] == DEAD_SOURCE]
    if d.empty:
        return []
    by_day = d.groupby("ts")["net_flow_usd"].apply(lambda s: bool((s == 0).all()))
    return list(by_day[by_day].index)


def drop_dead_sessions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """오염 날짜를 제거한 프레임과 제거된 날짜(ISO)를 돌려준다."""
    days = dead_sessions(df)
    if not days:
        return df, []
    mask = df["ts"].isin(days) & (df["source"] == DEAD_SOURCE)
    return (df[~mask].reset_index(drop=True),
            [pd.Timestamp(x).date().isoformat() for x in days])
