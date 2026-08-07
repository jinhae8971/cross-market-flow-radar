"""공통 자금흐름 스키마.

설계 핵심: 시장마다 데이터 해상도가 다르다(한국 일별 / 일본 주간 / 유럽 혼합).
이를 같은 신호엔진에 태우려면 lag_days와 confidence가 데이터의 1급 필드여야 한다.
신호 계층은 이 두 필드를 읽어 가중치를 조정한다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from datetime import date
from typing import Iterable

import pandas as pd

MARKETS = ("KR", "JP", "EU", "US")

# actor: 자금 주체
#   foreign      해외 투자자 (국가 ETF 유입 = 역외 배분)
#   institution  기관
#   pension      연기금 / 신탁은행
#   spec         투기적 포지션 (COT non-commercial)
#   insider      내부자
ACTORS = ("foreign", "institution", "pension", "spec", "insider", "retail")

SCHEMA = {
    "ts": "datetime64[ns]",     # 흐름이 '발생한' 날짜 (공표일 아님)
    "market": "string",
    "actor": "string",
    "instrument": "string",
    "net_flow_usd": "float64",  # + 유입 / - 유출
    "lag_days": "int16",        # 발생일 → 공표일 지연
    "confidence": "float64",    # 0~1. 추정치일수록 낮음
    "source": "string",
}


@dataclass(slots=True)
class FlowRecord:
    ts: date
    market: str
    actor: str
    instrument: str
    net_flow_usd: float
    lag_days: int
    confidence: float
    source: str
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.market not in MARKETS:
            raise ValueError(f"unknown market: {self.market}")
        if self.actor not in ACTORS:
            raise ValueError(f"unknown actor: {self.actor}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of range: {self.confidence}")


def to_frame(records: Iterable[FlowRecord]) -> pd.DataFrame:
    rows = [{k: v for k, v in asdict(r).items() if k != "meta"} for r in records]
    if not rows:
        return pd.DataFrame(columns=list(SCHEMA)).astype(SCHEMA)
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.astype(SCHEMA)


class FlowStore:
    """파티션 없는 단일 parquet. 중복은 (ts, market, actor, instrument, source)로 제거.

    GitHub Actions에서 커밋백하므로 파일 하나로 유지하는 편이 diff·복구에 유리하다.
    """

    KEY = ["ts", "market", "actor", "instrument", "source"]

    def __init__(self, path: str = "data/flows.parquet") -> None:
        self.path = path

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.path):
            return pd.DataFrame(columns=list(SCHEMA)).astype(SCHEMA)
        return pd.read_parquet(self.path)

    def upsert(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        base = self.load()
        merged = pd.concat([base, df], ignore_index=True)
        # 뒤에 온 레코드(재수집분)를 우선 채택 — 확정치가 추정치를 덮어쓴다
        merged = merged.drop_duplicates(subset=self.KEY, keep="last")
        merged = merged.sort_values(["ts", "market", "actor"]).reset_index(drop=True)
        added = len(merged) - len(base)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = f"{self.path}.tmp"
        merged.to_parquet(tmp, index=False)
        os.replace(tmp, self.path)
        return added
