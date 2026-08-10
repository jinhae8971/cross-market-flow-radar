"""L0 수집 → L1 저장 → L2 신호 → L3 로테이션 → L4 배포 산출물.

수집기 하나가 죽어도 나머지는 돈다(우아한 성능저하). 대신 어떤 수집기가
죽었는지는 산출물의 health 필드에 남겨 대시보드에서 바로 보이게 한다.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import traceback

import pandas as pd

from . import detail, repair, rotation, signals
from .calendar_mask import masked
from .collectors import cot, etf_flow, korea
from .schema import FlowStore, to_frame

MARKET_KO = {"KR": "한국", "JP": "일본", "EU": "유럽", "US": "미국"}


def _safe(name: str, fn, *a, **kw):
    try:
        out = fn(*a, **kw)
        return out, {"collector": name, "ok": True, "records": len(out)}
    except Exception as exc:
        traceback.print_exc()
        return [], {"collector": name, "ok": False, "error": str(exc)[:200]}


def run(seed: bool = False, store_path: str = "data/flows.parquet",
        out_path: str = "docs/data.json") -> dict:
    health = []
    records = []

    if seed:
        r, h = _safe("etf_backfill", etf_flow.backfill, "1y")
        records += r
        health.append(h)

    r, h = _safe("etf_aum", etf_flow.collect)
    records += r
    if h.get("ok") and not r:
        # 0건은 실패가 아니다 — 아직 새 세션이 없다는 정상 상태다.
        h["error"] = "새 세션 없음 — 수집 생략"
    health.append(h)

    r, h = _safe("cot", cot.collect, 26)
    records += r
    health.append(h)

    r, h = _safe("krx", korea.collect)
    records += r
    health.append(h)

    store = FlowStore(store_path)
    added = store.upsert(to_frame(records))
    df = store.load()

    # 자가 복구 — 과거 결함이 남긴 '전 종목 0' 날짜를 걷어낸다(멱등).
    df, purged = repair.drop_dead_sessions(df)
    if purged:
        store.replace(df)
        health.append({"collector": "repair", "ok": True,
                       "records": len(purged),
                       "error": f"미갱신 세션 제거: {', '.join(purged)}"})

    sig = signals.build(df)
    alerts = signals.alerts(sig)

    # 캘린더 마스크 — 기계적 매매일의 알림은 사유를 달아 억제한다
    kept = []
    for a in alerts:
        flag, why = masked(dt.date.fromisoformat(a["ts"]))
        if flag:
            a["suppressed"] = why
        else:
            kept.append(a)

    rot = rotation.matrix(sig)

    recent = sig[sig["ts"] >= sig["ts"].max() - pd.Timedelta(days=120)]
    series = {
        m: [
            {"d": r.ts.date().isoformat(), "f": round(r.net_flow_usd / 1e9, 3),
             "z": None if pd.isna(r.z20) else round(float(r.z20), 2)}
            for r in g.itertuples()
        ]
        for m, g in recent.groupby("market")
    }

    payload = {
        "dashboard_url": os.environ.get("DASHBOARD_URL", ""),
        "detail": detail.build(df, sig),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "as_of": sig["ts"].max().date().isoformat() if not sig.empty else None,
        "markets": MARKET_KO,
        "alerts": kept,
        "suppressed": [a for a in alerts if "suppressed" in a],
        "rotation": rot,
        "series": series,
        "health": health,
        "rows_total": int(len(df)),
        "rows_added": int(added),
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = f"{out_path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)
    return payload
