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


def _sessions_of(store_path: str, source: str) -> set:
    """이미 확보한 날짜 집합. 같은 날을 반복 조회하지 않기 위한 입력."""
    try:
        df = FlowStore(store_path).load()
    except Exception:
        return set()
    d = df[df["source"] == source]
    return set() if d.empty else set(pd.to_datetime(d["ts"]).dt.date)


def _last_session(store_path: str, source: str) -> dt.date | None:
    """저장소에 남은 특정 source의 마지막 관측일. 대리지표 시작점을 정한다."""
    try:
        df = FlowStore(store_path).load()
    except Exception:
        return None
    d = df[df["source"] == source]
    return None if d.empty else pd.to_datetime(d["ts"]).max().date()


def run(seed: bool = False, store_path: str = "data/flows.parquet",
        out_path: str = "docs/data.json") -> dict:
    health = []
    records = []

    if seed:
        r, h = _safe("etf_backfill", etf_flow.backfill, "1y")
        records += r
        health.append(h)

    closes, ch = _safe("etf_closes", etf_flow.load_closes)
    latest_session = (closes.index[-1].date()
                      if hasattr(closes, "empty") and not closes.empty else None)
    if not ch.get("ok"):
        health.append(ch)

    r, h = _safe("etf_aum", etf_flow.collect,
                 closes=closes if latest_session else None)
    records += r
    if not r and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as previous_file:
            previous = json.load(previous_file)
        if previous.get("quality_warnings"):
            # 이전 경고를 유지하되 이번 회차의 실제 사유를 덮어쓰지 않는다.
            # 덮어쓰면 근본 원인이 health에서 사라져 장애가 눈에 띄지 않는다
            # (2026-09-08 동결이 7일간 발견되지 않은 직접적 원인).
            h.update(ok=False, latched=True,
                     error=h.get("error") or "새 ETF 관측 없음 — 이전 품질 경고 유지")
    if h.get("ok") and not r:
        # 0건은 실패가 아니다 — 아직 새 세션이 없다는 정상 상태다.
        h["error"] = "추가 관측 없음 또는 기준점 설정 — 신규 흐름 없음"
    health.append(h)

    # 백본이 동결됐으면 대리지표로 계열을 이어간다. 멈춘 계열을 최신인 척
    # 내보내는 것보다, 해상도가 낮다고 밝히고 최신 날짜를 유지하는 편이 안전하다.
    degraded = False
    if not r and latest_session:
        last_real = _last_session(store_path, "etf_aum_delta")
        if last_real is None or last_real < latest_session:
            pr, ph = _safe("etf_proxy", etf_flow.proxy_collect, since=last_real)
            if pr:
                records += pr
                degraded = True
                ph.update(degraded=True,
                          error="ETF AUM 동결 — 대리지표(OHLCV) 모드로 계열 유지")
            health.append(ph)

    r, h = _safe("cot", cot.collect, 26)
    records += r
    health.append(h)

    known_kr = _sessions_of(store_path, "krx")
    r, h = _safe("krx", korea.collect, known=known_kr)
    records += r
    if h.get("ok") and not r:
        h.update(status="unavailable",
                 error="KRX 신규 공시 없음 — 한국은 ETF 대리지표")
    elif not h.get("ok") and "KRX_API_KEY" in h.get("error", ""):
        # 미설정은 장애가 아니다. 실패로 세면 quality_warnings가 상시 채워져
        # 알림·로테이션이 영구히 보류되고, 동시에 진짜 장애가 묻힌다.
        h.update(ok=True, status="unconfigured", records=0)
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

    quality = [h.get("error", h["collector"]) for h in health if not h.get("ok")]
    if degraded:
        # 대리지표는 계열을 잇기 위한 것이지 발화 근거가 아니다.
        quality.append("대리지표 모드 — 신규 신호·로테이션 판단 보류")
    if quality:
        kept = []
        rot = {"ready": False, "rows": [], "from": None, "to": None}

    as_of = sig["ts"].max().date() if not sig.empty else None
    # 신선도는 달력일이 아니라 '놓친 세션 수'로 잰다. 발송 게이트의 입력값이다.
    stale_sessions = 0
    if as_of and latest_session and hasattr(closes, "index"):
        stale_sessions = sum(1 for d in closes.index if d.date() > as_of)

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
        "as_of": as_of.isoformat() if as_of else None,
        "latest_session": latest_session.isoformat() if latest_session else None,
        "stale_sessions": int(stale_sessions),
        "mode": "degraded" if degraded else "normal",
        "markets": MARKET_KO,
        "alerts": kept,
        "suppressed": [a for a in alerts if "suppressed" in a],
        "rotation": rot,
        "series": series,
        "health": health,
        "quality_warnings": quality,
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

