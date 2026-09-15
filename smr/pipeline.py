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
from .collectors import cot, etf_flow, korea, naver_kr
from .schema import FlowStore, to_frame

MARKET_KO = {"KR": "한국", "JP": "일본", "EU": "유럽", "US": "미국"}

# 이 점수 아래로 떨어지면 산출물을 갱신하지 않고 작업 자체를 중단한다.
# repo variable이 비어 있으면 빈 문자열이 들어오므로 or로 받아낸다.
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE") or "0.60")


class LowConfidence(RuntimeError):
    """신뢰도 미달 — 산출물을 쓰지 않고 실행을 중단시킨다.

    낮은 신뢰도의 값에 경고를 달아 내보내는 방식은 이미 한 번 실패했다.
    경고는 며칠이면 배경이 되고 숫자는 그대로 읽힌다. 그래서 경고가 아니라
    중단으로 처리한다 — data.json을 덮지 않으므로 대시보드는 마지막 정상
    상태를 유지하고, 종료코드가 0이 아니므로 워크플로우가 실패 통지를 보낸다.
    """

    def __init__(self, score: float, breakdown: dict) -> None:
        self.score = score
        self.breakdown = breakdown
        super().__init__(
            f"신뢰도 {score:.2f} < 기준 {MIN_CONFIDENCE:.2f} — 산출물 갱신 중단: "
            + ", ".join(f"{k} {v}" for k, v in breakdown.items()))


def confidence_score(df, as_of):
    """최신 관측일 기준 신뢰도 = 관측 품질 x 시장 커버리지.

    곱하는 이유: 네 시장 중 하나만 고품질로 들어와도 평균 신뢰도는 높게
    나온다. 크로스마켓 비교가 목적이므로 커버리지 결손은 품질 저하와 같은
    무게로 다뤄야 한다.
    """
    if df.empty or as_of is None:
        return 0.0, {"관측": "없음"}
    d = df[(pd.to_datetime(df["ts"]).dt.date == as_of)
           & (df["actor"].isin(signals.PRIMARY_ACTORS))]
    if d.empty:
        return 0.0, {"관측": "없음"}
    # 시장별로 먼저 평균을 낸 뒤 시장 간 평균을 낸다. 행 단위 평균을 쓰면
    # ETF를 6개 담는 유럽이 1개 계열인 한국보다 6배 무겁게 반영된다.
    quality = float(d.groupby("market")["confidence"].mean().mean())
    covered = sorted(set(d["market"]))
    coverage = len(covered) / len(MARKET_KO)
    missing = [MARKET_KO[m] for m in MARKET_KO if m not in covered]
    return quality * coverage, {"품질": f"{quality:.2f}",
                                "커버리지": f"{len(covered)}/{len(MARKET_KO)}",
                                "결손": ",".join(missing) or "없음"}


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


def _comparable_session(sig: pd.DataFrame, window: int = 10) -> dt.date | None:
    """네 시장을 나란히 비교할 수 있는 가장 최근 날짜.

    단순 최대 날짜를 쓰면 안 된다. 소스마다 공시 시점이 다르므로(한국 원천은
    당일, 미국 ETF는 마감 후) 한 시장만 하루 앞서 들어오는 일이 흔하고,
    그날을 기준일로 삼으면 크로스마켓 비교가 성립하지 않는다.
    최근 window 세션 안에 전 시장이 모인 날이 없으면 최대 날짜를 돌려주고,
    커버리지 결손은 신뢰도 게이트가 판단하게 둔다.
    """
    if sig.empty:
        return None
    per_day = sig.groupby("ts")["market"].nunique().sort_index(ascending=False)
    for ts, n in per_day.head(window).items():
        if n >= len(MARKET_KO):
            return ts.date()
    return per_day.index[0].date()


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

    known_kr = (_sessions_of(store_path, "krx")
                | _sessions_of(store_path, "naver_kr"))
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

    # KRX 원천이 없으면 키가 필요 없는 네이버 표로 같은 해상도를 확보한다.
    if not r:
        nr, nh = _safe("naver_kr", naver_kr.collect, known=known_kr)
        records += nr
        health.append(nh)

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

    as_of = _comparable_session(sig)
    # 신선도는 달력일이 아니라 '놓친 세션 수'로 잰다. 발송 게이트의 입력값이다.
    stale_sessions = 0
    if as_of and latest_session and hasattr(closes, "index"):
        stale_sessions = sum(1 for d in closes.index if d.date() > as_of)

    score, breakdown = confidence_score(df, as_of)
    if score < MIN_CONFIDENCE:
        # 중단 사유만 별도 파일로 남긴다. data.json은 손대지 않으므로
        # 대시보드는 마지막 정상 상태를 그대로 보여준다.
        halt = {"halted_at": dt.datetime.now(dt.timezone.utc).isoformat(
                    timespec="seconds"),
                "as_of": as_of.isoformat() if as_of else None,
                "confidence": round(score, 3),
                "threshold": MIN_CONFIDENCE,
                "breakdown": breakdown,
                "health": health}
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(os.path.join(os.path.dirname(out_path) or ".", "halt.json"),
                  "w", encoding="utf-8") as f:
            json.dump(halt, f, ensure_ascii=False, indent=1)
        raise LowConfidence(score, breakdown)

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
        "confidence": round(score, 3),
        "confidence_breakdown": breakdown,
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

