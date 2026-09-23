"""L0 수집 → L1 저장 → L2 신호 → L3 로테이션 → L4 배포 산출물.

수집기 하나가 죽어도 나머지는 돈다(우아한 성능저하). 어떤 수집기가 죽었는지는
산출물의 health 필드에 남겨 대시보드에서 바로 보이게 한다.

v2 운영 원칙 (2026-09-23):
  · 저장과 발행을 분리한다. v1은 신뢰도 미달이면 예외로 작업을 끊었고, 그 탓에
    커밋 단계가 건너뛰어져 새 관측이 한 번도 저장되지 않았다. 원천이 살아나도
    기준점이 갱신되지 않으니 스스로 복구할 길이 없는 흡수 상태였다(9/15~9/23
    14회 연속 실패). 이제 관측은 항상 저장하고, 게이트는 '발행'만 막는다.
  · 게이트 판정은 status 필드로 내보낸다: ok / warming(첫 관측 누적 중) / halted.
    텔레그램은 이 상태의 '전환'에만 반응한다(notify.py).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import traceback

import pandas as pd

from . import detail, repair, rotation, signals
from .calendar_mask import masked
from .collectors import cot, issuer, korea, kr_investor
from .schema import MARKETS, FlowStore, to_frame

MARKET_KO = {"KR": "한국", "JP": "일본", "EU": "유럽", "US": "미국"}
KST = dt.timezone(dt.timedelta(hours=9))

# repo variable이 비어 있으면 빈 문자열이 들어오므로 or로 받아낸다.
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE") or "0.60")
# 이 세션 수를 넘어 뒤처지면 발행하지 않는다.
MAX_STALE_SESSIONS = 2
# 크로스마켓 비교가 성립하는 최소 시장 수. 한 시장 휴장·결손은 표기하고 발행한다.
MIN_MARKETS = 3
# 발행사 공시 도착 시각(KST). SSGA navhist Last-Modified ≈ T+1 13:56 KST.
ISSUER_READY_HOUR_KST = 15

# v1 잔재: 기준일 없는 AUM 추정치·OHLCV 대리지표·폐기된 네이버 PC 페이지.
# 대조 검증에서 잡음으로 판명됐거나 다음 금융으로 대체됐으므로 저장소에서 걷어낸다.
LEGACY_SOURCES = ("etf_aum_delta", "etf_moneyflow_proxy", "naver_kr")


def _safe(name: str, fn, *a, **kw):
    try:
        out = fn(*a, **kw)
        return out, {"collector": name, "ok": True, "records": len(out)}
    except Exception as exc:  # noqa: BLE001 — 수집기 격리
        traceback.print_exc()
        return [], {"collector": name, "ok": False,
                    "error": f"{type(exc).__name__}: {str(exc)[:180]}"}


def _comparable_session(sig: pd.DataFrame, window: int = 10,
                        min_markets: int = MIN_MARKETS) -> dt.date | None:
    """시장을 나란히 비교할 수 있는 가장 최근 날짜.

    전 시장이 모인 날을 우선하되, 그보다 최근에 min_markets 이상 모인 날이 있으면
    그날을 쓴다(한 시장 휴장·결손을 표기하고 발행하기 위해). 둘 다 없으면 창 안에서
    가장 많은 시장이 모인 날 중 최신일.
    """
    if sig.empty:
        return None
    per_day = sig.groupby("ts")["market"].nunique().sort_index(ascending=False).head(window)
    full = next((ts for ts, n in per_day.items() if n >= len(MARKETS)), None)
    part = next((ts for ts, n in per_day.items() if n >= min_markets), None)
    if part is not None and (full is None or part > full):
        return part.date()
    if full is not None:
        return full.date()
    best = per_day.max()
    return next(ts for ts, n in per_day.items() if n == best).date()


def expected_session(now: dt.datetime) -> dt.date:
    """지금쯤 발행사 공시가 올라와 있어야 할 가장 최근 미국 세션(평일 근사)."""
    now = now.astimezone(KST)
    day = now.date() - dt.timedelta(days=1)
    if now.hour < ISSUER_READY_HOUR_KST:
        day -= dt.timedelta(days=1)
    while day.weekday() >= 5:
        day -= dt.timedelta(days=1)
    return day


def stale_sessions(as_of: dt.date | None, calendar: list[dt.date],
                   now: dt.datetime) -> int:
    """기준일 이후 놓친 세션 수 — 달력일이 아니라 세션으로 센다.

    과거 구간은 발행사 이력(휴장일 반영)으로, 아직 이력에 없는 최근 구간은 평일로 센다.
    원천이 통째로 멈춰도 '시계'로 뒤처짐을 잡아내기 위해서다 — 관측 날짜끼리만
    비교하면 전부 같이 멈춘 날 뒤처짐이 0으로 보인다.
    """
    if as_of is None:
        return 0
    last_cal = calendar[-1] if calendar else as_of
    n = sum(1 for d in calendar if d > as_of)
    day, end = max(last_cal, as_of), expected_session(now)
    while day < end:
        day += dt.timedelta(days=1)
        if day.weekday() < 5:
            n += 1
    return n


def confidence_score(sig: pd.DataFrame, core: pd.DataFrame, as_of: dt.date | None):
    """기준일 신뢰도 = 관측 품질 x 시장 커버리지.

    곱하는 이유: 한 시장만 고품질이어도 평균은 높게 나온다. 크로스마켓 비교가
    목적이므로 시장 결손은 품질 저하와 같은 무게로 다룬다. 시장 품질은 소스 신뢰도에
    그날의 순자산 커버리지를 곱한 값이다.
    """
    if sig.empty or as_of is None:
        return 0.0, {"관측": "없음"}, {}
    day = pd.Timestamp(as_of)
    rows = sig[sig["ts"] == day]
    cov = {r.market: float(r.coverage) for r in rows.itertuples()}
    if not cov:
        return 0.0, {"관측": "없음"}, {}
    c = core[pd.to_datetime(core["ts"]) == day]
    conf = c.groupby("market")["confidence"].mean().to_dict()
    per = {m: conf.get(m, 0.0) * min(cov[m], 1.0) for m in cov}
    quality = sum(per.values()) / len(per)
    missing = [MARKET_KO[m] for m in MARKETS if m not in cov]
    score = quality * len(cov) / len(MARKETS)
    return score, {"품질": f"{quality:.2f}", "커버리지": f"{len(cov)}/{len(MARKETS)}",
                   "결손": ",".join(missing) or "없음"}, cov


def _market_status(sig: pd.DataFrame, as_of, cov: dict, store: issuer.ObsStore) -> dict:
    out = {}
    for m in MARKETS:
        g = sig[sig["market"] == m]
        last = g["ts"].max().date().isoformat() if len(g) else None
        if m in cov:
            state = "ok"
        elif m in issuer.UNIVERSE and not len(g) and any(
                store.series(s) for s in issuer.UNIVERSE[m]):
            state = "warming"      # 첫 관측은 확보, 직전 관측이 없어 아직 흐름을 못 낸다
        else:
            state = "missing"
        entry = {"state": state, "last": last,
                 "coverage": round(cov[m], 3) if m in cov else None}
        if m in issuer.UNIVERSE:
            entry["obs"] = {s: len(store.series(s)) for s in issuer.UNIVERSE[m]}
        out[m] = entry
    return out


def _status(as_of, score, breakdown, stale, markets: dict, previous: dict) -> dict:
    missing = [m for m, v in markets.items() if v["state"] != "ok"]
    if as_of is None:
        state, reason = "halted", "관측 없음"
    elif stale > MAX_STALE_SESSIONS:
        state, reason = "halted", f"기준일 {as_of} — 최신 세션 대비 {stale}세션 뒤처짐"
    elif score < MIN_CONFIDENCE:
        warming = missing and all(markets[m]["state"] == "warming" for m in missing)
        state = "warming" if warming else "halted"
        reason = (f"신뢰도 {score:.2f} < 기준 {MIN_CONFIDENCE:.2f} ("
                  + " / ".join(f"{k} {v}" for k, v in breakdown.items()) + ")")
    else:
        state, reason = "ok", None
    prev = previous.get("status") or {}
    since = prev.get("since") if prev.get("state") == state else None
    last_ok = (previous.get("as_of") if prev.get("state") == "ok"
               else prev.get("last_ok_as_of"))
    return {"state": state, "reason": reason,
            "since": since or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "last_ok_as_of": as_of.isoformat() if state == "ok" and as_of else last_ok,
            "threshold": MIN_CONFIDENCE}


def _write_if_changed(path: str, payload: dict, previous: dict) -> bool:
    """실행 시각성 필드 외에 바뀐 게 없으면 쓰지 않는다(빈 커밋 방지)."""
    volatile = ("generated_at", "rows_added")

    def norm(p: dict) -> str:
        q = {k: v for k, v in p.items() if k not in volatile}
        if isinstance(q.get("status"), dict):
            q["status"] = {k: v for k, v in q["status"].items() if k != "since"}
        return json.dumps(q, ensure_ascii=False, sort_keys=True, default=str)

    if previous and norm(previous) == norm(payload):
        return False
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return True


def run(seed: bool = False, store_path: str = "data/flows.parquet",
        out_path: str = "docs/data.json", obs_path: str = issuer.OBS_PATH,
        now: dt.datetime | None = None) -> dict:
    now = (now or dt.datetime.now(dt.timezone.utc)).astimezone(KST)
    health: list[dict] = []
    records = []

    # L0-a  발행사 공시 → 관측 저장소 → 흐름 재계산 (일본·유럽·미국 + EWY 맥락)
    obs = issuer.ObsStore(obs_path)
    r, hs = issuer.collect(obs, today=now.date())
    records += r
    health += hs
    obs.save()

    # L0-b  한국 핵심 계열. 저장 이력이 짧으면 깊게 받아 z60·CUSUM 창을 채운다.
    store = FlowStore(store_path)
    known = store.load()
    kr_days = known[known["source"] == "daum_kr"]["ts"].nunique() if len(known) else 0
    per_page = 100 if (seed or kr_days < 80) else 30
    r, h = _safe("daum_kr", kr_investor.collect, per_page=per_page, now=now)
    records += r
    health.append(h)
    if os.environ.get("KRX_API_KEY"):          # 키가 있을 때만 — 없으면 조용히 생략
        kr_known = set(pd.to_datetime(known[known["source"] == "krx"]["ts"]).dt.date) \
            if len(known) else set()
        r, h = _safe("krx", korea.collect, known=kr_known)
        records += r
        health.append(h)

    # L0-c  맥락 지표 — 실패해도 게이트에 영향 없음
    r, h = _safe("cot", cot.collect, 26)
    records += r
    health.append(h)

    # L1  저장 + 자가 복구
    added = store.upsert(to_frame(records))
    df = store.load()
    legacy = df["source"].isin(LEGACY_SOURCES)
    if legacy.any():
        health.append({"collector": "repair", "ok": True, "records": int(legacy.sum()),
                       "error": "v1 잔재(기준일 없는 추정치·대리지표) 제거"})
        df = df[~legacy].reset_index(drop=True)
        store.replace(df)
    df, purged = repair.drop_dead_sessions(df)
    if purged:
        store.replace(df)

    # L2  핵심 계열 → 신호
    universe, weights = issuer.UNIVERSE, issuer.weights(obs)
    core = signals.core_rows(df, universe)
    sig = signals.build(df, weights=weights, universe=universe)
    as_of = _comparable_session(sig)
    score, breakdown, cov = confidence_score(sig, core, as_of)
    calendar = [d for d in obs.calendar() if d <= now.date()]
    stale = stale_sessions(as_of, calendar, now)
    markets = _market_status(sig, as_of, cov, obs)

    previous = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as f:
                previous = json.load(f)
        except (json.JSONDecodeError, OSError):
            previous = {}
    status = _status(as_of, score, breakdown, stale, markets, previous)

    # L3  발화·로테이션 — 기준일에 관측된 시장만, 게이트 통과 시에만
    alerts = signals.alerts(sig, as_of=as_of) if status["state"] == "ok" else []
    kept = []
    for a in alerts:
        flag, why = masked(dt.date.fromisoformat(a["ts"]))
        if flag:
            a["suppressed"] = why
        else:
            kept.append(a)
    rot = rotation.matrix(sig) if len(sig) else {"ready": False, "rows": []}
    if rot.get("ready") and as_of and rot.get("as_of") != as_of.isoformat():
        rot = {"ready": False, "rows": [],
               "reason": f"4개 시장 동시 관측일({rot.get('as_of')})이 기준일과 다름"}
    if status["state"] != "ok":
        rot = {"ready": False, "rows": [], "reason": "발행 보류 중"}

    core_bad = [h for h in health
                if not h.get("ok") and h["collector"] in ("ishares", "ssga", "daum_kr", "krx")]
    quality = [f"{h['collector']}: {h.get('error', '실패')}" for h in core_bad]

    recent = sig[sig["ts"] >= sig["ts"].max() - pd.Timedelta(days=120)] if len(sig) else sig
    series = {
        m: [{"d": r.ts.date().isoformat(), "f": round(r.net_flow_usd / 1e9, 3),
             "z": None if pd.isna(r.z20) else round(float(r.z20), 2)}
            for r in g.itertuples()]
        for m, g in recent.groupby("market")
    } if len(recent) else {}

    payload = {
        "dashboard_url": os.environ.get("DASHBOARD_URL", ""),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "as_of": as_of.isoformat() if as_of else None,
        "latest_session": calendar[-1].isoformat() if calendar else None,
        "stale_sessions": int(stale),
        "confidence": round(score, 3),
        "confidence_breakdown": breakdown,
        "status": status,
        "coverage": markets,
        "mode": "normal",
        "method": {"KR": "KOSPI 외국인 순매수 (다음 금융, 원 → USD·ECB)",
                   "ETF": "발행사 공시 발행좌수 x NAV (iShares·SSGA)"},
        "universe": {m: list(v) for m, v in universe.items()},
        "markets": MARKET_KO,
        "alerts": kept,
        "suppressed": [a for a in alerts if "suppressed" in a],
        "rotation": rot,
        "series": series,
        "detail": detail.build(df, sig, universe, markets, as_of=as_of),
        "health": health,
        "quality_warnings": quality,
        "rows_total": int(len(df)),
        "rows_added": int(added),
    }
    _write_if_changed(out_path, payload, previous)
    return payload
