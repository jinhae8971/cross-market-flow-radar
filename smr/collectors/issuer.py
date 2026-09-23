"""발행사 공시 기반 ETF 순설정(creation/redemption) — 일본·유럽·미국 백본 v2.

v1은 yfinance totalAssets로 flow = AUM_t - AUM_{t-1}(1+r)을 추정했다. 그 값에는
기준일이 없었다. 2026-09-08 이후 값이 멈춘 것도 문제였지만, 2026-09-23에
SSGA 공시와 대조해 보니 동결 이전 구간(08-11~09-08)조차 SPY 상관계수 -0.11,
20세션 합계 +$25.9B(실제 -$0.1B)로 사실상 잡음이었다. 날짜 없는 값은
가격수익률을 유령 흐름으로 바꾼다. 그래서 v2는 기준일이 명시된 원천만 쓴다.

    iShares product screener  1회 요청으로 전 iShares 펀드의 순자산·NAV·기준일
    SSGA NAV history (xlsx)   펀드별 일별 발행좌수·NAV 전체 이력 → 소급 가능

순설정 = (발행좌수_t - 발행좌수_prev) x NAV_t
  · 직전 관측이 바로 앞 세션일 때만 계산한다. 수집 공백을 한 세션에 몰아
    기록하면 z-score가 부풀므로, 그 구간은 버린다(합계보다 신호 무결성 우선).
  · 발행좌수가 급변했는데 순자산이 그대로면 액면분할이다 — 흐름이 아니다.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import math
import os
from typing import Iterable

import requests

from ..schema import FlowRecord

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# 시장별 핵심 ETF. 기준: 날짜가 찍힌 발행사 원천이 있는 펀드만.
# VGK(Vanguard)·QQQ(Invesco)·BBJP(JPM)·DXJ(WisdomTree)·FLKR(Franklin)은 일별
# 기준일 원천을 확보하지 못해 제외했다. 과거 이력 없는 대체 종목을 새로 넣으면
# 구성이 바뀌는 날 z-score 분모가 단절되므로 추가하지 않는다.
UNIVERSE: dict[str, tuple[str, ...]] = {
    "JP": ("EWJ",),
    "EU": ("EZU", "FEZ", "EWG", "EWU", "EWQ"),
    "US": ("SPY", "IWM", "DIA"),
}
# 한국은 KOSPI 외국인 순매수가 핵심 계열이고, EWY 순설정은 맥락 지표로만 싣는다.
CONTEXT: dict[str, tuple[str, ...]] = {"KR": ("EWY",)}

ISHARES = ("EWY", "EWJ", "EZU", "EWG", "EWU", "EWQ", "IWM")
SSGA = ("SPY", "DIA", "FEZ")

SYM_TO_MARKET = {s: m for d in (UNIVERSE, CONTEXT) for m, t in d.items() for s in t}

ISHARES_URL = ("https://www.ishares.com/us/product-screener/product-screener-v3.1.jsn"
               "?dcrPath=/templatedata/config/product-screener-v3/data/en/us-ishares/"
               "ishares-product-screener-backend-config&siteEntryPassthrough=true")
SSGA_URL = ("https://www.ssga.com/us/en/intermediary/library-content/products/"
            "fund-data/etfs/us/navhist-us-en-{sym}.xlsx")

OBS_PATH = "data/fund_obs.json"
KEEP_DAYS = 550            # 약 1.5년 — z60·CUSUM(60) 창을 채우고도 남는 길이
CONFIDENCE = 0.9           # 발행사 공식 발행좌수 — 추정식이 아니라 원천 공시
SHARE_NOISE = 2            # 순자산/NAV 역산 시 부동소수 오차(주)
SPLIT_RATIO = 1.5          # 발행좌수가 하루에 1.5배/0.67배를 넘으면 분할 의심


class SourceError(RuntimeError):
    """원천 응답이 기대 형식이 아닐 때. 조용히 0을 내보내지 않기 위해 던진다."""


# ── 관측 저장소 ─────────────────────────────────────────────────────────────
class ObsStore:
    """펀드별 (기준일 → 발행좌수, NAV, 출처). 흐름은 매 실행 여기서 재계산한다.

    흐름 대신 원관측을 저장하는 이유: 계산식이나 가드를 고쳐도 과거 흐름을
    다시 만들 수 있고, 원천이 소급 불가(iShares)여도 관측 자체는 남는다.
    """

    def __init__(self, path: str = OBS_PATH) -> None:
        self.path = path
        self.funds: dict[str, dict[str, list]] = {}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.funds = json.load(f).get("funds", {})
            except (json.JSONDecodeError, OSError):
                self.funds = {}
        self._snapshot = self._dump()

    def put(self, sym: str, day: dt.date, shares: float, nav: float, src: str) -> None:
        self.funds.setdefault(sym, {})[day.isoformat()] = [
            int(round(shares)), round(float(nav), 6), src]

    def series(self, sym: str) -> list[tuple[dt.date, int, float]]:
        rows = self.funds.get(sym, {})
        return [(dt.date.fromisoformat(k), v[0], v[1]) for k, v in sorted(rows.items())]

    def latest(self, sym: str) -> tuple[dt.date, int, float] | None:
        s = self.series(sym)
        return s[-1] if s else None

    def calendar(self) -> list[dt.date]:
        """관측된 모든 기준일의 합집합 = 미국 펀드 영업일 달력.

        SSGA 이력이 매일 들어오므로 휴장일까지 정확하다. 달력을 코드에 박지 않는다.
        """
        days = {k for rows in self.funds.values() for k in rows}
        return sorted(dt.date.fromisoformat(d) for d in days)

    def prune(self, today: dt.date) -> None:
        cut = (today - dt.timedelta(days=KEEP_DAYS)).isoformat()
        for sym in list(self.funds):
            self.funds[sym] = {k: v for k, v in self.funds[sym].items() if k >= cut}

    def _dump(self) -> str:
        # 날짜 한 줄씩 — git diff가 '어느 펀드 어느 날짜가 바뀌었나'로 읽히게.
        lines = ['{"schema": 1, "funds": {']
        syms = sorted(self.funds)
        for i, sym in enumerate(syms):
            lines.append(f' "{sym}": {{')
            items = sorted(self.funds[sym].items())
            for j, (k, v) in enumerate(items):
                tail = "," if j < len(items) - 1 else ""
                lines.append(f'  "{k}": {json.dumps(v, ensure_ascii=False)}{tail}')
            lines.append(" }" + ("," if i < len(syms) - 1 else ""))
        lines.append("}}")
        return "\n".join(lines) + "\n"

    def save(self) -> bool:
        """내용이 바뀐 경우에만 쓴다(멱등). 반환값: 실제로 썼는지."""
        text = self._dump()
        if text == self._snapshot and os.path.exists(self.path):
            return False
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        self._snapshot = text
        return True


# ── 원천 파서 ───────────────────────────────────────────────────────────────
def _num(field) -> float | None:
    if isinstance(field, dict):
        field = field.get("r")
    try:
        v = float(field)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def _yyyymmdd(field) -> dt.date | None:
    if isinstance(field, dict):
        field = field.get("r")
    try:
        s = str(int(field))
        return dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except (TypeError, ValueError):
        return None


def parse_ishares(payload: dict, tickers: Iterable[str] = ISHARES
                  ) -> dict[str, tuple[dt.date, float, float]]:
    """screener JSON → {티커: (기준일, 발행좌수, NAV)}.

    순자산과 NAV의 기준일이 다르면 그 펀드는 버린다 — 서로 다른 날의 값을
    나누면 발행좌수가 가격 변동만큼 틀어지고, 그 오차가 곧 유령 흐름이다.
    """
    want = set(tickers)
    out: dict[str, tuple[dt.date, float, float]] = {}
    for fund in payload.values():
        if not isinstance(fund, dict):
            continue
        sym = fund.get("localExchangeTicker")
        if sym not in want:
            continue
        tna = _num(fund.get("totalNetAssets")) or _num(fund.get("totalNetAssetsFund"))
        nav = _num(fund.get("navAmount"))
        tna_day = _yyyymmdd(fund.get("totalNetAssetsFundAsOf"))
        nav_day = _yyyymmdd(fund.get("navAmountAsOf"))
        if not (tna and nav and tna_day and nav_day) or tna_day != nav_day:
            continue
        out[sym] = (tna_day, tna / nav, nav)
    return out


def parse_ssga(blob: bytes) -> list[tuple[dt.date, float, float]]:
    """navhist xlsx → [(기준일, 발행좌수, NAV)] 오름차순.

    헤더 위치를 고정하지 않고 'Date' 행을 찾는다. 순자산 = 발행좌수 x NAV 항등식이
    깨지는 행은 컬럼이 밀린 것이므로 버린다.
    """
    import pandas as pd

    raw = pd.read_excel(io.BytesIO(blob), header=None)
    first = raw[0].astype(str).str.strip().str.lower()
    hits = raw.index[first == "date"]
    if len(hits) == 0:
        raise SourceError("SSGA NAV history: 'Date' 헤더를 찾지 못함")
    head = [str(x).strip().lower() for x in raw.iloc[hits[0]].tolist()]
    try:
        c_nav = head.index("nav")
        c_sh = head.index("shares outstanding")
        c_tna = head.index("total net assets")
    except ValueError as exc:
        raise SourceError(f"SSGA NAV history 컬럼 구조 변경: {head[:6]}") from exc
    body = raw.iloc[hits[0] + 1:]
    days = pd.to_datetime(body[0], format="%d-%b-%Y", errors="coerce")
    out = []
    for day, nav, sh, tna in zip(days, body[c_nav], body[c_sh], body[c_tna]):
        if pd.isna(day):
            continue
        try:
            nav, sh, tna = float(nav), float(sh), float(tna)
        except (TypeError, ValueError):
            continue
        if not (nav > 0 and sh > 0 and tna > 0):
            continue
        if abs(sh * nav / tna - 1.0) > 0.005:
            continue
        out.append((day.date(), sh, nav))
    if not out:
        raise SourceError("SSGA NAV history: 유효 행 없음")
    return sorted(out)


def fetch_ishares(session: requests.Session | None = None) -> dict:
    s = session or requests
    r = s.get(ISHARES_URL, headers={"User-Agent": UA, "Accept": "application/json"},
              timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_ssga(sym: str, session: requests.Session | None = None) -> bytes:
    s = session or requests
    r = s.get(SSGA_URL.format(sym=sym.lower()), headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    if not r.content.startswith(b"PK"):
        raise SourceError(f"SSGA {sym}: xlsx가 아닌 응답")
    return r.content


# ── 흐름 계산 ───────────────────────────────────────────────────────────────
def flows(store: ObsStore, syms: Iterable[str]) -> tuple[list[FlowRecord], dict]:
    """저장된 관측으로 흐름을 재계산한다. 반환: (레코드, 진단)."""
    cal = store.calendar()
    prev_of = {d: (cal[i - 1] if i else None) for i, d in enumerate(cal)}
    records: list[FlowRecord] = []
    diag = {"gaps": [], "splits": []}
    for sym in syms:
        series = store.series(sym)
        for (d0, s0, n0), (d1, s1, n1) in zip(series, series[1:]):
            if prev_of.get(d1) != d0:
                diag["gaps"].append(f"{sym} {d0}→{d1}")
                continue
            ratio = s1 / s0 if s0 else math.inf
            if ratio > SPLIT_RATIO or ratio < 1 / SPLIT_RATIO:
                # 순자산이 연속이면 분할, 아니면 원천 이상 — 어느 쪽이든 흐름이 아니다.
                diag["splits"].append(f"{sym} {d1} x{ratio:.2f}")
                continue
            delta = s1 - s0
            if abs(delta) <= SHARE_NOISE:
                delta = 0
            records.append(FlowRecord(
                ts=d1, market=SYM_TO_MARKET[sym], actor="foreign", instrument=sym,
                net_flow_usd=round(delta * n1, 2), lag_days=0,
                confidence=CONFIDENCE, source="etf_flow"))
    return records, diag


def weights(store: ObsStore) -> dict[str, float]:
    """최신 순자산(발행좌수 x NAV) — 시장 내 커버리지 가중치."""
    out = {}
    for sym in SYM_TO_MARKET:
        last = store.latest(sym)
        if last:
            out[sym] = float(last[1]) * float(last[2])
    return out


def collect(store: ObsStore, today: dt.date | None = None
            ) -> tuple[list[FlowRecord], list[dict]]:
    """두 원천을 각각 격리해 수집한다 — 한쪽 장애가 다른 쪽을 막지 않는다."""
    today = today or dt.date.today()
    health = []
    session = requests.Session()

    try:
        snap = parse_ishares(fetch_ishares(session))
        for sym, (day, shares, nav) in snap.items():
            store.put(sym, day, shares, nav, "ishares")
        missing = [s for s in ISHARES if s not in snap]
        asof = max((v[0] for v in snap.values()), default=None)
        health.append({"collector": "ishares", "ok": not missing, "records": len(snap),
                       "asof": asof.isoformat() if asof else None,
                       **({"error": "기준일 불일치·결측: " + ",".join(missing)}
                          if missing else {})})
    except Exception as exc:  # noqa: BLE001 — 원천 격리
        health.append({"collector": "ishares", "ok": False,
                       "error": f"{type(exc).__name__}: {str(exc)[:160]}"})

    fails, latest = [], None
    for sym in SSGA:
        try:
            rows = parse_ssga(fetch_ssga(sym, session))
            cut = today - dt.timedelta(days=KEEP_DAYS)
            for day, shares, nav in rows:
                if day >= cut:
                    store.put(sym, day, shares, nav, "ssga")
            latest = max(latest or rows[-1][0], rows[-1][0])
        except Exception as exc:  # noqa: BLE001
            fails.append(f"{sym}: {type(exc).__name__} {str(exc)[:80]}")
    health.append({"collector": "ssga", "ok": not fails, "records": len(SSGA) - len(fails),
                   "asof": latest.isoformat() if latest else None,
                   **({"error": " / ".join(fails)} if fails else {})})

    store.prune(today)
    recs, diag = flows(store, SYM_TO_MARKET)
    if diag["splits"]:
        health.append({"collector": "issuer_guard", "ok": True, "records": len(diag["splits"]),
                       "error": "분할 의심 제외: " + ", ".join(diag["splits"][-3:])})
    return recs, health
