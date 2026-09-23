"""한국 투자자별 순매수(KOSPI) — 다음 금융 API, 인증키 불필요.

2026-09-23 네이버 '일자별 순매수' PC 페이지가 러너에서 HTTP 410(Gone)을 반환하기
시작했다(Npay 증권 개편). KRX legacy는 로그인을 요구하고 Open API 키는 미등록.
다음 금융 investor API는 러너·컨테이너 모두에서 열리며, 저장돼 있던 네이버 값과
환율 오차 이내로 일치했다(2026-09-01~14 대조). 수천 영업일 이력도 준다.

응답은 원 단위이고 details에 기관 세부(연기금 포함)가 들어 있다. 스크래핑과
달리 JSON이지만 필드 의미가 바뀌어도 숫자는 그럴듯하게 나오므로, 회계
항등식 두 개로 매번 검산한다(개인+외국인+기관+기타법인=0, 기관 세부 합=기관).
"""
from __future__ import annotations

import datetime as dt

import requests

from ..fx import prefetch, to_usd
from ..schema import FlowRecord

URL = "https://finance.daum.net/api/investor/KOSPI/days"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    # Referer가 없으면 403 — 다음 금융은 자기 화면에서 부른 요청만 받는다.
    "Referer": "https://finance.daum.net/domestic/investors",
}
INSTITUTION_PARTS = ("FINANCIAL_INVESTOR", "INSURANCE_COMPANIES", "MUTUAL_FUND", "BANK",
                     "ETC_FINANCIAL_INSTITUTION", "PENSION_FUND", "PRIVATE_EQUITY_FUND")
TOLERANCE_KRW = 1e9        # 10억원 — 표기 반올림 허용폭
CONFIDENCE = 0.9           # 거래소 원천의 2차 배포처
CLOSE_HOUR_KST = 16        # 15:30 마감 + 정산. 그 전 '오늘' 행은 장중 잠정치다.
KST = dt.timezone(dt.timedelta(hours=9))


class LayoutChanged(RuntimeError):
    """필드 의미가 바뀌어 항등식이 깨진 상태 — 값을 내보내지 않는다."""


def ceiling(now: dt.datetime | None = None) -> dt.date:
    """확정치로 인정할 수 있는 마지막 날짜.

    실행 시각의 '오늘'을 그대로 쓰면 장중 실행분이 잠정치를 확정치처럼 저장한다
    (kospi-tracker가 같은 이유로 조용히 멈췄던 전례가 있다).
    """
    now = (now or dt.datetime.now(KST)).astimezone(KST)
    day = now.date()
    if now.hour < CLOSE_HOUR_KST:
        day -= dt.timedelta(days=1)
    while day.weekday() >= 5:
        day -= dt.timedelta(days=1)
    return day


def parse(payload: dict, now: dt.datetime | None = None) -> list[dict]:
    """API 응답 → 검산을 통과한 확정 행 [{date, foreign, institution, pension, retail}]."""
    rows = payload.get("data") or []
    if not rows:
        raise LayoutChanged("다음 금융 투자자 응답에 data가 없음")
    last_ok = ceiling(now)
    out = []
    for i, row in enumerate(rows):
        try:
            day = dt.date.fromisoformat(str(row["date"])[:10])
            fore = float(row["foreignStraightPurchasePrice"])
            indi = float(row["individualStraightPurchasePrice"])
            inst = float(row["institutionStraightPurchasePrice"])
            det = row.get("details") or {}
            etc = float(det.get("ETC_CORPORATION") or 0.0)
            parts = sum(float(det.get(k) or 0.0) for k in INSTITUTION_PARTS)
            pension = float(det.get("PENSION_FUND") or 0.0)
        except (KeyError, TypeError, ValueError) as exc:
            if i == 0:
                raise LayoutChanged(f"필드 구조 변경: {exc}") from exc
            continue
        broken = (abs(indi + fore + inst + etc) > TOLERANCE_KRW
                  or (det and abs(parts - inst) > TOLERANCE_KRW))
        if broken:
            if i == 0:   # 최신 행이 깨지면 전체를 신뢰할 수 없다
                raise LayoutChanged("주체별 순매수 항등식 불성립 — 필드 의미 확인 필요")
            continue
        if day > last_ok:
            continue
        out.append({"date": day, "foreign": fore, "institution": inst,
                    "pension": pension if det else None, "retail": indi})
    return out


def collect(per_page: int = 30, now: dt.datetime | None = None) -> list[FlowRecord]:
    r = requests.get(URL, params={"page": 1, "perPage": per_page, "details": "true"},
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    rows = parse(r.json(), now)
    if not rows:
        return []
    prefetch(min(x["date"] for x in rows), max(x["date"] for x in rows))
    records: list[FlowRecord] = []
    for x in rows:
        for actor in ("foreign", "institution", "pension", "retail"):
            krw = x[actor]
            if krw is None:
                continue
            records.append(FlowRecord(
                ts=x["date"], market="KR", actor=actor, instrument="KOSPI",
                net_flow_usd=round(to_usd(krw, "KRW", x["date"]), 2), lag_days=0,
                confidence=CONFIDENCE, source="daum_kr"))
    return records
