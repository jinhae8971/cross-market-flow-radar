"""한국 투자자별 순매수 — 인증키가 필요 없는 경로.

KRX는 2026년 들어 Open API(AUTH_KEY) 또는 계정 로그인을 요구하도록 바뀌었고,
legacy getJsonData 엔드포인트는 세션을 심어도 LOGOUT을 돌려준다. 반면
네이버 증권의 '일자별 순매수' 표는 키 없이 열리고 한 번에 최근 20여 영업일을
준다. 해상도는 KRX 원천과 사실상 동일하다(주체별 × 일별 순매수).

표 구조 (2026-09 확인):
    날짜 | 개인 | 외국인 | 기관계 | [기관 6분류] | 기타법인
    기관 6분류 = 금융투자 · 보험 · 투신 · 은행 · 기타금융기관 · 연기금등
    단위: 억원

HTML 스크래핑이므로 레이아웃 변경이 곧 조용한 오류가 된다. 그래서 파싱 직후
두 개의 항등식으로 자가 검산한다 — 회계적으로 반드시 성립해야 하는 관계라
컬럼이 밀리면 즉시 깨진다. 검산에 실패하면 값을 내보내지 않고 예외를 던진다.
"""
from __future__ import annotations

import datetime as dt
import re

import requests

from ..schema import FlowRecord
from ..fx import to_usd

URL = "https://finance.naver.com/sise/investorDealTrendDay.naver"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Referer": "https://finance.naver.com/sise/investorDealTrend.naver",
}

EOKWON = 1e8  # 억원 → 원

# 표의 데이터 컬럼 순서. 헤더가 2단(기관이 colspan=6)이라 이름으로 맞출 수
# 없으므로 순서로 고정하고, 대신 아래 검산으로 정합성을 매번 확인한다.
COLUMNS = ("개인", "외국인", "기관계", "금융투자", "보험", "투신",
           "은행", "기타금융기관", "연기금등", "기타법인")

# 순매수는 시장 전체에서 상계되므로 모든 주체를 합산하면 항상 0이다.
# 따라서 시장 순유입으로 쓸 수 있는 것은 역외 주체뿐이다.
ACTOR_MAP = {"외국인": "foreign", "기관계": "institution",
             "연기금등": "pension", "개인": "retail"}

CHECK_TOLERANCE = 3.0  # 억원. 표기 반올림 오차 허용폭


class LayoutChanged(RuntimeError):
    """표 구조가 바뀌어 컬럼 정합성 검산이 깨진 상태.

    스크래퍼의 최대 위험은 못 읽는 것이 아니라 '밀린 컬럼을 읽고도
    그럴듯한 숫자를 내놓는 것'이다. 그 경우 값을 버린다.
    """


def _rows(html: str) -> list[tuple[dt.date, dict[str, float]]]:
    # 이 페이지에는 <tbody>가 없다. find()가 -1을 돌려주면 html[-1:]이 되어
    # 표 전체를 잃으므로, 찾지 못하면 문서 전체를 훑는다.
    i = html.find("<tbody")
    body = html[i:] if i >= 0 else html
    out = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S):
        cells = [re.sub(r"<[^>]+>", "", c).replace("\xa0", " ").strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
        cells = [c for c in cells if c]
        if len(cells) != len(COLUMNS) + 1:
            continue
        m = re.match(r"(\d{2})\.(\d{2})\.(\d{2})", cells[0])
        if not m:
            continue
        day = dt.date(2000 + int(m.group(1)), int(m.group(2)), int(m.group(3)))
        vals = {}
        for name, raw in zip(COLUMNS, cells[1:]):
            try:
                vals[name] = float(raw.replace(",", "").replace("+", ""))
            except ValueError:
                vals = {}
                break
        if vals:
            out.append((day, vals))
    return out


def _verify(vals: dict[str, float]) -> None:
    """회계 항등식 두 개로 컬럼 정렬을 검산한다."""
    zero_sum = (vals["개인"] + vals["외국인"] + vals["기관계"] + vals["기타법인"])
    if abs(zero_sum) > CHECK_TOLERANCE * 10:
        raise LayoutChanged(
            f"주체별 순매수 합계가 0이 아님({zero_sum:,.0f}억) — 컬럼 정렬 확인 필요")
    parts = sum(vals[k] for k in ("금융투자", "보험", "투신", "은행",
                                  "기타금융기관", "연기금등"))
    if abs(parts - vals["기관계"]) > CHECK_TOLERANCE:
        raise LayoutChanged(
            f"기관 세부 합({parts:,.0f})과 기관계({vals['기관계']:,.0f}) 불일치")


def collect(known: set | None = None, market: str = "01",
            max_days: int = 20) -> list[FlowRecord]:
    """최근 영업일 순매수. known에 있는 날짜는 건너뛴다. market 01=KOSPI."""
    # bizdate를 명시해야 최신 영업일 표가 안정적으로 내려온다.
    bizdate = (dt.date.today() - dt.timedelta(days=1))
    while bizdate.weekday() >= 5:
        bizdate -= dt.timedelta(days=1)
    r = requests.get(URL, params={"bizdate": bizdate.strftime("%Y%m%d"),
                                  "sosok": market},
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    r.encoding = "euc-kr"
    rows = _rows(r.text)
    if not rows:
        raise LayoutChanged("일자별 순매수 표를 찾지 못함")

    _verify(rows[0][1])          # 최신 행으로 레이아웃 검증

    records: list[FlowRecord] = []
    for day, vals in rows[:max_days]:
        if known and day in known:
            continue
        for label, actor in ACTOR_MAP.items():
            krw = vals[label] * EOKWON
            records.append(
                FlowRecord(
                    ts=day, market="KR", actor=actor, instrument="KOSPI",
                    net_flow_usd=to_usd(krw, "KRW", day),
                    lag_days=0,
                    # 거래소 원천(0.95)보다는 낮게 — 2차 배포처이고 스크래핑이다.
                    confidence=0.9,
                    source="naver_kr",
                )
            )
    return records
