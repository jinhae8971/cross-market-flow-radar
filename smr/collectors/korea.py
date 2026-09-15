"""한국 — 투자자별 매매동향 (외국인 / 기관 / 연기금).

4개 시장 중 유일하게 '주체별 × 일별'이 공시된다. 신호엔진 검증의 기준 시장.

소스 체인 (앞에서 실패하면 다음으로):
  1. KRX Open API (data-dbg.krx.co.kr)  — 공식. 무료 AUTH_KEY 필요. 가장 안정적.
  2. data.krx.co.kr getJsonData.cmd     — 키 불필요하나 데이터센터 IP를 차단할 수 있음.
  전부 실패 시 빈 리스트를 반환한다. ETF 백본이 계속 돌기 때문에
  파이프라인 전체가 멈추지는 않는다 — 해상도만 낮아진다.
"""
from __future__ import annotations

import datetime as dt
import os

import requests

from ..schema import FlowRecord
from ..fx import to_usd

OPENAPI = "https://data-dbg.krx.co.kr/svc/apis/sto/stk_bydd_trd"
LEGACY = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
UA = {"User-Agent": "Mozilla/5.0", "Referer": "http://data.krx.co.kr/"}

# 놓친 회차를 되메우는 창. KRX는 AUM과 달리 소급 조회가 되므로,
# 하루 실패했다고 그날이 영구 결측이 될 이유가 없다. 수집기가 매 실행마다
# 최근 구간의 구멍을 스스로 확인하고 메운다.
LOOKBACK_SESSIONS = 7


class KrxUnconfigured(RuntimeError):
    """인증키 미등록 — 장애가 아니라 미설정 상태.

    '키가 없어서 못 부름'과 '불러봤는데 거부당함'은 대응이 완전히 다르다.
    (전자는 5분짜리 등록 작업, 후자는 코드/네트워크 조사)
    지금까지는 둘 다 같은 문구로 뭉개져서, 키가 없다는 사실이
    'KRX 원천 수급 미수집'이라는 결과 문구 뒤에 가려져 있었다.
    """

# KRX 표기 → 공통 actor
ACTOR_MAP = {
    "외국인": "foreign",
    "외국인합계": "foreign",
    "기관합계": "institution",
    "연기금등": "pension",
    "개인": "retail",
}


def _via_openapi(day: dt.date, key: str) -> list[dict]:
    r = requests.get(
        OPENAPI,
        params={"basDd": day.strftime("%Y%m%d")},
        headers={"AUTH_KEY": key},
        timeout=30,
    )
    if r.status_code == 401:
        # KRX는 401을 두 갈래로 준다. 원인이 다르므로 구분해서 알려야 한다.
        msg = ""
        try:
            msg = r.json().get("respMsg", "")
        except ValueError:
            pass
        if "API Call" in msg:
            raise PermissionError(
                "인증키는 유효하나 이 서비스가 미승인 상태입니다 — "
                "openapi.krx.co.kr 마이페이지에서 해당 API 개별 이용신청 후 승인 필요"
            )
        raise PermissionError("인증키가 유효하지 않습니다 (AUTH_KEY 헤더/값 확인)")
    r.raise_for_status()
    return r.json().get("OutBlock_1", [])


def _via_legacy(day: dt.date, mkt: str = "STK") -> list[dict]:
    payload = {
        "bld": "dbms/MDC/STAT/standard/MDCSTAT02203",
        "mktId": mkt,
        "inqTpCd": "1",
        "trdVolVal": "2",
        "askBid": "3",
        "strtDd": day.strftime("%Y%m%d"),
        "endDd": day.strftime("%Y%m%d"),
        "money": "1",
        "csvxls_isNo": "false",
    }
    r = requests.post(LEGACY, data=payload, headers=UA, timeout=30)
    if r.status_code != 200 or r.text.strip() == "LOGOUT":
        raise RuntimeError(f"legacy KRX 거부 (status={r.status_code})")
    return r.json().get("output", [])


def _candidate_days(lookback: int, today: dt.date | None = None) -> list[dt.date]:
    """최근 영업일 후보. 공휴일은 여기서 거르지 않는다.

    KRX 휴장일 달력을 코드에 박으면 매년 손봐야 하고 임시휴장에 틀린다.
    대신 조회 결과가 비면 그날을 휴장으로 간주한다 — 달력을 유지보수하는
    대신 거래소의 응답을 신뢰하는 쪽이 깨질 여지가 적다.
    """
    today = today or dt.date.today()
    out, day = [], today - dt.timedelta(days=1)
    while len(out) < lookback:
        if day.weekday() < 5:          # 0=월 ... 4=금
            out.append(day)
        day -= dt.timedelta(days=1)
    return sorted(out)


def _parse(rows: list[dict], day: dt.date) -> list[FlowRecord]:
    records: list[FlowRecord] = []
    for row in rows:
        label = str(row.get("INVST_NM") or row.get("INVST_TP_NM") or "").strip()
        actor = ACTOR_MAP.get(label)
        if not actor:
            continue
        raw = str(row.get("NETASK_TRDVAL") or row.get("TRDVAL") or "0")
        try:
            krw = float(raw.replace(",", ""))
        except ValueError:
            continue
        records.append(
            FlowRecord(
                ts=day,
                market="KR",
                actor=actor,
                instrument="KOSPI",
                net_flow_usd=to_usd(krw, "KRW", day),
                lag_days=0,
                confidence=0.95,  # 거래소 원천 공시
                source="krx",
            )
        )
    return records


def _fetch_day(day: dt.date, key: str) -> list[dict]:
    if key:
        try:
            return _via_openapi(day, key)
        except PermissionError:
            raise                      # 인증 문제는 다음 날짜로 넘겨도 똑같다
        except Exception as exc:
            print(f"[korea] {day} Open API 실패: {exc}")
    try:
        return _via_legacy(day)
    except Exception as exc:
        print(f"[korea] {day} legacy 실패: {exc}")
        return []


def collect(day: dt.date | None = None, known: set | None = None,
            lookback: int = LOOKBACK_SESSIONS) -> list[FlowRecord]:
    """최근 영업일 중 저장소에 없는 날짜만 채워 넣는다.

    day를 주면 그 하루만 조회한다(수동 재수집·테스트용).
    known에는 이미 확보한 날짜를 넘긴다 — 매 실행마다 같은 날을 다시
    긁어 거래소에 부하를 주지 않기 위해서다.
    """
    key = os.environ.get("KRX_API_KEY", "")
    if not key and day is None:
        # 키 없이도 legacy를 시도는 하되, 전부 실패하면 '미설정'으로 보고한다.
        probe = _fetch_day(_candidate_days(1)[0], "")
        if not probe:
            raise KrxUnconfigured(
                "KRX_API_KEY 미등록 — openapi.krx.co.kr에서 발급 후 "
                "Actions secret에 추가하면 한국 트랙이 원천 공시로 전환됩니다"
                " (현재는 ETF 대리지표)"
            )

    days = [day] if day else [d for d in _candidate_days(lookback)
                              if not known or d not in known]
    records: list[FlowRecord] = []
    holidays = []
    for d in days:
        rows = _fetch_day(d, key)
        if not rows:
            holidays.append(d.isoformat())
            continue
        records += _parse(rows, d)
    if holidays:
        print(f"[korea] 데이터 없음(휴장 또는 미공시): {', '.join(holidays)}")
    return records
