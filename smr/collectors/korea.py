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


def collect(day: dt.date | None = None) -> list[FlowRecord]:
    day = day or dt.date.today() - dt.timedelta(days=1)
    key = os.environ.get("KRX_API_KEY", "")

    rows: list[dict] = []
    if key:
        try:
            rows = _via_openapi(day, key)
        except Exception as exc:
            print(f"[korea] Open API 실패: {exc}")
    if not rows:
        try:
            rows = _via_legacy(day)
        except Exception as exc:
            print(f"[korea] legacy 실패: {exc} — ETF 백본으로 대체 진행")
            return []

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
