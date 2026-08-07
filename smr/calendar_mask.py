"""가짜 신호 마스크.

실전에서 알림 품질을 좌우하는 건 신호식이 아니라 이 마스크다.
지수 리밸런싱·만기·배당락일의 기계적 매매는 '스마트머니'가 아니다.
"""
from __future__ import annotations

import datetime as dt


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    d = dt.date(year, month, 1)
    d += dt.timedelta(days=(weekday - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


def masked(day: dt.date) -> tuple[bool, str]:
    """(마스크 여부, 사유)"""
    # 쿼드러플 위칭 — 3·6·9·12월 세 번째 금요일
    if day.month in (3, 6, 9, 12):
        w = _nth_weekday(day.year, day.month, 4, 3)
        if abs((day - w).days) <= 1:
            return True, "선물옵션 동시만기"
    # MSCI 분기 리뷰 반영 — 2·5·8·11월 마지막 영업일 부근
    if day.month in (2, 5, 8, 11) and day.day >= 25:
        return True, "MSCI 분기 리밸런싱"
    # 한국 12월 배당락 직전 기관 매수 왜곡
    if day.month == 12 and day.day >= 26:
        return True, "배당락 전후"
    return False, ""
