#!/usr/bin/env python3
"""실행 오류(코드·환경 장애) 알림 — 연속 2회부터, 20시간에 1회만.

발행 보류(데이터 품질)는 notify.py가 상태 전환으로 다룬다. 여기는 워크플로우가
'실패'로 끝난 경우 — 테스트 실패, 예외, 푸시 실패 — 만 본다. 1회 실패는 러너
일시 장애일 수 있으므로 기록만 하고, 연속될 때 알린다. 성공하면 run.py가
카운터를 0으로 되돌린다.
"""
from __future__ import annotations

import datetime as dt
import json
import os

import requests

STATE = "data/run_state.json"
KST = dt.timezone(dt.timedelta(hours=9))
ALERT_AFTER = 2
COOLDOWN_H = 20


def load() -> dict:
    try:
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save(s: dict) -> None:
    os.makedirs(os.path.dirname(STATE) or ".", exist_ok=True)
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, sort_keys=True)


def decide(state: dict, now: dt.datetime) -> tuple[bool, dict]:
    s = dict(state)
    s["consecutive_failures"] = int(s.get("consecutive_failures") or 0) + 1
    s["last_failure"] = now.isoformat(timespec="seconds")
    if s["consecutive_failures"] < ALERT_AFTER:
        return False, s
    h = now.astimezone(KST).hour
    if h >= 22 or h < 7:
        return False, s                     # 야간 — 다음 실패 때 알린다
    last = s.get("last_alert")
    if last and now - dt.datetime.fromisoformat(last) < dt.timedelta(hours=COOLDOWN_H):
        return False, s
    s["last_alert"] = now.isoformat(timespec="seconds")
    return True, s


def reset_on_success() -> None:
    """성공 실행에서 호출 — 카운터가 0이 아닐 때만 써서 빈 커밋을 만들지 않는다."""
    s = load()
    if int(s.get("consecutive_failures") or 0) == 0:
        return
    s["consecutive_failures"] = 0
    s["recovered_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    save(s)


def main() -> None:
    now = dt.datetime.now(dt.timezone.utc)
    alert, s = decide(load(), now)
    save(s)
    n = s["consecutive_failures"]
    if not alert:
        print(f"[ops] 실행 오류 {n}회 연속 — 알림 조건 미충족(기록만)")
        return
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("[ops] 자격증명 없음")
        return
    text = (f"⚠️ Flow Radar 실행 오류 {n}회 연속 — 코드·환경 점검 필요\n"
            f"(데이터 보류가 아니라 워크플로우 자체 실패)\n{os.environ.get('RUN_URL', '')}")
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": text}, timeout=20)
    print(f"[ops] 알림 발송 HTTP {r.status_code}")


if __name__ == "__main__":
    main()
