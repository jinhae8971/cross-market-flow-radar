#!/usr/bin/env python3
"""텔레그램 브리프. 알림이 없으면 하루 1회 스냅샷만 보낸다(중복 방지 상태 파일)."""
from __future__ import annotations

import datetime as dt
import json
import os

import requests

STATE = "data/notify_state.json"
KO = {"KR": "한국", "JP": "일본", "EU": "유럽", "US": "미국"}


def load_config() -> dict:
    cfg = {
        "telegram_token": os.environ.get("TELEGRAM_TOKEN", ""),
        "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID", ""),
    }
    path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                key = k.lower()
                if key in cfg and not cfg[key]:
                    cfg[key] = v
        except (json.JSONDecodeError, OSError) as e:
            print(f"[config] 읽기 실패 - 환경변수만 사용: {e}")
    return cfg


def dashboard_url(d: dict) -> str:
    """Secret로 준 값 우선, 없으면 GITHUB_REPOSITORY에서 Pages 주소를 조립한다."""
    url = d.get("dashboard_url") or os.environ.get("DASHBOARD_URL", "")
    if url:
        return url.rstrip("/")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if "/" in repo:
        owner, name = repo.split("/", 1)
        return f"https://{owner}.github.io/{name}/"
    return ""


def build_message(d: dict) -> str:
    """텔레그램은 결론만. 근거는 대시보드에서 본다."""
    lines = [f"<b>🌐 Cross-Market Flow Radar</b>  <code>{d['as_of']}</code>", ""]

    rot = d.get("rotation", {})
    if rot.get("ready"):
        src, dst = rot.get("from"), rot.get("to")
        lines.append(f"<b>로테이션</b> {KO[src]} → {KO[dst]}" if src and dst
                     else "<b>로테이션</b> 방향성 뚜렷하지 않음")
        for r in rot["rows"]:
            arrow = "▲" if r["delta"] >= 0 else "▼"
            lines.append(f"  {KO[r['market']]} {r['share']:>5.1f}% {arrow}{abs(r['delta'])}")
        lines.append("")

    if d["alerts"]:
        lines.append("<b>발화 신호</b>")
        for a in d["alerts"]:
            lines.append(
                f"  {'🟢' if a['flow_usd'] > 0 else '🔴'} {KO[a['market']]} {a['direction']} "
                f"${a['flow_usd']/1e9:+.2f}B · z {a['z20']:.2f} · {'+'.join(a['triggers'])}"
            )
    else:
        lines.append("발화 조건을 충족한 시장 없음")

    for s in d.get("suppressed", []):
        lines.append(f"  ⚪ {KO[s['market']]} — {s['suppressed']}로 억제")

    # 시장별 한 줄 요약 — 상세는 대시보드로
    lines.append("")
    lines.append("<b>시장별 순유입</b>")
    for m in ("KR", "JP", "EU", "US"):
        det = d.get("detail", {}).get(m)
        if not det:
            continue
        z = det["signal"]["z20"]
        lines.append(
            f"  {KO[m]} ${det['latest']:+.2f}B "
            f"(20일 ${det['cum']['d20']:+.2f}B · z {z if z is not None else '–'})"
        )

    bad = [h for h in d.get("health", []) if not h["ok"]]
    if bad:
        lines.append("")
        lines.append("⚠️ 수집 실패: " + ", ".join(h["collector"] for h in bad))

    url = dashboard_url(d)
    if url:
        lines.append("")
        lines.append(f'📊 <a href="{url}">대시보드에서 상세 보기</a>')
        lines.append("<i>시장을 누르면 기여 종목 · 주체 · 신호값이 펼쳐집니다</i>")
    return "\n".join(lines)


def main() -> None:
    with open("docs/data.json", encoding="utf-8") as f:
        d = json.load(f)

    today = dt.date.today().isoformat()
    state = {}
    if os.path.exists(STATE):
        try:
            with open(STATE, encoding="utf-8") as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    if state.get("last_sent") == today and not d["alerts"]:
        print("[notify] 오늘 스냅샷 이미 발송 - 생략")
        return

    cfg = load_config()
    if not cfg["telegram_token"] or not cfg["telegram_chat_id"]:
        print("[telegram] 자격증명 없음 - 발송 생략")
        print(build_message(d))
        return

    r = requests.post(
        f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
        json={"chat_id": cfg["telegram_chat_id"], "text": build_message(d),
              "parse_mode": "HTML", "disable_web_page_preview": True},
        timeout=20,
    )
    r.raise_for_status()

    os.makedirs("data", exist_ok=True)
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump({"last_sent": today}, f)


if __name__ == "__main__":
    main()
