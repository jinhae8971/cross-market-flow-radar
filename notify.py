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


# 이 세션 수를 넘어 뒤처지면 숫자를 '참고값'으로도 내보내지 않는다.
MAX_STALE_SESSIONS = 2


def build_stale_message(d: dict) -> str:
    """신선도 게이트에 걸렸을 때의 장애 통지.

    통상 브리프와 형식을 일부러 다르게 간다. 같은 레이아웃에 '(참고값)'만
    덧붙이면 수신자는 습관적으로 최신치로 읽는다. 실제로 7세션 묵은 수치가
    매일 정상 브리프처럼 발송됐다. 숫자는 아예 싣지 않는다.
    """
    bad = [h for h in d.get("health", []) if not h.get("ok")]
    lines = [
        "<b>🛑 Cross-Market Flow Radar — 발송 보류</b>",
        "",
        f"최신 관측 <code>{d.get('as_of')}</code> · "
        f"직전 세션 <code>{d.get('latest_session')}</code>",
        f"<b>{d.get('stale_sessions')}세션 뒤처짐</b> — 브리프 수치는 게시하지 않습니다.",
        "",
        "<b>원인</b>",
    ]
    lines += [f"  · {h['collector']}: {h.get('error', '실패')}" for h in bad] or \
             ["  · 사유 미기록 — Actions 로그 확인 필요"]
    lines += ["", "AUM 스냅샷은 소급 수집이 불가하므로 해당 구간은 영구 결측입니다."]
    url = dashboard_url(d)
    if url:
        lines += ["", f'📊 <a href="{url}">대시보드</a>']
    return "\n".join(lines)


def is_stale(d: dict) -> bool:
    return int(d.get("stale_sessions") or 0) > MAX_STALE_SESSIONS


def build_message(d: dict) -> str:
    """텔레그램은 결론만. 근거는 대시보드에서 본다."""
    if is_stale(d):
        return build_stale_message(d)

    banner = " · 대리지표" if d.get("mode") == "degraded" else ""
    lines = [f"<b>🌐 Cross-Market Flow Radar</b>  "
             f"<code>{d['as_of']}</code>{banner}", ""]

    quality = d.get("quality_warnings", [])
    if quality:
        lines.append("⚠️ 데이터 품질 확인 필요 — 신규 신호·로테이션 판단 보류")
    rot = {} if quality else d.get("rotation", {})
    if rot.get("ready"):
        src, dst = rot.get("from"), rot.get("to")
        lines.append(f"<b>로테이션</b> {KO[src]} → {KO[dst]}" if src and dst
                     else "<b>로테이션</b> 방향성 뚜렷하지 않음")
        for r in rot["rows"]:
            arrow = "▲" if r["delta"] >= 0 else "▼"
            lines.append(f"  {KO[r['market']]} {r['share']:>5.1f}% {arrow}{abs(r['delta'])}")
        lines.append("")

    if d["alerts"] and not quality:
        lines.append("<b>발화 신호</b>")
        for a in d["alerts"]:
            lines.append(
                f"  {'🟢' if a['flow_usd'] > 0 else '🔴'} {KO[a['market']]} {a['direction']} "
                f"${a['flow_usd']/1e9:+.2f}B · z {a['z20']:.2f} · {'+'.join(a['triggers'])}"
            )
    else:
        lines.append("신호 판단 보류" if quality else "발화 조건을 충족한 시장 없음")

    for s in d.get("suppressed", []):
        lines.append(f"  ⚪ {KO[s['market']]} — {s['suppressed']}로 억제")

    # 시장별 한 줄 요약 — 상세는 대시보드로
    lines.append("")
    if d.get("mode") == "degraded":
        note = " (대리지표 · 해상도 낮음)"     # 날짜는 최신, 추정 방식만 다르다
    elif quality:
        note = " (이전 관측 참고값)"
    else:
        note = ""
    lines.append("<b>시장별 순유입</b>" + note)
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
        for h in bad:
            # 수집기 이름만 적으면 원인을 보려고 매번 Actions 로그를 열어야 한다.
            lines.append(f"⚠️ 수집 실패 · {h['collector']}: {h.get('error', '사유 미기록')}")

    status = {h.get("collector"): h.get("status") for h in d.get("health", [])}
    if status.get("krx") == "unconfigured":
        # 조치 가능한 상태는 조치 방법까지 적는다. '미수집'만 적으면
        # 매일 같은 줄을 보면서도 무엇을 해야 할지 알 수 없다.
        lines.append("⚠️ KRX 인증키 미등록 — 한국은 ETF 대리지표 "
                     "(openapi.krx.co.kr 발급 후 KRX_API_KEY secret 등록)")
    elif status.get("krx") == "unavailable":
        lines.append("⚠️ KRX 신규 공시 없음 — 한국은 ETF 대리지표")

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
    force = os.environ.get("NOTIFY_FORCE", "").lower() == "true"
    if state.get("last_sent") == today and not d["alerts"] and not force:
        print("[notify] 오늘 스냅샷 이미 발송 - 생략 "
              "(재발송하려면 workflow_dispatch의 force_notify=true)")
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

