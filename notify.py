#!/usr/bin/env python3
"""텔레그램 브리프 — '세션당 1회' + '상태 전환 시에만' 알림.

v1은 실패 통지를 실행마다(하루 2회) 보냈다. 같은 경고가 8일간 반복되면 경고는
배경이 된다. v2는 상태 기계로 바꾼다:
  ok       새 기준일이 생겼을 때만 브리프 1회. 직전이 보류였다면 '재개'를 붙인다.
  halted   보류로 '전환'될 때 1회, 이후 7일마다 재알림. 복구되면 브리프로 알린다.
  warming  첫 관측 누적 중 — 전환 시 1회만 안내.
공통: KST 22:00~07:00에는 보내지 않는다(다음 실행으로 이월). 수동 강제 발송은 예외.
"""
from __future__ import annotations

import datetime as dt
import json
import os

import requests

STATE = "data/notify_state.json"
KO = {"KR": "한국", "JP": "일본", "EU": "유럽", "US": "미국"}
KST = dt.timezone(dt.timedelta(hours=9))
NIGHT = (22, 7)                 # [22시, 7시) 발송 금지
REMIND_DAYS = 7
PARTIAL_HOLD = (12, 16)         # 공시 도착 대기창 — 이 시간대에는 결손 브리프를 미룬다


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


def _link(d: dict) -> list[str]:
    url = dashboard_url(d)
    return ["", f'📊 <a href="{url}">대시보드에서 상세 보기</a>'] if url else []


def _market_lines(d: dict) -> list[str]:
    lines = []
    cov = d.get("coverage", {})
    for m in ("KR", "JP", "EU", "US"):
        det = (d.get("detail") or {}).get(m)
        st = (cov.get(m) or {}).get("state")
        if st == "warming":
            lines.append(f"  {KO[m]} — 첫 관측 확보, 다음 세션부터 산출")
            continue
        if not det or (d.get("as_of") and det.get("ts") != d.get("as_of")):
            lines.append(f"  {KO[m]} — 기준일 미관측 (휴장 또는 원천 미갱신)")
            continue
        z = det["signal"]["z20"]
        ztxt = f"z {z}" if z is not None else f"z 누적 {det.get('observations', 0)}/8"
        lines.append(f"  {KO[m]} ${det['latest']:+.2f}B "
                     f"(20일 ${det['cum']['d20']:+.2f}B · {ztxt})")
    return lines


def build_message(d: dict, recovered: bool = False) -> str:
    """정상 브리프. 텔레그램은 결론만, 근거는 대시보드에서 본다."""
    lines = [f"<b>🌐 Cross-Market Flow Radar</b>  <code>{d['as_of']}</code>", ""]
    if recovered:
        lines += ["✅ 발행 재개 — 발행사 공시 기반 v2 원천으로 복구", ""]

    rot = d.get("rotation", {})
    if rot.get("ready"):
        src, dst = rot.get("from"), rot.get("to")
        lines.append(f"<b>로테이션</b> {KO[src]} → {KO[dst]}" if src and dst
                     else "<b>로테이션</b> 방향성 뚜렷하지 않음")
        for r in rot["rows"]:
            arrow = "▲" if r["delta"] >= 0 else "▼"
            lines.append(f"  {KO[r['market']]} {r['share']:>5.1f}% {arrow}{abs(r['delta'])}")
        lines.append("")

    if d.get("alerts"):
        lines.append("<b>발화 신호</b>")
        for a in d["alerts"]:
            lines.append(
                f"  {'🟢' if a['flow_usd'] > 0 else '🔴'} {KO[a['market']]} {a['direction']} "
                f"${a['flow_usd']/1e9:+.2f}B · z {a['z20']:.2f} · {'+'.join(a['triggers'])}")
    else:
        lines.append("발화 조건을 충족한 시장 없음")
    for s in d.get("suppressed", []):
        lines.append(f"  ⚪ {KO[s['market']]} — {s['suppressed']}로 억제")

    lines += ["", "<b>시장별 순유입</b> (한국=외국인 순매수 · 일·유·미=ETF 순설정)"]
    lines += _market_lines(d)

    for q in d.get("quality_warnings", []):
        lines.append(f"⚠️ 수집 실패 · {q}")
    return "\n".join(lines + _link(d))


def build_halt_message(d: dict, reminder: bool = False) -> str:
    st = d.get("status", {})
    head = "⏰ 발행 보류 지속" if reminder else "🛑 발행 보류"
    lines = [f"<b>{head} — Cross-Market Flow Radar</b>", "",
             f"사유: {st.get('reason') or '미기록'}",
             f"기준일 <code>{d.get('as_of')}</code> · 마지막 정상 발행 "
             f"<code>{st.get('last_ok_as_of') or '없음'}</code>"]
    bad = [h for h in d.get("health", []) if not h.get("ok")]
    if bad:
        lines += ["", "<b>원인 후보</b>"]
        lines += [f"  · {h['collector']}: {h.get('error', '실패')}" for h in bad]
    lines += ["", f"복구되면 브리프로 바로 알려드리고, 미복구 시 {REMIND_DAYS}일 뒤 다시 알립니다."]
    return "\n".join(lines + _link(d))


def build_warming_message(d: dict) -> str:
    lines = ["<b>🔄 Cross-Market Flow Radar — v2 전환 · 워밍업</b>", "",
             "ETF 백본을 날짜가 찍힌 발행사 공시(발행좌수 x NAV)로 교체했습니다.",
             "한국은 KOSPI 외국인 순매수(다음 금융)로 전환했습니다.", "",
             f"<b>시장별 준비 상태</b> (기준일 <code>{d.get('as_of')}</code>)"]
    lines += _market_lines(d)
    warm = [KO[m] for m, v in (d.get("coverage") or {}).items() if v.get("state") == "warming"]
    tail = (f"{'·'.join(warm)}은(는) 발행사 공시가 한 세션 더 쌓이면 흐름이 계산됩니다. "
            if warm else "")
    lines += ["", tail + "신뢰도 기준을 넘는 첫 기준일에 정상 브리프가 자동으로 재개됩니다."]
    return "\n".join(lines + _link(d))


def _load_state() -> dict:
    if not os.path.exists(STATE):
        return {}
    try:
        with open(STATE, encoding="utf-8") as f:
            s = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    if "last_status" not in s and "last_sent" in s:
        # v1 상태 파일({"last_sent": 날짜}) — v1은 2026-09-15부터 보류 상태였다.
        s = {"last_status": "halted", "last_sent_as_of": None,
             "last_status_notice": s.get("last_sent")}
    return s


def _save_state(s: dict) -> None:
    os.makedirs(os.path.dirname(STATE) or ".", exist_ok=True)
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, sort_keys=True)


def is_night(now: dt.datetime) -> bool:
    h = now.astimezone(KST).hour
    return h >= NIGHT[0] or h < NIGHT[1]


def decide(d: dict, state: dict, now: dt.datetime, force: bool = False):
    """(보낼 메시지 or None, 갱신될 상태, 사유). 부작용 없음 — 테스트 대상."""
    st = (d.get("status") or {}).get("state", "ok")
    prev = state.get("last_status")
    new = dict(state)
    today = now.astimezone(KST).date().isoformat()

    if is_night(now) and not force:
        return None, state, "야간(22~07시) — 다음 실행으로 이월"

    if st == "ok":
        fresh = d.get("as_of") and d.get("as_of") != state.get("last_sent_as_of")
        if not (fresh or force):
            return None, state, "이 기준일 브리프는 이미 발송"
        full = sum(1 for v in (d.get("coverage") or {}).values() if v.get("state") == "ok")
        hour = now.astimezone(KST).hour
        if full < 4 and PARTIAL_HOLD[0] <= hour < PARTIAL_HOLD[1] and not force:
            return None, state, "결손 시장 공시 도착 대기 — 다음 실행에서 재판정"
        msg = build_message(d, recovered=prev in ("halted", "warming"))
        new.update(last_status="ok", last_sent_as_of=d.get("as_of"))
        return msg, new, "브리프"

    if st == "halted":
        last = state.get("last_status_notice")
        overdue = (not last) or (dt.date.fromisoformat(today)
                                 - dt.date.fromisoformat(last[:10])).days >= REMIND_DAYS
        if prev != "halted" or overdue or force:
            msg = build_halt_message(d, reminder=(prev == "halted" and not force))
            new.update(last_status="halted", last_status_notice=today)
            return msg, new, "보류 통지"
        return None, state, "보류 지속 — 재알림 주기 전"

    # warming
    if prev != "warming" or force:
        new.update(last_status="warming", last_status_notice=today)
        return build_warming_message(d), new, "워밍업 안내"
    return None, state, "워밍업 지속 — 안내 완료"


def send(text: str, cfg: dict) -> int | None:
    r = requests.post(
        f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
        json={"chat_id": cfg["telegram_chat_id"], "text": text,
              "parse_mode": "HTML", "disable_web_page_preview": True},
        timeout=20,
    )
    r.raise_for_status()
    try:
        return (r.json().get("result") or {}).get("message_id")
    except ValueError:
        return None


def main() -> None:
    with open("docs/data.json", encoding="utf-8") as f:
        d = json.load(f)
    force = os.environ.get("NOTIFY_FORCE", "").lower() == "true"
    now = dt.datetime.now(dt.timezone.utc)
    state = _load_state()
    msg, new_state, why = decide(d, state, now, force=force)
    if msg is None:
        print(f"[notify] 발송 생략 — {why}")
        if new_state != state:
            _save_state(new_state)
        return

    cfg = load_config()
    if not cfg["telegram_token"] or not cfg["telegram_chat_id"]:
        print("[telegram] 자격증명 없음 - 발송 생략")
        print(msg)
        return
    mid = send(msg, cfg)
    print(f"[telegram] 발송 완료 ({why}, message_id={mid})")
    _save_state(new_state)


if __name__ == "__main__":
    main()
