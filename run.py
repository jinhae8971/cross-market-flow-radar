#!/usr/bin/env python3
"""엔트리포인트.  python run.py [--seed]"""
import sys

from smr.pipeline import LowConfidence, run

if __name__ == "__main__":
    seed = "--seed" in sys.argv
    try:
        p = run(seed=seed)
    except LowConfidence as exc:
        # 비정상 종료로 끝내야 워크플로우가 실패 통지를 보내고,
        # 이어지는 텔레그램 브리프 단계가 건너뛰어진다.
        print(f"HALT: {exc}", file=sys.stderr)
        sys.exit(2)
    print(f"as_of={p['as_of']}  rows={p['rows_total']} (+{p['rows_added']})  "
          f"confidence={p['confidence']} {p['confidence_breakdown']}")
    for h in p["health"]:
        print("  ", h)
    for a in p["alerts"]:
        print(f"  ALERT {a['market']} {a['direction']} z={a['z20']:.2f} {a['triggers']}")
