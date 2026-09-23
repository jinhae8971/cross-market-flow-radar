#!/usr/bin/env python3
"""엔트리포인트.  python run.py [--seed]

발행 보류는 정상 종료(0)다 — 관측은 저장됐고, 보류 여부는 status로 전달된다.
비정상 종료는 코드·환경 장애일 때뿐이며, 그때만 ops_alert.py가 동작한다.
"""
import sys

import ops_alert
from smr.pipeline import run

if __name__ == "__main__":
    p = run(seed="--seed" in sys.argv)
    st = p["status"]
    print(f"as_of={p['as_of']}  status={st['state']}  rows={p['rows_total']} "
          f"(+{p['rows_added']})  confidence={p['confidence']} {p['confidence_breakdown']}")
    if st.get("reason"):
        print(f"  reason: {st['reason']}")
    for m, v in p["coverage"].items():
        print(f"  {m}: {v}")
    for h in p["health"]:
        print("  ", h)
    for a in p["alerts"]:
        print(f"  ALERT {a['market']} {a['direction']} z={a['z20']:.2f} {a['triggers']}")
    ops_alert.reset_on_success()
