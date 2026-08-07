#!/usr/bin/env python3
"""엔트리포인트.  python run.py [--seed]"""
import sys

from smr.pipeline import run

if __name__ == "__main__":
    seed = "--seed" in sys.argv
    p = run(seed=seed)
    print(f"as_of={p['as_of']}  rows={p['rows_total']} (+{p['rows_added']})")
    for h in p["health"]:
        print("  ", h)
    for a in p["alerts"]:
        print(f"  ALERT {a['market']} {a['direction']} z={a['z20']:.2f} {a['triggers']}")
