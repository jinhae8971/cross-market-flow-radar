import datetime as dt
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smr import rotation, signals
from smr.calendar_mask import masked
from smr.schema import FlowRecord, FlowStore, to_frame


def rec(day, market, flow, actor="foreign", conf=0.8, source="t"):
    return FlowRecord(ts=day, market=market, actor=actor, instrument="X",
                      net_flow_usd=flow, lag_days=0, confidence=conf, source=source)


class TestSchema(unittest.TestCase):
    def test_rejects_unknown_market(self):
        with self.assertRaises(ValueError):
            rec(dt.date(2026, 1, 2), "CN", 1.0)

    def test_rejects_bad_confidence(self):
        with self.assertRaises(ValueError):
            rec(dt.date(2026, 1, 2), "KR", 1.0, conf=1.5)

    def test_upsert_dedups_and_prefers_latest(self):
        with tempfile.TemporaryDirectory() as d:
            store = FlowStore(os.path.join(d, "f.parquet"))
            day = dt.date(2026, 1, 2)
            store.upsert(to_frame([rec(day, "KR", 100.0)]))
            store.upsert(to_frame([rec(day, "KR", 250.0)]))  # 같은 키 재수집
            df = store.load()
            self.assertEqual(len(df), 1)
            self.assertEqual(df["net_flow_usd"].iloc[0], 250.0)


class TestSignals(unittest.TestCase):
    def _series(self, values, market="KR"):
        base = dt.date(2026, 1, 5)
        return to_frame([rec(base + dt.timedelta(days=i), market, v)
                         for i, v in enumerate(values)])

    def test_confidence_weighting_applied(self):
        df = to_frame([rec(dt.date(2026, 1, 5), "KR", 100.0, conf=0.5)])
        agg = signals.aggregate(df)
        self.assertAlmostEqual(agg["net_flow_usd"].iloc[0], 50.0)

    def test_zscore_flags_outlier(self):
        vals = [1.0] * 20 + [50.0]
        df = self._series(vals)
        sig = signals.build(df)
        self.assertGreater(sig["z20"].iloc[-1], 3.0)

    def test_persistence_uses_deviation_not_raw_sign(self):
        # 상시 순유입이어도 평소 수준이면 run이 쌓이면 안 된다
        s = pd.Series([10.0] * 10)
        self.assertEqual(int(signals.persistence(s).iloc[-1]), 0)
        # 평소보다 큰 유입이 이어지면 run이 쌓인다
        s2 = pd.Series([10.0] * 10 + [20.0, 21.0, 22.0])
        self.assertGreaterEqual(int(signals.persistence(s2).iloc[-1]), 3)

    def test_alert_requires_two_triggers(self):
        # 단발 급등: 강도만 충족 → 발화하면 안 됨
        df = self._series([1.0] * 20 + [40.0])
        self.assertEqual(signals.alerts(signals.build(df)), [])

    def test_alert_fires_on_sustained_move(self):
        df = self._series([1.0] * 20 + [30.0, 32.0, 35.0, 38.0])
        out = signals.alerts(signals.build(df))
        self.assertTrue(out)
        self.assertEqual(out[0]["direction"], "유입")


class TestRotation(unittest.TestCase):
    def test_share_sums_to_100(self):
        base = dt.date(2026, 1, 5)
        recs = []
        for i in range(30):
            d = base + dt.timedelta(days=i)
            recs += [rec(d, "KR", 10.0 + i), rec(d, "US", 20.0),
                     rec(d, "JP", 5.0), rec(d, "EU", -3.0)]
        sig = signals.build(to_frame(recs))
        sh = rotation.share(sig)
        self.assertAlmostEqual(float(sh.iloc[-1].sum()), 100.0, places=0)

    def test_matrix_identifies_direction(self):
        base = dt.date(2026, 1, 5)
        recs = []
        for i in range(30):
            d = base + dt.timedelta(days=i)
            recs += [rec(d, "KR", 5.0), rec(d, "US", 5.0),
                     rec(d, "JP", 1.0 + i * 3), rec(d, "EU", 5.0)]
        m = rotation.matrix(signals.build(to_frame(recs)))
        self.assertTrue(m["ready"])
        self.assertEqual(m["to"], "JP")


class TestNotify(unittest.TestCase):
    def test_url_falls_back_to_pages_address(self):
        import notify
        os.environ.pop("DASHBOARD_URL", None)
        os.environ["GITHUB_REPOSITORY"] = "owner/repo-name"
        self.assertEqual(notify.dashboard_url({}),
                         "https://owner.github.io/repo-name/")

    def test_explicit_url_wins(self):
        import notify
        self.assertEqual(notify.dashboard_url({"dashboard_url": "https://x.io/d/"}),
                         "https://x.io/d")

    def test_message_contains_link_and_summary(self):
        import notify
        d = {
            "as_of": "2026-08-07", "alerts": [], "suppressed": [],
            "rotation": {"ready": True, "rows": [
                {"market": "JP", "share": 60.0, "delta": 5.0}], "from": "US", "to": "JP"},
            "detail": {"KR": {"latest": 0.2, "cum": {"d20": -0.7},
                              "signal": {"z20": 0.16}}},
            "health": [{"collector": "krx", "ok": False}],
            "dashboard_url": "https://x.io/d",
        }
        msg = notify.build_message(d)
        self.assertIn("https://x.io/d", msg)
        self.assertIn("한국", msg)
        self.assertIn("수집 실패", msg)


class TestMask(unittest.TestCase):
    def test_quad_witching_masked(self):
        flag, why = masked(dt.date(2026, 6, 19))  # 6월 셋째 금요일
        self.assertTrue(flag)
        self.assertIn("만기", why)

    def test_ordinary_day_not_masked(self):
        self.assertFalse(masked(dt.date(2026, 4, 8))[0])


if __name__ == "__main__":
    unittest.main()
