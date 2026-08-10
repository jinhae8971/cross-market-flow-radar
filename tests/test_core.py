import datetime as dt
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smr import repair, rotation, signals
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


class TestRegressions(unittest.TestCase):
    """2026-08-11 진단에서 나온 결함 4건의 재발 방지."""

    def _flat(self, values_by_market, days=30):
        base = dt.date(2026, 1, 5)
        recs = []
        for i in range(days):
            d = base + dt.timedelta(days=i)
            for m, v in values_by_market.items():
                recs.append(rec(d, m, v(i) if callable(v) else v))
        return signals.build(to_frame(recs))

    def test_share_survives_all_markets_outflowing(self):
        # 4개 시장 전부 순유출 — 이전 구현은 분모 0으로 전 시장 0%로 붕괴했다
        sig = self._flat({"KR": -5.0, "JP": -9.0, "EU": -3.0, "US": -20.0})
        sh = rotation.share(sig)
        last = sh.iloc[-1]
        self.assertAlmostEqual(float(last.sum()), 100.0, places=0)
        self.assertTrue((last > 0).all(), "전 시장 유출일에도 배분율은 0이 아니어야 한다")

    def test_no_direction_invented_when_markets_are_tied(self):
        # 완전 동률 — 이전 구현은 정렬 안정성 때문에 알파벳 순을 방향으로 내보냈다
        sig = self._flat({"KR": 4.0, "JP": 4.0, "EU": 4.0, "US": 4.0})
        m = rotation.matrix(sig)
        self.assertIsNone(m["from"])
        self.assertIsNone(m["to"])

    def test_zero_does_not_overwrite_real_value(self):
        # 새 세션 없이 돈 실행이 같은 날 실측값을 0으로 지우면 안 된다
        with tempfile.TemporaryDirectory() as d:
            store = FlowStore(os.path.join(d, "f.parquet"))
            day = dt.date(2026, 8, 11)
            store.upsert(to_frame([rec(day, "JP", -5.9e8)]))
            store.upsert(to_frame([rec(day, "JP", 0.0)]))
            df = store.load()
            self.assertEqual(len(df), 1)
            self.assertAlmostEqual(df["net_flow_usd"].iloc[0], -5.9e8)

    def test_repair_drops_all_zero_sessions_only(self):
        good = dt.date(2026, 8, 7)
        dead = dt.date(2026, 8, 10)
        df = to_frame([
            rec(good, "KR", 100.0, source="etf_aum_delta"),
            rec(good, "US", 0.0, source="etf_aum_delta"),   # 실제 0 — 보존돼야 함
            rec(dead, "KR", 0.0, source="etf_aum_delta"),
            rec(dead, "US", 0.0, source="etf_aum_delta"),
        ])
        out, purged = repair.drop_dead_sessions(df)
        self.assertEqual(purged, ["2026-08-10"])
        self.assertEqual(len(out), 2)
        self.assertEqual(set(out["ts"].dt.date), {good})

    def test_repair_is_idempotent(self):
        df = to_frame([rec(dt.date(2026, 8, 10), "KR", 0.0, source="etf_aum_delta")])
        once, _ = repair.drop_dead_sessions(df)
        twice, purged = repair.drop_dead_sessions(once)
        self.assertEqual(purged, [])
        self.assertEqual(len(once), len(twice))

    def test_collector_labels_with_session_date_not_today(self):
        # 종가 인덱스의 마지막 세션일로 라벨링되어야 한다(KST 오늘 날짜가 아니라)
        from smr.collectors import etf_flow
        idx = pd.to_datetime(["2026-08-07", "2026-08-10"])
        closes = pd.DataFrame({s: [100.0, 101.0]
                               for t in etf_flow.UNIVERSE.values() for s in t}, index=idx)
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "aum.json")
            prev = {"session": "2026-08-07",
                    "latest": {s: {"aum": 1.0e10}
                               for t in etf_flow.UNIVERSE.values() for s in t}}
            with open(cache, "w", encoding="utf-8") as f:
                json.dump(prev, f)

            def fake_snapshot(tickers):
                return {s: {"aum": 1.02e10} for s in tickers}

            orig = etf_flow._snapshot
            etf_flow._snapshot = fake_snapshot
            try:
                out = etf_flow.collect(cache_path=cache, closes=closes)
            finally:
                etf_flow._snapshot = orig
        self.assertTrue(out)
        self.assertEqual({r.ts for r in out}, {dt.date(2026, 8, 10)})

    def test_collector_reseeds_on_legacy_cache_without_session(self):
        # 세션 라벨 없는 구 캐시로 delta를 계산하면 부호가 뒤집힌 유령 흐름이 생긴다
        from smr.collectors import etf_flow
        idx = pd.to_datetime(["2026-08-07", "2026-08-10"])
        closes = pd.DataFrame({s: [100.0, 101.0]
                               for t in etf_flow.UNIVERSE.values() for s in t}, index=idx)
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "aum.json")
            legacy = {"date": "2026-08-10",  # 구 포맷 — session 키 없음
                      "latest": {s: {"aum": 1.0e10, "px": 100.0}
                                 for t in etf_flow.UNIVERSE.values() for s in t}}
            with open(cache, "w", encoding="utf-8") as f:
                json.dump(legacy, f)

            orig = etf_flow._snapshot
            etf_flow._snapshot = lambda tickers: {s: {"aum": 1.02e10} for s in tickers}
            try:
                out = etf_flow.collect(cache_path=cache, closes=closes)
                with open(cache, encoding="utf-8") as f:
                    after = json.load(f)
            finally:
                etf_flow._snapshot = orig
        self.assertEqual(out, [], "레거시 캐시 회차는 레코드를 만들지 않아야 한다")
        self.assertEqual(after["session"], "2026-08-10", "기준점은 새 포맷으로 재설정")

    def test_collector_skips_when_no_new_session(self):
        from smr.collectors import etf_flow
        idx = pd.to_datetime(["2026-08-07", "2026-08-10"])
        closes = pd.DataFrame({s: [100.0, 101.0]
                               for t in etf_flow.UNIVERSE.values() for s in t}, index=idx)
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "aum.json")
            with open(cache, "w", encoding="utf-8") as f:
                json.dump({"session": "2026-08-10", "latest": {}}, f)
            self.assertEqual(etf_flow.collect(cache_path=cache, closes=closes), [])


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

    def test_force_flag_overrides_daily_dedup(self):
        # 중복 방지 가드가 강제 발송까지 막으면 안 된다 (스텝은 success인데 미발송)
        import notify
        payload = {"as_of": "2026-08-11", "alerts": [], "suppressed": [],
                   "rotation": {"ready": False, "rows": []},
                   "detail": {}, "health": [], "dashboard_url": "https://x.io/d"}
        today = dt.date.today().isoformat()
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "docs"))
            os.makedirs(os.path.join(d, "data"))
            with open(os.path.join(d, "docs", "data.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f)
            with open(os.path.join(d, "data", "notify_state.json"), "w", encoding="utf-8") as f:
                json.dump({"last_sent": today}, f)

            cwd = os.getcwd()
            sent = []
            orig_post = notify.requests.post
            notify.requests.post = lambda *a, **k: sent.append(k) or type(
                "R", (), {"raise_for_status": lambda self: None})()
            os.environ["TELEGRAM_TOKEN"] = "t"
            os.environ["TELEGRAM_CHAT_ID"] = "c"
            try:
                os.chdir(d)
                os.environ.pop("NOTIFY_FORCE", None)
                notify.main()
                self.assertEqual(len(sent), 0, "같은 날 재실행은 기본적으로 생략")
                os.environ["NOTIFY_FORCE"] = "true"
                notify.main()
                self.assertEqual(len(sent), 1, "force=true면 발송돼야 한다")
            finally:
                os.chdir(cwd)
                notify.requests.post = orig_post
                os.environ.pop("NOTIFY_FORCE", None)
                for k in ("TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID"):
                    os.environ.pop(k, None)

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
