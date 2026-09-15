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


class TestAumQuality(unittest.TestCase):
    def check_collect(self, changed=True, gap=False, invalid=False):
        from unittest.mock import patch
        from smr.collectors import etf_flow
        syms = [s for group in etf_flow.UNIVERSE.values() for s in group]
        closes = pd.DataFrame({s: [100., 101.] for s in syms},
                              index=pd.to_datetime(['2026-09-04', '2026-09-08']))
        snap = {s: {'aum': 1020. if changed else 1000.} for s in syms}
        if invalid:
            closes.iloc[-1, 0] = float('nan')
        before = {'session': '2026-09-03' if gap else '2026-09-04',
                  'latest': {s: {'aum': 1000.} for s in syms}}
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'aum.json')
            with open(path, 'w') as f:
                json.dump(before, f)
            with patch.object(etf_flow, '_snapshot', return_value=snap):
                with self.assertRaises(ValueError):
                    etf_flow.collect(cache_path=path, closes=closes)
            with open(path) as f:
                after = json.load(f)
            self.assertEqual(after['session'], '2026-09-08' if gap else before['session'])
            if not gap:
                # 불변식은 '파일이 한 글자도 안 바뀐다'가 아니라
                # '기준점(session/latest)이 보존된다'이다. 동결 카운터 같은
                # 진단용 메타데이터는 기록돼야 다음 회차가 판단할 수 있다.
                self.assertEqual(after['latest'], before['latest'])

    def test_unchanged_aum_does_not_create_inverse_price_flow(self):
        self.check_collect(changed=False)

    def test_session_gap_rebaselines_without_daily_flow(self):
        self.check_collect(gap=True)

    def test_nan_price_preserves_cache(self):
        self.check_collect(invalid=True)

    def test_quality_warning_suppresses_rotation_in_message(self):
        import notify
        d = {'as_of': '2026-09-08', 'alerts': [], 'detail': {},
             'quality_warnings': ['AUM 갱신 미확인'],
             'rotation': {'ready': True, 'from': 'EU', 'to': 'US', 'rows': []}}
        msg = notify.build_message(d)
        self.assertIn('판단 보류', msg)
        self.assertNotIn('유럽 → 미국', msg)


class TestSourceFreeze(unittest.TestCase):
    """AUM 소스 동결 감지 — 2026-09-08 장애의 회귀 테스트."""

    def _run(self, sessions, cached_session):
        from unittest.mock import patch
        from smr.collectors import etf_flow
        syms = [s for group in etf_flow.UNIVERSE.values() for s in group]
        closes = pd.DataFrame({s: [100.0 + i for i in range(len(sessions))]
                               for s in syms},
                              index=pd.to_datetime(sessions))
        snap = {s: {'aum': 1000.0} for s in syms}          # 전 종목 값 불변
        cache = {'session': cached_session,
                 'latest': {s: {'aum': 1000.0} for s in syms}}
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'aum.json')
            with open(path, 'w') as f:
                json.dump(cache, f)
            with patch.object(etf_flow, '_snapshot', return_value=snap):
                with self.assertRaises(Exception) as ctx:
                    etf_flow.collect(cache_path=path, closes=closes)
            with open(path) as f:
                after = json.load(f)
            return ctx.exception, after

    def test_one_session_behind_is_transient_not_frozen(self):
        from smr.collectors.etf_flow import SourceFrozen
        exc, after = self._run(['2026-09-04', '2026-09-08'], '2026-09-04')
        self.assertNotIsInstance(exc, SourceFrozen)
        self.assertEqual(after['unchanged_streak'], 1)

    def test_two_sessions_behind_escalates_to_frozen(self):
        from smr.collectors.etf_flow import SourceFrozen
        exc, after = self._run(
            ['2026-09-04', '2026-09-08', '2026-09-09'], '2026-09-04')
        self.assertIsInstance(exc, SourceFrozen)
        self.assertEqual(exc.last_good, '2026-09-04')
        self.assertEqual(after['session'], '2026-09-04')   # 기준점 보존

    def test_proxy_collect_only_fills_after_last_real_session(self):
        from unittest.mock import patch
        from smr.collectors import etf_flow
        from smr.schema import FlowRecord
        made = [FlowRecord(ts=dt.date(2026, 9, d), market='US', actor='foreign',
                           instrument='SPY', net_flow_usd=1.0, lag_days=0,
                           confidence=0.5, source='etf_moneyflow_proxy')
                for d in (7, 8, 9, 10)]
        with patch.object(etf_flow, 'backfill', return_value=made):
            out = etf_flow.proxy_collect(since=dt.date(2026, 9, 8))
        self.assertEqual([r.ts.day for r in out], [9, 10])


class TestFreshnessGate(unittest.TestCase):
    """오래된 수치가 통상 브리프 형태로 발송되지 않아야 한다."""

    def _payload(self, stale):
        return {'as_of': '2026-09-08', 'latest_session': '2026-09-15',
                'stale_sessions': stale, 'alerts': [], 'quality_warnings': ['x'],
                'detail': {'US': {'latest': 3.27, 'cum': {'d20': 37.0},
                                  'signal': {'z20': 0.12}}},
                'health': [{'collector': 'etf_aum', 'ok': False,
                            'error': 'ETF AUM 소스 동결'}],
                'rotation': {'ready': False, 'rows': []}}

    def test_stale_payload_withholds_numbers(self):
        import notify
        msg = notify.build_message(self._payload(5))
        self.assertIn('발송 보류', msg)
        self.assertIn('동결', msg)
        self.assertNotIn('3.27', msg)
        self.assertNotIn('시장별 순유입', msg)

    def test_fresh_payload_keeps_normal_brief(self):
        import notify
        msg = notify.build_message(self._payload(1))
        self.assertIn('시장별 순유입', msg)
        self.assertNotIn('발송 보류', msg)


class TestSourceDedup(unittest.TestCase):
    def test_proxy_does_not_double_count_with_real(self):
        from smr import signals
        day = dt.date(2026, 9, 9)
        df = to_frame([
            rec(day, 'US', 100.0, conf=0.75, source='etf_aum_delta'),
            rec(day, 'US', 900.0, conf=0.50, source='etf_moneyflow_proxy'),
        ])
        agg = signals.aggregate(df)
        self.assertEqual(len(agg), 1)
        self.assertAlmostEqual(agg['net_flow_usd'].iloc[0], 75.0)


class TestKoreaGapFill(unittest.TestCase):
    def test_candidate_days_skips_weekends(self):
        from smr.collectors import korea
        # 2026-09-15는 화요일 → 직전 5영업일은 주말을 건너뛴다
        days = korea._candidate_days(5, today=dt.date(2026, 9, 15))
        self.assertTrue(all(d.weekday() < 5 for d in days))
        self.assertEqual(days[-1], dt.date(2026, 9, 14))
        self.assertNotIn(dt.date(2026, 9, 13), days)   # 일요일

    def test_collect_only_requests_missing_days(self):
        from unittest.mock import patch
        from smr.collectors import korea
        asked = []

        def fake(day, key):
            asked.append(day)
            return []

        known = {dt.date(2026, 9, 10), dt.date(2026, 9, 11)}
        with patch.dict(os.environ, {"KRX_API_KEY": "k"}):
            with patch.object(korea, "_fetch_day", fake):
                korea.collect(known=known, lookback=5)
        self.assertTrue(set(asked).isdisjoint(known))
        self.assertTrue(asked)

    def test_missing_key_raises_unconfigured_not_silent_empty(self):
        from unittest.mock import patch
        from smr.collectors import korea
        with patch.dict(os.environ, {"KRX_API_KEY": ""}):
            with patch.object(korea, "_fetch_day", lambda d, k: []):
                with self.assertRaises(korea.KrxUnconfigured):
                    korea.collect()

    def test_holiday_empty_response_is_not_an_error(self):
        from unittest.mock import patch
        from smr.collectors import korea
        with patch.dict(os.environ, {"KRX_API_KEY": "k"}):
            with patch.object(korea, "_fetch_day", lambda d, k: []):
                self.assertEqual(korea.collect(lookback=3), [])


class TestKrxStatusInBrief(unittest.TestCase):
    def _msg(self, status):
        import notify
        return notify.build_message({
            "as_of": "2026-09-11", "stale_sessions": 0, "alerts": [],
            "detail": {}, "quality_warnings": [], "rotation": {"ready": False},
            "health": [{"collector": "krx", "ok": True, "status": status}]})

    def test_unconfigured_tells_the_operator_what_to_do(self):
        msg = self._msg("unconfigured")
        self.assertIn("KRX_API_KEY", msg)
        self.assertIn("openapi.krx.co.kr", msg)

    def test_unavailable_does_not_claim_a_key_problem(self):
        msg = self._msg("unavailable")
        self.assertNotIn("KRX_API_KEY", msg)


NAVER_HTML = """<table class="type_1"><thead></thead><tbody>
<tr><td>26.09.11</td><td>18,675</td><td>-22,984</td><td>-12,184</td>
<td>-10,345</td><td>-17</td><td>-4,703</td><td>7</td><td>216</td>
<td>2,659</td><td>16,493</td></tr>
<tr><td>26.09.10</td><td>3,802</td><td>-26,217</td><td>5,744</td>
<td>11,628</td><td>155</td><td>-4,803</td><td>24</td><td>184</td>
<td>-1,444</td><td>16,671</td></tr>
</tbody></table>"""


class TestNaverKr(unittest.TestCase):
    def test_parses_rows_with_confirmed_column_order(self):
        from smr.collectors import naver_kr
        rows = naver_kr._rows(NAVER_HTML)
        self.assertEqual(len(rows), 2)
        day, vals = rows[0]
        self.assertEqual(day, dt.date(2026, 9, 11))
        self.assertEqual(vals["외국인"], -22984.0)
        self.assertEqual(vals["연기금등"], 2659.0)

    def test_verify_accepts_real_layout(self):
        from smr.collectors import naver_kr
        naver_kr._verify(naver_kr._rows(NAVER_HTML)[0][1])   # 예외 없어야 함

    def test_shifted_columns_are_rejected_not_silently_used(self):
        from smr.collectors import naver_kr
        shifted = NAVER_HTML.replace("<td>18,675</td>", "<td>99,999</td>")
        with self.assertRaises(naver_kr.LayoutChanged):
            naver_kr._verify(naver_kr._rows(shifted)[0][1])


class TestConfidenceGate(unittest.TestCase):
    def _df(self, pairs):
        return to_frame([rec(dt.date(2026, 9, 11), m, 1.0, conf=c,
                             source="s") for m, c in pairs])

    def test_partial_coverage_scales_the_score_down(self):
        from smr import pipeline
        score, br = pipeline.confidence_score(
            self._df([("US", 0.95), ("EU", 0.95)]), dt.date(2026, 9, 11))
        self.assertAlmostEqual(score, 0.95 * 0.5)
        self.assertIn("한국", br["결손"])

    def test_full_coverage_high_quality_passes(self):
        from smr import pipeline
        score, _ = pipeline.confidence_score(
            self._df([("KR", 0.95), ("JP", 0.95), ("EU", 0.95), ("US", 0.95)]),
            dt.date(2026, 9, 11))
        self.assertGreaterEqual(score, pipeline.MIN_CONFIDENCE)

    def test_proxy_only_run_falls_below_threshold(self):
        from smr import pipeline
        score, _ = pipeline.confidence_score(
            self._df([("KR", 0.5), ("JP", 0.5), ("EU", 0.5), ("US", 0.5)]),
            dt.date(2026, 9, 11))
        self.assertLess(score, pipeline.MIN_CONFIDENCE)

    def test_market_quality_is_not_weighted_by_instrument_count(self):
        from smr import pipeline
        # 유럽 ETF 6개 vs 한국 1개 — 행 단위 평균이면 유럽이 6배 무거워진다
        pairs = [("EU", 0.5)] * 6 + [("KR", 0.9), ("JP", 0.9), ("US", 0.9)]
        score, _ = pipeline.confidence_score(self._df(pairs),
                                             dt.date(2026, 9, 11))
        self.assertAlmostEqual(score, (0.5 + 0.9 * 3) / 4)


class TestPrimaryActors(unittest.TestCase):
    def test_domestic_actors_do_not_cancel_the_market_to_zero(self):
        from smr import signals
        day = dt.date(2026, 9, 11)
        df = to_frame([
            rec(day, "KR", -2298.0, actor="foreign", conf=0.9, source="naver_kr"),
            rec(day, "KR", 1867.0, actor="retail", conf=0.9, source="naver_kr"),
            rec(day, "KR", -1218.0, actor="institution", conf=0.9, source="naver_kr"),
        ])
        agg = signals.aggregate(df)
        self.assertEqual(len(agg), 1)
        self.assertAlmostEqual(agg["net_flow_usd"].iloc[0], -2298.0 * 0.9)

    def test_higher_confidence_source_supersedes_proxy_for_same_actor(self):
        from smr import signals
        day = dt.date(2026, 9, 11)
        df = to_frame([
            rec(day, "KR", -2298.0, actor="foreign", conf=0.9, source="naver_kr"),
            rec(day, "KR", 500.0, actor="foreign", conf=0.5,
                source="etf_moneyflow_proxy"),
        ])
        agg = signals.aggregate(df)
        self.assertAlmostEqual(agg["net_flow_usd"].iloc[0], -2298.0 * 0.9)
