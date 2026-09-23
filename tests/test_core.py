import datetime as dt
import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smr import pipeline, repair, rotation, signals  # noqa: E402
from smr.calendar_mask import masked  # noqa: E402
from smr.collectors import issuer, kr_investor  # noqa: E402
from smr.schema import FlowRecord, FlowStore, to_frame  # noqa: E402

KST = dt.timezone(dt.timedelta(hours=9))
CORE = {"KR": "daum_kr", "JP": "etf_flow", "EU": "etf_flow", "US": "etf_flow"}


def rec(day, market, flow, actor="foreign", conf=0.9, source=None, instrument="X"):
    return FlowRecord(ts=day, market=market, actor=actor, instrument=instrument,
                      net_flow_usd=flow, lag_days=0, confidence=conf,
                      source=source or CORE[market])


def kst(y, m, d, h, mi=0):
    return dt.datetime(y, m, d, h, mi, tzinfo=KST)


# ── 공통 계층 ───────────────────────────────────────────────────────────────
class TestSchema(unittest.TestCase):
    def test_rejects_unknown_market(self):
        with self.assertRaises(ValueError):
            rec(dt.date(2026, 1, 2), "CN", 1.0, source="x")

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

    def test_identical_upsert_does_not_touch_the_file(self):
        # 같은 데이터를 다시 넣을 때 파일을 다시 쓰면 매 실행 빈 커밋이 생긴다
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.parquet")
            store = FlowStore(path)
            frame = to_frame([rec(dt.date(2026, 1, 2), "KR", 1.0)])
            store.upsert(frame)
            before = os.stat(path).st_mtime_ns
            self.assertEqual(store.upsert(frame), 0)
            self.assertEqual(os.stat(path).st_mtime_ns, before)


class TestSignals(unittest.TestCase):
    def _series(self, values, market="KR"):
        base = dt.date(2026, 1, 5)
        return to_frame([rec(base + dt.timedelta(days=i), market, v)
                         for i, v in enumerate(values)])

    def test_aggregate_is_unweighted_dollars(self):
        # v1은 신뢰도를 곱해 같은 달러가 소스에 따라 다른 크기로 기록됐다
        df = to_frame([rec(dt.date(2026, 1, 5), "US", 100.0, conf=0.5)])
        self.assertAlmostEqual(signals.aggregate(df)["net_flow_usd"].iloc[0], 100.0)

    def test_zscore_flags_outlier(self):
        sig = signals.build(self._series([1.0] * 20 + [50.0]))
        self.assertGreater(sig["z20"].iloc[-1], 3.0)

    def test_persistence_uses_deviation_not_raw_sign(self):
        s = pd.Series([10.0] * 10)
        self.assertEqual(int(signals.persistence(s).iloc[-1]), 0)
        s2 = pd.Series([10.0] * 10 + [20.0, 21.0, 22.0])
        self.assertGreaterEqual(int(signals.persistence(s2).iloc[-1]), 3)

    def test_alert_requires_two_triggers(self):
        self.assertEqual(signals.alerts(signals.build(self._series([1.0] * 20 + [40.0]))), [])

    def test_alert_fires_on_sustained_move(self):
        out = signals.alerts(signals.build(self._series([1.0] * 20 + [30.0, 32.0, 35.0, 38.0])))
        self.assertTrue(out)
        self.assertEqual(out[0]["direction"], "유입")

    def test_alert_skips_market_not_observed_on_as_of(self):
        # 원천이 멈춘 시장의 마지막 행으로 발화하면 묵은 신호가 오늘 신호처럼 나간다
        sig = signals.build(self._series([1.0] * 20 + [30.0, 32.0, 35.0, 38.0]))
        last = sig["ts"].max().date()
        self.assertTrue(signals.alerts(sig, as_of=last))
        self.assertEqual(signals.alerts(sig, as_of=last + dt.timedelta(days=1)), [])


class TestCoreSeries(unittest.TestCase):
    day = dt.date(2026, 9, 22)

    def test_korea_uses_investor_flow_not_ewy(self):
        df = to_frame([rec(self.day, "KR", 5e8),
                       rec(self.day, "KR", 3e7, source="etf_flow", instrument="EWY")])
        agg = signals.aggregate(df)
        self.assertAlmostEqual(agg["net_flow_usd"].iloc[0], 5e8)

    def test_krx_supersedes_daum_for_same_day(self):
        df = to_frame([rec(self.day, "KR", 1.0, instrument="KOSPI"),
                       rec(self.day, "KR", 2.0, conf=0.95, source="krx", instrument="KOSPI")])
        self.assertAlmostEqual(signals.aggregate(df)["net_flow_usd"].iloc[0], 2.0)

    def test_domestic_actors_and_cot_are_not_summed(self):
        # 국내 주체는 상계되어 합이 0, COT는 주간 명목 — 더하면 척도가 튄다
        df = to_frame([rec(self.day, "KR", 100.0),
                       rec(self.day, "KR", -60.0, actor="institution"),
                       rec(self.day, "KR", -40.0, actor="retail"),
                       rec(self.day, "US", 5.0, instrument="SPY"),
                       rec(self.day, "US", 9e9, actor="spec", source="cftc_cot")])
        agg = signals.aggregate(df).set_index("market")["net_flow_usd"]
        self.assertAlmostEqual(agg["KR"], 100.0)
        self.assertAlmostEqual(agg["US"], 5.0)

    def test_low_coverage_day_is_dropped_not_underreported(self):
        uni = {"EU": ("A", "B")}
        w = {"A": 80.0, "B": 20.0}
        df = to_frame([rec(self.day, "EU", 1.0, instrument="A"),
                       rec(self.day, "EU", 1.0, instrument="B"),
                       rec(self.day + dt.timedelta(days=1), "EU", 1.0, instrument="B")])
        agg = signals.aggregate(df, weights=w, universe=uni)
        self.assertEqual(list(agg["ts"].dt.date), [self.day])
        self.assertAlmostEqual(agg["coverage"].iloc[0], 1.0)

    def test_legacy_sources_are_ignored(self):
        df = to_frame([rec(self.day, "US", 9e9, source="etf_aum_delta"),
                       rec(self.day, "US", 9e9, source="etf_moneyflow_proxy")])
        self.assertTrue(signals.aggregate(df).empty)


class TestRotation(unittest.TestCase):
    def test_share_sums_to_100(self):
        base = dt.date(2026, 1, 5)
        recs = []
        for i in range(30):
            d = base + dt.timedelta(days=i)
            recs += [rec(d, "KR", 10.0 + i), rec(d, "US", 20.0),
                     rec(d, "JP", 5.0), rec(d, "EU", -3.0)]
        sh = rotation.share(signals.build(to_frame(recs)))
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
    """2026-08-11 진단에서 나온 결함의 재발 방지."""

    def _flat(self, values_by_market, days=30):
        base = dt.date(2026, 1, 5)
        recs = [rec(base + dt.timedelta(days=i), m, v)
                for i in range(days) for m, v in values_by_market.items()]
        return signals.build(to_frame(recs))

    def test_share_survives_all_markets_outflowing(self):
        sh = rotation.share(self._flat({"KR": -5.0, "JP": -9.0, "EU": -3.0, "US": -20.0}))
        last = sh.iloc[-1]
        self.assertAlmostEqual(float(last.sum()), 100.0, places=0)
        self.assertTrue((last > 0).all())

    def test_no_direction_invented_when_markets_are_tied(self):
        m = rotation.matrix(self._flat({"KR": 4.0, "JP": 4.0, "EU": 4.0, "US": 4.0}))
        self.assertIsNone(m["from"])
        self.assertIsNone(m["to"])

    def test_zero_does_not_overwrite_real_value(self):
        with tempfile.TemporaryDirectory() as d:
            store = FlowStore(os.path.join(d, "f.parquet"))
            day = dt.date(2026, 8, 11)
            store.upsert(to_frame([rec(day, "JP", -5.9e8)]))
            store.upsert(to_frame([rec(day, "JP", 0.0)]))
            df = store.load()
            self.assertEqual(len(df), 1)
            self.assertAlmostEqual(df["net_flow_usd"].iloc[0], -5.9e8)

    def test_repair_drops_all_zero_sessions_only(self):
        good, dead = dt.date(2026, 8, 7), dt.date(2026, 8, 10)
        df = to_frame([
            rec(good, "KR", 100.0, source="etf_aum_delta"),
            rec(good, "US", 0.0, source="etf_aum_delta"),
            rec(dead, "KR", 0.0, source="etf_aum_delta"),
            rec(dead, "US", 0.0, source="etf_aum_delta"),
        ])
        out, purged = repair.drop_dead_sessions(df)
        self.assertEqual(purged, ["2026-08-10"])
        self.assertEqual(set(out["ts"].dt.date), {good})

    def test_repair_is_idempotent(self):
        df = to_frame([rec(dt.date(2026, 8, 10), "KR", 0.0, source="etf_aum_delta")])
        once, _ = repair.drop_dead_sessions(df)
        twice, purged = repair.drop_dead_sessions(once)
        self.assertEqual(purged, [])
        self.assertEqual(len(once), len(twice))


# ── 발행사 원천 ─────────────────────────────────────────────────────────────
def _ishares_fund(sym, tna, nav, day=20260922, nav_day=None):
    return {"localExchangeTicker": sym, "totalNetAssets": {"r": tna},
            "totalNetAssetsFundAsOf": {"r": day}, "navAmount": {"r": nav},
            "navAmountAsOf": {"r": nav_day or day}}


def _ssga_xlsx(rows):
    """SSGA navhist와 같은 모양(상단 메타 3행 + 헤더 + 최신일 우선)의 xlsx."""
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.append(["Fund Name:", "Test ETF"])
    ws.append(["Ticker Symbol:", "TST"])
    ws.append([])
    ws.append(["Date", "NAV", "Shares Outstanding", "Total Net Assets"])
    for day, nav, sh in rows:
        ws.append([day, nav, sh, nav * sh])
    ws.append([])
    ws.append(["Before investing ..."])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


class TestIssuerParsers(unittest.TestCase):
    def test_ishares_implied_shares_from_dated_tna_and_nav(self):
        payload = {"1": _ishares_fund("EWY", 27_420_786_707.34, 191.218875),
                   "2": _ishares_fund("ZZZ", 1.0, 1.0)}
        out = issuer.parse_ishares(payload, ["EWY"])
        day, shares, nav = out["EWY"]
        self.assertEqual(day, dt.date(2026, 9, 22))
        self.assertAlmostEqual(shares, 143_400_000, delta=1)   # 상품 페이지 공시값

    def test_ishares_drops_fund_when_tna_and_nav_dates_differ(self):
        # 서로 다른 날의 순자산과 NAV를 나누면 발행좌수가 가격만큼 틀어진다
        payload = {"1": _ishares_fund("EWJ", 1e9, 100.0, 20260922, nav_day=20260921)}
        self.assertEqual(issuer.parse_ishares(payload, ["EWJ"]), {})

    def test_ssga_history_parsed_and_sorted(self):
        blob = _ssga_xlsx([("22-Sep-2026", 773.442942, 1042332116),
                           ("21-Sep-2026", 773.446844, 1038632116)])
        rows = issuer.parse_ssga(blob)
        self.assertEqual([r[0] for r in rows], [dt.date(2026, 9, 21), dt.date(2026, 9, 22)])
        self.assertEqual(rows[-1][1], 1042332116)

    def test_ssga_missing_header_is_an_error_not_empty(self):
        from openpyxl import Workbook
        wb = Workbook()
        wb.active.append(["nothing here"])
        buf = io.BytesIO()
        wb.save(buf)
        with self.assertRaises(issuer.SourceError):
            issuer.parse_ssga(buf.getvalue())


class TestIssuerFlows(unittest.TestCase):
    def _store(self, d, series):
        store = issuer.ObsStore(os.path.join(d, "obs.json"))
        for sym, rows in series.items():
            for day, sh, nav in rows:
                store.put(sym, day, sh, nav, "test")
        return store

    def test_flow_is_share_change_times_nav(self):
        with tempfile.TemporaryDirectory() as d:
            store = self._store(d, {"SPY": [(dt.date(2026, 9, 21), 1_038_632_116, 773.446844),
                                            (dt.date(2026, 9, 22), 1_042_332_116, 773.442942)]})
            recs, _ = issuer.flows(store, ["SPY"])
        self.assertEqual(len(recs), 1)
        self.assertAlmostEqual(recs[0].net_flow_usd, 3_700_000 * 773.442942, delta=1)
        self.assertEqual((recs[0].ts, recs[0].market), (dt.date(2026, 9, 22), "US"))

    def test_gap_between_observations_is_not_collapsed_into_one_day(self):
        # SPY가 달력(9/21)을 제공하는데 EWJ는 9/18→9/22로 건너뛰었다 → 계산 생략
        with tempfile.TemporaryDirectory() as d:
            store = self._store(d, {
                "SPY": [(dt.date(2026, 9, 18), 1, 1.0), (dt.date(2026, 9, 21), 1, 1.0),
                        (dt.date(2026, 9, 22), 1, 1.0)],
                "EWJ": [(dt.date(2026, 9, 18), 100_000_000, 98.0),
                        (dt.date(2026, 9, 22), 105_000_000, 98.7)]})
            recs, diag = issuer.flows(store, ["EWJ"])
        self.assertEqual(recs, [])
        self.assertTrue(diag["gaps"])

    def test_split_is_not_a_flow(self):
        with tempfile.TemporaryDirectory() as d:
            store = self._store(d, {"EWQ": [(dt.date(2026, 9, 21), 8_000_000, 44.0),
                                            (dt.date(2026, 9, 22), 16_000_000, 22.0)]})
            recs, diag = issuer.flows(store, ["EWQ"])
        self.assertEqual(recs, [])
        self.assertTrue(diag["splits"])

    def test_rounding_noise_is_zero_flow(self):
        with tempfile.TemporaryDirectory() as d:
            store = self._store(d, {"EWU": [(dt.date(2026, 9, 21), 77_600_000, 47.1),
                                            (dt.date(2026, 9, 22), 77_600_001, 47.4)]})
            recs, _ = issuer.flows(store, ["EWU"])
        self.assertEqual(recs[0].net_flow_usd, 0.0)

    def test_obs_store_save_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "obs.json")
            s = issuer.ObsStore(path)
            s.put("SPY", dt.date(2026, 9, 22), 10, 1.5, "ssga")
            self.assertTrue(s.save())
            again = issuer.ObsStore(path)
            again.put("SPY", dt.date(2026, 9, 22), 10, 1.5, "ssga")
            self.assertFalse(again.save(), "같은 관측이면 파일을 다시 쓰지 않는다")
            self.assertEqual(again.series("SPY")[0][1], 10)


# ── 한국 원천 ───────────────────────────────────────────────────────────────
def _daum(rows):
    data = []
    for day, fore, ind, inst, etc in rows:
        det = {"FOREIGN": fore, "INDIVIDUAL": ind, "ETC_CORPORATION": etc,
               "FINANCIAL_INVESTOR": inst - 10.0, "PENSION_FUND": 10.0}
        data.append({"date": f"{day} 00:00:00", "foreignStraightPurchasePrice": fore,
                     "individualStraightPurchasePrice": ind,
                     "institutionStraightPurchasePrice": inst, "details": det})
    return {"code": 200, "data": data}


class TestKrInvestor(unittest.TestCase):
    rows = [("2026-09-23", -4.98e11, -1.454e12, 3.23e11, 1.629e12),
            ("2026-09-22", 4.8e10, -1.563e12, -1.29e11, 1.644e12)]

    def test_parses_confirmed_rows_after_close(self):
        out = kr_investor.parse(_daum(self.rows), now=kst(2026, 9, 23, 17))
        self.assertEqual([x["date"] for x in out], [dt.date(2026, 9, 23), dt.date(2026, 9, 22)])
        self.assertEqual(out[0]["pension"], 10.0)

    def test_intraday_today_row_is_excluded(self):
        # 장중 실행분이 잠정치를 확정치처럼 저장하면 안 된다
        out = kr_investor.parse(_daum(self.rows), now=kst(2026, 9, 23, 10))
        self.assertEqual([x["date"] for x in out], [dt.date(2026, 9, 22)])

    def test_broken_identity_on_latest_row_rejects_everything(self):
        bad = [("2026-09-23", 1e12, -1.454e12, 3.23e11, 1.629e12)] + self.rows[1:]
        with self.assertRaises(kr_investor.LayoutChanged):
            kr_investor.parse(_daum(bad), now=kst(2026, 9, 23, 17))

    def test_ceiling_skips_weekend(self):
        self.assertEqual(kr_investor.ceiling(kst(2026, 9, 21, 9)), dt.date(2026, 9, 18))


class TestFx(unittest.TestCase):
    def test_nearest_previous_ecb_rate(self):
        from smr import fx
        body = {"rates": {"2026-09-18": {"KRW": 1350.0, "JPY": 150.0, "EUR": 0.9, "GBP": 0.8},
                          "2026-09-21": {"KRW": 1356.15, "JPY": 157.18, "EUR": 0.87, "GBP": 0.75}}}

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return body

        fx._CACHE.clear()
        with patch.object(fx.requests, "get", lambda *a, **k: R()):
            self.assertTrue(fx.prefetch(dt.date(2026, 9, 18), dt.date(2026, 9, 21)))
        # 9/20(일)은 직전 기준일(9/18) 값을 쓴다
        self.assertAlmostEqual(fx.to_usd(1350.0, "KRW", dt.date(2026, 9, 20)), 1.0)
        self.assertAlmostEqual(fx.to_usd(0.87, "EUR", dt.date(2026, 9, 21)), 1.0)
        fx._CACHE.clear()


# ── 발행 게이트 ─────────────────────────────────────────────────────────────
class TestComparableSession(unittest.TestCase):
    def _sig(self, days_markets):
        recs = [rec(d, m, 1.0) for d, ms in days_markets.items() for m in ms]
        return signals.aggregate(to_frame(recs))

    def test_prefers_newer_three_market_day_over_older_full_day(self):
        # 한국 휴장(추석) — 3개 시장이 모인 날을 결손 표기하고 발행한다
        sig = self._sig({dt.date(2026, 9, 23): "KR JP EU US".split(),
                         dt.date(2026, 9, 24): "JP EU US".split()})
        self.assertEqual(pipeline._comparable_session(sig), dt.date(2026, 9, 24))

    def test_single_market_day_does_not_advance_as_of(self):
        # 한국은 미국 공시보다 하루 빨리 마감된다 — 한국만 있는 날을 기준일로 삼지 않는다
        sig = self._sig({dt.date(2026, 9, 22): "KR JP EU US".split(),
                         dt.date(2026, 9, 23): ["KR"]})
        self.assertEqual(pipeline._comparable_session(sig), dt.date(2026, 9, 22))


class TestStaleness(unittest.TestCase):
    cal = [dt.date(2026, 9, 21), dt.date(2026, 9, 22)]

    def test_expected_session_waits_for_issuer_posting(self):
        self.assertEqual(pipeline.expected_session(kst(2026, 9, 24, 15, 17)), dt.date(2026, 9, 23))
        self.assertEqual(pipeline.expected_session(kst(2026, 9, 24, 8, 17)), dt.date(2026, 9, 22))
        self.assertEqual(pipeline.expected_session(kst(2026, 9, 28, 8, 17)), dt.date(2026, 9, 25))

    def test_clock_detects_sources_frozen_together(self):
        # 관측 날짜끼리만 비교하면 원천이 전부 같이 멈춘 날 뒤처짐이 0으로 보인다
        self.assertEqual(pipeline.stale_sessions(dt.date(2026, 9, 22), self.cal,
                                                 kst(2026, 9, 23, 16)), 0)
        # 월 16시 기준 기대 세션은 금(9/25) — 화 이후 수·목·금 3세션 뒤처짐 → 보류
        self.assertEqual(pipeline.stale_sessions(dt.date(2026, 9, 22), self.cal,
                                                 kst(2026, 9, 28, 16)), 3)


class TestStatus(unittest.TestCase):
    def _mk(self, states):
        return {m: {"state": s} for m, s in zip("KR JP EU US".split(), states)}

    def test_warming_only_when_every_missing_market_is_warming(self):
        st = pipeline._status(dt.date(2026, 9, 22), 0.43, {"커버리지": "2/4"}, 0,
                              self._mk(["ok", "warming", "warming", "ok"]), {})
        self.assertEqual(st["state"], "warming")
        st = pipeline._status(dt.date(2026, 9, 22), 0.43, {"커버리지": "2/4"}, 0,
                              self._mk(["ok", "missing", "warming", "ok"]), {})
        self.assertEqual(st["state"], "halted")

    def test_stale_halts_even_with_good_score(self):
        st = pipeline._status(dt.date(2026, 9, 22), 0.9, {}, 3, self._mk(["ok"] * 4), {})
        self.assertEqual(st["state"], "halted")
        self.assertIn("뒤처짐", st["reason"])

    def test_last_ok_as_of_is_carried_through_a_halt(self):
        prev = {"as_of": "2026-09-22", "status": {"state": "ok"}}
        st = pipeline._status(dt.date(2026, 9, 23), 0.2, {}, 0, self._mk(["ok"] + ["missing"] * 3), prev)
        self.assertEqual(st["last_ok_as_of"], "2026-09-22")


class TestPipelinePersistsEvenWhenHalted(unittest.TestCase):
    """v1의 흡수 상태 재발 방지 — 보류돼도 관측은 저장되고 예외로 끊기지 않는다."""

    def test_halted_run_still_writes_store_and_payload(self):
        day = dt.date(2026, 9, 22)
        kr = [rec(day - dt.timedelta(days=i), "KR", 1e8 * (i % 3 - 1), instrument="KOSPI")
              for i in range(5)]
        with tempfile.TemporaryDirectory() as d:
            paths = {k: os.path.join(d, k) for k in ("flows.parquet", "data.json", "obs.json")}
            with patch.object(pipeline.issuer, "collect",
                              lambda store, today=None: ([], [{"collector": "ishares", "ok": False,
                                                                "error": "down"}])), \
                 patch.object(pipeline.kr_investor, "collect", lambda **k: kr), \
                 patch.object(pipeline.cot, "collect", lambda *a, **k: []):
                p = pipeline.run(store_path=paths["flows.parquet"], out_path=paths["data.json"],
                                 obs_path=paths["obs.json"], now=kst(2026, 9, 23, 16))
                self.assertEqual(p["status"]["state"], "halted")
                self.assertTrue(os.path.exists(paths["flows.parquet"]))
                self.assertEqual(FlowStore(paths["flows.parquet"]).load()["ts"].nunique(), 5)
                mtime = os.stat(paths["data.json"]).st_mtime_ns
                pipeline.run(store_path=paths["flows.parquet"], out_path=paths["data.json"],
                             obs_path=paths["obs.json"], now=kst(2026, 9, 23, 16, 30))
                self.assertEqual(os.stat(paths["data.json"]).st_mtime_ns, mtime,
                                 "내용이 같으면 data.json을 다시 쓰지 않는다")


# ── 알림 ────────────────────────────────────────────────────────────────────
def _payload(state="ok", as_of="2026-09-22", cov=("ok", "ok", "ok", "ok")):
    return {"as_of": as_of, "alerts": [], "suppressed": [],
            "rotation": {"ready": False, "rows": []},
            "status": {"state": state, "reason": "사유", "last_ok_as_of": "2026-09-11"},
            "coverage": {m: {"state": s} for m, s in zip("KR JP EU US".split(), cov)},
            "detail": {"KR": {"ts": as_of, "latest": 0.2, "cum": {"d20": -0.7},
                              "observations": 30, "signal": {"z20": 0.16}}},
            "health": [], "quality_warnings": [], "dashboard_url": "https://x.io/d"}


class TestNotify(unittest.TestCase):
    day = kst(2026, 9, 24, 15, 17)

    def test_url_falls_back_to_pages_address(self):
        import notify
        with patch.dict(os.environ, {"GITHUB_REPOSITORY": "owner/repo-name"}, clear=False):
            os.environ.pop("DASHBOARD_URL", None)
            self.assertEqual(notify.dashboard_url({}), "https://owner.github.io/repo-name/")

    def test_new_session_sends_once(self):
        import notify
        msg, st, _ = notify.decide(_payload(), {"last_status": "ok"}, self.day)
        self.assertIn("Cross-Market Flow Radar", msg)
        self.assertIn("https://x.io/d", msg)
        again, _, why = notify.decide(_payload(), st, self.day)
        self.assertIsNone(again, why)

    def test_recovery_is_announced_in_the_brief(self):
        import notify
        msg, st, _ = notify.decide(_payload(), {"last_status": "halted"}, self.day)
        self.assertIn("발행 재개", msg)
        self.assertEqual(st["last_status"], "ok")

    def test_halt_notified_on_transition_then_weekly(self):
        import notify
        msg, st, _ = notify.decide(_payload("halted"), {"last_status": "ok"}, self.day)
        self.assertIn("발행 보류", msg)
        quiet, st2, _ = notify.decide(_payload("halted"), st, self.day + dt.timedelta(days=3))
        self.assertIsNone(quiet, "실행마다 같은 경고를 보내지 않는다")
        remind, _, _ = notify.decide(_payload("halted"), st2, self.day + dt.timedelta(days=7))
        self.assertIn("지속", remind)

    def test_night_defers_but_force_overrides(self):
        import notify
        night = kst(2026, 9, 23, 22, 40)
        msg, st, why = notify.decide(_payload(), {}, night)
        self.assertIsNone(msg)
        self.assertEqual(st, {}, "이월 시 상태를 바꾸지 않아야 다음 실행이 보낸다")
        forced, _, _ = notify.decide(_payload(), {}, night, force=True)
        self.assertIsNotNone(forced)

    def test_partial_brief_waits_for_late_issuer_posting(self):
        import notify
        p = _payload(cov=("ok", "ok", "missing", "ok"))
        held, _, why = notify.decide(p, {}, kst(2026, 9, 24, 15, 17))
        self.assertIsNone(held, why)
        sent, _, _ = notify.decide(p, {}, kst(2026, 9, 24, 16, 47))
        self.assertIn("기준일 미관측", sent)

    def test_warming_notice_only_once(self):
        import notify
        p = _payload("warming", cov=("ok", "warming", "warming", "ok"))
        msg, st, _ = notify.decide(p, {"last_status": "halted"}, self.day)
        self.assertIn("워밍업", msg)
        again, _, _ = notify.decide(p, st, self.day + dt.timedelta(hours=2))
        self.assertIsNone(again)

    def test_v1_state_file_is_migrated(self):
        import notify
        with tempfile.TemporaryDirectory() as d:
            cwd = os.getcwd()
            os.chdir(d)
            try:
                os.makedirs("data")
                with open(notify.STATE, "w", encoding="utf-8") as f:
                    json.dump({"last_sent": "2026-09-15"}, f)
                self.assertEqual(notify._load_state()["last_status"], "halted")
            finally:
                os.chdir(cwd)


class TestOpsAlert(unittest.TestCase):
    def test_alerts_from_second_consecutive_failure_with_cooldown(self):
        import ops_alert
        t = kst(2026, 9, 24, 15, 20)
        a1, s = ops_alert.decide({}, t)
        self.assertFalse(a1, "1회 실패는 기록만")
        a2, s = ops_alert.decide(s, t + dt.timedelta(hours=1))
        self.assertTrue(a2)
        a3, s = ops_alert.decide(s, t + dt.timedelta(hours=3))
        self.assertFalse(a3, "20시간 안에는 다시 알리지 않는다")

    def test_no_alert_at_night(self):
        import ops_alert
        alert, _ = ops_alert.decide({"consecutive_failures": 3}, kst(2026, 9, 24, 23, 30))
        self.assertFalse(alert)


class TestMask(unittest.TestCase):
    def test_quad_witching_masked(self):
        flag, why = masked(dt.date(2026, 6, 19))
        self.assertTrue(flag)
        self.assertIn("만기", why)

    def test_ordinary_day_not_masked(self):
        self.assertFalse(masked(dt.date(2026, 4, 8))[0])


class TestKoreaKrx(unittest.TestCase):
    """KRX Open API 경로(키 등록 시에만 동작)."""

    def test_candidate_days_skips_weekends(self):
        from smr.collectors import korea
        days = korea._candidate_days(5, today=dt.date(2026, 9, 15))
        self.assertTrue(all(d.weekday() < 5 for d in days))
        self.assertEqual(days[-1], dt.date(2026, 9, 14))

    def test_collect_only_requests_missing_days(self):
        from smr.collectors import korea
        asked = []
        known = {dt.date(2026, 9, 10), dt.date(2026, 9, 11)}
        with patch.dict(os.environ, {"KRX_API_KEY": "k"}):
            with patch.object(korea, "_fetch_day", lambda d, k: asked.append(d) or []):
                korea.collect(known=known, lookback=5)
        self.assertTrue(asked)
        self.assertTrue(set(asked).isdisjoint(known))


if __name__ == "__main__":
    unittest.main()
