"""
selftest_v25.py — v25 변경(정산 정렬, price_date 스키마, 요약 리포트, 외부 데이터 방어)의 오프라인 검증.

네트워크 없이 합성(가짜) 데이터로만 돈다. 실행: python selftest_v25.py
- 실제 logs/ 는 임시 폴더로 복사해서 쓰므로 원본은 건드리지 않는다.
- 하나라도 실패하면 AssertionError 로 즉시 멈춘다.
"""
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

import common
import easy_report as er

REPO = os.path.dirname(os.path.abspath(__file__))
PASSED = []


def ok(name):
    PASSED.append(name)
    print(f"PASS  {name}")


def synth_prices(start='2026-06-01', end='2026-12-31', seed=1, drop_dates=()):
    idx = pd.bdate_range(start, end)
    idx = idx.difference(pd.DatetimeIndex(drop_dates))
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, len(idx))))
    return pd.DataFrame({'High': close, 'Low': close, 'Close': close}, index=idx)


# ---------------------------------------------------------------- T1 정산 정렬
def test_settlement_alignment():
    px = synth_prices()
    for h in (1, 5, 20):
        t = pd.Timestamp('2026-08-12')                        # 예측에 쓴 마지막 종가일(수)
        entry = float(px.loc[t, 'Close'])
        rec = common.make_prediction_record(pd.Timestamp('2026-08-13 01:00'), 'QQQ', 'TQQQ', h,
                                            60.0, 55.0, entry, price_date=t)
        assert rec['target_date'] == t + BDay(h)
        pos = px.index.get_loc(t)
        res = common.compute_aligned_outcome(px, t, h, entry)
        assert abs(res['actual_price'] - float(px['Close'].iloc[pos + h])) < 1e-12
        exp_ret = float(px['Close'].iloc[pos + h]) / entry - 1
        assert abs(res['actual_return'] - exp_ret) < 1e-12
        assert res['actual_label'] == int(exp_ret > 0)
    ok("T1a 정산 가격 = 예측일 종가 기준 정확히 h거래일 뒤 (h=1,5,20)")

    # 공휴일(영업일인데 거래 없는 날)이 끼면 BDay 가 아니라 '거래일' 기준이어야 함
    hol = [pd.Timestamp('2026-08-14')]
    px2 = synth_prices(drop_dates=hol)
    t = pd.Timestamp('2026-08-12')
    res = common.compute_aligned_outcome(px2, t, 5, float(px2.loc[t, 'Close']))
    pos = px2.index.get_loc(t)
    assert res['target_used'] == px2.index[pos + 5]
    assert res['target_used'] != t + BDay(5)                  # 공휴일 때문에 BDay 와 달라야 정상
    ok("T1b 중간에 휴장일이 있어도 '거래일' 기준으로 정산")

    # 아직 h거래일이 안 지났으면 None
    short = px.loc[:'2026-08-14']
    assert common.compute_aligned_outcome(short, pd.Timestamp('2026-08-12'), 5, None) is None
    ok("T1c 미래 데이터가 없으면 정산 보류(None)")

    # 기록가와 1.5% 넘게 어긋나면 기록가 사용(날짜 불일치 방어)
    t = pd.Timestamp('2026-08-12')
    bad = float(px.loc[t, 'Close']) * 1.10
    res = common.compute_aligned_outcome(px, t, 1, bad)
    assert abs(res['actual_return'] - (res['actual_price'] / bad - 1)) < 1e-12
    ok("T1d entry 가격 불일치 시 로그 기록가로 폴백")


# ---------------------------------------------------------------- T2/T3 로그 하위호환
def test_backward_compat_with_real_logs():
    tmp = tempfile.mkdtemp()
    cwd = os.getcwd()
    try:
        shutil.copytree(os.path.join(REPO, 'logs'), os.path.join(tmp, 'logs'))
        os.chdir(tmp)
        before = {}
        for name in ['scalp', 'swing', 'position', 'stock_scalp', 'stock_swing', 'stock_position']:
            df = common.load_log(name)
            assert 'price_date' in df.columns and df['price_date'].isna().all()
            assert len(df) > 0
            before[name] = df.copy()
        ok("T2a 구 로그(price_date 없음) 로드 OK, 6개 파일 모두")

        # 합성 가격으로 settle 호출: 예외 없이 돌고, 이미 정산된 행은 그대로여야 함
        px = synth_prices(start='2026-07-01', end='2026-12-31', seed=3)
        orig = common.fetch_ticker_data
        common.fetch_ticker_data = lambda t, start_date='2022-01-01': px.copy()
        try:
            for name, b in before.items():
                after = common.settle_predictions(name)
                done_before = b[b['actual_label'].notna()].reset_index(drop=True)
                # 이미 정산된 행(같은 키)의 결과 컬럼이 변하지 않았는지
                key = ['run_date', 'ticker']
                m = done_before.merge(after, on=key, suffixes=('_b', '_a'))
                assert len(m) == len(done_before)
                for col in ['actual_price', 'actual_return', 'actual_label', 'correct']:
                    assert np.allclose(m[col + '_b'].astype(float), m[col + '_a'].astype(float), equal_nan=True), col
                # 미정산이었던 행 중 price_date 가 채워지고 target_date 가 교정됐는지
                pend = after[after['price_date'].notna()]
                assert len(pend) > 0
                tgt_ok = all(pd.Timestamp(r['target_date']) == pd.Timestamp(r['price_date']) + BDay(int(r['horizon_days']))
                             for _, r in after[after['price_date'].notna()].iterrows())
                assert tgt_ok
                # 저장된 CSV 에 컬럼이 들어갔는지
                raw = pd.read_csv(os.path.join('logs', f'predictions_{name}.csv'))
                assert 'price_date' in raw.columns
        finally:
            common.fetch_ticker_data = orig
        ok("T2b settle: 기존 정산 행 불변, 미정산 행 price_date 보강 + target_date 교정, CSV 컬럼 추가")

        # 새 기록 추가 + 같은 종가 기준 중복 스킵
        name = 'scalp'
        t = pd.Timestamp('2026-10-16')
        r1 = common.make_prediction_record(pd.Timestamp('2026-10-17 04:00'), 'QQQ', 'TQQQ', 1, 58.0, 50.0, 70.0, price_date=t)
        n0 = len(common.load_log(name))
        common.append_predictions(name, [r1])
        assert len(common.load_log(name)) == n0 + 1
        r2 = common.make_prediction_record(pd.Timestamp('2026-10-19 04:00'), 'QQQ', 'TQQQ', 1, 58.0, 50.0, 70.0, price_date=t)
        common.append_predictions(name, [r2])          # 토 실행분과 월 실행분 = 같은 금요일 종가 기준
        assert len(common.load_log(name)) == n0 + 1
        ok("T2c 새 기록 추가 OK, 같은 종가 기준(토/월 실행) 중복은 스킵")

        # 재실행 시 변경 없으면 파일 쓰기 없음(불필요한 커밋 방지)
        p = os.path.join('logs', 'predictions_swing.csv')
        common.fetch_ticker_data = lambda t, start_date='2022-01-01': px.copy()
        try:
            common.settle_predictions('swing')
            m1 = os.path.getmtime(p)
            import time; time.sleep(0.05)
            common.settle_predictions('swing')
            assert os.path.getmtime(p) == m1
        finally:
            common.fetch_ticker_data = orig
        ok("T2d 정산할 게 없으면 로그 파일을 다시 쓰지 않음")
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def test_fallback_price_date():
    f = common._fallback_price_date
    assert f(pd.Timestamp('2026-10-14 05:00')) == pd.Timestamp('2026-10-13')   # 수 실행 → 화 종가
    assert f(pd.Timestamp('2026-10-17 05:00')) == pd.Timestamp('2026-10-16')   # 토 실행 → 금 종가
    assert f(pd.Timestamp('2026-10-18 05:00')) == pd.Timestamp('2026-10-16')   # 일 실행 → 금 종가
    assert f(pd.Timestamp('2026-10-19 05:00')) == pd.Timestamp('2026-10-16')   # 월 실행 → 금 종가
    rec = common.make_prediction_record(pd.Timestamp('2026-10-14 05:00'), 'QQQ', 'TQQQ', 1, 55, 50, 70.0)
    assert rec['price_date'] == pd.Timestamp('2026-10-13') and rec['target_date'] == pd.Timestamp('2026-10-14')
    ok("T3 price_date 폴백(화~금/토/일/월 실행) 및 target_date 계산")


# ---------------------------------------------------------------- T4 장중 봉 제거
def test_partial_bar():
    ET = ZoneInfo('America/New_York')
    idx = pd.to_datetime(['2026-10-14', '2026-10-15', '2026-10-16'])
    df = pd.DataFrame({'Close': [1.0, 2.0, 3.0]}, index=idx)
    assert len(common.drop_partial_us_bar(df, datetime(2026, 10, 16, 10, 30, tzinfo=ET))) == 2   # 금 장중 → 제거
    assert len(common.drop_partial_us_bar(df, datetime(2026, 10, 16, 17, 0, tzinfo=ET))) == 3    # 금 마감 후 → 유지
    df2 = df.iloc[:-1]
    assert len(common.drop_partial_us_bar(df2, datetime(2026, 10, 17, 9, 0, tzinfo=ET))) == 2    # 토 → 유지
    assert len(common.drop_partial_us_bar(df2, datetime(2026, 10, 16, 9, 0, tzinfo=ET))) == 2    # 개장 전(오늘 봉 없음) → 유지
    assert common.drop_partial_us_bar(None) is None
    ok("T4 장중 미완성 봉 제거(장중만 제거, 마감 후·주말·개장 전은 유지)")


# ---------------------------------------------------------------- T5 신호 판정
def _mk_log(n, h, acc, up_rate, pred_up=1.0, seed=0):
    rng = np.random.default_rng(seed)
    actual = (rng.random(n) < up_rate).astype(int)
    pred = np.where(rng.random(n) < pred_up, 1, 0)
    correct = np.zeros(n, dtype=int)
    k = int(round(acc * n))
    correct[rng.permutation(n)[:k]] = 1
    # correct 와 pred/actual 일관성은 평가 함수에 필요 없으므로 correct 를 직접 지정
    return pd.DataFrame({'ticker': 'X', 'horizon_days': h, 'predicted_label': pred,
                         'actual_label': actual, 'correct': correct})


def test_evaluate_signal():
    # (i) 항상 상승 찍기: 적중률 = 상승비율 → 통과 못 함
    df = _mk_log(60, 1, 0.56, 0.56)
    r = common.evaluate_signal(df, 'X')
    assert r['status'] == 'fail', r
    # (ii) 20일 예측 20행 전부 적중이어도 독립 관측 1 → 판정 불가
    df = _mk_log(20, 20, 1.0, 0.8)
    r = common.evaluate_signal(df, 'X')
    assert r['n_eff'] == 1 and r['status'] == 'insufficient', r
    # (iii) 진짜 실력(80%/기준 55%, 표본 100) → 통과
    df = _mk_log(100, 1, 0.80, 0.55)
    r = common.evaluate_signal(df, 'X')
    assert r['status'] == 'pass', r
    # (iv) 빈 로그 / 정산 없음
    assert common.evaluate_signal(pd.DataFrame(columns=['ticker', 'correct']), 'X') is None
    # (v) 한쪽 몰빵 감지
    r = common.evaluate_signal(_mk_log(60, 1, 0.5, 0.5, pred_up=1.0), 'X')
    assert r['one_sided'] is True
    ok("T5 신호 판정: 찍기=fail, 겹침표본=insufficient, 진짜 실력=pass, 빈 로그=None, 한쪽몰빵 감지")


# ---------------------------------------------------------------- T6~T9 핫이슈·일정·외부 데이터
def test_hot_issues_and_events():
    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, 'h.json')
        json.dump({'updated': '2026-10-10', 'issues': [
            {'title': '살아있음', 'detail': 'd', 'until': '2026-10-20', 'priority': 2},
            {'title': '만료됨', 'detail': 'x', 'until': '2026-10-15', 'priority': 1},
            {'title': '깨진 항목', 'until': 'not-a-date'},
            {'detail': '제목없음', 'until': '2026-10-30'},
            {'title': '우선순위1', 'until': '2026-10-17', 'priority': 1},
        ]}, open(p, 'w', encoding='utf-8'))
        act, age = er.load_hot_issues(p, today='2026-10-16')
        assert [a['title'] for a in act] == ['우선순위1', '살아있음'], act
        assert act[0]['days_left'] == 1 and act[1]['days_left'] == 4 and age == 6
        act2, _ = er.load_hot_issues(p, today='2026-10-20')          # 만료일 당일까지 유지
        assert [a['title'] for a in act2] == ['살아있음']
        act3, _ = er.load_hot_issues(p, today='2026-10-21')
        assert act3 == []
        assert er.load_hot_issues(os.path.join(d, 'nope.json')) == ([], None)
        open(os.path.join(d, 'bad.json'), 'w').write('{not json')
        assert er.load_hot_issues(os.path.join(d, 'bad.json')) == ([], None)
        json.dump({'issues': 'oops'}, open(os.path.join(d, 'odd.json'), 'w'))
        assert er.load_hot_issues(os.path.join(d, 'odd.json'))[0] == []
    finally:
        shutil.rmtree(d, ignore_errors=True)
    ok("T6 핫이슈: 만료일 자동 제거(당일까지 유지), 깨진 항목·파일 없음·깨진 JSON·이상한 형식 모두 안전")

    ev = er.upcoming_events(common.MARKET_EVENTS, common.STOCK_EARNINGS_DATES, today='2026-10-16', window=14)
    descs = ' | '.join(e['desc'] for e in ev)
    assert 'TSLA' in descs and 'FOMC' in descs and 'AAPL' in descs
    assert all(0 <= e['days_left'] <= 14 for e in ev)
    assert ev == sorted(ev, key=lambda e: e['date'])
    assert next(e for e in ev if 'TSLA' in e['desc'])['days_left'] == 5      # 10/21
    ok("T7 일정: 14일 창, 날짜순, TSLA 10/21(D-5)·FOMC·AAPL 포함")


def test_external_defenses():
    rss = b"""<?xml version="1.0"?><rss><channel>
      <item><title>Stocks Slide as Yields Climb</title></item>
      <item><title>  Chip Rout Deepens  </title></item>
      <item><title></title></item>
      <item><title>Third &amp; Final</title></item>
      <item><title>Fourth</title></item></channel></rss>"""
    assert er.fetch_news_headlines(3, opener=lambda: rss) == ['Stocks Slide as Yields Climb', 'Chip Rout Deepens', 'Third & Final']
    assert er.fetch_news_headlines(3, opener=lambda: b'<html>not xml') == []
    assert er.fetch_news_headlines(3, opener=lambda: b'') == []

    def boom():
        raise OSError('network down')
    assert er.fetch_news_headlines(3, opener=boom) == []
    ok("T8 RSS: 정상 파싱, 깨진 XML·빈 응답·네트워크 예외는 [] (리포트 유지)")

    close = pd.Series(np.linspace(90, 100, 40))
    btc = er.fetch_btc_context(downloader=lambda: pd.DataFrame({'Close': close}))
    assert btc and abs(btc['ret1'] - (close.iloc[-1] / close.iloc[-2] - 1)) < 1e-12
    assert abs(btc['ret7'] - (close.iloc[-1] / close.iloc[-8] - 1)) < 1e-12
    assert abs(btc['from_high30']) < 1e-12
    assert er.fetch_btc_context(downloader=lambda: pd.DataFrame({'Close': close.iloc[:5]})) is None
    assert er.fetch_btc_context(downloader=lambda: pd.DataFrame()) is None
    assert er.fetch_btc_context(downloader=boom) is None
    ok("T9 BTC 컨텍스트: 정상 계산, 데이터 부족·빈 데이터·예외는 None")


# ---------------------------------------------------------------- T10 수식 검증
def test_math():
    assert abs(er.recovery_needed(-0.35) - 0.5385) < 1e-3 and er.recovery_needed(-0.5) == 1.0
    assert er.recovery_needed(0.1) is None and er.recovery_needed(None) is None
    # 변동성 잠식 공식을 몬테카를로로 검증: 지수 로그수익 기댓값 0 → 3배 ETF 평균 로그수익 ≈ -(L^2-L)/2 s^2 t
    rng = np.random.default_rng(42)
    L, s_idx, days = 3, 0.35, 21
    sd = s_idx / np.sqrt(252)
    paths = 200_000
    z = rng.normal(0, sd, size=(paths, days))
    r_idx = np.exp(z - 0.5 * sd ** 2) - 1            # 일간 지수 수익률, 로그수익 평균 ≈ 0(제자리)
    letf = np.log1p(L * r_idx).sum(axis=1)            # 일간 리밸런싱 3배 ETF 로그수익
    idx_log = np.log1p(r_idx).sum(axis=1)
    sim_drag = float(letf.mean() - L * idx_log.mean())
    formula = -er.decay_estimate_monthly(L * s_idx, leverage=L, trading_days=days)
    # 몬테카를로 vs 공식(이산 보정으로 소폭 차이 허용)
    assert abs(sim_drag - formula) < 0.006, (sim_drag, formula)
    ok(f"T10 수식: 회복필요율 정확, 변동성 잠식 공식 vs 몬테카를로 (공식 {formula*100:.2f}% / 시뮬 {sim_drag*100:.2f}%)")


# ---------------------------------------------------------------- T11/T12 리포트 조립
def fake_item(ticker, bench, direction, up, edge, base, price, vol, dd, ret5, status, breadth=None,
              pdate='2026-10-15'):
    sig = None if status is None else {'status': status, 'n': 40, 'n_eff': 40, 'horizon': 1, 'accuracy': 52.0,
                                       'up_rate': 52.0, 'naive_best': 52.0, 'edge_vs_naive': 0.0,
                                       'ci_low': 40.0, 'ci_high': 64.0, 'pred_up_share': 100.0, 'one_sided': True}
    return dict(ticker=ticker, benchmark=bench, desc=ticker, direction=direction, up_prob=up, raw_up_prob=up,
                base_rate=base, edge=edge, price=price, vol=vol, drawdown=dd, ret1=-0.03, ret5=ret5,
                price_date=pdate, breadth=breadth, signal=sig)


def make_summaries():
    S = {}
    bre = {'up_share': 0.0, 'momentum5': -0.095}
    def lev(h, qqq, smh, spy, status):
        items = []
        for tk, b, up, edge in [('TQQQ', 'QQQ', *qqq), ('SOXL', 'SMH', *smh), ('SPXL', 'SPY', *spy)]:
            dd = {'TQQQ': -0.12, 'SOXL': -0.347, 'SPXL': -0.03}[tk]
            px = {'TQQQ': 73.82, 'SOXL': 107.35, 'SPXL': 290.0}[tk]
            vol = {'TQQQ': 0.487, 'SOXL': 1.052, 'SPXL': 0.289}[tk]
            items.append(fake_item(tk, b, 1, up, edge, up - edge, px, vol, dd, -0.08, status,
                                   bre if b == 'SMH' else None))
            items.append(fake_item(tk + '_inv', b, -1, 100 - up, edge, up - edge, 30.0, vol, -0.5, 0.1, status))
        return {'log': 'x', 'horizon_days': h, 'generated_at': datetime.now().isoformat(), 'items': items}
    S['scalp'] = lev(1, (59.8, 3.3), (61.3, 5.6), (60.1, 3.2), 'fail')
    S['swing'] = lev(5, (55.3, -4.7), (60.5, -0.3), (63.4, 0.4), 'insufficient')
    S['position'] = lev(20, (66.4, -4.4), (71.2, 1.4), (73.4, 0.8), 'insufficient')
    for name, h, vals in [('stock_scalp', 1, [(54.1, 0.0), (51.0, -3.8), (46.6, -3.6)]),
                          ('stock_swing', 5, [(61.0, 2.7), (57.6, -2.5), (50.0, -1.1)]),
                          ('stock_position', 20, [(57.1, -6.1), (63.3, -6.0), (51.7, 2.0)])]:
        items = [fake_item(t, t, 1, u, e, u - e, 300.0, 0.2, -0.05, 0.0, 'fail' if h == 1 else 'insufficient')
                 for t, (u, e) in zip(['AAPL', 'NVDA', 'TSLA'], vals)]
        S[name] = {'log': name, 'horizon_days': h, 'generated_at': datetime.now().isoformat(), 'items': items}
    return S


def test_report_build():
    S = make_summaries()
    now = datetime(2026, 10, 16, 13, 40, tzinfo=er.KST)
    hot = [{'title': '금리 <쇼크> & 인상', 'detail': 'CPI > 예상', 'until': pd.Timestamp('2026-10-29'),
            'days_left': 13, 'priority': 1}]
    events = er.upcoming_events(common.MARKET_EVENTS, common.STOCK_EARNINGS_DATES, today=now.date(), window=14)
    btc = {'price': 81500.0, 'ret1': -0.012, 'ret7': -0.03, 'from_high30': -0.07}
    heads = ['Stocks <Slide> & Yields Climb', 'Chip Rout Deepens']
    secs = er.build_sections(S, [], hot, 6, events, btc, heads, now)
    msgs = er.chunk_sections(secs)
    text = '\n\n'.join(msgs)
    assert all(len(m) <= er.TELEGRAM_LIMIT for m in msgs), [len(m) for m in msgs]
    # 텔레그램 HTML 안전성: 허용 태그는 <b></b> 뿐이고, 본문의 < > & 는 모두 이스케이프돼야 함
    stripped = re.sub(r'</?b>', '', text)
    assert '<' not in stripped and '>' not in stripped, "이스케이프 안 된 꺾쇠가 있음"
    assert not re.search(r'&(?!amp;|lt;|gt;)', stripped), "이스케이프 안 된 &"
    assert text.count('<b>') == text.count('</b>')
    # 내용 점검
    assert '오늘 밤 미국장' in text and '10/16(금)' in text
    assert '검증 통과 0/18' in text
    assert '본전까지 +53% 필요' in text or '본전까지 +54% 필요' in text
    assert 'SOXL (보유)' in text and '고점대비 -34.7%' in text
    assert 'D-5 10/21(수) TSLA 실적' in text
    assert '반도체 구성종목 상승비율 0%' in text
    assert 'BTC $81,500' in text
    assert '매수' not in text and '매도' not in text and '사라' not in text
    ok(f"T11 리포트 조립: 길이 {[len(m) for m in msgs]}자(한도 {er.TELEGRAM_LIMIT}), HTML 이스케이프 안전, 핵심 문구·수치 확인, 매수/매도 문구 없음")
    return text


def test_report_robustness():
    now = datetime(2026, 10, 18, 9, 0, tzinfo=er.KST)             # 일요일: 가격일은 금요일
    # 요약이 전혀 없어도, 외부 데이터가 전부 실패해도 예외 없이 메시지가 나와야 함
    secs = er.build_sections({}, ['scalp', 'swing'], [], None, [], None, [], now)
    msgs = er.chunk_sections(secs)
    assert msgs and all(len(m) <= er.TELEGRAM_LIMIT for m in msgs)
    S = make_summaries()
    for v in S.values():
        for it in v['items']:
            it['price_date'] = '2026-10-16'
    secs = er.build_sections(S, [], [], None, [], None, [], now)
    text = '\n'.join(secs)
    assert '10/19(월) 미국장' in text            # 일요일 아침 → 다음 장은 월요일
    # 신호/낙폭 값이 None·NaN 이어도 안 죽음
    S2 = make_summaries()
    for v in S2.values():
        for it in v['items']:
            it['drawdown'] = None
            it['vol'] = float('nan')
            it['ret5'] = None
            it['signal'] = None
    er.build_sections(S2, [], [], None, [], None, [], now)
    # 뉴스·일정·핫이슈가 비어 생략돼도 섹션 번호가 1부터 연속이어야 함
    import re as _re
    nums = [int(n) for n in _re.findall(r'<b>(\d+)\. ', '\n'.join(secs))]
    assert nums == list(range(1, len(nums) + 1)), nums
    ok("T12 방어: 요약 없음/외부 데이터 전부 실패/None·NaN 값에도 정상 출력, 주말엔 '다음 미국장' 날짜 표기")


def test_finish_report_and_load():
    tmp = tempfile.mkdtemp()
    cwd = os.getcwd()
    try:
        os.chdir(tmp)
        common._SUMMARY_ITEMS.clear()
        common.add_summary_item(ticker='SOXL', benchmark='SMH', direction=1, up_prob=np.float64(61.3),
                                price_date=pd.Timestamp('2026-10-15'), signal=None, edge=np.float64(5.6))
        common.finish_report('scalp', 1, "detail text")
        assert open('reports/detail_scalp.txt', encoding='utf-8').read() == "detail text"
        data = json.load(open('reports/summary_scalp.json', encoding='utf-8'))
        assert data['items'][0]['up_prob'] == 61.3 and data['items'][0]['price_date'].startswith('2026-10-15')
        sums, stale = er.load_summaries('reports', now=datetime.now())
        assert 'scalp' in sums and 'swing' in stale          # 없는 파일은 stale
        sums, stale = er.load_summaries('reports', now=datetime.now() + timedelta(hours=30))
        assert 'scalp' in stale                              # 오래된 파일은 stale 표시
        os.environ.pop('SEND_DETAIL', None)
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)
    ok("T13 finish_report: 상세 파일·요약 JSON(numpy/Timestamp 직렬화) 저장, 없는 파일·오래된 파일은 stale 처리")


def test_swap_invariant_untouched():
    # 페어 대칭 원칙: apply_direction 은 그대로여야 함(상승확률 합 = 100)
    for up in (0.0, 12.5, 50.0, 73.4, 100.0):
        a, _ = common.apply_direction(up, 100 - up, +1)
        b, _ = common.apply_direction(up, 100 - up, -1)
        assert abs(a + b - 100) < 1e-12
    ok("T14 페어 대칭(롱+인버스 상승확률=100) 원칙 유지 — apply_direction 변경 없음")


if __name__ == '__main__':
    test_settlement_alignment()
    test_backward_compat_with_real_logs()
    test_fallback_price_date()
    test_partial_bar()
    test_evaluate_signal()
    test_hot_issues_and_events()
    test_external_defenses()
    test_math()
    sample = test_report_build()
    test_report_robustness()
    test_finish_report_and_load()
    test_swap_invariant_untouched()
    print(f"\n전체 {len(PASSED)}개 검증 통과")
    if os.environ.get('SHOW_SAMPLE') == '1':
        print("\n" + "=" * 50 + "\n" + sample)
