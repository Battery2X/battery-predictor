"""
stress_test2.py — 2차 극한 시뮬레이션 (v28, 합성 데이터·네트워크 불필요). 실행: python stress_test2.py

1차(stress_test.py)가 '입력이 깨졌을 때'를 봤다면, 2차는 '시간·기업행위·사람의 손·규모·정합성'을 본다.
  A 기업행위(분할/역분할)  B 달력(휴일·연말·요일)  C 사람 손(엑셀 저장·충돌 마커·열 순서)
  D 규모(수만 행 성능)    E 정합성(미래 가격일=룩어헤드, 시계 어긋남)  F 환경(토큰 없음 등)  G 모델 극단
판정: OK / SILENT(조용히 틀림) / CRASH(예외)
"""
import io
import os
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm
import push_logs as P
import health_check as hc

RESULTS = []
_ORIG_FETCH = cm.fetch_ticker_data


def case(group, name, fn, judge=None):
    t0 = time.time()
    try:
        out = fn()
        verdict, memo = ("OK", "") if judge is None else judge(out)
    except Exception as e:
        verdict, memo = "CRASH", f"{type(e).__name__}: {str(e)[:110]}"
    RESULTS.append((group, name, verdict, memo, time.time() - t0))


def fresh():
    d = tempfile.mkdtemp()
    cm.LOG_DIR = d
    return d


def px(vals, start="2026-09-01", idx=None):
    vals = np.asarray(vals, dtype=float)
    return pd.DataFrame({'Close': vals}, index=idx if idx is not None else pd.bdate_range(start, periods=len(vals)))


def rec(run, tk="SOXS", h=5, up=40.0, price=10.0, pdate=None):
    return cm.make_prediction_record(pd.Timestamp(run), 'SMH', tk, h, up, up, price, price_date=pdate)


def settle_one(record, prices_df):
    """레코드 1건을 로그에 넣고 정산해서 정산된 행을 돌려준다."""
    fresh()
    cm.append_predictions('x', [record])
    orig = cm.fetch_ticker_data
    cm.fetch_ticker_data = lambda t, **k: prices_df
    try:
        df = cm.settle_predictions('x')
    finally:
        cm.fetch_ticker_data = orig
    return df


def ok(cond, memo=""):
    return ("OK", "") if cond else ("SILENT", memo or "기대와 다름")


# ======================================================================= A. 기업행위
# 로그 진입가는 '당시 실제 가격'(비조정). yfinance 의 과거 시세는 분할 후 '소급 조정'된다.
pd_ = pd.Timestamp("2026-09-01")
n = 12
true_ret = np.r_[np.zeros(1), np.full(n - 1, -0.002)]
base = 10.0 * np.cumprod(1 + true_ret)            # 분할 전 가격(10 근처, 서서히 -0.2%/일) → 5일 후 약 -1%
idx = pd.bdate_range(pd_, periods=n)

adj_rev = px(base * 10, idx=idx)                  # 1:10 역분할 → 과거가 ×10 로 소급 조정
case("A.split", "역분할 1:10(인버스 ETF) 정산: 진짜는 -1% → 하락(0)이어야",
     lambda: settle_one(rec("2026-09-02 09:17", price=10.0, h=5, pdate="2026-09-01"), adj_rev),
     lambda d: ok(len(d) == 1 and d.iloc[0]['actual_label'] == 0 and abs(d.iloc[0]['actual_return']) < 0.05,
                  f"label={d.iloc[0]['actual_label']}, ret={d.iloc[0]['actual_return']:.2f}"))
adj_fwd = px(base / 3, idx=idx)                   # 3:1 정분할 → 과거가 ÷3 로 소급 조정
case("A.split", "정분할 3:1 정산: 진짜는 -1% → 하락(0)이어야",
     lambda: settle_one(rec("2026-09-02 09:17", price=10.0, h=5, pdate="2026-09-01"), adj_fwd),
     lambda d: ok(d.iloc[0]['actual_label'] == 0 and abs(d.iloc[0]['actual_return']) < 0.05,
                  f"label={d.iloc[0]['actual_label']}, ret={d.iloc[0]['actual_return']:.2f}"))
up_move = base.copy(); up_move[5:] = base[0] * 1.45     # 분할 아님: 실제 +45% 급등
case("A.split", "분할 아닌 진짜 +45% 급등은 그대로 +45%로 채점",
     lambda: settle_one(rec("2026-09-02 09:17", price=10.0, h=5, pdate="2026-09-01"), px(up_move, idx=idx)),
     lambda d: ok(abs(d.iloc[0]['actual_return'] - 0.45) < 0.01, f"ret={d.iloc[0]['actual_return']:.3f}"))
div = px(base * 0.992, idx=idx)                    # 배당 소급 조정 0.8%
case("A.split", "배당 조정 0.8% 차이는 시리즈 진입가 사용(정상)",
     lambda: settle_one(rec("2026-09-02 09:17", price=10.0, h=5, pdate="2026-09-01"), div),
     lambda d: ok(abs(d.iloc[0]['actual_return'] - (base[5] / base[0] - 1)) < 1e-6, f"ret={d.iloc[0]['actual_return']:.4f}"))

# ======================================================================= B. 달력
hol = pd.DatetimeIndex(["2026-11-23", "2026-11-24", "2026-11-25", "2026-11-27", "2026-11-30", "2026-12-01", "2026-12-02"])  # 11/26 추수감사절 휴장
case("B.cal", "추수감사절 휴장 낀 5거래일 정산: 거래 봉 기준 +5번째",
     lambda: cm.compute_aligned_outcome(px(np.arange(100, 107), idx=hol), "2026-11-23", 5, 100.0),
     lambda r: ok(r is not None and r['target_used'] == pd.Timestamp("2026-12-01") and abs(r['actual_return'] - 0.05) < 1e-9, str(r)))
case("B.cal", "굿프라이데이·연말·새해 휴장 낀 1일 정산",
     lambda: cm.compute_aligned_outcome(px([100, 101], idx=pd.DatetimeIndex(["2026-12-31", "2027-01-04"])), "2026-12-31", 1, 100.0),
     lambda r: ok(r is not None and r['target_used'] == pd.Timestamp("2027-01-04")))
case("B.cal", "윤일(2028-02-29) 포함 정산", lambda: cm.compute_aligned_outcome(
    px([100, 101, 102], idx=pd.DatetimeIndex(["2028-02-28", "2028-02-29", "2028-03-01"])), "2028-02-28", 2, 100.0),
     lambda r: ok(r is not None and r['target_used'] == pd.Timestamp("2028-03-01")))
case("B.cal", "토요일 실행(금요일 종가 기준) price_date 폴백",
     lambda: cm._fallback_price_date(pd.Timestamp("2026-10-17 00:17")),
     lambda r: ok(r == pd.Timestamp("2026-10-16"), str(r)))
case("B.cal", "화요일 00:17 UTC 실행(월요일 종가 기준) price_date 폴백",
     lambda: cm._fallback_price_date(pd.Timestamp("2026-10-20 00:17")),
     lambda r: ok(r == pd.Timestamp("2026-10-19"), str(r)))
case("B.cal", "월요일이 공휴일이면 화요일 실행분의 종가일=금요일(시세에 월요일 없음)이어도 정산 가능",
     lambda: cm.compute_aligned_outcome(px([100, 101, 102], idx=pd.DatetimeIndex(["2026-05-22", "2026-05-26", "2026-05-27"])), "2026-05-25", 1, 100.0),
     lambda r: ok(r is not None and r['target_used'] == pd.Timestamp("2026-05-26"), str(r)))

# ======================================================================= C. 사람의 손
GOOD = pd.DataFrame([rec(f"2026-10-{d:02d} 09:17", tk="SOXL", h=1, up=60, price=100.0) for d in (12, 13, 14)])


def write_variant(kind):
    d = fresh()
    p = os.path.join(d, 'predictions_x.csv')
    GOOD.to_csv(p, index=False)
    raw = open(p, 'rb').read().decode('utf-8')
    if kind == 'bom':
        raw = '﻿' + raw
    elif kind == 'crlf':
        raw = raw.replace('\n', '\r\n')
    elif kind == 'blank_tail':
        raw = raw + '\n\n,,,,,,,,,,,,,,\n'
    elif kind == 'conflict':
        lines = raw.splitlines()
        raw = '\n'.join(lines[:2] + ['<<<<<<< HEAD'] + lines[2:3] + ['======='] + lines[3:] + ['>>>>>>> branch']) + '\n'
    elif kind == 'dup_header':
        lines = raw.splitlines()
        raw = '\n'.join(lines + [lines[0]] + lines[1:2]) + '\n'
    elif kind == 'reordered':
        df = pd.read_csv(p)
        df[list(reversed(df.columns))].to_csv(p, index=False)
        return cm.load_log('x')
    elif kind == 'extra_col':
        df = pd.read_csv(p)
        df['memo'] = '메모'
        df.to_csv(p, index=False)
        return cm.load_log('x')
    elif kind == 'excel_dates':
        df = pd.read_csv(p)
        df['run_date'] = pd.to_datetime(df['run_date']).dt.strftime('%m/%d/%Y %H:%M')
        df.to_csv(p, index=False)
        return cm.load_log('x')
    elif kind == 'no_price_date':
        df = pd.read_csv(p).drop(columns=['price_date'])
        df.to_csv(p, index=False)
        return cm.load_log('x')
    with open(p, 'w', encoding='utf-8', newline='') as f:
        f.write(raw)
    return cm.load_log('x')


for kind, label in [('bom', "엑셀 UTF-8 BOM"), ('crlf', "윈도우 CRLF 줄바꿈"), ('blank_tail', "끝에 빈 줄·쉼표만 있는 줄"),
                    ('conflict', "git 충돌 마커(<<<<<<<) 섞임"), ('dup_header', "헤더가 중간에 한 번 더 있음(파일 이어붙임)"),
                    ('reordered', "열 순서가 뒤집힘"), ('extra_col', "사용자가 열 추가"), ('excel_dates', "엑셀 날짜서식(10/12/2026)"),
                    ('no_price_date', "price_date 열 없는 구형 로그")]:
    case("C.human", label, (lambda k=kind: write_variant(k)),
         lambda df: ok(len(df) >= 3 and df['ticker'].eq('SOXL').all(), f"행 수 {len(df)}"))

case("C.human", "충돌 마커 섞인 원격 CSV를 push_logs 병합에 넣어도 죽지 않음",
     lambda: P.merge_csv_text("<<<<<<< HEAD\nrun_date,ticker\n=======\n>>>>>>> x\n", os.path.join(fresh(), 'none.csv')) if False else
     (lambda d: (GOOD.to_csv(os.path.join(d, 'l.csv'), index=False),
                 P.merge_csv_text("<<<<<<< HEAD\nrun_date,ticker,horizon_days\n2026-10-12,SOXL,1\n=======\n>>>>>>> x\n", os.path.join(d, 'l.csv')))[1])(fresh()),
     lambda df: ok(len(df) >= 3))

# ======================================================================= D. 규모
def big_log(n_rows):
    days = pd.bdate_range("2016-01-04", periods=n_rows // 6)
    tks = ['TQQQ', 'SQQQ', 'SOXL', 'SOXS', 'SPXL', 'SPXS']
    rows = []
    for d in days:
        for t in tks:
            r = cm.make_prediction_record(d + pd.Timedelta(hours=9, minutes=17), 'QQQ', t, 5, 55.0, 55.0, 50.0 + (hash(t) % 10))
            rows.append(r)
    df = pd.DataFrame(rows)
    df['actual_label'] = 1.0; df['correct'] = 1.0; df['actual_return'] = 0.01; df['actual_price'] = 51.0
    return df


case("D.scale", "10년치(약 15,000행) 로그 로드+append+저장 5초 이내", lambda: (lambda d: (
    big_log(15000).to_csv(os.path.join(d, 'predictions_x.csv'), index=False), time.time(),
    cm.append_predictions('x', [rec("2026-10-16 09:17", tk="SOXL", h=5, up=60, price=100.0, pdate="2026-10-15")]), time.time())[-1])(fresh()) and True,
     lambda x: ok(True))
_t = {}
def timed_append():
    d = fresh()
    big_log(15000).to_csv(os.path.join(d, 'predictions_x.csv'), index=False)
    t0 = time.time()
    cm.append_predictions('x', [rec("2026-10-16 09:17", tk="SOXL", h=5, up=60, price=100.0, pdate="2026-10-15")])
    _t['append'] = time.time() - t0
    t0 = time.time(); cm.load_log('x'); _t['load'] = time.time() - t0
    return _t['append']
case("D.scale", "15,000행 append 소요", timed_append, lambda s: ok(s < 5.0, f"{s:.1f}s"))


def timed_merge():
    a = big_log(15000)
    b = a.copy()
    t0 = time.time()
    m = P.merge_log_frames(a, b)
    return time.time() - t0, len(m), len(a)
case("D.scale", "15,000행 병합(push_logs) 소요·행 수 불변", timed_merge, lambda r: ok(r[0] < 20 and r[1] == r[2], f"{r[0]:.1f}s, {r[1]}/{r[2]}행"))


def timed_resettle():
    import resettle_logs as R
    df = big_log(3000)
    df['price_date'] = df['run_date'].apply(cm._fallback_price_date)
    idx = pd.bdate_range("2015-12-01", periods=900)
    p_ = px(np.linspace(50, 80, 900), idx=idx)
    t0 = time.time()
    R.resettle_df(df, lambda t: p_)
    return time.time() - t0
case("D.scale", "재채점 3,000행 소요(행 단위 루프)", timed_resettle, lambda s: ok(s < 60, f"{s:.1f}s"))

# ======================================================================= E. 정합성(룩어헤드·시계)
def reject_future():
    fresh()
    now = pd.Timestamp.now()
    rows = [rec(now + pd.Timedelta(days=30), tk="SOXL", h=1, up=60, price=100.0, pdate=(now + pd.Timedelta(days=29)).normalize()),
            rec(now, tk="SOXS", h=1, up=40, price=100.0, pdate=(now - pd.Timedelta(days=1)).normalize())]
    cm.append_predictions('x', rows)
    return cm.load_log('x')
case("E.integrity", "30일 뒤 날짜(시계 오류) 행은 기록 거부", reject_future,
     lambda df: ok(len(df) == 1 and df.iloc[0]['ticker'] == 'SOXS', f"기록된 {list(df['ticker'])}"))


def reject_lookahead():
    fresh()
    r = rec("2026-10-14 09:17", tk="SOXL", h=1, up=60, price=100.0, pdate="2026-10-16")     # 가격일이 실행일보다 미래 = 룩어헤드
    cm.append_predictions('x', [r])
    return cm.load_log('x')
case("E.integrity", "가격일(price_date)이 실행일보다 미래인 룩어헤드 행은 기록 거부", reject_lookahead,
     lambda df: ok(len(df) == 0, f"그대로 기록됨 {len(df)}행"))


def settle_future_target():
    fresh()
    cm.append_predictions('x', [rec("2026-10-14 09:17", tk="SOXL", h=20, up=60, price=100.0, pdate="2026-10-13")])
    cm.fetch_ticker_data = lambda t, **k: px(np.linspace(100, 101, 5), start="2026-10-05")
    return cm.settle_predictions('x')
case("E.integrity", "아직 20거래일 안 지난 행은 정산하지 않고 대기", settle_future_target,
     lambda df: ok(df['actual_label'].isna().all()))

# ======================================================================= F. 환경
def no_token():
    os.environ.pop('TELEGRAM_TOKEN', None); os.environ.pop('TELEGRAM_CHAT_ID', None)
    return cm.send_telegram("테스트")
case("F.env", "텔레그램 토큰 없음 → 조용히 False", no_token, lambda r: ok(r is False))


def tg_timeout():
    import requests
    os.environ['TELEGRAM_TOKEN'] = 't'; os.environ['TELEGRAM_CHAT_ID'] = 'c'
    orig = cm.requests.post
    def boom(*a, **k): raise requests.exceptions.ReadTimeout("timeout")
    cm.requests.post = boom
    try:
        return cm.send_telegram("테스트")
    finally:
        cm.requests.post = orig
        os.environ.pop('TELEGRAM_TOKEN'); os.environ.pop('TELEGRAM_CHAT_ID')
case("F.env", "텔레그램 타임아웃 → 예외 없이 False", tg_timeout, lambda r: ok(r is False))


def tg_429():
    import requests
    os.environ['TELEGRAM_TOKEN'] = 't'; os.environ['TELEGRAM_CHAT_ID'] = 'c'
    class R:  # 429 → 전달 실패로 취급하되 죽지 않아야
        status_code = 429; text = '{"ok":false,"error_code":429}'
        def json(self): return {"ok": False, "error_code": 429}
    orig = cm.requests.post
    cm.requests.post = lambda *a, **k: R()
    try:
        return cm.send_telegram("테스트")
    finally:
        cm.requests.post = orig
        os.environ.pop('TELEGRAM_TOKEN'); os.environ.pop('TELEGRAM_CHAT_ID')
case("F.env", "텔레그램 429(속도 제한) → 예외 없이 False", tg_429, lambda r: ok(r is False))


def health_future_rows():
    d = fresh()
    now = pd.Timestamp.now().normalize()
    rows = [dict(rec(now - pd.Timedelta(days=20), tk=t, h=1, up=60, price=10.0, pdate=(now - pd.Timedelta(days=21)))) for t in 'AB']
    rows += [dict(rec(now + pd.Timedelta(days=400), tk='Z', h=1, up=60, price=10.0, pdate=(now + pd.Timedelta(days=399))))]
    pd.DataFrame(rows).to_csv(os.path.join(d, 'predictions_scalp.csv'), index=False)
    return hc.check_health(d, today=now, expected={'scalp': 2})
case("F.env", "미래 날짜 행 하나가 '최근 기록 정상'으로 위장하지 못함(건강점검)", health_future_rows,
     lambda iss: ok(len(iss) > 0, "미래 행 때문에 정체를 못 잡음"))

# ======================================================================= G. 모델 극단
def macro(n=900):
    idx = pd.bdate_range("2022-01-03", periods=n)
    r = np.random.default_rng(0)
    return pd.DataFrame({'QQQ': 100.0, 'SMH': 100.0, 'SPY': 100.0, 'VIX': 20 + r.normal(0, 2, n), 'TNX': 4 + r.normal(0, .1, n)}, index=idx)


def train_with(close, m):
    idx = m.index
    c = pd.Series(close, index=idx)
    cm.fetch_ticker_data = lambda t, **k: pd.DataFrame({'High': c * 1.01, 'Low': c * 0.99, 'Close': c})
    return cm.train_benchmark_model('QQQ', m, 5, 0.04, min_train_rows=300)


r0 = np.random.default_rng(5)
base_close = 100 * np.exp(np.cumsum(r0.normal(0.0004, .015, 900)))
case("G.model", "마지막 봉 -35% 폭락(분할 아님) 학습·예측", lambda: train_with(np.r_[base_close[:-1], base_close[-2] * 0.65], macro()),
     lambda r: ok(r is None or np.isfinite(r['calib_clamped'] if 'calib_clamped' in r else 50)))
m2 = macro(); m2['VIX'] = np.nan
case("G.model", "VIX 열이 통째로 NaN", lambda: train_with(base_close, m2), lambda r: ok(True))
m3 = macro().drop(columns=['TNX'])
case("G.model", "매크로 열(TNX) 누락", lambda: train_with(base_close, m3), lambda r: ok(True))
m4 = macro(); m4.iloc[-10:, :] = np.nan
case("G.model", "매크로 마지막 10행 전부 NaN(데이터 지연)", lambda: train_with(base_close, m4), lambda r: ok(True))
case("G.model", "6,000행(24년) 긴 이력 학습 90초 이내", lambda: (lambda t0: (train_with(100 * np.exp(np.cumsum(r0.normal(0.0003, .014, 6000))), macro(6000)), time.time() - t0)[1])(time.time()),
     lambda s: ok(s < 90, f"{s:.0f}s"))


# ======================================================================= H. 데이터 공급·누수 (후임 AI 제안 반영, v29)
def stale_record():
    fresh()
    cm.append_predictions('x', [rec("2026-10-16 09:17", tk="SOXL", h=5, up=60, price=100.0, pdate="2026-10-02"),   # 14일 묵은 시세
                                rec("2026-10-16 09:17", tk="SOXS", h=5, up=40, price=100.0, pdate="2026-10-15"),
                                rec("2026-10-20 09:17", tk="TQQQ", h=5, up=60, price=100.0, pdate="2026-10-14")]) if False else None
    now = pd.Timestamp.now()
    cm.append_predictions('x', [rec(now, tk="SOXL", h=5, up=60, price=100.0, pdate=(now - pd.Timedelta(days=14)).normalize()),
                                rec(now, tk="SOXS", h=5, up=40, price=100.0, pdate=(now - pd.Timedelta(days=1)).normalize()),
                                rec(now, tk="TQQQ", h=5, up=60, price=100.0, pdate=(now - pd.Timedelta(days=5)).normalize())])  # 긴 연휴 5일은 허용
    return cm.load_log('x')
case("H.data", "14일 묵은 시세로 만든 예측은 기록 거부(연휴 5일은 허용)", stale_record,
     lambda df: ok(sorted(df['ticker']) == ['SOXS', 'TQQQ'], f"기록된 {sorted(df['ticker'])}"))


def overdue_pending():
    d = fresh()
    now = pd.Timestamp.now().normalize()
    rows = [dict(rec(now - pd.Timedelta(days=80), tk='GONE', h=5, up=60, price=10.0, pdate=now - pd.Timedelta(days=81))) for _ in range(1)]
    rows += [dict(rec(now - pd.Timedelta(days=1), tk='OK', h=5, up=60, price=10.0, pdate=now - pd.Timedelta(days=2)))]
    pd.DataFrame(rows).to_csv(os.path.join(d, 'predictions_scalp.csv'), index=False)
    return hc.check_health(d, today=now, expected={'scalp': 1}, max_lag_days=4)
case("H.data", "상장폐지 등으로 45일 넘게 정산 안 된 행을 건강점검이 잡음", overdue_pending,
     lambda iss: ok(any('정산 안 된' in i for i in iss), str(iss)))


def run_with_download(df):
    cm.fetch_ticker_data = _ORIG_FETCH        # G 그룹이 바꿔 둔 가짜 시세 함수를 원복
    orig = cm._yf_download_with_retry
    cm._yf_download_with_retry = lambda *a, **k: df
    try:
        return cm.fetch_ticker_data('SOXL')
    finally:
        cm._yf_download_with_retry = orig


_idx = pd.bdate_range("2026-09-01", periods=20)
_v = np.linspace(10, 12, 20)
_flat = pd.DataFrame({'High': _v * 1.01, 'Low': _v * .99, 'Close': _v, 'Volume': 1}, index=_idx)
_multi = _flat.copy(); _multi.columns = pd.MultiIndex.from_product([_flat.columns, ['SOXL']])
_multi_sw = _multi.copy(); _multi_sw.columns = _multi_sw.columns.swaplevel(0, 1)
_adj = _flat.drop(columns=['Close']).assign(**{'Adj Close': _v})
case("H.data", "yfinance 열이 MultiIndex(('Close','SOXL'))로 바뀌어도 동일 결과", lambda: run_with_download(_multi),
     lambda r: ok(r is not None and np.allclose(r['Close'].values, _v)))
case("H.data", "yfinance 열 순서가 ('SOXL','Close')로 뒤집혀도 안전(None 또는 동일)", lambda: run_with_download(_multi_sw),
     lambda r: ok(r is None or np.allclose(r['Close'].values, _v)))
case("H.data", "Close 열 없고 Adj Close 만 있으면 폴백 사용", lambda: run_with_download(_adj),
     lambda r: ok(r is not None and np.allclose(r['Close'].values, _v)))
case("H.data", "Close/Adj Close 모두 없음 → None(예외 없음)", lambda: run_with_download(_flat[['High', 'Low']]), lambda r: ok(r is None))
case("H.data", "High/Low 열만 없음 → Close 로 대체해 동작", lambda: run_with_download(_flat[['Close']]),
     lambda r: ok(r is not None and len(r) == 20))
case("H.data", "빈 DataFrame/None 응답", lambda: (run_with_download(pd.DataFrame()), run_with_download(None)), lambda r: ok(r == (None, None)))


def macro_missing_close():
    orig = cm._yf_download_with_retry
    cm._yf_download_with_retry = lambda *a, **k: _flat[['High']]
    try:
        return cm.fetch_macro_data()
    finally:
        cm._yf_download_with_retry = orig
case("H.data", "매크로 응답에 Close 없음 → 예외 없이 빈 결과", macro_missing_close, lambda r: ok(r is not None and len(r) == 0))

# --- 누수(룩어헤드) 검정: 시점 t 의 피처는 t 이후 가격을 쓰면 안 된다 → 시계열을 t 에서 잘라도 t 의 피처가 같아야 한다.
def leak_truncation():
    rr = np.random.default_rng(11)
    n = 700
    idx = pd.bdate_range("2023-01-02", periods=n)
    c = pd.Series(100 * np.exp(np.cumsum(rr.normal(0.0003, .014, n))), index=idx)
    df = pd.DataFrame({'High': c * 1.01, 'Low': c * .99, 'Close': c})
    mc = macro(n); mc.index = idx
    full = cm.build_features(df.join(mc), benchmark_col='SPY')
    bad = []
    for cut in (300, 450, 600):
        part = cm.build_features(df.iloc[:cut].join(mc.iloc[:cut]), benchmark_col='SPY')
        cols = [c_ for c_ in cm.FEATURE_COLUMNS if c_ in full.columns and c_ in part.columns]
        a, b = full[cols].iloc[cut - 1], part[cols].iloc[-1]
        diff = [c_ for c_ in cols if not (np.isclose(a[c_], b[c_], equal_nan=True, rtol=1e-9, atol=1e-12))]
        bad += [(cut, d) for d in diff]
    return bad
case("H.leak", "피처 누수 검정: 시계열을 t에서 잘라도 t시점 피처가 동일(미래 가격 미사용)", leak_truncation,
     lambda bad: ok(not bad, f"미래에 의존하는 피처 {bad[:6]}"))


def target_tail_nan():
    rr = np.random.default_rng(12)
    n = 400
    idx = pd.bdate_range("2023-01-02", periods=n)
    c = pd.Series(100 * np.exp(np.cumsum(rr.normal(0, .014, n))), index=idx)
    df = pd.DataFrame({'High': c * 1.01, 'Low': c * .99, 'Close': c})
    out = cm.build_target(df, 5, 0.04)
    return out['Target'].iloc[-5:].isna().all(), out['Target'].iloc[:-5].notna().sum()
case("H.leak", "라벨 정렬: 마지막 h개 행의 Target 은 비어 있어야(미래 미확정)", target_tail_nan, lambda r: ok(bool(r[0]), str(r)))


def null_signal_calibration():
    """무작위 보행에서는 확률이 50% 근처여야 한다(보정 붕괴·누수 점검). 여러 시드의 평균 엣지 절댓값."""
    edges = []
    for sd in range(4):
        rr = np.random.default_rng(100 + sd)
        c = 100 * np.exp(np.cumsum(rr.normal(0, .014, 900)))
        r = train_with(c, macro())
        if r is not None:
            edges.append(abs(r['edge']))
    return edges
case("H.leak", "무작위 보행 4개: 모델 엣지 평균 절댓값 15%p 미만(허구의 강한 신호 없음)", null_signal_calibration,
     lambda e: ok(len(e) > 0 and float(np.mean(e)) < 15, f"엣지 {np.round(e, 1).tolist()}"))


# ======================================================================= R. 머신러닝·해석 (Gemini 제안 반영)
def regime_shift():
    rr = np.random.default_rng(21)
    r = np.r_[rr.normal(0.0004, .008, 650), rr.normal(-0.004, .09, 250)]      # 저변동 → 변동성 10배+하락 레짐
    return train_with(100 * np.exp(np.cumsum(r)), macro())
case("R.ml", "레짐 전환(변동성 10배·급락 추세) 직후 학습: 확률은 1~99%·유한", regime_shift,
     lambda r: ok(r is None or (1 <= r['up_prob'] <= 99 and 1 <= r['down_prob'] <= 99 and abs(r['up_prob'] + r['down_prob'] - 100) < 1e-6),
                  f"{None if r is None else (r['up_prob'], r['down_prob'])}"))


def symmetry_extremes():
    bad = []
    for p in (0, 0.01, 1, 49.99, 50, 50.01, 99, 99.99, 100):
        lu, ld = cm.apply_direction(p, 100 - p, 1)
        iu, idn = cm.apply_direction(p, 100 - p, -1)
        if abs(lu + iu - 100) > 1e-9 or abs(ld + idn - 100) > 1e-9:
            bad.append(p)
    return bad
case("R.ml", "페어 대칭성: 확률 0~100 극단에서도 롱+인버스 상승확률 = 100", symmetry_extremes, lambda b: ok(not b, f"깨진 입력 {b}"))


def overconfident_but_wrong():
    rr = np.random.default_rng(31)
    n = 400
    d = pd.DataFrame({'ticker': 'SOXL', 'horizon_days': 5, 'predicted_label': 1, 'predicted_up_prob': 92.0,
                      'actual_label': (rr.random(n) < 0.45).astype(float)})
    d['correct'] = (d['actual_label'] == d['predicted_label']).astype(float)
    d['run_date'] = pd.bdate_range('2025-01-02', periods=n)
    return cm.evaluate_signal(d, 'SOXL')
case("R.ml", "확률 92%로 자신만만하지만 실제 적중 45%인 로그 → 신호 통과 금지", overconfident_but_wrong,
     lambda r: ok(r is not None and r['status'] != 'pass', str(r and r['status'])))


def always_up_market():
    n = 400
    d = pd.DataFrame({'ticker': 'TQQQ', 'horizon_days': 1, 'predicted_label': 1, 'predicted_up_prob': 60.0,
                      'actual_label': 1.0, 'correct': 1.0, 'run_date': pd.bdate_range('2025-01-02', periods=n)})
    return cm.evaluate_signal(d, 'TQQQ')
case("R.ml", "항상 상승만 찍는 시스템이 상승장에서 100% 적중 → 실력으로 인정 금지(기준선=100%)", always_up_market,
     lambda r: ok(r is not None and r['status'] != 'pass' and r['one_sided'], str(r)))


def zigzag_drag():
    """기초지수 ±2% 지그재그 → 3배 레버리지는 횡보인데 손실(변동성 끌림). 정산이 ETF 자체 가격으로 되는지 확인."""
    idx = pd.bdate_range('2026-01-02', periods=60)
    under = 100 * np.cumprod(np.where(np.arange(60) % 2 == 0, 1.02, 1 / 1.02))
    lev = 100 * np.cumprod(1 + 3 * (pd.Series(under).pct_change().fillna(0).values))
    res = cm.compute_aligned_outcome(px(lev, idx=idx), idx[0], 20, lev[0])
    und = cm.compute_aligned_outcome(px(under, idx=idx), idx[0], 20, under[0])
    return res, und
case("R.ml", "변동성 끌림: 지수는 ±0 횡보여도 3배 ETF 20일 정산은 ETF 가격 기준(지수가 아닌)으로 계산", zigzag_drag,
     lambda r: ok(r[0] is not None and abs(r[1]['actual_return']) < 1e-9 < 1 and r[0]['actual_return'] < -0.01, f"ETF {r[0]['actual_return']:.3f} / 지수 {r[1]['actual_return']:.3f}"))

# ----------------------------------------------------------------------- 요약
order = {'CRASH': 0, 'SILENT': 1, 'OK': 2}
for g, n_, v, memo, dt in RESULTS:
    print(f"[{v}] {g:12s} {n_}" + (f"  ← {memo}" if memo else "") + f"  ({dt:.1f}s)")
cnt = {k: sum(1 for r in RESULTS if r[2] == k) for k in order}
print(f"\n총 {len(RESULTS)}건: OK {cnt['OK']} / SILENT {cnt['SILENT']} / CRASH {cnt['CRASH']}")
sys.exit(0 if cnt['SILENT'] + cnt['CRASH'] == 0 else 1)
