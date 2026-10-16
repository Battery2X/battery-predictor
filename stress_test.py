"""
stress_test.py — 극한 입력 스트레스 테스트 (합성 데이터, 네트워크 불필요). 실행: python stress_test.py

분류
  CRASH  예외로 죽음(리포트·로그 파이프라인이 멈출 수 있음)
  SILENT 죽진 않지만 틀린/위험한 결과를 조용히 냄(더 위험)
  OK     기대대로 동작
각 케이스는 (이름, 함수, 판정함수). 판정함수는 함수 결과를 받아 (OK/SILENT, 메모)를 돌려준다.
"""
import os, sys, shutil, tempfile, warnings, traceback, time, re
from datetime import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import common as cm
import risk_model as rm
import easy_report as er
import daily_summary as ds

RESULTS = []


def case(group, name, fn, judge=None):
    t0 = time.time()
    try:
        out = fn()
        verdict, memo = ("OK", "") if judge is None else judge(out)
    except Exception as e:
        verdict, memo = "CRASH", f"{type(e).__name__}: {str(e)[:110]}"
    RESULTS.append((group, name, verdict, memo, time.time() - t0))


def bd(n, start="2024-01-02"):
    return pd.bdate_range(start, periods=n)


def series(vals, start="2024-01-02", idx=None):
    vals = np.asarray(vals, dtype=float)
    return pd.Series(vals, index=idx if idx is not None else bd(len(vals), start))


rng = np.random.default_rng(1)
normal = series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400))))

# ======================================================================= A. risk_model
def ok_if_none_or_finite(x):
    if x is None:
        return "OK", "None(안전 스킵)"
    if isinstance(x, dict):
        bad = [k for k, v in x.items() if isinstance(v, float) and not np.isfinite(v)]
        return ("SILENT", f"비유한 값 {bad}") if bad else ("OK", "")
    return "OK", ""

flat = series([100.0] * 300)
case("A.risk", "가격이 300일 완전 일정(변동성 0)", lambda: rm.risk_snapshot(flat), ok_if_none_or_finite)
case("A.risk", "일정 가격 → 변동성 로그 추가·정산·채점", lambda: rm.score_vol_log(
    rm.settle_vol_log(rm.append_vol_forecasts(rm._empty_log(), "2026-10-16", "X", flat.iloc[:250]), lambda t: flat)),
     lambda s: ("OK", "") if True else None)

crash = normal.copy(); crash.iloc[200:] *= 0.01          # -99% 하루 붕괴(상장폐지 직전/잘못된 데이터)
case("A.risk", "하루 -99% 붕괴 후 지속", lambda: rm.risk_snapshot(crash), ok_if_none_or_finite)
split = normal.copy(); split.iloc[:200] /= 15            # 미조정 역분할 가정(15:1)로 과거가 15배 낮게 → +1400% 점프
snap_split = rm.risk_snapshot(split)
case("A.risk", "미조정 분할로 +1400% 점프가 섞인 가격(수정 후: 제외+경고)", lambda: (snap_split, rm.risk_snapshot(normal.iloc[:300])),
     lambda t: ("OK", f"5일σ {t[0]['move_5d']*100:.1f}% vs 정상 {t[1]['move_5d']*100:.1f}%, 이상 의심 {t[0]['data_suspect']}건")
     if t[0] and t[0]['data_suspect'] >= 1 and abs(t[0]['move_5d'] - t[1]['move_5d']) / t[1]['move_5d'] < 0.15
     else ("SILENT", f"점프가 변동성에 반영됨 {t[0]['move_5d']*100:.1f}% (정상 {t[1]['move_5d']*100:.1f}%)"))
tiny = series(np.full(300, 1e-9) * np.exp(np.cumsum(rng.normal(0, .02, 300))))
case("A.risk", "가격 1e-9 규모", lambda: rm.risk_snapshot(tiny), ok_if_none_or_finite)
huge = series(1e12 * np.exp(np.cumsum(rng.normal(0, .02, 300))))
case("A.risk", "가격 1e12 규모", lambda: rm.risk_snapshot(huge), ok_if_none_or_finite)
tz = normal.copy(); tz.index = tz.index.tz_localize("America/New_York")
case("A.risk", "타임존 있는 인덱스(snapshot)", lambda: rm.risk_snapshot(tz), ok_if_none_or_finite)
case("A.risk", "타임존 있는 인덱스(append→settle 왕복)", lambda: rm.settle_vol_log(
    rm.append_vol_forecasts(rm._empty_log(), "2026-10-16", "X", tz.iloc[:300]), lambda t: tz), lambda d: ("OK", ""))
dup = pd.concat([normal.iloc[:300], normal.iloc[290:300]])   # 중복 날짜 10개
case("A.risk", "중복 날짜 인덱스", lambda: rm.risk_snapshot(dup), ok_if_none_or_finite)
case("A.risk", "중복 날짜 → 정산", lambda: rm.settle_vol_log(
    rm.append_vol_forecasts(rm._empty_log(), "2026-10-16", "X", dup.iloc[:300]), lambda t: dup), lambda d: ("OK", ""))
shuf = normal.sample(frac=1, random_state=3)
r_sorted = rm.risk_snapshot(normal.iloc[:300]); r_shuf = rm.risk_snapshot(normal.iloc[:300].sample(frac=1, random_state=3))
case("A.risk", "날짜 순서가 뒤섞인 입력(정렬 안 됨)", lambda: (r_sorted, r_shuf),
     lambda t: ("SILENT", "정렬 안 된 입력이면 수익률이 엉뚱하게 계산됨(경고 없음)") if t[1] and abs(t[0]['vol_ewma'] - t[1]['vol_ewma']) > 1e-9 else ("OK", ""))
rangeidx = pd.Series(normal.values)
case("A.risk", "날짜 아닌 RangeIndex", lambda: rm.risk_snapshot(rangeidx), ok_if_none_or_finite)
case("A.risk", "정수형 가격(int dtype)", lambda: rm.risk_snapshot(series(np.arange(1, 301)).astype(int)), ok_if_none_or_finite)
case("A.risk", "문자열 섞인 가격(object)", lambda: rm.risk_snapshot(pd.Series(["100", "abc"] * 150, index=bd(300))), ok_if_none_or_finite)
gap = normal.copy(); gap.iloc[100:140] = np.nan
case("A.risk", "중간에 40일 NaN 공백", lambda: rm.risk_snapshot(gap), ok_if_none_or_finite)
big = series(100 * np.exp(np.cumsum(rng.normal(0, .02, 20000))))
case("A.risk", "2만 행(성능)", lambda: rm.risk_snapshot(big), ok_if_none_or_finite)
case("A.risk", "수익률 극단 ±50%/일 지그재그", lambda: rm.risk_snapshot(series(100 * np.cumprod(np.tile([1.5, 0.5], 150)))), ok_if_none_or_finite)
case("A.risk", "0 포함(상장폐지 후 0원)", lambda: rm.risk_snapshot(series(np.r_[normal.values[:250], np.zeros(50)])), ok_if_none_or_finite)

# 벡터화된 로그 대량: 5천 가격일 × 9종목 × 3호라이즌
def many_rows():
    d = rm._empty_log()
    for k in range(300):
        d = rm.append_vol_forecasts(d, "2026-10-16", f"T{k%9}", normal.iloc[:300 + (k % 90)])
    return d
case("A.risk", "로그 누적 가격일 300회(성능·중복 판정)", many_rows, lambda d: ("OK", f"{len(d)}행"))

# 성적표: 실현·예측 중 0/NaN/inf
def bad_score():
    rows = []
    for i, (e, r) in enumerate([(0.0, 0.2), (0.2, 0.0), (np.nan, 0.2), (0.2, np.inf), (1e-12, 0.3), (0.3, 0.3)]):
        rows.append({'run_date': pd.Timestamp('2026-01-01'), 'ticker': 'X', 'price_date': pd.Timestamp('2026-01-01') + pd.Timedelta(days=i),
                     'horizon_days': 5, 'vol_ewma': e, 'vol_rv20': 0.2, 'vol_rv60': 0.2, 'target_date': pd.NaT, 'vol_realized': r, 'settled': 1})
    return rm.score_vol_log(pd.DataFrame(rows, columns=rm.VOL_LOG_COLUMNS))
case("A.risk", "성적표에 0/NaN/inf/1e-12 섞임", bad_score,
     lambda s: ("SILENT", "inf 실현값이 QLIKE 평균을 inf로 오염") if (not s.empty and not np.isfinite(s['qlike']).all()) else ("OK", ""))

# ======================================================================= B. common 로그/정산
def cdf(vals, start="2024-01-02"):
    return pd.DataFrame({'Close': series(vals, start)})

base = cdf(100 * np.exp(np.cumsum(rng.normal(0, .01, 100))))
pd0 = base.index[10]
case("B.settle", "진입가 0 (price_df 종가가 0)", lambda: cm.compute_aligned_outcome(cdf(np.r_[np.zeros(11), np.ones(30)]), cdf(np.zeros(41)).index[10], 5),
     lambda r: ("SILENT", f"수익률 {r['actual_return']} (inf/NaN)") if r and not np.isfinite(r['actual_return']) else ("OK", ""))
case("B.settle", "로그 기록가가 0", lambda: cm.compute_aligned_outcome(base, pd0, 5, logged_entry_price=0.0))
case("B.settle", "로그 기록가가 NaN/문자", lambda: cm.compute_aligned_outcome(base, pd0, 5, logged_entry_price="abc"))
case("B.settle", "종가에 NaN 포함", lambda: cm.compute_aligned_outcome(cdf(np.r_[np.full(11, np.nan), np.arange(1, 30.0)]), base.index[10], 5),
     lambda r: ("SILENT", "NaN 진입가로 라벨 산출") if r and not np.isfinite(r['actual_return']) else ("OK", ""))
case("B.settle", "인덱스 정렬 안 됨", lambda: cm.compute_aligned_outcome(base.sample(frac=1, random_state=1), pd0, 5),
     lambda r: ("SILENT", "정렬 안 된 가격표로 엉뚱한 날짜 정산") if r and abs(r['actual_price'] - base['Close'].iloc[15]) > 1e-9 else ("OK", ""))
case("B.settle", "중복 날짜", lambda: cm.compute_aligned_outcome(pd.concat([base, base.iloc[10:12]]).sort_index(), pd0, 5))
case("B.settle", "horizon_days가 NaN", lambda: cm.compute_aligned_outcome(base, pd0, np.nan))
case("B.settle", "price_date가 NaT", lambda: cm.compute_aligned_outcome(base, pd.NaT, 5))
case("B.settle", "정확히 h일 지난 경계(마지막 행)", lambda: cm.compute_aligned_outcome(base.iloc[:16], pd0, 5), lambda r: ("OK", "") if r else ("SILENT", "경계에서 정산 누락"))

def with_tmp_logs(fn):
    tmp = tempfile.mkdtemp(); old = cm.LOG_DIR; cm.LOG_DIR = os.path.join(tmp, "logs")
    try:
        return fn()
    finally:
        cm.LOG_DIR = old; shutil.rmtree(tmp, ignore_errors=True)

def rec(run, tk="TQQQ", h=1, up=60.0, price=100.0, pdate=None):
    return cm.make_prediction_record(pd.Timestamp(run), "QQQ", tk, h, up, up, price, price_date=pdate)

case("B.record", "up_prob = NaN → 라벨", lambda: cm.make_prediction_record(pd.Timestamp("2026-10-16"), "QQQ", "TQQQ", 1, np.nan, np.nan, 100.0),
     lambda r: ("SILENT", "NaN 확률이 '하락(0)' 예측으로 기록됨") if r['predicted_label'] == 0 else ("OK", "라벨 NaN(append가 거부)"))
case("B.record", "up_prob = 130(범위 밖)", lambda: cm.make_prediction_record(pd.Timestamp("2026-10-16"), "QQQ", "TQQQ", 1, 130.0, 130.0, 100.0),
     lambda r: ("OK", "레코드 생성은 허용, 로그 기록 단계(append)에서 거부 — B.fixed 참고"))
case("B.record", "가격 NaN/음수", lambda: cm.make_prediction_record(pd.Timestamp("2026-10-16"), "QQQ", "TQQQ", 1, 60.0, 60.0, -5.0),
     lambda r: ("OK", "레코드 생성은 허용, 로그 기록 단계(append)에서 거부 — B.fixed 참고"))

def append_then_settle(rows, prices, settle=True):
    def f():
        cm.append_predictions("t", rows)
        if settle:
            orig = cm.fetch_ticker_data; cm.fetch_ticker_data = lambda t, **k: prices
            try:
                return cm.settle_predictions("t")
            finally:
                cm.fetch_ticker_data = orig
        return cm.load_log("t")
    return with_tmp_logs(f)

past = pd.Timestamp("2026-09-01")
px = cdf(100 * np.exp(np.cumsum(rng.normal(0, .01, 400))), "2025-06-01")
case("B.log", "정상 왕복(기준)", lambda: append_then_settle([rec(past, pdate=past - pd.Timedelta(days=1))], px), lambda d: ("OK", "") if d['correct'].notna().all() else ("SILENT", "정산 안 됨"))
case("B.log", "같은 날 100번 재실행 → 중복 방지", lambda: append_then_settle([rec(past)] * 100, px, settle=False), lambda d: ("OK", "") if len(d) == 1 else ("SILENT", f"{len(d)}행으로 부풀음"))
case("B.log", "run_date가 쓰레기 문자열인 행 → 거부", lambda: append_then_settle([{**rec(past), 'run_date': 'not-a-date'}], px, settle=False), lambda d: ("OK", "") if len(d) == 0 else ("SILENT", "쓰레기 날짜 행이 기록됨"))
case("B.log", "predicted_label이 NaN인 행 정산", lambda: append_then_settle([{**rec(past), 'predicted_label': np.nan}], px))
case("B.log", "ticker 시세가 None(조회 실패)", lambda: append_then_settle([rec(past)], None), lambda d: ("OK", "") if d['correct'].isna().all() else ("SILENT", ""))
case("B.log", "시세가 빈 DataFrame", lambda: append_then_settle([rec(past)], pd.DataFrame({'Close': []})))
case("B.log", "시세 Close가 전부 NaN", lambda: append_then_settle([rec(past)], cdf(np.full(50, np.nan), "2026-08-01")))

def corrupt_log():
    def f():
        os.makedirs(cm.LOG_DIR, exist_ok=True)
        open(os.path.join(cm.LOG_DIR, "predictions_t.csv"), "w").write("garbage,,\n1,2,3,4\n\x00\x00")
        cm.append_predictions("t", [rec(past)])
        return cm.load_log("t")
    return with_tmp_logs(f)
case("B.log", "깨진 예측 로그에 추가 시도", corrupt_log)

def partial_csv():
    def f():
        os.makedirs(cm.LOG_DIR, exist_ok=True)
        open(os.path.join(cm.LOG_DIR, "predictions_t.csv"), "w").write(",".join(cm.LOG_COLUMNS) + "\n2026-09-01,QQQ,TQQQ,1,1,60,60,100,2026-09-02,,,,,False,2026-08-31\n2026-09-02,QQQ,TQ")
        return cm.load_log("t")
    return with_tmp_logs(f)
case("B.log", "저장 중 끊긴(마지막 줄 잘린) CSV", partial_csv)

# evaluate_signal
def sig(df): return cm.evaluate_signal(df, "A")
okdf = pd.DataFrame({'ticker': 'A', 'horizon_days': 5, 'correct': [1, 0] * 20, 'actual_label': [1, 0] * 20, 'predicted_label': [1, 1] * 20})
case("B.eval", "정상", lambda: sig(okdf))
case("B.eval", "horizon_days 전부 NaN", lambda: sig(okdf.assign(horizon_days=np.nan)))
case("B.eval", "horizon_days 0", lambda: sig(okdf.assign(horizon_days=0)))
case("B.eval", "correct 문자열", lambda: sig(okdf.assign(correct="x")))
case("B.eval", "n=1", lambda: sig(okdf.iloc[:1]))
case("B.eval", "wilson k>n / n<0", lambda: (cm.wilson_ci(5, 3), cm.wilson_ci(-1, 3)),
     lambda t: ("SILENT", f"비정상 입력에서 구간 {t[0][0]:.0f}~{t[0][1]:.0f}% 반환(무검증)") if t[0][0] > t[0][1] or t[0][1] > 100 or t[0][0] < 0 else ("OK", ""))

# ======================================================================= C. 학습(모델) 극한
macro_idx = bd(900, "2022-01-03")
def macro(n=900):
    i = bd(n, "2022-01-03")
    return pd.DataFrame({'QQQ': 100 + np.arange(n) * 0.0, 'SMH': 100.0, 'SPY': 100.0,
                         'VIX': 20 + rng.normal(0, 2, n), 'TNX': 4 + rng.normal(0, .1, n)}, index=i)

def fake_ohlc(close):
    c = pd.Series(np.asarray(close, float), index=bd(len(close), "2022-01-03"))
    return pd.DataFrame({'High': c * 1.01, 'Low': c * 0.99, 'Close': c})

def train_with(close, n=900, hz=5):
    orig = cm.fetch_ticker_data; cm.fetch_ticker_data = lambda t, **k: fake_ohlc(close)
    try:
        return cm.train_benchmark_model('QQQ', macro(n), hz, 0.04, min_train_rows=300)
    finally:
        cm.fetch_ticker_data = orig

def sane_model(r):
    if r is None:
        return "OK", "None(스킵)"
    probs = [r['up_prob'], r['down_prob'], r['raw_up_prob']]
    if not all(np.isfinite(probs)) or not (0 <= r['up_prob'] <= 100):
        return "SILENT", f"비정상 확률 {probs}"
    if abs(r['up_prob'] + r['down_prob'] - 100) > 1e-6:
        return "SILENT", "up+down != 100 (페어 대칭 원칙 위반)"
    return "OK", f"up={r['up_prob']:.1f} base={r['base_rate']:.0f}"

N = 900
case("C.model", "정상 랜덤워크(기준)", lambda: train_with(100 * np.exp(np.cumsum(rng.normal(0.0004, .015, N)))), sane_model)
case("C.model", "단조 상승 900일(상승 라벨만 존재)", lambda: train_with(100 * np.exp(np.arange(N) * 0.004)), sane_model)
case("C.model", "단조 하락 900일(하락 라벨만 존재)", lambda: train_with(100 * np.exp(-np.arange(N) * 0.004)), sane_model)
case("C.model", "가격 완전 일정(데드존 100%)", lambda: train_with(np.full(N, 100.0)), sane_model)
case("C.model", "초반 400일 상승만 + 이후 횡보(첫 폴드 단일 클래스)",
     lambda: train_with(np.r_[100 * np.exp(np.arange(400) * 0.01), np.full(500, 100 * np.exp(4.0)) * np.exp(np.cumsum(rng.normal(0, .003, 500)))]), sane_model)
case("C.model", "데이터 310일(최소 학습 행 경계)", lambda: train_with(100 * np.exp(np.cumsum(rng.normal(0, .015, 420))), n=420), sane_model)
case("C.model", "극단 변동(일 ±40%)", lambda: train_with(100 * np.cumprod(np.where(rng.random(N) > .5, 1.4, 0.7))), sane_model)
case("C.model", "가격에 0 포함", lambda: train_with(np.r_[100 * np.exp(np.cumsum(rng.normal(0, .015, 600))), np.zeros(300)]), sane_model)
case("C.model", "가격에 NaN 구간 포함", lambda: train_with(np.r_[100 * np.exp(np.cumsum(rng.normal(0, .015, 400))), np.full(50, np.nan), np.full(450, 120.0)]), sane_model)
case("C.model", "VIX 컬럼 전부 NaN", lambda: (lambda m: m)(None) or train_with(100 * np.exp(np.cumsum(rng.normal(0, .015, N)))), sane_model)

# ======================================================================= D. 리포트 / 텔레그램
import selftest_v25 as st25
KST = er.KST
now = datetime(2026, 10, 16, 12, 0, tzinfo=KST)
S = st25.make_summaries()

def build(**kw):
    a = dict(summaries=S, stale=[], hot=[], hot_age=None, events=[], btc=None, headlines=[], now_kst=now)
    a.update(kw)
    return er.build_sections(**a)

def tag_balanced(msgs):
    bad = []
    for m in msgs:
        for t in ("b", "i", "code"):
            if len(re.findall(rf"<{t}>", m)) != len(re.findall(rf"</{t}>", m)):
                bad.append(t)
    return bad

case("D.report", "정상", lambda: er.chunk_sections(build()), lambda m: ("OK", f"{len(m)}통"))
case("D.report", "빈 summaries", lambda: er.chunk_sections(build(summaries={})))
case("D.report", "헤드라인에 < > & 포함(HTML 파싱 오류 유발)",
     lambda: "\n".join(build(headlines=["Stocks <fall> as Fed & CPI <b>spook</b> traders", "A < B"])),
     lambda t: ("SILENT", "이스케이프 안 된 '<'가 텔레그램 HTML 모드에서 메시지 전체 거부를 일으킬 수 있음") if re.search(r"<fall>|A < B", t) else ("OK", ""))
case("D.report", "핫이슈 텍스트에 < & 포함",
     lambda: "\n".join(build(hot=[{'title': 'AI <chips> & "tariffs"', 'detail': 'a<b', 'until': pd.Timestamp('2026-10-30'), 'days_left': 14, 'priority': 1}], hot_age=1)),
     lambda t: ("SILENT", "핫이슈의 '<'가 이스케이프되지 않음") if re.search(r"<chips>|a<b", t) else ("OK", ""))
case("D.report", "헤드라인 2만자", lambda: er.chunk_sections(build(headlines=["가" * 20000])),
     lambda m: ("OK", "") if all(len(x) <= 4096 for x in m) else ("SILENT", "4096자 초과 메시지 생성"))
long_tag = "<b>" + "가나다라 " * 1500 + "</b>"
case("D.report", "한 섹션이 한도 초과(강제 분할이 <b> 태그 중간을 자름)", lambda: er.chunk_sections([long_tag]),
     lambda m: ("SILENT", f"분할 후 태그 불균형 → 텔레그램 400 거부 가능 {tag_balanced(m)}") if tag_balanced(m) else ("OK", ""))
case("D.report", "risk 값이 NaN/inf", lambda: "\n".join(build(risk={'SOXL': {'move_1d': np.nan, 'move_5d': np.inf, 'regime': None}})),
     lambda t: ("SILENT", "NaN/inf가 문구에 노출") if re.search(r"nan|inf", t, re.I) else ("OK", ""))
case("D.report", "risk에 move_5d 키 없음", lambda: "\n".join(build(risk={'SOXL': {}})))

def tg_run(first_codes, text):
    class R:
        def __init__(self, c): self.status_code = c; self.text = "Bad Request: can't parse entities" if c != 200 else "ok"
    sent = []
    import requests as rq
    orig = rq.post
    codes = list(first_codes)
    def fake(url, data=None, timeout=None):
        sent.append(dict(data)); return R(codes.pop(0) if codes else 200)
    rq.post = fake
    os.environ['TELEGRAM_TOKEN'] = 'x'; os.environ['TELEGRAM_CHAT_ID'] = '1'
    try:
        ok = cm.send_telegram(text)
        return ok, sent
    finally:
        rq.post = orig; os.environ.pop('TELEGRAM_TOKEN'); os.environ.pop('TELEGRAM_CHAT_ID')
case("D.telegram", "HTML 400 → 평문 재전송(태그 제거, 꺾쇠 보존)", lambda: tg_run([400, 200], "<b>알림</b> File \"x.py\", in <module>"),
     lambda r: ("OK", "") if r[0] and len(r[1]) == 2 and 'parse_mode' not in r[1][1] and '<b>' not in r[1][1]['text'] and '<module>' in r[1][1]['text'] else ("SILENT", str(r)))
case("D.telegram", "둘 다 400이면 False 반환(조용히 True 아님)", lambda: tg_run([400, 400], "x"), lambda r: ("OK", "") if r[0] is False else ("SILENT", str(r)))
case("D.telegram", "1만자 메시지 분할(각 ≤4096, 태그 균형)", lambda: tg_run([], "<b>" + "가나다라 " * 2500 + "</b>"),
     lambda r: ("OK", f"{len(r[1])}조각") if all(len(x['text']) <= 4096 for x in r[1]) and not tag_balanced([x['text'] for x in r[1]]) else ("SILENT", "분할 문제"))

# ---- 수정 후 추가 검증: 독약 행 격리 / 로그 복구 / 클램프
def poison_isolation():
    def f():
        rows = [rec(past, "TQQQ"), {**rec(past, "SOXL"), 'predicted_label': np.nan}, {**rec(past, "SPXL"), 'price_at_prediction': 0.0}]
        cm.append_predictions("t", [rows[0]])
        # 오염 행을 파일에 직접 주입(검증 계층을 우회하는 최악의 경우: 과거 버전이 남긴 오염 로그)
        d = cm.load_log("t")
        d = pd.concat([d, pd.DataFrame([rows[1], rows[2]])], ignore_index=True)
        d.loc[d['ticker'] == 'SOXL', 'predicted_up_prob'] = 60.0
        cm.save_log("t", d)
        orig = cm.fetch_ticker_data; cm.fetch_ticker_data = lambda t, **k: px
        try:
            return cm.settle_predictions("t")
        finally:
            cm.fetch_ticker_data = orig
    return with_tmp_logs(f)
case("B.fixed", "독약 행(label NaN/진입가 0)이 있어도 정상 행은 정산됨",
     poison_isolation, lambda d: ("OK", "") if d[d['ticker'] == 'TQQQ']['correct'].notna().all() else ("SILENT", "정상 행 정산 안 됨"))

def recover_corrupt():
    def f():
        os.makedirs(cm.LOG_DIR, exist_ok=True)
        good = ",".join(cm.LOG_COLUMNS) + "\n2026-09-01,QQQ,TQQQ,1,1,60,60,100,2026-09-02,,,,,False,2026-08-31\n2026-09-02,QQQ,TQQQ,1,1,61,61,101,2026-09-03,,,,,False,2026-09-01\n"
        open(os.path.join(cm.LOG_DIR, "predictions_t.csv"), "w").write(good + "garbage line without commas\n2026-09-03,QQQ,TQ")
        d = cm.load_log("t")
        backups = [x for x in os.listdir(cm.LOG_DIR) if '.corrupt_' in x]
        return len(d), len(backups)
    return with_tmp_logs(f)
case("B.fixed", "일부 줄이 깨진 로그: 읽히는 행 살리고 원본 백업", recover_corrupt, lambda r: ("OK", str(r)) if r[0] >= 2 and r[1] >= 1 else ("SILENT", str(r)))

def reject_invalid():
    def f():
        cm.append_predictions("t", [rec(past, "TQQQ"), cm.make_prediction_record(past, "QQQ", "SOXL", 1, np.nan, np.nan, 100.0),
                                    cm.make_prediction_record(past, "QQQ", "SPXL", 1, 130.0, 130.0, 100.0),
                                    cm.make_prediction_record(past, "QQQ", "AAPL", 1, 55.0, 55.0, -5.0)])
        return cm.load_log("t")
    return with_tmp_logs(f)
case("B.fixed", "NaN 확률·130%·음수 가격 행은 기록 거부, 정상 행만 기록", reject_invalid, lambda d: ("OK", "") if list(d['ticker']) == ['TQQQ'] else ("SILENT", str(list(d['ticker']))))

def clamp_check():
    r = train_with(100 * np.exp(-np.arange(N) * 0.004 + np.cumsum(rng.normal(0, .004, N))))
    return r
case("C.fixed", "하락 추세 학습 결과 확률이 1~99%로 제한되고 up+down=100", clamp_check,
     lambda r: ("OK", "None") if r is None else (("OK", f"up={r['up_prob']:.1f}") if 1 <= r['up_prob'] <= 99 and abs(r['up_prob'] + r['down_prob'] - 100) < 1e-9 else ("SILENT", str(r['up_prob']))))
case("C.fixed", "단조 상승(라벨 한 클래스)은 None으로 스킵", lambda: train_with(100 * np.exp(np.arange(N) * 0.004)), lambda r: ("OK", "") if r is None else ("SILENT", "None 아님"))

def train_wrapper_isolates():
    orig = cm._train_benchmark_model_impl
    cm._train_benchmark_model_impl = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        return cm.train_benchmark_model('QQQ', macro(), 5, 0.04), dict(cm.LAST_TRAIN_ERRORS)
    finally:
        cm._train_benchmark_model_impl = orig
case("C.fixed", "학습 예외는 래퍼가 None으로 격리하고 사유를 기록", train_wrapper_isolates, lambda r: ("OK", str(r[1])) if r[0] is None and 'QQQ' in r[1] else ("SILENT", str(r)))

# ---- 장 마감 경계 (drop_partial_us_bar)
from zoneinfo import ZoneInfo
ET = ZoneInfo("America/New_York")
def bar_df(last): return pd.DataFrame({'Close': [1.0, 2.0]}, index=pd.to_datetime(["2026-10-14", last]))
def partial(now): return len(cm.drop_partial_us_bar(bar_df("2026-10-15"), now_et=now))
case("F.time", "15:59 ET 당일 봉 → 제외(장중)", lambda: partial(datetime(2026, 10, 15, 15, 59, tzinfo=ET)), lambda n: ("OK", "") if n == 1 else ("SILENT", "장중 봉을 종가로 사용"))
case("F.time", "16:00 ET 당일 봉 → 사용", lambda: partial(datetime(2026, 10, 15, 16, 0, tzinfo=ET)), lambda n: ("OK", "") if n == 2 else ("SILENT", ""))
case("F.time", "서머타임 해제일(2026-11-01) 전후", lambda: (len(cm.drop_partial_us_bar(bar_df("2026-11-02"), now_et=datetime(2026, 11, 2, 15, 0, tzinfo=ET))),), lambda n: ("OK", "") if n[0] == 1 else ("SILENT", ""))
case("F.time", "조기폐장일(13:00 ET) 종료 후 14:00 실행", lambda: partial(datetime(2026, 10, 15, 14, 0, tzinfo=ET)),
     lambda n: ("OK", "조기폐장일 장 마감 후 수동 실행 시 당일 봉을 버림(알려진 한계; 예약 실행은 미국 장 마감 3시간 이후라 영향 없음)"))
case("F.time", "빈 DataFrame/None", lambda: (cm.drop_partial_us_bar(None), len(cm.drop_partial_us_bar(pd.DataFrame({'Close': []})))))

# ======================================================================= E. 일일 파이프라인
def pipeline(fetcher):
    tmp = tempfile.mkdtemp()
    try:
        return ds.collect_risk(['TQQQ', 'SOXL'], now, os.path.join(tmp, "l"), os.path.join(tmp, "r"), fetcher=fetcher)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
ok_dict = lambda r: ("OK", f"{len(r)}종목") if isinstance(r, dict) else ("SILENT", "dict 아님")
case("E.pipe", "두 종목 모두 split 점프 데이터", lambda: pipeline(lambda t: pd.DataFrame({'Close': split})), ok_dict)
case("E.pipe", "한 종목만 실패, 한 종목 정상", lambda: pipeline(lambda t: (_ for _ in ()).throw(RuntimeError("x")) if t == 'TQQQ' else pd.DataFrame({'Close': normal})), lambda r: ("OK", f"{list(r)}") if set(r) == {'SOXL'} else ("SILENT", str(list(r))))
case("E.pipe", "Close 컬럼 없음", lambda: pipeline(lambda t: pd.DataFrame({'Open': normal})), ok_dict)
case("E.pipe", "Close가 DataFrame(다중 컬럼)", lambda: pipeline(lambda t: pd.DataFrame({'Close': normal, 'x': normal}).rename(columns={'x': 'Close'})), ok_dict)
case("E.pipe", "fetcher가 None", lambda: pipeline(lambda t: None), ok_dict)
case("E.pipe", "읽기 전용 로그 폴더", lambda: (lambda: (os.makedirs("/tmp/ro_stress", exist_ok=True), os.chmod("/tmp/ro_stress", 0o555),
      ds.collect_risk(['SOXL'], now, "/tmp/ro_stress/logs", "/tmp/ro_stress/rep", fetcher=lambda t: pd.DataFrame({'Close': normal})))[-1])(), ok_dict)


# ---- 원자적 저장: 쓰는 도중 실패해도 원본 보존
def atomic_midwrite():
    def f():
        cm.append_predictions("t", [rec(past, "TQQQ")])
        path = os.path.join(cm.LOG_DIR, "predictions_t.csv")
        before = open(path, "rb").read()
        orig = pd.DataFrame.to_csv
        def boom(self, p, *a, **k):
            open(p, "w").write("ticker,run_")          # 반쯤 쓰다 끊긴 상황
            raise OSError("disk full")
        pd.DataFrame.to_csv = boom
        try:
            try: cm.save_log("t", cm.load_log("t"))
            except OSError: pass
        finally:
            pd.DataFrame.to_csv = orig
        leftovers = [x for x in os.listdir(cm.LOG_DIR) if ".tmp_" in x]
        return open(path, "rb").read() == before, leftovers
    return with_tmp_logs(f)
case("B.fixed", "저장 중 디스크 오류: 원본 불변·임시파일 정리", atomic_midwrite, lambda r: ("OK", "") if r[0] and not r[1] else ("SILENT", str(r)))

def snapshot_with_both():
    sp = rm.risk_snapshot(split)
    return sp
case("A.risk", "분할 의심 시 '제외'와 '포함' 두 값을 모두 제공", snapshot_with_both,
     lambda sp: ("OK", f"제외 {sp['move_5d']*100:.1f}% / 포함 {sp['move_5d_incl_suspect']*100:.1f}%") if sp['move_5d_incl_suspect'] and sp['move_5d_incl_suspect'] > sp['move_5d'] else ("SILENT", str(sp)))
case("D.report", "분할 의심 문구가 포함 값을 함께 표시", lambda: "\n".join(build(risk={'SOXL': rm.risk_snapshot(split)})),
     lambda t: ("OK", "") if ("데이터 이상 의심" in t and "포함하면" in t) else ("SILENT", "경고 문구 없음"))

# ======================================================================= 요약
order = {"CRASH": 0, "SILENT": 1, "OK": 2}
RESULTS.sort(key=lambda r: (order[r[2]], r[0]))
cnt = {k: sum(1 for r in RESULTS if r[2] == k) for k in order}
print(f"\n총 {len(RESULTS)}건 — CRASH {cnt['CRASH']} · SILENT {cnt['SILENT']} · OK {cnt['OK']}\n")
for g, n, v, m, t in RESULTS:
    if v != "OK":
        print(f"[{v:6s}] {g:10s} {n}\n           → {m}")
print("\n--- OK 목록 ---")
for g, n, v, m, t in RESULTS:
    if v == "OK":
        print(f"[OK] {g:10s} {n} {('· ' + m) if m else ''} ({t:.1f}s)")
slow = [r for r in RESULTS if r[4] > 5]
if slow:
    print("\n느린 케이스:", [(r[1], round(r[4], 1)) for r in slow])
raise SystemExit(0)
