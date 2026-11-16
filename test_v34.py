"""test_v34.py — v34 읽기 전용 도구(방향 성적표·외부 종가 대조)와 핫이슈 갱신 검증(합성 데이터). 방향 모델·피처·임계값·로그 스키마는 건드리지 않는다."""
import hashlib, json, os, tempfile
import numpy as np
import pandas as pd
import audit_external_prices as ax
import direction_scorecard as ds
import easy_report as er

fails = []
def ck(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)

COLS = ['run_date', 'benchmark', 'ticker', 'horizon_days', 'predicted_label', 'predicted_up_prob', 'raw_up_prob',
        'price_at_prediction', 'target_date', 'actual_price', 'actual_return', 'actual_label', 'correct', 'near_earnings', 'price_date']
days = pd.bdate_range('2026-08-03', periods=60)
rng = np.random.default_rng(7)
px = {t: 100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(days)))) for t in ['TQQQ', 'SOXL', 'SPXL', 'SQQQ']}

def make_log(h, prob_fn, tickers=('TQQQ', 'SOXL', 'SPXL', 'SQQQ')):
    rows = []
    for i, d in enumerate(days):
        for t in tickers:
            base = px[t][i]
            settled = i + h < len(days)
            tgt = px[t][i + h] if settled else np.nan
            p = prob_fn(t, i)
            lab = int(p >= 50)
            al = (1.0 if tgt > base else 0.0) if settled else np.nan
            rows.append([str(d + pd.Timedelta(hours=30)), 'B', t, h, lab, p, p, base,
                         str(days[i + h].date()) if settled else str((d + pd.offsets.BDay(h)).date()),
                         tgt, (tgt / base - 1) if settled else np.nan, al, (float(lab == al) if settled else np.nan), 0.0, str(d.date())])
    return pd.DataFrame(rows, columns=COLS)

# ---------------- 1) 방향 성적표
always_up = make_log(1, lambda t, i: 58.0)
d = ds._prep(always_up, 'scalp')
s = ds.score(d)
ck("인버스(SQQQ)는 독립 계열에서 제외", set(d['ticker']) == {'TQQQ', 'SOXL', 'SPXL'})
ck("항상 상승만 찍으면 적중률 = 항상상승 기준선(차 0)", abs(s['edge_vs_always_up']) < 1e-9 and s['pred_up_share'] == 100)
ck("1일: 독립 구간 = 기준일 수(종목 3개를 3배로 세지 않음)", s['n_indep'] == s['n_days'] == 59 and s['n_rows'] == 177)
ck("상수 58% 예측의 Brier 는 사후 상승비율 상수보다 낮을 수 없음", s['brier'] >= s['brier_clim'] - 1e-12 and s['skill_clim'] <= 1e-12)

def oracle(t, i):
    return 90.0 if (i + 1 < len(days) and px[t][i + 1] > px[t][i]) else 10.0
so = ds.score(ds._prep(make_log(1, oracle), 'scalp'))
ck("정답을 아는 예측 → 적중 100%, skill>0, '상수 예측보다 나음'", so['accuracy'] == 100 and so['skill_half'] > 0.9 and so['verdict'].startswith('상수 예측보다 나음'))

s5 = ds.score(ds._prep(make_log(5, lambda t, i: 60.0), 'swing'))
ck("5일: 독립 구간 = 기준일//5, 15 미만이면 판정 불가", s5['n_indep'] == s5['n_days'] // 5 and s5['verdict'].startswith('판정 불가'))
ck("비중첩 적중률은 최소<=최대", s5['nonoverlap_min'] <= s5['nonoverlap_max'])

stock = make_log(1, lambda t, i: 55.0, tickers=('TQQQ', 'SQQQ'))
ck("개별주 로그는 종목을 거르지 않음", len(ds._prep(stock, 'stock_scalp')['ticker'].unique()) == 2)
ck("빈 로그 → None", ds.score(ds._prep(pd.DataFrame(columns=COLS), 'scalp')) is None)
ck("필수 컬럼 없는 로그 → 빈 결과(예외 없음)", len(ds._prep(pd.DataFrame({'ticker': ['TQQQ']}), 'scalp')) == 0)
dirty = always_up.copy()
dirty['predicted_up_prob'] = dirty['predicted_up_prob'].astype(object)
dirty.loc[dirty.index[:5], 'predicted_up_prob'] = 'oops'
ck("오염된 숫자 행은 건너뛰고 계속", ds.score(ds._prep(dirty, 'scalp'))['n_rows'] < 177)
old = always_up.drop(columns=['price_date'])
ck("price_date 없는 옛 스키마도 run_date 로 채점", ds.score(ds._prep(old, 'scalp'))['n_rows'] == 177)

with tempfile.TemporaryDirectory() as td:
    logd, outd = os.path.join(td, 'logs'), os.path.join(td, 'rep')
    os.makedirs(logd)
    p = os.path.join(logd, 'predictions_scalp.csv')
    always_up.to_csv(p, index=False)
    with open(os.path.join(logd, 'predictions_swing.csv'), 'w') as f:
        f.write("깨진,파일\n1")
    h0 = hashlib.md5(open(p, 'rb').read()).hexdigest()
    text, out = ds.run(logd, outd)
    ck("run: 없는 로그·깨진 로그가 있어도 끝까지 실행", 'scalp|전체' in out and os.path.exists(os.path.join(outd, 'direction_scorecard.json')))
    ck("run: 로그 파일을 바꾸지 않음(읽기 전용)", hashlib.md5(open(p, 'rb').read()).hexdigest() == h0)
    ck("run: v31 이후 구간을 따로 냄", any(k.startswith('scalp|v31') for k in out))

    # ---------------- 2) 외부 종가 대조
    ext_rows = [(t, str(dd.date()), px[t][i]) for t in px for i, dd in enumerate(days)]
    ep = os.path.join(td, 'ext.csv')
    pd.DataFrame(ext_rows, columns=['ticker', 'date', 'close']).to_csv(ep, index=False)
    ext = ax.load_external(ep)
    s1, bad1 = ax.audit_log(always_up, ext)
    ck("일치하는 로그 → 불일치 0", len(bad1) == 0 and s1['base_checked'] == len(always_up) and s1['settle_checked'] > 0)
    stale = always_up.copy()
    i_row = stale.index[40]
    stale.loc[i_row, 'price_at_prediction'] *= 1.05                # 날짜 밀린(하루 전) 가격이 기록된 상황
    _, bad2 = ax.audit_log(stale, ext)
    ck("기준가가 5% 어긋난 행을 잡아냄", any(b[0] == i_row and b[3] == 'base' for b in bad2))
    flip = always_up.copy()
    j = flip.index[flip['actual_label'].notna()][10]
    flip.loc[j, 'actual_label'] = 1 - flip.loc[j, 'actual_label']
    _, bad3 = ax.audit_log(flip, ext)
    big = abs(flip.loc[j, 'actual_return']) > 0.005
    ck("뒤집힌 정산 라벨을 잡아냄(0 근처 수익률은 판정 제외)", (not big) or any(b[0] == j and b[3] == 'label' for b in bad3))
    s4, bad4 = ax.audit_log(always_up, {})
    ck("외부 종가가 없으면 전부 '대조 불가'(불일치로 세지 않음)", len(bad4) == 0 and s4['uncovered'] == len(always_up))
    ck("옛 스키마(price_date 없음) → 예외 없이 건너뜀", ax.audit_log(old, ext)[1] == [])
    ck("없는 외부 파일 → 빈 dict", ax.load_external(os.path.join(td, 'nope.csv')) == {})
    bp = os.path.join(td, 'bad.csv')
    open(bp, 'w').write("a,b\n1,2\n")
    ck("형식 틀린 외부 파일 → 빈 dict", ax.load_external(bp) == {})
    t2 = ax.run(ep, logd, 0.01, outd)
    ck("run: 보고서 저장, 로그 불변", os.path.exists(os.path.join(outd, 'audit_external.txt')) and hashlib.md5(open(p, 'rb').read()).hexdigest() == h0)
    ck("run: 외부 파일이 없어도 예외 없이 안내", '건너뜁니다' in ax.run(os.path.join(td, 'nope.csv'), logd))

# ---------------- 3) 핫이슈·일정 (표시 전용)
hot, age = er.load_hot_issues(today='2026-11-17')
ck("hot_issues.json: 11/17 기준 살아있는 이슈 3건 이상, 우선순위 순", len(hot) >= 3 and [h['priority'] for h in hot] == sorted(h['priority'] for h in hot))
ck("hot_issues.json: NVDA 이슈는 실적 뒤(11/20) 자동 만료", all('NVIDIA' not in h['title'] for h in er.load_hot_issues(today='2026-11-20')[0]))
ck("hot_issues.json: 12/11 이후 전부 만료", er.load_hot_issues(today='2026-12-11')[0] == [])
import common
ck("MARKET_EVENTS: 11/25 PCE 추가, 날짜 형식 유효", '2026-11-25' in common.MARKET_EVENTS and all(pd.Timestamp(k) for k in common.MARKET_EVENTS))
ck("기존 일정·실적일 불변", common.STOCK_EARNINGS_DATES['NVDA'] == '2026-11-17' and '2026-12-09' in common.MARKET_EVENTS)

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + str(fails)} ({len(fails)} fail)")
raise SystemExit(1 if fails else 0)
