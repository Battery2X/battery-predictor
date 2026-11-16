"""test_v35.py — v35 검증(합성 데이터, 네트워크 없음): 공포탐욕지수 · Alpha Vantage 교차확인 · 엣지/맥락 로그 · 그림자 재보정 · 날짜 정합성 가드.
방향 모델·피처·임계값·예측 로그 스키마는 건드리지 않는다는 것도 함께 확인한다."""
import hashlib, json, os, tempfile
import numpy as np
import pandas as pd

import common
import direction_scorecard as ds
import easy_report as er
import market_context as mc

fails = []
def ck(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)

nosleep = lambda s: None

# ============ 1) 공포탐욕지수 해석
days = pd.bdate_range('2024-01-02', periods=600)
rng = np.random.default_rng(3)
sp = 5000 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(days))))
score = np.clip(50 + 25 * np.sin(np.arange(len(days)) / 9) + rng.normal(0, 5, len(days)), 1, 99)
ms = lambda d: float(pd.Timestamp(d).value // 10**6)
def payload(last_score=38.6, with_sp=True, n=len(days)):
    p = {"fear_and_greed": {"score": last_score, "rating": "fear", "timestamp": "2026-11-13T23:59:56+00:00",
                            "previous_close": 39.8, "previous_1_week": 24.3, "previous_1_month": 45.6, "previous_1_year": 24.5},
         "fear_and_greed_historical": {"score": last_score, "data": [{"x": ms(d), "y": float(s), "rating": "x"} for d, s in zip(days[:n], score[:n])]},
         "put_call_options": {"score": 25.6, "rating": "fear", "data": []},
         "market_volatility_vix": {"score": 50, "rating": "neutral", "data": []},
         "junk_bond_demand": {"score": "oops", "rating": "greed"}}
    if with_sp:
        p["market_momentum_sp500"] = {"score": 14.6, "rating": "extreme fear", "data": [{"x": ms(d), "y": float(v)} for d, v in zip(days[:n], sp[:n])]}
    return p

fg = mc.parse_fear_greed(payload())
ck("F&G: 점수·구간·기준일(뉴욕 날짜) 해석", fg and fg['score'] == 38.6 and fg['bucket'] == '공포' and str(fg['asof'].date()) == '2026-11-13')
ck("F&G: 구성요소는 숫자인 것만(깨진 값 제외)", set(fg['components']) == {'market_momentum_sp500', 'put_call_options', 'market_volatility_vix'})
ck("F&G: 이력에 score·sp500 열", list(fg['history'].columns) == ['score', 'sp500'] and len(fg['history']) == 600)
ck("F&G: 구간 경계 0/24.9/25/45/55/75/100", [mc.fg_bucket(v) for v in (0, 24.9, 25, 45, 55, 75, 100)] ==
   ['극단적 공포', '극단적 공포', '공포', '중립', '탐욕', '극단적 탐욕', '극단적 탐욕'])
for bad in (None, [], {}, {"fear_and_greed": {}}, {"fear_and_greed": {"score": "NaN"}}, {"fear_and_greed": {"score": 140}}, "text"):
    if mc.parse_fear_greed(bad) is not None:
        fails.append(f"bad payload {bad!r}")
ck("F&G: 빈·깨진·범위 밖 응답 → None(예외 없음)", not any(f.startswith('bad payload') for f in fails))
ck("F&G: 이력이 없어도 현재값은 해석", mc.parse_fear_greed({"fear_and_greed": {"score": 50, "rating": "neutral"}})['history'] is None)

calls = []
def getter_fallback(url):
    calls.append(url)
    if url.endswith(mc.FG_HISTORY_START):
        raise RuntimeError("500")
    return payload()
ck("F&G: 긴 이력 주소 실패 → 기본 주소로 재시도", mc.fetch_fear_greed(getter=getter_fallback)['score'] == 38.6 and len(calls) == 2)
ck("F&G: 전부 실패 → None", mc.fetch_fear_greed(getter=lambda u: (_ for _ in ()).throw(RuntimeError('x'))) is None)

# ---- 과거 통계: 룩어헤드 없음 / 독립 구간 / 표본 부족
hist = fg['history']
st = mc.fg_forward_stats(hist, 30.0)
ck("F&G 통계: 5·20일 결과, 독립 구간 수 <= 행 수", set(st) == {5, 20} and all(v['episodes'] <= v['n'] for v in st.values()))
i = np.where((score >= 25) & (score < 45))[0]
i20 = i[i + 20 < len(days)]
ck("F&G 통계: 20일 평균이 직접 계산과 일치(미래 20일 수익률)", abs(st[20]['mean'] - np.mean(sp[i20 + 20] / sp[i20] - 1)) < 1e-12)
hist_cut = hist.iloc[:400]
st_cut = mc.fg_forward_stats(hist_cut, 30.0)
hist_fut = hist.copy(); hist_fut.iloc[400:, hist_fut.columns.get_loc('sp500')] *= 3.0
ck("F&G 통계: 400일 이후 값을 바꿔도 앞 400일 통계는 그 안의 값만 사용", abs(mc.fg_forward_stats(hist_fut.iloc[:400], 30.0)[20]['mean'] - st_cut[20]['mean']) < 1e-12)
ck("F&G 통계: 이력 250일 미만 → 빈 결과", mc.fg_forward_stats(hist.iloc[:200], 30.0) == {})
ck("F&G 통계: sp500 열 없음·None → 빈 결과", mc.fg_forward_stats(hist[['score']], 30.0) == {} and mc.fg_forward_stats(None, 30.0) == {})
const = hist.copy(); const['score'] = 60.0
ck("F&G 통계: 한 번도 없던 구간(독립 구간 부족) → 그 구간 결과 없음", mc.fg_forward_stats(const, 10.0) == {})

# ============ 2) Alpha Vantage (가짜 응답)
def reset_av(key='K'):
    mc._AV_STATE.update(calls=0, last=0.0, blocked=False)
    if key is None:
        os.environ.pop(mc.AV_KEY_ENV, None)
    else:
        os.environ[mc.AV_KEY_ENV] = key

reset_av(None)
ck("AV: 키 없으면 호출 자체를 안 함", mc.av_get("GLOBAL_QUOTE", getter=lambda q: 1 / 0, symbol='X') is None and mc._AV_STATE['calls'] == 0)
reset_av()
def quote_getter(q):
    if q['function'] == 'GLOBAL_QUOTE':
        px = {'TQQQ': ('57.68', '2026-11-13'), 'SOXL': ('58.32', '2026-11-13'), 'SPXL': ('265.10', '2026-11-13')}.get(q['symbol'])
        if px is None:
            return json.dumps({"Global Quote": {}})
        return json.dumps({"Global Quote": {"01. symbol": q['symbol'], "05. price": px[0], "07. latest trading day": px[1], "08. previous close": "1.0"}})
    if q['function'] == 'EARNINGS_CALENDAR':
        if q['symbol'] == 'NVDA':
            return "symbol,name,reportDate,fiscalDateEnding,estimate,currency,timeOfTheDay\nNVDA,NVIDIA,2026-11-17,2026-10-31,2.49,USD,post-market\nNVDA,NVIDIA,2027-02-24,2027-01-31,,USD,post-market\n"
        return "symbol,name,reportDate,fiscalDateEnding,estimate,currency,timeOfTheDay\n"
    if q['function'] == mc.AV_PCR_FUNCTION:
        if q['symbol'] == 'SOXL':
            return json.dumps({"symbol": "SOXL", "date": "2026-11-13", "put_call_ratio_full_chain": "0.89", "put_call_ratio_by_expiration": []})
        if q['symbol'] == 'QQQ':
            return json.dumps({"data": [{"put_call_ratio_full_chain": 1.31}]})
        return json.dumps({"unexpected": True})
    return None
qs = mc.av_quotes(['TQQQ', 'SOXL', 'SPXL', 'ZZZZ'], getter=quote_getter, sleeper=nosleep)
ck("AV 시세: 응답 있는 종목만, 빈 응답 종목은 제외", set(qs) == {'TQQQ', 'SOXL', 'SPXL'} and qs['SOXL']['price'] == 58.32)
items = [{'ticker': 'TQQQ', 'price': 57.68, 'price_date': '2026-11-13'},
         {'ticker': 'SOXL', 'price': 65.53, 'price_date': '2026-11-12'},      # 하루 전 시세로 예측(미갱신)
         {'ticker': 'SPXL', 'price': 271.36, 'price_date': '2026-11-13'},     # 같은 날인데 2.4% 다름(장중가)
         {'ticker': 'SOXL', 'price': 1.0, 'price_date': '2026-11-13'},        # 같은 종목 중복 항목은 한 번만
         {'ticker': 'NVDA', 'price': 1.0, 'price_date': '2026-11-13'}]        # 대조 대상 없음
w, n = mc.crosscheck_prices(items, qs)
ck("교차확인: 일치 종목은 경고 없음, 미갱신·가격 불일치만 경고", n == 3 and len(w) == 2 and '미갱신' in w[0] and '출처마다 다름' in w[1])
ck("교차확인: 우리 쪽이 더 최신이면 경고 안 함", mc.crosscheck_prices([{'ticker': 'TQQQ', 'price': 60.0, 'price_date': '2026-11-16'}], qs)[0] == [])
ck("교차확인: 빈 입력·깨진 항목에도 예외 없음", mc.crosscheck_prices(None, qs) == ([], 0) and mc.crosscheck_prices([{'ticker': 'TQQQ'}], qs)[0] == [])

en = mc.av_next_earnings(['NVDA', 'AAPL'], today='2026-11-18', getter=quote_getter, sleeper=nosleep)
ck("AV 실적: 오늘 이후 가장 가까운 날짜, 캘린더에 없는 종목은 제외", en == {'NVDA': '2027-02-24'})
notes = mc.earnings_calendar_notes({'NVDA': '2026-11-17', 'AAPL': '2026-10-29'}, en, today='2026-11-18')
ck("실적 안내: 수동 달력이 지난 종목에 후보일 제시(자동 변경 없음)", len(notes) == 1 and '2027-02-24' in notes[0] and common.STOCK_EARNINGS_DATES['NVDA'] == '2026-11-17')
ck("실적 안내: 수동값과 다르면 불일치 안내, 같으면 없음", len(mc.earnings_calendar_notes({'NVDA': '2027-02-20'}, en, today='2026-11-18')) == 1
   and mc.earnings_calendar_notes({'NVDA': '2027-02-24'}, en, today='2026-11-18') == [])
pcr = mc.av_put_call(['SOXL', 'QQQ', 'SPY'], date='2026-11-13', getter=quote_getter, sleeper=nosleep)
ck("AV 풋콜: 두 가지 응답 형식 해석, 모르는 형식은 조용히 제외", pcr == {'SOXL': 0.89, 'QQQ': 1.31})

reset_av()
lim = lambda q: json.dumps({"Information": "Thank you... rate limit is 25 requests per day."})
r1 = mc.av_get("GLOBAL_QUOTE", getter=lim, sleeper=nosleep, symbol='A')
r2 = mc.av_get("GLOBAL_QUOTE", getter=lambda q: 1 / 0, sleeper=nosleep, symbol='B')
ck("AV: 한도 안내문을 받으면 이번 실행에선 더 부르지 않음", r1 is None and r2 is None and mc._AV_STATE['calls'] == 1 and mc._AV_STATE['blocked'])
reset_av()
for _ in range(mc.AV_DAILY_BUDGET + 5):
    mc.av_get("GLOBAL_QUOTE", getter=quote_getter, sleeper=nosleep, symbol='TQQQ')
ck("AV: 하루 호출 예산을 넘지 않음", mc._AV_STATE['calls'] == mc.AV_DAILY_BUDGET)
reset_av()
ck("AV: 예외·HTML·None 응답 → None", mc.av_get("X", getter=lambda q: 1 / 0, sleeper=nosleep) is None and mc.av_get("X", getter=lambda q: "<html>", sleeper=nosleep) is None
   and mc.av_get("X", getter=lambda q: None, sleeper=nosleep) is None)
slept = []
reset_av()
mc.av_get("GLOBAL_QUOTE", getter=quote_getter, sleeper=slept.append, symbol='TQQQ')
mc.av_get("GLOBAL_QUOTE", getter=quote_getter, sleeper=slept.append, symbol='TQQQ')
ck("AV: 연속 호출 사이 간격을 둠(초당 1회 제한)", len(slept) >= 1 and 0 < slept[-1] <= mc.AV_MIN_INTERVAL)

# ============ 3) 로그
def summ(pdate, edge=2.0):
    mk = lambda t, b, d, up: {'ticker': t, 'benchmark': b, 'direction': d, 'up_prob': up, 'raw_up_prob': up + 1, 'base_rate': 58.0,
                              'edge': up - 58.0, 'price': 57.68 if t == 'TQQQ' else 58.32, 'price_date': pdate}
    return {'scalp': {'horizon_days': 1, 'items': [mk('TQQQ', 'QQQ', 1, 58.0 + edge), mk('SQQQ', 'QQQ', -1, 42.0 - edge)]},
            'swing': {'horizon_days': 5, 'items': [mk('SOXL', 'SMH', 1, 61.0), {'ticker': 'BAD', 'price_date': None}]}}

with tempfile.TemporaryDirectory() as td:
    n1 = mc.append_edge_history(summ('2026-11-13'), td)
    n2 = mc.append_edge_history(summ('2026-11-13', edge=9.0), td)      # 같은 가격일 재실행
    n3 = mc.append_edge_history(summ('2026-11-16'), td)
    e = pd.read_csv(os.path.join(td, mc.EDGE_LOG))
    ck("엣지 이력: 깨진 항목 제외, 재실행은 처음 값 유지, 새 가격일은 추가", (n1, n2, n3) == (3, 0, 3) and len(e) == 6
       and float(e[(e.ticker == 'TQQQ') & (e.price_date == '2026-11-13')].edge.iloc[0]) == 2.0)
    ck("엣지 이력: push_logs 합치기용 키 열 보유", {'run_date', 'ticker', 'horizon_days', 'price_date'} <= set(e.columns))
    import push_logs
    m = push_logs.merge_log_frames(e.iloc[:4].copy(), e.copy())
    ck("엣지 이력: push_logs.merge_log_frames 로 합쳐도 행이 늘거나 사라지지 않음", len(m) == 6)
    ck("엣지 이력: 빈 요약·None → 0(예외 없음)", mc.append_edge_history({}, td) == 0 and mc.append_edge_history(None, td) == 0)
    open(os.path.join(td, mc.EDGE_LOG), 'wb').write(b'\x00\xff\x00broken"\n"')
    ck("엣지 이력: 깨진 파일이어도 죽지 않고 새로 기록", mc.append_edge_history(summ('2026-11-17'), td) >= 0 and os.path.exists(os.path.join(td, mc.EDGE_LOG)))

    cfg = {k: v for k, v in fg.items() if k != 'history'}
    ck("맥락 로그: 첫 기록", mc.append_context('2026-11-13', fg=cfg, pcr={'SOXL': 0.89}, log_dir=td))
    c1 = pd.read_csv(os.path.join(td, mc.CONTEXT_LOG))
    ck("맥락 로그: F&G 점수·구성요소·풋콜 열", float(c1.fg_score.iloc[0]) == 38.6 and 'fg_put_call_options' in c1.columns and float(c1.pcr_SOXL.iloc[0]) == 0.89)
    cfg2 = dict(cfg, score=99.0)
    mc.append_context('2026-11-13', fg=cfg2, pcr={'SOXL': 5.0, 'QQQ': 1.31}, log_dir=td)
    c2 = pd.read_csv(os.path.join(td, mc.CONTEXT_LOG))
    ck("맥락 로그: 같은 날 재실행은 기록된 값을 덮지 않고 빈 칸만 채움", len(c2) == 1 and float(c2.fg_score.iloc[0]) == 38.6 and float(c2.pcr_SOXL.iloc[0]) == 0.89 and float(c2.pcr_QQQ.iloc[0]) == 1.31)
    mc.append_context('2026-11-16', fg=None, pcr={'SPY': 1.1}, log_dir=td)
    c3 = pd.read_csv(os.path.join(td, mc.CONTEXT_LOG))
    ck("맥락 로그: 새 날짜 행 추가(새 열이 생겨도 옛 행 보존)", len(c3) == 2 and float(c3.fg_score.iloc[0]) == 38.6 and pd.isna(c3.fg_score.iloc[1]))
    ck("맥락 로그: 기록할 값이 없거나 날짜가 깨지면 False", not mc.append_context('2026-11-17', log_dir=td) and not mc.append_context('nope', fg=cfg, log_dir=td))

# ---- collect: 전부 실패 / 전부 성공
with tempfile.TemporaryDirectory() as td:
    reset_av(None)
    ctx = mc.collect(summ('2026-11-13'), td, fg_getter=lambda u: 1 / 0, sleeper=nosleep)
    ck("collect: F&G 실패 + AV 키 없음 → 예외 없이 빈 맥락, 엣지 이력은 기록", ctx['fg'] is None and not ctx['av_used'] and os.path.exists(os.path.join(td, mc.EDGE_LOG)))
    ck("collect: 보여줄 게 없으면 리포트 줄 없음", mc.context_lines(ctx) == [] and mc.context_lines(None) == [] and mc.context_lines({}) == [])
    reset_av()
    ctx = mc.collect(summ('2026-11-13'), td, earnings_manual={'NVDA': '2026-11-17', 'AAPL': '2026-10-29', 'TSLA': '2026-10-21'}, today='2026-11-18',
                     fg_getter=lambda u: payload(), av_getter=quote_getter, sleeper=nosleep)
    c = pd.read_csv(os.path.join(td, mc.CONTEXT_LOG))
    ck("collect: F&G·교차확인·풋콜·실적 안내가 모두 채워짐", ctx['fg']['score'] == 38.6 and ctx['price_checked'] == 2 and ctx['pcr'] == {'SOXL': 0.89, 'QQQ': 1.31} and len(ctx['earnings_notes']) == 1)
    ck("collect: 맥락 로그에 가격일 1행", len(c) == 1 and c.price_date.iloc[0] == '2026-11-13' and float(c.fg_score.iloc[0]) == 38.6)
    ck("collect: AV 호출 수가 예산 이내", mc._AV_STATE['calls'] <= mc.AV_DAILY_BUDGET)
    lines = mc.context_lines(ctx)
    ck("리포트 줄: 지수·구성·과거 집계·교차확인·풋콜", any('공포탐욕지수 39' in l for l in lines) and any('독립 구간' in l for l in lines) and any('교차확인' in l for l in lines) and any('풋/콜' in l for l in lines))
    reset_av(None)
    with tempfile.TemporaryDirectory() as td2:
        ctx2 = mc.collect(summ('2026-11-12'), td2, fg_getter=lambda u: payload(), sleeper=nosleep)
        ck("collect: F&G 기준일(11/13)이 가격일(11/12)과 다르면 그 날짜 행에 F&G 를 적지 않음", ctx2['fg'] is not None and not os.path.exists(os.path.join(td2, mc.CONTEXT_LOG)))

    # ---- 요약 리포트 조립
    now = pd.Timestamp('2026-11-17 09:20')
    sums = {'scalp': {'horizon_days': 1, 'generated_at': str(now), 'items': [
        {'ticker': 'SOXL', 'benchmark': 'SMH', 'desc': 'x', 'direction': 1, 'up_prob': 56.0, 'raw_up_prob': 60.0, 'base_rate': 55.0, 'edge': 1.0,
         'price': 58.32, 'vol': 1.5, 'drawdown': -0.6, 'ret1': -0.1, 'ret5': 0.2, 'price_date': '2026-11-13', 'signal': None}]}}
    base = er.build_sections(sums, [], [], None, [], None, [], now, risk={})
    withc = er.build_sections(sums, [], [], None, [], None, [], now, risk={'_context': ctx})
    ck("요약: 맥락이 없으면 기존과 동일한 섹션 수, 있으면 1개 추가", len(withc) == len(base) + 1 and any('시장 심리' in s for s in withc))
    ck("요약: 섹션 번호가 건너뛰지 않음", [int(x) for s in withc for x in __import__('re').findall(r'<b>(\d+)\. ', s)] == list(range(1, 1 + sum('<b>' in s and __import__('re').search(r'<b>\d+\. ', s) is not None for s in withc))))
    ck("요약: 깨진 맥락(dict 아님)에도 예외 없음", len(er.build_sections(sums, [], [], None, [], None, [], now, risk={'_context': 'oops'})) == len(base))
    ck("요약: HTML 특수문자 이스케이프", '<script>' not in ''.join(er.build_sections(sums, [], [], None, [], None, [], now, risk={'_context': {'price_warn': ['<script>x'], 'price_checked': 1}})))
reset_av(None)

# ============ 4) 그림자 재보정 (룩어헤드 없음)
COLS = ['run_date', 'benchmark', 'ticker', 'horizon_days', 'predicted_label', 'predicted_up_prob', 'raw_up_prob',
        'price_at_prediction', 'target_date', 'actual_price', 'actual_return', 'actual_label', 'correct', 'near_earnings', 'price_date']
d2 = pd.bdate_range('2026-06-01', periods=120)
px = 100 * np.exp(np.cumsum(np.random.default_rng(5).normal(0, 0.02, len(d2))))
def make_log(h, prob=60.0):
    rows = []
    for i, dd in enumerate(d2):
        settled = i + h < len(d2)
        tgt = px[i + h] if settled else np.nan
        al = (1.0 if tgt > px[i] else 0.0) if settled else np.nan
        rows.append([str(dd + pd.Timedelta(hours=30)), 'QQQ', 'TQQQ', h, 1, prob, prob, px[i], str(d2[i + h].date()) if settled else str((dd + pd.offsets.BDay(h)).date()),
                     tgt, tgt / px[i] - 1 if settled else np.nan, al, al, 0.0, str(dd.date())])
    return pd.DataFrame(rows, columns=COLS)
L5 = make_log(5)
P = ds.add_shadows(ds._prep(L5, 'swing'))
y = P['actual_label'].values
exp = []
for k in range(len(P)):
    known = y[:max(k - 5 + 1, 0)]          # 목표일 <= 이 행의 가격일: 5일 전까지의 행만 결과가 알려짐
    exp.append((ds.PRIOR_A + known.sum()) / (ds.PRIOR_A + ds.PRIOR_B + len(known)))
ck("그림자 prior: 그 가격일까지 정산이 끝난 결과만 사용(직접 계산과 일치)", np.allclose(P['sh_prior'].values, exp))
L5b = L5.copy(); L5b.loc[L5b.index[60:], 'actual_label'] = 1 - L5b.loc[L5b.index[60:], 'actual_label']
Pb = ds.add_shadows(ds._prep(L5b, 'swing'))
ck("그림자 prior: 뒤쪽 결과를 뒤집어도 앞쪽 값은 불변", np.allclose(P['sh_prior'].values[:60], Pb['sh_prior'].values[:60]))
ck("그림자: 첫 행 prior = 0.5, shrink50 = 모델과 50%의 중간, half = 0.5", P['sh_prior'].iloc[0] == 0.5 and np.allclose(P['sh_shrink50'], 0.55) and (P['sh_half'] == 0.5).all())
sc = ds.shadow_scores(P)
ck("그림자 점수: 차이 = 모델 Brier - 후보 Brier", all(abs(v['diff'] - (v['brier_model'] - v['brier'])) < 1e-12 for v in sc.values()) and 'base' not in sc)
edge = pd.DataFrame({'log': 'swing', 'ticker': 'TQQQ', 'day': P['day'].values[:30], 'base_rate': 55.0, 'edge': np.linspace(-5, 5, 30)})
Pe = ds.add_shadows(ds._prep(L5, 'swing'), edge, 'swing')
ck("그림자 base·엣지: 이력이 있는 행에만 붙음(행 수·순서 불변)", Pe['sh_base'].notna().sum() == 30 and len(Pe) == len(P) and (Pe['day'].values == P['day'].values).all())
et = ds.edge_table(Pe)
ck("엣지 표: 부호별 행 수 합 = 이력 붙은 행 수", et[0][1] + et[1][1] == 30)
lo, hi = ds._block_ci(np.full(40, 0.01), 5)
ck("블록 부트스트랩: 상수 계열이면 구간이 그 값, 표본 부족이면 NaN", abs(lo - 0.01) < 1e-12 and abs(hi - 0.01) < 1e-12 and np.isnan(ds._block_ci(np.zeros(10), 5)[0]))
z = np.random.default_rng(1).normal(0, 1, 400)
lo, hi = ds._block_ci(z, 5)
ck("블록 부트스트랩: 구간이 표본 평균을 포함하고 폭이 표준오차 수준", lo < z.mean() < hi and 0.1 < (hi - lo) < 0.4)
rel = ds.reliability(P)
ck("신뢰도표: 60% 예측은 60~65% 구간 한 줄", len(rel) == 1 and rel[0][0].startswith('60') and rel[0][1] == len(P))
ck("그림자: 빈 로그·열 없는 로그에도 예외 없음", ds.shadow_scores(ds.add_shadows(ds._prep(pd.DataFrame(columns=COLS), 'swing'))) == {} and ds.edge_table(pd.DataFrame()) == [])
dupL = pd.concat([L5, L5.iloc[:10].assign(run_date=lambda x: (pd.to_datetime(x['run_date']) + pd.Timedelta(days=2)).astype(str))], ignore_index=True)
ck("성적표: 같은 가격일·종목 중복 행은 한 번만 셈", len(ds._prep(dupL, 'swing')) == len(ds._prep(L5, 'swing')))
with tempfile.TemporaryDirectory() as td:
    lg = os.path.join(td, 'logs'); os.makedirs(lg)
    p = os.path.join(lg, 'predictions_swing.csv'); L5.to_csv(p, index=False)
    h0 = hashlib.md5(open(p, 'rb').read()).hexdigest()
    text, out = ds.run(lg, os.path.join(td, 'rep'))
    ck("성적표 run: 그림자 구간 출력, 판정 보류 문구, 예측 로그 불변", '그림자' in text and '보류' in text and hashlib.md5(open(p, 'rb').read()).hexdigest() == h0)
    ck("성적표 run: 엣지 이력 파일이 없어도 동작", any(k.startswith('swing|shadow') for k in out))

# ============ 5) 날짜 정합성 가드 + 모델 불변
m = common.data_date_mismatch
ck("가드: 같은 날 → None, 다른 날 → 안내 문자열", m({'asof': pd.Timestamp('2026-11-13')}, {'price_date': pd.Timestamp('2026-11-13 00:00')}) is None
   and '불일치' in m({'asof': pd.Timestamp('2026-11-12')}, {'price_date': pd.Timestamp('2026-11-13')}))
ck("가드: 값이 없거나 깨지면 기록을 막지 않음(None)", m({}, {'price_date': pd.Timestamp('2026-11-13')}) is None and m({'asof': None}, {}) is None
   and m({'asof': 'x'}, {'price_date': 'y'}) is None and m({'asof': pd.NaT}, {'price_date': pd.NaT}) is None)
ck("모델 불변: 피처 수·라벨 방식·로그 스키마 그대로", len(common.FEATURE_COLUMNS) > 0 and common.LABEL_MODE == 'all' and common.LOG_COLUMNS[-1] == 'price_date' and len(common.LOG_COLUMNS) == 15)
ck("페어 대칭: 롱+인버스 = 100", sum(common.apply_direction(61.3, 38.7, +1)[:1] + common.apply_direction(61.3, 38.7, -1)[:1]) == 100.0)

# ---- 가드가 실제 예측 스크립트 흐름에서 동작하는지(외부 호출은 전부 가짜)
import predict_swing as psw, predict_stock_scalp as pss
def run_script(mod, asof_by_bench, stock=False):
    got = []
    res = lambda b: {'up_prob': 61.0, 'down_prob': 39.0, 'raw_up_prob': 63.0, 'base_rate': 58.0, 'edge': 3.0, 'f1': .5, 'prec': .5, 'rec': .5,
                     'dead_zone': 0.1, 'top_features': [('RSI', 0.2)], 'calib_unclamped': 61.0, 'calib_clamped': False, **({'asof': asof_by_bench[b]} if b in asof_by_bench else {})}
    met = lambda t: {'price': 50.0, 'vol': 0.5, 'df': None, 'price_date': pd.Timestamp('2026-11-13'), 'high_60d': 60.0, 'drawdown': -0.1, 'ret1': 0.0, 'ret5': 0.0}
    saved = {k: getattr(mod, k) for k in dir(mod)}
    try:
        mod.settle_predictions = lambda n: pd.DataFrame(columns=common.LOG_COLUMNS)
        mod.fetch_macro_data = lambda *a, **k: pd.DataFrame()
        if hasattr(mod, 'fetch_constituent_breadth_features'):
            mod.fetch_constituent_breadth_features = lambda *a, **k: pd.DataFrame()
        if hasattr(mod, 'fetch_analyst_momentum_features'):
            mod.fetch_analyst_momentum_features = lambda *a, **k: None
        if hasattr(mod, 'macro_shock_warning'):
            mod.macro_shock_warning = lambda *a, **k: None
        mod.train_benchmark_model = lambda b, *a, **k: res(b)
        mod.fetch_display_metrics = met
        mod.append_predictions = lambda n, rows: got.extend(rows)
        mod.finish_report = lambda *a, **k: None
        mod.check_upcoming_events = lambda *a, **k: []
        common._SUMMARY_ITEMS.clear()
        mod.run()
    finally:
        for k, v in saved.items():
            setattr(mod, k, v)
    return got
d13, d12 = pd.Timestamp('2026-11-13'), pd.Timestamp('2026-11-12')
g = run_script(psw, {'QQQ': d13, 'SMH': d12, 'SPY': d13})
ck("가드(스윙): 날짜가 맞는 벤치마크의 4종목만 기록, 어긋난 SMH 쌍(SOXL·SOXS)은 기록 안 함", sorted(r['ticker'] for r in g) == ['SPXL', 'SPXS', 'SQQQ', 'TQQQ'])
ck("가드(스윙): 기록된 쌍은 대칭(합 100)", all(abs(sum(r['predicted_up_prob'] for r in g if r['benchmark'] == b) - 100) < 1e-9 for b in ('QQQ', 'SPY')))
ck("가드(스윙): asof 가 없는 결과(옛 형식)는 막지 않음 → 6종목 모두 기록", len(run_script(psw, {})) == 6)
g2 = run_script(pss, {'AAPL': d13, 'NVDA': d12, 'TSLA': d13}, stock=True)
ck("가드(종목 단타): 어긋난 NVDA 만 제외", sorted(r['ticker'] for r in g2) == ['AAPL', 'TSLA'])
common._SUMMARY_ITEMS.clear()

# ============ 6) 실행 환경 기록
with tempfile.TemporaryDirectory() as td:
    v1 = {"python": "3.10.1", "xgboost": "2.1.4", "lightgbm": "4.5.0"}
    ck("환경 기록: 첫 기록은 변화 없음으로 처리하고 파일 생성", mc.record_env(td, versions=v1, today='2026-11-17') == [] and os.path.exists(os.path.join(td, mc.ENV_FILE)))
    ck("환경 기록: 같은 버전이면 변화 없음", mc.record_env(td, versions=dict(v1), today='2026-11-18') == [])
    ch = mc.record_env(td, versions=dict(v1, xgboost="3.0.0"), today='2026-11-19')
    j = json.load(open(os.path.join(td, mc.ENV_FILE), encoding='utf-8'))
    ck("환경 기록: 버전이 바뀌면 알리고 이력에 남김", ch == ["xgboost 2.1.4 → 3.0.0"] and j['history'][-1]['date'] == '2026-11-19' and j['versions']['xgboost'] == '3.0.0')
    open(os.path.join(td, mc.ENV_FILE), 'w').write("{broken")
    ck("환경 기록: 깨진 파일이어도 예외 없이 새로 기록", mc.record_env(td, versions=v1) == [])
    real = mc.record_env(os.path.join(td, 'r2'))
    ck("환경 기록: 실제 설치 버전 읽기", real == [] and json.load(open(os.path.join(td, 'r2', mc.ENV_FILE)))['versions'].get('xgboost'))
    ck("리포트 줄: 버전 변경 경고 표시", any('라이브러리 버전' in l for l in mc.context_lines({'env_changes': ch})))

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + str(fails)} ({len(fails)} fail)")
raise SystemExit(1 if fails else 0)
