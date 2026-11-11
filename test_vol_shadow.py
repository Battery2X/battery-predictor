"""v32 그림자 후보(HAR/GJR)·성적표 정정 검증 — 합성 데이터만 사용. 실행: python test_vol_shadow.py"""
import tempfile
import numpy as np
import pandas as pd
import risk_model as rm

FAIL = []
def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))
    if not cond:
        FAIL.append(name)

def mk(returns, start='2020-01-02'):
    idx = pd.bdate_range(start, periods=len(returns))
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=idx)

def garch(n, seed, omega=2e-6, a=0.08, g=0.08, b=0.85):
    rng = np.random.default_rng(seed); s2 = omega / (1 - a - g / 2 - b); out = []
    for _ in range(n):
        e = np.sqrt(s2) * rng.standard_t(6) / np.sqrt(6 / 4)
        out.append(e); s2 = omega + (a + (g if e < 0 else 0)) * e * e + b * s2
    return np.array(out)

# T1 알려진 값
c = mk(np.tile([0.01, -0.01], 400))
exp = 0.01 * np.sqrt(252)
for h in (1, 5, 20):
    v = rm.har_vol(c, h)
    check(f"T1a HAR 일정크기 수익 h={h} 정확", v is not None and abs(v - exp) < 1e-6, f"{v}")
    g = rm.gjr_vol(c, h)
    check(f"T1b GJR 일정크기 수익 h={h} 근사(±20%)", g is not None and abs(g / exp - 1) < 0.2, f"{g}")

# T2 방어
bad = [None, [], pd.Series([], dtype=float), pd.Series([100.0] * 10), pd.Series([np.nan] * 400), pd.Series([0.0] * 400),
       pd.Series([-5.0] * 400), "abc", 123, pd.Series(np.linspace(100, 101, 400))]
res = []
for b in bad:
    for fn in (rm.har_vol, rm.gjr_vol):
        try:
            fn(b, 5); res.append(True)
        except Exception:
            res.append(False)
check("T2a 이상 입력에서도 예외 없음", all(res))
check("T2b 데이터 부족(150일)이면 HAR None, GJR 값", rm.har_vol(mk(garch(150, 1)), 5) is None and rm.gjr_vol(mk(garch(150, 1)), 5) is not None)
check("T2c h<1 이면 None", rm.har_vol(mk(garch(500, 1)), 0) is None and rm.gjr_vol(mk(garch(500, 1)), 0) is None)
flat = pd.Series(100.0, index=pd.bdate_range('2020-01-02', periods=500))
check("T2d 가격이 완전 일정(변동성 0)이어도 예외·무한값 없음",
      all(v is None or np.isfinite(v) for v in (rm.har_vol(flat, 5), rm.gjr_vol(flat, 5))))

# T3 GJR 기간구조: 충격 직후 h 가 길수록 장기평균으로 회귀
r = garch(800, 2); r[-3:] = [-0.12, -0.10, -0.09]
c3 = mk(r)
g1, g5, g20, g60 = (rm.gjr_vol(c3, h) for h in (1, 5, 20, 60))
check("T3 충격 직후 GJR: 1일 > 5일 > 20일 > 60일 (평균회귀)", g1 > g5 > g20 > g60, f"{g1:.2f}>{g5:.2f}>{g20:.2f}>{g60:.2f}")

# T4 룩어헤드: 기준일 T 의 예측은 T 이후 데이터를 바꿔도 불변(함수는 잘린 시계열만 본다 — 잘라서 호출하는 방식 검증)
c4 = mk(garch(900, 3)); c4b = c4.copy(); c4b.iloc[700:] *= 1.5
check("T4 T 이후 변조해도 T 시점 예측 동일",
      all(rm.har_vol(c4.iloc[:600], h) == rm.har_vol(c4b.iloc[:600], h) and rm.gjr_vol(c4.iloc[:600], h) == rm.gjr_vol(c4b.iloc[:600], h) for h in (1, 5, 20)))

# T5 시뮬레이션 비교(참고용 + 최소 안전선): 상수 변동성 대비 열위가 아니어야 한다
def qlike(real, pred):
    q = (real / pred) ** 2
    return float(np.mean(q - np.log(q) - 1))
wins = {'har': 0, 'gjr': 0, 'ewma': 0}; tot = 0; agg = {'har': [], 'gjr': [], 'ewma': [], 'const': []}
for seed in range(6):
    ret = garch(1600, 100 + seed); c5 = mk(ret); h = 5
    rows = {'har': [], 'gjr': [], 'ewma': [], 'const': [], 'real': []}
    for T in range(800, 1590, 10):
        cs = c5.iloc[:T + 1]
        rv = float(np.sqrt((np.diff(np.log(c5.values[T:T + h + 1])) ** 2).mean() * 252))
        rows['har'].append(rm.har_vol(cs, h)); rows['gjr'].append(rm.gjr_vol(cs, h))
        rows['ewma'].append(rm.ewma_vol(cs)); rows['const'].append(rm.rolling_rv(cs, 250)); rows['real'].append(rv)
    q = {k: qlike(np.array(rows['real']), np.array(rows[k])) for k in ('har', 'gjr', 'ewma', 'const')}
    for k in q: agg[k].append(q[k])
print("   시뮬 5일 QLIKE 평균(낮을수록 좋음):", {k: round(float(np.mean(v)), 3) for k, v in agg.items()})
check("T5 시뮬: HAR·GJR 평균 QLIKE 가 상수 변동성보다 낮음", np.mean(agg['har']) < np.mean(agg['const']) and np.mean(agg['gjr']) < np.mean(agg['const']))

# T6 로그 호환·기록
old_cols = ['run_date', 'ticker', 'price_date', 'horizon_days', 'vol_ewma', 'vol_rv20', 'vol_rv60', 'target_date', 'vol_realized', 'settled']
d = tempfile.mkdtemp()
old = pd.DataFrame([{'run_date': '2026-10-16', 'ticker': 'X', 'price_date': '2026-10-15', 'horizon_days': 1, 'vol_ewma': 0.5, 'vol_rv20': 0.5,
                     'vol_rv60': 0.6, 'target_date': '2026-10-16', 'vol_realized': 0.4, 'settled': 1}])[old_cols]
old.to_csv(f"{d}/{rm.VOL_LOG_NAME}", index=False)
lg = rm.load_vol_log(d)
check("T6a 옛 로그(10컬럼) 로드: 새 컬럼 NaN, 스키마 일치", list(lg.columns) == rm.VOL_LOG_COLUMNS and lg[['vol_har', 'vol_gjr', 'vol_ewmaf']].isna().all().all())
c6 = mk(garch(700, 7), start='2023-01-02')
lg2 = rm.append_vol_forecasts(lg, '2026-11-09', 'X', c6)
new = lg2.iloc[1:]
check("T6b 신규 행 3개, 그림자 값 채워짐", len(new) == 3 and new[['vol_har', 'vol_gjr', 'vol_ewmaf']].notna().all().all(), str(new[['horizon_days', 'vol_har', 'vol_gjr']].values.tolist()))
check("T6c 기존 행·기존 후보 값은 불변", lg2.iloc[0]['vol_ewma'] == 0.5 and all(abs(new.iloc[i]['vol_ewma'] - rm.forecast_vol(c6)['vol_ewma']) < 1e-12 for i in range(3)))
rm.save_vol_log(lg2, d); lg3 = rm.load_vol_log(d)
check("T6d 저장→재로딩 왕복", len(lg3) == 4 and lg3[['vol_har', 'vol_gjr']].iloc[1:].notna().all().all())
short = rm.append_vol_forecasts(rm._empty_log(), '2026-11-09', 'Y', mk(garch(150, 8), start='2026-03-02'))
check("T6e 데이터 150일: 기존 3후보는 기록, HAR만 NaN(조용히 스킵)",
      len(short) == 3 and short['vol_har'].isna().all() and short['vol_gjr'].notna().all() and short['vol_ewmaf'].notna().all() and short['vol_ewma'].notna().all())

# T6f 빠른 EWMA: 항상 ≥ 느린 EWMA, 충격 직후엔 더 큼, 일정 수익이면 동일
ce = mk(np.tile([0.01, -0.01], 200))
check("T6f ewmaf = ewma (일정 크기 수익)", abs(rm.ewma_fast_vol(ce) - rm.ewma_vol(ce)) < 1e-9)
rs = garch(500, 4); rs[-1] = -0.2; cs = mk(rs)
check("T6g 충격 직후 ewmaf > ewma", rm.ewma_fast_vol(cs) > rm.ewma_vol(cs) * 1.05, f"{rm.ewma_fast_vol(cs):.2f} vs {rm.ewma_vol(cs):.2f}")
check("T6h ewmaf 항상 ≥ ewma", all(rm.ewma_fast_vol(mk(garch(200, k))) >= rm.ewma_vol(mk(garch(200, k))) for k in range(10)))

# T7 성적표: 새 지표, 일관성, 그림자 NaN 행 제외
sc_df = lg3.copy(); sc_df['settled'] = 1; sc_df['vol_realized'] = [0.4, 0.5, 0.6, 0.7]
sc_df.loc[sc_df.index[1:], 'horizon_days'] = [1, 1, 1]
sc = rm.score_vol_log(sc_df)
check("T7a bias_var 컬럼 존재 + 구 bias 유지", {'bias_var', 'bias'}.issubset(sc.columns))
e = sc[sc.candidate == 'vol_ewma'].iloc[0]
exp_bv = float((sc_df['vol_realized'] ** 2).sum() / (sc_df['vol_ewma'] ** 2).sum())
check("T7b bias_var = 수동 계산", abs(e['bias_var'] - exp_bv) < 1e-12)
check("T7c 그림자 후보는 NaN 행(옛 행) 제외하고 n=3", int(sc[sc.candidate == 'vol_har'].iloc[0]['n']) == 3 and int(e['n']) == 4)
# 보정된 예측(실현 = 예측·z, z~N(0,1) 의 1일 값)에서는 bias_var≈1 이지만 구 bias(산술평균 비율)는 ≈0.8 → 구 지표가 오해를 부르는 것을 확인
rng = np.random.default_rng(5); N = 20000; pred = np.full(N, 0.5); real = np.abs(rng.standard_normal(N)) * pred
cal = pd.DataFrame({'run_date': pd.Timestamp('2026-10-20'), 'ticker': 'Z', 'price_date': pd.bdate_range('2000-01-03', periods=N),
                    'horizon_days': 1, 'vol_ewma': pred, 'vol_rv20': pred, 'vol_rv60': pred, 'target_date': pd.Timestamp('2026-10-21'),
                    'vol_realized': real, 'settled': 1, 'vol_har': np.nan, 'vol_gjr': np.nan, 'vol_ewmaf': np.nan})
sc2 = rm.score_vol_log(cal); c0 = sc2[sc2.candidate == 'vol_ewma'].iloc[0]
check("T7d 보정된 예측: bias_var≈1.0(±5%), 구 bias≈0.8", abs(c0['bias_var'] - 1) < 0.05 and abs(c0['bias'] - 0.798) < 0.02, f"bias_var={c0['bias_var']:.3f} bias={c0['bias']:.3f}")

print("\n실패 없음" if not FAIL else f"\n실패 {len(FAIL)}건: {FAIL}")
raise SystemExit(1 if FAIL else 0)
