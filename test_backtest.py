"""backtest_2026 합성 검증: 무신호(랜덤워크)에서는 우위가 없어야 하고, 심은 신호는 잡아야 한다. 누수가 없는지도 본다."""
import sys, numpy as np, pandas as pd
import backtest_2026 as B
import common as cm
res = []
def check(n, c): res.append(bool(c)); print(("OK   " if c else "FAIL ") + n)

def macro(idx):
    r = np.random.default_rng(0)
    return pd.DataFrame({'QQQ': 100.0, 'SMH': 100.0, 'SPY': 100.0, 'VIX': 20 + r.normal(0, 2, len(idx)), 'TNX': 4 + r.normal(0, .1, len(idx))}, index=idx)

def make(ar, seed, n=1500):
    r = np.random.default_rng(seed); e = r.normal(0, .012, n); x = np.zeros(n)
    for i in range(1, n): x[i] = ar * x[i - 1] + e[i]
    idx = pd.bdate_range("2020-01-01", periods=n); c = pd.Series(100 * np.exp(np.cumsum(x)), index=idx)
    return pd.DataFrame({'High': c * 1.01, 'Low': c * .99, 'Close': c}), idx

H = {1: dict(cap=0.04, params={'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.08})}
def run(ar, seed):
    pm_df, idx = make(ar, seed)
    pm = {'QQQ': pm_df}; m = macro(idx)
    start = idx[-260]
    p = B.run_backtest(pm, m, start, 2, benchmarks=['QQQ'], horizons=H, log=lambda *_: None)
    p = B.attach_outcomes(p, pm)
    return B.summarize_group(p[(p.benchmark == 'QQQ') & (p.h == 1)], 1, 2), p

import time; t0 = time.time()
null_aucs = []; sig_aucs = []
for sd in (1, 2):
    s0, p0 = run(0.0, sd); null_aucs.append(s0['auc'])
    s1, p1 = run(0.6, 10 + sd); sig_aucs.append(s1['auc'])
print('무신호 AUC', np.round(null_aucs, 3), '/ 심은 신호(AR 0.6) AUC', np.round(sig_aucs, 3), f'({time.time()-t0:.0f}s)')
check('무신호 랜덤워크: 평균 AUC가 0.5 근처(0.42~0.58)', 0.42 < np.mean(null_aucs) < 0.58)
check('강한 심은 모멘텀 신호(AR 0.6, 130표본)는 평균 AUC 0.6 이상으로 포착', np.mean(sig_aucs) > 0.60)
check('심은 신호가 무신호보다 유의하게 높음', np.mean(sig_aucs) - np.mean(null_aucs) > 0.08)
# 누수: 기준일 T 의 예측은 T 이후 데이터를 바꿔도 변하지 않아야 한다
pm_df, idx = make(0.0, 5); m = macro(idx); T0 = idx[-120]
a = B.run_backtest({'QQQ': pm_df}, m, T0, 200, benchmarks=['QQQ'], horizons=H, log=lambda *_: None)
alt = pm_df.copy(); alt.loc[alt.index > T0, ['High', 'Low', 'Close']] *= np.linspace(1, 3, int((alt.index > T0).sum()))[:, None]
b = B.run_backtest({'QQQ': alt}, m, T0, 200, benchmarks=['QQQ'], horizons=H, log=lambda *_: None)
check('미래 가격을 3배로 조작해도 T 시점 예측 동일(누수 없음)', len(a) == 1 and len(b) == 1 and abs(a.up_prob[0] - b.up_prob[0]) < 1e-9)
# 통계 함수
check('AUC: 완벽 분리=1, 반대=0', auc_ok := (B.auc_score([.1, .2, .8, .9], [0, 0, 1, 1]) == 1 and B.auc_score([.9, .8, .2, .1], [0, 0, 1, 1]) == 0))
d = np.r_[np.full(60, -0.05), np.full(60, 0.0)] + np.random.default_rng(1).normal(0, .01, 120)
mn, lo, hi = B.block_bootstrap_diff(d, 5)
check('블록 부트스트랩: 평균 -0.025 부근 & 구간이 0을 포함하지 않음', -0.04 < mn < -0.01 and hi < 0)
mn, lo, hi = B.block_bootstrap_diff(np.random.default_rng(2).normal(0, .05, 200), 5)
check('블록 부트스트랩: 무차이 데이터는 구간이 0을 포함', lo < 0 < hi)
# ex-ante 기준선: 미래 라벨을 쓰지 않음
c = pd.Series(np.r_[np.linspace(100, 110, 30), np.linspace(110, 80, 30)], index=pd.bdate_range('2024-01-01', periods=60))
base = B.ex_ante_base_rates(c, 5, [c.index[10], c.index[55]])
check('기준선은 그날 확정된 과거만 사용(상승구간 초반=100%, 하락 반영 후 감소)', base[c.index[10]] >= 0.99 and base[c.index[55]] < 0.8)
# 빈/짧은 입력
check('데이터 부족 시 빈 결과(예외 없음)', B.run_backtest({'QQQ': pm_df.iloc[:100]}, macro(pm_df.index[:100]), '2020-03-01', 3, log=lambda *_: None).empty)
print(f"{sum(res)}/{len(res)} 통과"); sys.exit(0 if all(res) else 1)
