"""btc_leadlag_test 합성 검증: 귀무(선행 없음)에서는 기각 안 하고, 심은 선행(lag1, 계수 0.3)은 찾아야 한다."""
import numpy as np
import pandas as pd
from btc_leadlag_test import leadlag_stats, half_split_consistent

ok = True
def check(n, c):
    global ok
    print("PASS" if c else "FAIL", n); ok &= bool(c)

idx = pd.bdate_range('2020-01-01', periods=1500)
# 귀무: 독립 + 변동성 군집(GARCH 비슷)
def garchy(rng, n):
    r = np.zeros(n); v = 1.0
    for i in range(n):
        r[i] = rng.normal() * np.sqrt(v)
        v = 0.1 + 0.15 * r[i] ** 2 + 0.8 * v
    return r

false_pos = 0
trials = 20
for s in range(trials):
    rng = np.random.default_rng(100 + s)
    b = pd.Series(garchy(rng, len(idx)), index=idx)
    t = pd.Series(garchy(rng, len(idx)), index=idx)
    res = leadlag_stats(b, t, max_lag=3, n_perm=300, seed=s)
    false_pos += res['p_max'] < 0.05
print(f"귀무 오탐 {false_pos}/{trials} (기대 ≈ 5% = 1건 안팎)")
check("귀무 오탐률 낮음(<=3/20)", false_pos <= 3)

# 심은 선행: t[i] = 0.3*b[i-1] + noise
rng = np.random.default_rng(7)
b = pd.Series(rng.normal(size=len(idx)), index=idx)
t = 0.3 * b.shift(1).fillna(0) + pd.Series(rng.normal(size=len(idx)), index=idx)
res = leadlag_stats(b, t, max_lag=3, n_perm=500, seed=1)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res['lags'][0].items()}, 'p_max', res['p_max'])
check("심은 lag1 상관 탐지(p_max<0.05)", res['p_max'] < 0.05)
check("lag1 상관 ≈ 0.29 (0.2~0.4)", 0.2 < res['lags'][0]['corr'] < 0.4)
c1, c2, same = half_split_consistent(b, t, 1)
check("심은 선행은 전반/후반 일관", same)

# 방어: 표본 부족 / 상수열
check("표본 부족 시 None", leadlag_stats(b.iloc[:50], t.iloc[:50]) is None)
const = pd.Series(1.0, index=idx)
r = leadlag_stats(const, t, n_perm=50)
check("상수열에서 예외 없이 상관 0", r is not None and all(L['corr'] == 0.0 for L in r['lags']))
print("\nALL PASS" if ok else "\nSOME FAIL"); raise SystemExit(0 if ok else 1)
