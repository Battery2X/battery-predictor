"""backtest_vol.py 합성 데이터 검증 (네트워크 불필요). 실행: python test_backtest_vol.py"""
import os, tempfile
import numpy as np, pandas as pd
import backtest_vol as bv
import risk_model as rm

FAIL = []
def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))
    if not cond: FAIL.append(name)

def mk(r, start='2020-01-02'):
    return pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range(start, periods=len(r)))
def garch(n, seed, omega=2e-6, a=0.08, g=0.08, b=0.85):
    rng = np.random.default_rng(seed); s2 = omega / (1 - a - g / 2 - b); out = []
    for _ in range(n):
        e = np.sqrt(s2) * rng.standard_t(6) / np.sqrt(1.5); out.append(e); s2 = omega + (a + (g if e < 0 else 0)) * e * e + b * s2
    return np.array(out)

check("T1 qlike: 완벽 예측 0, 틀리면 양수", abs(bv.qlike([1, 2], [1, 2]).sum()) < 1e-12 and bv.qlike([2], [1])[0] > 0 and bv.qlike([1], [2])[0] > 0)
m, lo, hi = bv.block_boot_mean(np.full(200, 0.3), 20)
check("T2 블록부트스트랩: 상수열이면 CI = 상수", abs(m - 0.3) < 1e-12 and abs(lo - 0.3) < 1e-9 and abs(hi - 0.3) < 1e-9)
cov = np.mean([(lambda r: r[1] < 0 < r[2])(bv.block_boot_mean(np.random.default_rng(s).normal(0, 1, 400), 20, nb=300)) for s in range(40)])
check("T2b 평균 0 잡음 40회: CI 가 0 을 포함하는 비율 ≥ 85% (명목 95%)", cov >= 0.85, f"{cov:.2f}")
check("T2c 표본 짧으면 NaN CI(예외 없음)", np.isnan(bv.block_boot_mean([1.0, 2.0], 20)[1]))

r = garch(800, 3); r[500] = -0.25
c = mk(r); ds, events = bv.days_since_shock(c)
near = [e for e in events if abs((e - c.index[500]).days) <= 4]
check("T3 주입한 충격(500)이 이벤트 1개로 묶이고, 충격일 이후 일수가 하루씩 증가", len(near) == 1 and ds.iloc[500] == 0 and ds.iloc[501] == 1 and ds.iloc[510] == 10, str(ds.iloc[498:503].tolist()))
check("T3b 충격 이전은 (대부분) 충격 없음 또는 더 이른 충격 기준", ds.iloc[:100].isna().mean() > 0.5 or ds.iloc[:100].dropna().min() >= 0)

# 룩어헤드: T 이후 가격을 바꿔도 T 행의 후보 예측은 불변
c2 = c.copy(); c2.iloc[601:] *= 2.0
w1 = bv.walk(c, step=50, start=300); w2 = bv.walk(c2, step=50, start=300)
j = w1.merge(w2, on=['date', 'h'], suffixes=('_a', '_b'))
j = j[j.date <= c.index[600]]
same = all(np.allclose(j[f'{k}_a'], j[f'{k}_b'], equal_nan=True) for k in ['ewma', 'rv20', 'rv60', 'har', 'gjr', 'ewmaf'])
check("T4 룩어헤드 없음(T 이후 변조해도 후보 불변)", same and len(j) > 0, f"행 {len(j)}")

# 전체 실행: 합성 3종목 + 데이터 부족 종목 + 쓰레기 종목
d = tempfile.mkdtemp()
closes = {'A': mk(garch(900, 11)), 'B': mk(garch(900, 12)), 'SHORT': mk(garch(100, 13)), 'BAD': pd.Series([np.nan] * 500)}
txt = bv.run(closes, step=10, out_dir=d)
check("T5 run: 결과 파일 생성·핵심 섹션 포함", os.path.exists(f"{d}/backtest_vol.txt") and '종목 간 일관성' in txt and '충격 후 조건부 편향' in txt)
check("T5b 데이터 부족/불량 종목은 예외 없이 '데이터 부족'으로 처리", txt.count('데이터 부족') >= 2, str(txt.count('데이터 부족')))
check("T5c 매수/매도/추천 문구 없음", not any(w in txt for w in ['매수', '매도', '사라', '팔아']))
check("T5d 빈 입력도 예외 없음", isinstance(bv.run({}, out_dir=None), str))

print("\n실패 없음" if not FAIL else f"\n실패 {len(FAIL)}건: {FAIL}")
raise SystemExit(1 if FAIL else 0)
