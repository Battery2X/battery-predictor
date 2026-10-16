"""risk_model 합성 데이터 검증 (네트워크 불필요). 실행: python test_risk_model.py"""
import os, shutil, tempfile
from math import sqrt
import numpy as np
import pandas as pd
import risk_model as rm

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS" if cond else "FAIL"), name, extra)
    ok &= bool(cond)

rng = np.random.default_rng(11)
TD = 252

def make_close(returns, start='2024-01-02'):
    idx = pd.bdate_range(start, periods=len(returns) + 1)
    return pd.Series(100 * np.exp(np.concatenate([[0], np.cumsum(returns)])), index=idx)

# T1 알려진 값: 매일 ±1% 로그수익 → 제곱평균 제곱근 변동성 = 0.01*sqrt(252)
r = np.tile([0.01, -0.01], 150)
c = make_close(r)
exp = 0.01 * sqrt(TD)
check("T1a rolling_rv 정확", abs(rm.rolling_rv(c, 20) - exp) < 1e-9, f"({rm.rolling_rv(c,20):.4f} vs {exp:.4f})")
check("T1b ewma 정확(일정 크기 수익 → 동일)", abs(rm.ewma_vol(c) - exp) < 1e-6)

# T2 미래 정보 미사용: t 이후 값을 바꿔도 t 시점 예측은 불변
base = make_close(rng.normal(0, 0.01, 400))
t = 300
f1 = rm.forecast_vol(base.iloc[:t])
mod = base.copy(); mod.iloc[t:] *= 5.0
f2 = rm.forecast_vol(mod.iloc[:t])
check("T2 룩어헤드 없음(미래 변조해도 예측 동일)", all(abs(f1[k] - f2[k]) < 1e-12 for k in f1))

# T3 방어: 빈/짧은/NaN/0·음수 가격/이상 타입
bad_inputs = [None, [], pd.Series(dtype=float), pd.Series([1.0]), pd.Series([np.nan] * 100),
              pd.Series([0.0, -1.0] * 50), "abc", pd.Series(range(1, 20))]
res = []
for b in bad_inputs:
    try:
        res.append(rm.forecast_vol(b)['vol_ewma'] is None and rm.risk_snapshot(b) is None)
    except Exception as e:
        res.append(False); print("  예외:", repr(b)[:30], e)
check("T3 이상 입력에서도 예외 없이 None", all(res))
dirty = base.copy(); dirty.iloc[50] = np.nan; dirty.iloc[60] = 0.0; dirty.iloc[70] = np.inf
check("T3b 일부 오염(NaN/0/inf)돼도 계산됨", rm.forecast_vol(dirty)['vol_ewma'] is not None)

# T4 정산 정렬: price_date 이후 정확히 h거래일의 수익률로 계산, 아직 안 지났으면 미정산
r = rng.normal(0, 0.012, 120)
c = make_close(r)
pdate = c.index[40]
h = 5
manual = sqrt(np.mean(np.diff(np.log(c.iloc[40:46].to_numpy())) ** 2) * TD)
check("T4a realized_vol_after = 수동 계산", abs(rm.realized_vol_after(c, pdate, h) - manual) < 1e-12)
fri_pos = [i for i in range(30, 80) if c.index[i].weekday() == 4][0]
sat = c.index[fri_pos] + pd.Timedelta(days=1)
check("T4b price_date가 토요일이면 직전 금요일 종가 기준",
      abs(rm.realized_vol_after(c, sat, h) - rm.realized_vol_after(c, c.index[fri_pos], h)) < 1e-12)
check("T4c 아직 h일 안 지났으면 None", rm.realized_vol_after(c.iloc[:44], pdate, h) is None)
check("T4d 딱 h일 지난 시점엔 계산됨", rm.realized_vol_after(c.iloc[:46], pdate, h) is not None)

# T5 로그: 추가·중복 방지·스키마·정산·빈 로그/없는 파일
tmp = tempfile.mkdtemp()
df = rm.load_vol_log(tmp)
check("T5a 파일 없으면 빈 로그(스키마 유지)", len(df) == 0 and list(df.columns) == rm.VOL_LOG_COLUMNS)
hist = make_close(rng.normal(0, 0.012, 300))
today = hist.index[-1] + pd.Timedelta(days=1)
df = rm.append_vol_forecasts(df, today, 'SOXL', hist)
check("T5b 호라이즌 3개 추가", len(df) == 3 and set(df['horizon_days']) == {1, 5, 20})
df = rm.append_vol_forecasts(df, today, 'SOXL', hist)
check("T5c 같은 가격일 재실행은 중복 추가 안 함", len(df) == 3)
df = rm.append_vol_forecasts(df, today, 'SOXL', hist.iloc[:10])
check("T5d 데이터 부족이면 추가 안 함(조용히 스킵)", len(df) == 3)
rm.save_vol_log(df, tmp)
df2 = rm.load_vol_log(tmp)
check("T5e 저장→재로딩 왕복", len(df2) == 3 and df2['price_date'].notna().all())

# 정산: 미래 6일치가 생기면 h=1,5는 정산, h=20은 보류
fut = make_close(rng.normal(0, 0.012, 306), start=str(hist.index[0].date()))
# fut 앞 300개 구간이 hist 와 같도록 구성
fut.iloc[:len(hist)] = hist.values
settled = rm.settle_vol_log(df2, lambda t: fut)
st = settled.set_index('horizon_days')['settled'].to_dict()
check("T5f 정산: h=1,5 완료·h=20 보류", st[1] == 1 and st[5] == 1 and st[20] == 0, str(st))
exp5 = rm.realized_vol_after(fut, hist.index[-1], 5)
check("T5g 정산값 = 직접 계산값", abs(settled.set_index('horizon_days').at[5, 'vol_realized'] - exp5) < 1e-12)
none_settle = rm.settle_vol_log(df2, lambda t: None)
check("T5h 시세 조회 실패 시 그대로 유지", (none_settle['settled'] == 0).all())
def boom(t): raise RuntimeError("network")
check("T5i 시세 조회 예외도 삼킴", (rm.settle_vol_log(df2, boom)['settled'] == 0).all())
shutil.rmtree(tmp)

# T6 성적표: 완벽한 예측 < 편향된 예측 (QLIKE 낮음), 편향 지표
rows = []
for i in range(60):
    real = 0.30 + 0.1 * rng.random()
    rows.append({'run_date': pd.Timestamp('2026-01-01'), 'ticker': 'X', 'price_date': pd.Timestamp('2026-01-01') + pd.Timedelta(days=i),
                 'horizon_days': 5, 'vol_ewma': real, 'vol_rv20': real * 1.5, 'vol_rv60': real * 0.6,
                 'target_date': pd.NaT, 'vol_realized': real, 'settled': 1})
sc = rm.score_vol_log(pd.DataFrame(rows, columns=rm.VOL_LOG_COLUMNS)).set_index('candidate')
check("T6 성적표: 정확한 후보 QLIKE≈0, 틀린 후보는 더 큼, 편향 방향 맞음",
      sc.at['vol_ewma', 'qlike'] < 1e-9 and sc.at['vol_rv20', 'qlike'] > 0.05 and sc.at['vol_rv20', 'bias'] < 1 < sc.at['vol_rv60', 'bias'])
check("T6b 정산 행 없으면 빈 성적표", rm.score_vol_log(rm.load_vol_log(tempfile.mkdtemp())).empty)

# T7 스냅샷 + 레짐(합성: 조용한 상승 vs 급락+고변동)
calm = make_close(rng.normal(0.0008, 0.006, 400))
crash_r = np.concatenate([rng.normal(0.0008, 0.006, 340), rng.normal(-0.01, 0.035, 60)])
crash = make_close(crash_r)
sc_, sk = rm.risk_snapshot(calm), rm.risk_snapshot(crash)
check("T7a 스냅샷 필드 존재", sc_ is not None and all(k in sc_ for k in ['move_1d', 'move_5d', 'vol_percentile', 'drawdown60', 'regime']))
check("T7b 급락장의 예상 변동폭이 조용한 장보다 큼", sk['move_5d'] > 2 * sc_['move_5d'], f"({sk['move_5d']*100:.1f}% vs {sc_['move_5d']*100:.1f}%)")
check("T7c 급락장 라벨: 고변동·200일선 아래 또는 큰 낙폭", sk['regime'] is not None and sk['regime'].startswith('고변동'), sk['regime'])
check("T7d 조용한 장 라벨: 저/중간변동", sc_['regime'] is not None and not sc_['regime'].startswith('고변동'), sc_['regime'])
check("T7e 5일 1σ = 연율/√(252/5) 관계", abs(sc_['move_5d'] - sc_['vol_ewma'] * sqrt(5 / TD)) < 1e-12)

# T8 [핵심 한계 검증] GARCH 합성: 변동성 예측 vs 방향 예측의 '검증에 필요한 표본'
from scipy import stats
def garch(n, omega=2e-6, a=0.08, b=0.90, seed=0):
    g = np.random.default_rng(seed); v = omega / (1 - a - b); r = np.empty(n)
    for i in range(n):
        r[i] = g.normal() * sqrt(v); v = omega + a * r[i] ** 2 + b * v
    return r
h = 5
def power_vol(n_windows, sims=150):
    """벡터화 EWMA(= rm.ewma_vol 과 동치, 아래에서 확인)로 겹치지 않는 h일 구간 n개의 검정력을 잰다."""
    hit = 0
    for s_ in range(sims):
        r = garch(300 + n_windows * h + h + 5, seed=9000 + s_)
        ew = np.sqrt((pd.Series(r) ** 2).ewm(alpha=0.06, adjust=False).mean() * TD).to_numpy()
        f, real = [], []
        for k in range(n_windows):
            t0 = 299 + k * h
            f.append(ew[t0]); real.append(np.sqrt((r[t0 + 1:t0 + 1 + h] ** 2).mean() * TD))
        rho, p = stats.spearmanr(f, real)
        hit += (rho > 0) and (p / 2 < 0.05)
    return hit / sims
_r = garch(500, seed=1); _c = make_close(_r)
_ew = np.sqrt((np.log(_c).diff().dropna() ** 2).ewm(alpha=0.06, adjust=False).mean() * TD)
check("T8-0 벡터화 EWMA == risk_model.ewma_vol", abs(_ew.iloc[299] - rm.ewma_vol(_c.iloc[:301])) < 1e-9)
print("  [T8] 변동성 예측 실력을 5% 수준으로 확인할 확률(검정력), 서로 안 겹치는 5일 구간 n개:")
pw = {n: power_vol(n) for n in (20, 40, 80)}
for n, p in pw.items():
    print(f"       n={n:3d}구간(≈{n*h/252:.1f}년): 검정력 {p*100:.0f}%")
check("T8a 검정력이 표본과 함께 증가하고, 약 1.6년(80구간)이면 80% 이상 (방향 3%p는 같은 조건에서 약 43년 필요)",
      pw[20] < pw[40] < pw[80] and pw[80] >= 0.80, f"({pw[20]*100:.0f}/{pw[40]*100:.0f}/{pw[80]*100:.0f}%)")
# EWMA가 '과거 250일 평균 변동성(상수 예측)'보다 QLIKE가 낮은가
def qlike(f, real):
    q = (np.array(real) / np.array(f)) ** 2
    return float((q - np.log(q) - 1).mean())
wins = 0; S = 100
for s in range(S):
    r = garch(700, seed=5000 + s); c = make_close(r)
    fe, fc, rr = [], [], []
    for t0 in range(300, 690, h):
        fe.append(rm.ewma_vol(c.iloc[:t0 + 1]))
        fc.append(float(np.sqrt((np.diff(np.log(c.iloc[t0 - 250:t0 + 1])) ** 2).mean() * TD)))
        rr.append(rm.realized_vol_after(c, c.index[t0], h))
    wins += qlike(fe, rr) < qlike(fc, rr)
print(f"  [T8] EWMA가 250일 상수 변동성보다 QLIKE 우수: {wins}/{S}")
check("T8b 지속적인 변동성(GARCH)에서 EWMA가 상수 예측을 이김(≥70%)", wins / S >= 0.70)

print("\nALL PASS" if ok else "\nSOME FAIL")
raise SystemExit(0 if ok else 1)
