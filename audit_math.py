"""audit_math.py — 방향예측 파이프라인 수학 감사 (합성데이터, 정답을 아는 상태에서 파이프라인을 압박 테스트).
모델/로직은 건드리지 않는다(읽기 전용 진단). 결과: reports/audit_math.txt
E1 라벨 불일치(횡보 제외 학습 vs ret>0 평가)가 기준선을 얼마나 부풀리나
E2 Isotonic 보정이 신호를 보존하나(원시 vs 보정 AUC, 납작해짐)
E3 TimeSeriesSplit gap=0 vs gap=h 의 OOF 누수 크기(무신호 데이터에서 AUC가 0.5를 넘는지)
E4 극한 입력(상수/급등락/단일클래스/짧은 데이터/강한 추세/폭락)에서 안 죽고 확률 범위·대칭 유지
"""
import sys, time, numpy as np, pandas as pd
import common as cm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from backtest_2026 import auc_score

OUT = []
def P(s=""):
    print(s); OUT.append(str(s))

def macro(idx, seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame({'QQQ': 100.0, 'SMH': 100.0, 'SPY': 100.0, 'VIX': 20 + r.normal(0, 2, len(idx)),
                         'TNX': 4 + r.normal(0, .1, len(idx))}, index=idx)

def make(ar=0.0, drift=0.0, vol=.012, n=1500, seed=0):
    r = np.random.default_rng(seed); e = r.normal(drift, vol, n); x = np.zeros(n)
    for i in range(1, n): x[i] = ar * x[i - 1] + e[i]
    idx = pd.bdate_range("2019-01-01", periods=n)
    c = pd.Series(100 * np.exp(np.cumsum(x)), index=idx)
    return pd.DataFrame({'High': c * 1.01, 'Low': c * .99, 'Close': c}), idx

def prepared(df, idx, h, cap, bench='QQQ'):
    d = df.join(macro(idx), how='left').ffill().dropna()
    d = cm.build_features(d, benchmark_col=cm.REL_COMPARISON[bench])
    d = cm.build_target(d, h, cap / 3)
    return d.replace([np.inf, -np.inf], np.nan)

# ---------------------------------------------------------------- E1
def e1():
    P("== E1. 라벨 불일치: 학습 라벨(횡보 제외) 상승비율 vs 평가 라벨(ret>0) 상승비율 ==")
    P("무신호 랜덤워크(드리프트만 다름). 학습 기준선이 평가 기준선보다 얼마나 높게 잡히나(%p)")
    P("drift/일  h   train_y평균  ret>0평균  횡보비율  차이(%p)")
    for drift in (0.0, 0.0004, 0.001):
        for h, cap in ((1, .04), (5, .08), (20, .15)):
            df, idx = make(drift=drift, seed=3)
            d = prepared(df, idx, h, cap)
            d = d.dropna(subset=['Target_Ret'])
            y = d['Target'].dropna()
            P(f"{drift:<9} {h:<3} {y.mean()*100:9.1f} {(d['Target_Ret']>0).mean()*100:10.1f} {d['DeadZone_Ratio'].iloc[-1]*100:9.1f} {(y.mean()-(d['Target_Ret']>0).mean())*100:9.1f}")

# ---------------------------------------------------------------- 공통: OOF 재현(gap 옵션)
def oof_fit(X, y, gap=0, params=None, n_splits=3):
    params = params or {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.08}
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    xs = XGBClassifier(eval_metric='logloss', random_state=42, **params)
    lg = LGBMClassifier(verbose=-1, random_state=42, **params)
    p, l = [], []
    for tr, te in tscv.split(X):
        if y.iloc[tr].nunique() < 2: continue
        sc = StandardScaler(); a = sc.fit_transform(X.iloc[tr]); b = sc.transform(X.iloc[te])
        xs.fit(a, y.iloc[tr]); lg.fit(a, y.iloc[tr])
        p.extend(((xs.predict_proba(b) + lg.predict_proba(b)) / 2)[:, 1].tolist()); l.extend(y.iloc[te].tolist())
    return np.array(p), np.array(l)

# ---------------------------------------------------------------- E2
def e2():
    P("\n== E2. Isotonic 보정이 신호를 보존하나 (심은 신호 강도별) ==")
    P("OOF로 보정기를 학습 → 뒤쪽 홀드아웃(마지막 25%) 예측에서 원시 vs 보정 비교. h=1")
    P("AR   원시AUC  보정AUC  원시sd  보정sd  보정고유값수")
    for ar in (0.0, 0.1, 0.3, 0.6):
        raws, cals, ys = [], [], []; sd_r = sd_c = nu = []
        res = []
        for seed in (1, 2, 3):
            df, idx = make(ar=ar, seed=seed, n=1800)
            d = prepared(df, idx, 1, .04).dropna(subset=['Target'] + cm.FEATURE_COLUMNS[:12])
            feats = [c for c in cm.FEATURE_COLUMNS if c in d.columns and d[c].notna().all()]
            X, y = d[feats], d['Target']
            cut = int(len(X) * .75)
            p, l = oof_fit(X.iloc[:cut], y.iloc[:cut])
            iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(p, l)
            sc = StandardScaler(); a = sc.fit_transform(X.iloc[:cut])
            xs = XGBClassifier(eval_metric='logloss', random_state=42, n_estimators=100, max_depth=3, learning_rate=.08).fit(a, y.iloc[:cut])
            lg = LGBMClassifier(verbose=-1, random_state=42, n_estimators=100, max_depth=3, learning_rate=.08).fit(a, y.iloc[:cut])
            b = sc.transform(X.iloc[cut:]); raw = ((xs.predict_proba(b) + lg.predict_proba(b)) / 2)[:, 1]
            cal = iso.predict(raw)
            res.append((auc_score(raw, y.iloc[cut:].values), auc_score(cal, y.iloc[cut:].values), raw.std() * 100, cal.std() * 100, len(np.unique(np.round(cal, 6)))))
        r = np.nanmean(np.array(res, float), axis=0)
        P(f"{ar:<4} {r[0]:.3f}    {r[1]:.3f}    {r[2]:5.1f}   {r[3]:5.1f}   {r[4]:.0f}")

# ---------------------------------------------------------------- E3
def e3():
    P("\n== E3. OOF 누수: 무신호 데이터에서 gap=0 vs gap=h 의 OOF AUC (0.5보다 높으면 누수/과대평가) ==")
    P("h   gap0 AUC(평균±sd)   gap=h AUC(평균±sd)   (5개 시드)")
    for h, cap in ((1, .04), (5, .08), (20, .15)):
        a0, a1 = [], []
        for seed in range(11, 16):
            df, idx = make(seed=seed, n=1600)
            d = prepared(df, idx, h, cap).dropna(subset=['Target'])
            feats = [c for c in cm.FEATURE_COLUMNS if c in d.columns and d[c].notna().all()]
            X, y = d[feats], d['Target']
            for gap, acc in ((0, a0), (h, a1)):
                p, l = oof_fit(X, y, gap=gap)
                acc.append(auc_score(p, l))
        P(f"{h:<3} {np.mean(a0):.3f}±{np.std(a0):.3f}        {np.mean(a1):.3f}±{np.std(a1):.3f}")

# ---------------------------------------------------------------- E4
def e4():
    P("\n== E4. 극한 입력 스트레스: 죽지 않고, 확률 1~99%, 롱+인버스=100% 유지되나 ==")
    cases = {}
    n = 900; idx = pd.bdate_range("2020-01-01", periods=n)
    def frame(c): c = pd.Series(c, index=idx[:len(c)]); return pd.DataFrame({'High': c * 1.01, 'Low': c * .99, 'Close': c})
    r = np.random.default_rng(7)
    cases['상수 가격'] = frame(np.full(n, 100.0))
    cases['매일 +1% 직진(항상 상승)'] = frame(100 * 1.01 ** np.arange(n))
    cases['폭락 -50% 후 반등'] = frame(100 * np.exp(np.cumsum(np.r_[r.normal(0, .01, 600), np.full(5, -.14), r.normal(0, .01, 295)])))
    cases['±10% 지그재그'] = frame(100 * np.exp(np.cumsum(np.where(np.arange(n) % 2 == 0, .1, -.1))))
    cases['초저변동(0.01%)'] = frame(100 * np.exp(np.cumsum(r.normal(0, .0001, n))))
    cases['초고변동(8%)'] = frame(100 * np.exp(np.cumsum(r.normal(0, .08, n))))
    cases['짧은 데이터(250행)'] = frame(100 * np.exp(np.cumsum(r.normal(0, .01, 250))))
    m = macro(idx)
    real = cm.fetch_ticker_data
    try:
        for name, df in cases.items():
            m = macro(df.index)
            cm.fetch_ticker_data = lambda t, _d=df, **k: _d.copy()
            line = []
            for h, cap in ((1, .04), (5, .08), (20, .15)):
                try:
                    res = cm.train_benchmark_model('QQQ', m, h, cap, min_train_rows=300, model_params={'n_estimators': 60, 'max_depth': 3, 'learning_rate': .1})
                except Exception as e:
                    line.append(f"h{h}: 예외 {type(e).__name__}"); continue
                if res is None: line.append(f"h{h}: 스킵(None)"); continue
                ok = 1 <= res['up_prob'] <= 99 and abs(res['up_prob'] + res['down_prob'] - 100) < 1e-9 and np.isfinite(res['edge'])
                line.append(f"h{h}: 상승{res['up_prob']:.0f}% 기준{res['base_rate']:.0f}% {'OK' if ok else '이상!'}")
            P(f"{name:<22} " + " | ".join(line))
    finally:
        cm.fetch_ticker_data = real

if __name__ == '__main__':
    which = sys.argv[1:] or ['e1', 'e2', 'e3', 'e4']
    t0 = time.time()
    for w in which:
        {'e1': e1, 'e2': e2, 'e3': e3, 'e4': e4}[w](); P(f"  ({w} {time.time()-t0:.0f}s)")
    import os; os.makedirs('reports', exist_ok=True)
    open('reports/audit_math.txt', 'w').write("\n".join(OUT) + "\n")
