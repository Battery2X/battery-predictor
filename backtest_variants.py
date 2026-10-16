"""backtest_variants.py — 라벨·보정 수정안을 '실전 코드를 건드리지 않고' 같은 워크포워드로 비교 (읽기 전용 진단).
V0 현행(횡보 제외 라벨 + Isotonic) / V1 라벨 통일(전 구간 ret>0 + Isotonic) /
V2 라벨 통일 + Platt(로지스틱) 보정 / V3 현행 라벨 + Platt.
평가는 모두 동일: 실제 h일 수익률>0, 시점-사전 기준선 대비 Brier 차이(블록 부트스트랩), AUC, IC.
※ 이 파일은 common.py 를 수정하지 않는다. 결과가 좋아도 실전 적용은 별도 결정(검증 시계 리셋)이다.
"""
import os, sys, json, numpy as np, pandas as pd
import common as cm
import backtest_2026 as B
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

VARIANTS = {'V0_현행': ('deadzone', 'iso'), 'V1_라벨통일': ('all', 'iso'),
            'V2_라벨통일+Platt': ('all', 'platt'), 'V3_현행라벨+Platt': ('deadzone', 'platt')}


def train_variant(label, calib, benchmark, macro_df, horizon_days, threshold_cap, benchmark_multiplier=3,
                  min_train_rows=300, model_params=None, extra_features_df=None, extra_feature_cols=None):
    """cm._train_benchmark_model_impl 과 동일 흐름. 바뀐 것은 라벨(label)과 보정기(calib)뿐. 실패 시 None."""
    model_params = model_params or {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05}
    df = cm.fetch_ticker_data(benchmark)
    if df is None:
        return None
    df = df.join(macro_df, how='left')
    if extra_features_df is not None:
        df = df.join(extra_features_df, how='left')
    df = df.ffill().dropna()
    df = cm.build_features(df, benchmark_col=cm.REL_COMPARISON[benchmark])
    df = cm.build_target(df, horizon_days, threshold_cap / benchmark_multiplier)
    df = df.replace([np.inf, -np.inf], np.nan)
    if label == 'all':   # 횡보 포함: 평가 라벨(ret>0)과 같은 정의
        df['Target'] = np.where(df['Target_Ret'].notna(), (df['Target_Ret'] > 0).astype(float), np.nan)
    cols = cm.FEATURE_COLUMNS + (extra_feature_cols or [])
    dt = df.dropna(subset=['Target'] + cols).copy()
    if len(dt) < min_train_rows:
        return None
    X, y = dt[cols], dt['Target']
    if y.nunique() < 2:
        return None
    xgb = XGBClassifier(eval_metric='logloss', random_state=42, **model_params)
    lgbm = LGBMClassifier(verbose=-1, random_state=42, **model_params)
    oof_p, oof_l, ok = [], [], 0
    for tr, te in TimeSeriesSplit(n_splits=3).split(X):
        if y.iloc[tr].nunique() < 2:
            continue
        ok += 1
        sc = StandardScaler(); a = sc.fit_transform(X.iloc[tr]); b = sc.transform(X.iloc[te])
        xgb.fit(a, y.iloc[tr]); lgbm.fit(a, y.iloc[tr])
        oof_p.extend((((xgb.predict_proba(b) + lgbm.predict_proba(b)) / 2)[:, 1]).tolist()); oof_l.extend(y.iloc[te].tolist())
    if ok == 0 or len(set(oof_l)) < 2:
        return None
    oof_p, oof_l = np.array(oof_p), np.array(oof_l)
    if calib == 'iso':
        cal = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0).fit(oof_p, oof_l)
        f = lambda v: float(cal.predict([v])[0])
    else:   # Platt: 원시확률의 logit 에 로지스틱 회귀 (단조·매끄러움)
        lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
        pl = LogisticRegression(C=1.0).fit(lg(oof_p).reshape(-1, 1), oof_l)
        f = lambda v: float(pl.predict_proba(lg(np.array([v])).reshape(-1, 1))[0, 1])
    sc = StandardScaler(); a = sc.fit_transform(X)
    xgb.fit(a, y); lgbm.fit(a, y)
    last = sc.transform(df[cols].iloc[[-1]])
    raw = float((((xgb.predict_proba(last) + lgbm.predict_proba(last)) / 2))[0][1])
    up = f(raw)
    if not np.isfinite(up):
        return None
    up = min(max(up, cm.PROB_FLOOR), 1 - cm.PROB_FLOOR)
    base = float(y.mean() * 100)
    return {'up_prob': up * 100, 'down_prob': (1 - up) * 100, 'raw_up_prob': raw * 100, 'base_rate': base, 'edge': up * 100 - base}


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else '2026-01-01'
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    outdir = sys.argv[3] if len(sys.argv) > 3 else 'reports'
    sel = [x.strip() for x in os.environ.get('BT_VARIANTS', '').split(',') if x.strip()]
    todo = {k: v for k, v in VARIANTS.items() if (not sel or any(k.startswith(x) for x in sel))} or VARIANTS
    per_variant = int(os.environ.get('BT_MAX_SECONDS', '4000')) // len(todo)
    os.makedirs(outdir, exist_ok=True)
    pm = {}
    for b in B.BENCHMARKS:
        d = cm.fetch_ticker_data(b)
        if d is not None and len(d):
            pm[b] = d
    if not pm:
        open(os.path.join(outdir, 'backtest_variants.txt'), 'w', encoding='utf-8').write('시세 조회 실패'); return
    macro = cm.fetch_macro_data()
    extra = {}
    try:
        br = cm.fetch_constituent_breadth_features(); extra['SMH'] = br if (br is not None and len(br)) else None
    except Exception:
        extra['SMH'] = None
    allp, L = [], [f"[라벨·보정 수정안 비교 백테스트] {start}~ · {step}거래일 간격 · 평가=실제 h일 수익률>0 · 실전 적중률 아님", ""]
    for name, (lab, cal) in todo.items():
        fn = lambda b, m, h, cap, _l=lab, _c=cal, **k: train_variant(_l, _c, b, m, h, cap, **k)
        p = B.run_backtest(pm, macro, start, step, train_fn=fn, extra_map=extra, max_seconds=per_variant, log=lambda *_: None)
        if p.empty:
            L.append(f"{name}: 예측 없음"); continue
        p = B.attach_outcomes(p, pm); p['variant'] = name; allp.append(p)
        L.append(f"== {name} (라벨={lab}, 보정={cal}) ==")
        agg = []
        for (b, h), g in sorted(p.groupby(['benchmark', 'h'])):
            s = B.summarize_group(g, h, step)
            if s.get('n', 0) < 5:
                L.append(f"  {b} {h:>2}일: 표본 부족"); continue
            agg.append(s)
            L.append(f"  {b} {h:>2}일 | 적중 {s['hit']:.0f}% (늘상승 {s['naive_always_up']:.0f}%·기준선 {s['ex_ante_base_hit']:.0f}%) | 평균예측 {s['avg_prob']:.0f}% vs 실제상승 {s['naive_always_up']:.0f}% "
                     f"| Brier차 {s['brier_diff']:+.4f} ({s['brier_diff_ci'][0]:+.4f}~{s['brier_diff_ci'][1]:+.4f}) | AUC {s['auc']:.2f} IC {s['ic_spearman']:+.2f} | 상승예측 {s['pred_up_share']:.0f}% → {s['verdict']}")
        if agg:
            L.append(f"  요약: 평균 Brier차 {np.mean([s['brier_diff'] for s in agg]):+.4f} · 평균 AUC {np.nanmean([s['auc'] for s in agg]):.3f} · "
                     f"평균 |예측-실제상승| {np.mean([abs(s['avg_prob']-s['naive_always_up']) for s in agg]):.1f}%p · 유의하게 우수 {sum('우수' in s['verdict'] for s in agg)}/{len(agg)} · 유의하게 나쁨 {sum('나쁨' in s['verdict'] for s in agg)}/{len(agg)}")
        L.append("")
    text = "\n".join(L); print(text)
    open(os.path.join(outdir, 'backtest_variants.txt'), 'w', encoding='utf-8').write(text)
    if allp:
        pd.concat(allp, ignore_index=True).to_csv(os.path.join(outdir, 'backtest_variants_preds.tsv'), index=False, sep='\t')


if __name__ == '__main__':
    main()
