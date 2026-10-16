"""
backtest_2026.py — 워크포워드 백테스트 + 올해 경향성 비교 (검증성 평가용, 읽기 전용).

무엇을 하나
- 각 기준일 T 마다 '그날까지의 데이터만' 잘라서 실전과 똑같은 학습·보정(train_benchmark_model)을 돌리고,
  h거래일 뒤 실제 방향과 비교한다. (모델·피처·임계값은 건드리지 않는다 → 검증 시계 영향 없음, 로그도 안 건드림)
- 평가는 '그날 알 수 있던 정보'로 만든 기준선(과거 상승비율)과 비교한다(사후 평균이 아니라 ex-ante).
- 올해(2026) 시장 경향(추세·변동성·월별)과 나눠서, 어떤 국면에서 맞고 틀렸는지 본다.

한계(꼭 읽을 것)
- 백테스트는 '실전 적중률'이 아니다. 설계·임계값을 보고 난 뒤의 과거 구간이라 낙관적일 수 있다.
- 벤치마크(QQQ/SMH/SPY) 방향을 평가한다(롱 ETF 방향의 대용). 인버스는 확률 반전이라 대칭.
- SMH 구성종목 폭 피처는 쓰지 않는다(가능하면 포함 시도, 실패 시 생략) — 실전과 약간 다를 수 있다.
- 겹치는 표본: 5일·20일 예측은 서로 겹친다 → 독립 관측 수(n_eff)로 판단.

실행: python backtest_2026.py [시작일=2026-01-01] [간격(거래일)=3] [출력폴더=reports]
"""
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import common as cm

BENCHMARKS = ['QQQ', 'SMH', 'SPY']
HORIZONS = {1: dict(cap=0.04, params={'n_estimators': 150, 'max_depth': 3, 'learning_rate': 0.08}),
            5: dict(cap=0.08, params=None),
            20: dict(cap=0.15, params=None)}


# ---------------------------------------------------------------- 통계 (순수 함수)
def wilson(k, n, z=1.96):
    return cm.wilson_ci(k, n, z)


def auc_score(p, y):
    p, y = np.asarray(p, float), np.asarray(y, int)
    if len(set(y)) < 2:
        return np.nan
    r = pd.Series(p).rank().values
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a, b):
    a, b = pd.Series(a), pd.Series(b)
    if len(a) < 5 or a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return float(a.rank().corr(b.rank()))


def block_bootstrap_diff(d, block, n_boot=2000, seed=0):
    """d: 시간순 손실차(모델-기준선) 배열. 블록 부트스트랩으로 평균의 95% 구간. 반환 (평균, 하한, 상한)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 8:
        return (float(np.mean(d)) if n else np.nan, np.nan, np.nan)
    block = max(1, min(int(block), n))
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        means[i] = np.concatenate([d[s:s + block] for s in starts])[:n].mean()
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def ex_ante_base_rates(close, h, asofs):
    """각 기준일 T 에서 '그때까지 결과가 확정된 과거'의 상승비율(0~1). 확정 = t+h <= T."""
    ret = close.shift(-h) / close - 1
    label = (ret > 0).astype(float).where(ret.notna())
    pos = {d: i for i, d in enumerate(close.index)}
    out = {}
    cum = label.cumsum()
    cnt = label.notna().cumsum()
    for T in asofs:
        i = pos[T] - h                       # t <= T-h 인 라벨만 확정
        if i < 0 or cnt.iloc[i] == 0:
            out[T] = 0.5
        else:
            out[T] = float(cum.iloc[i] / cnt.iloc[i])
    return out


# ---------------------------------------------------------------- 백테스트 본체
def run_backtest(price_map, macro_df, start, step, benchmarks=BENCHMARKS, horizons=HORIZONS, train_fn=None,
                 extra_map=None, log=print, max_seconds=None):
    """price_map: {벤치마크: Close 포함 DataFrame(High/Low/Close)}. train_fn 기본=실전 학습 함수. 반환: 예측 DataFrame."""
    train_fn = train_fn or cm.train_benchmark_model
    rows = []
    real_fetch = cm.fetch_ticker_data
    t0 = time.time()
    try:
        for b in benchmarks:
            full = price_map.get(b)
            if full is None or len(full) < 400:
                log(f"⚠️ {b}: 데이터 부족 — 건너뜀")
                continue
            idx = full.index
            asofs = [d for d in idx[idx >= pd.Timestamp(start)]][::max(1, int(step))]
            for T in asofs:
                cm.fetch_ticker_data = (lambda t, _full=full, _T=T, **k: _full.loc[:_T].copy())
                m = macro_df.loc[:T] if macro_df is not None else macro_df
                ex = None
                if extra_map and b in extra_map and extra_map[b] is not None:
                    ex = extra_map[b].loc[:T]
                for h, cfg in horizons.items():
                    if max_seconds and time.time() - t0 > max_seconds:
                        log("⏱ 시간 한도 도달 — 여기까지의 결과로 마무리")
                        return pd.DataFrame(rows)
                    kw = dict(min_train_rows=300)
                    if cfg['params']:
                        kw['model_params'] = cfg['params']
                    if ex is not None and len(ex):
                        kw['extra_features_df'] = ex
                    try:
                        r = train_fn(b, m, h, cfg['cap'], **kw)
                    except Exception as e:
                        r = None
                    if r is None:
                        continue
                    rows.append({'asof': T, 'benchmark': b, 'h': h, 'up_prob': float(r['up_prob']),
                                 'raw_up_prob': float(r['raw_up_prob']), 'edge': float(r['edge']),
                                 'model_base_rate': float(r['base_rate'])})
            log(f"  {b}: {sum(1 for x in rows if x['benchmark'] == b)}건 완료 ({time.time() - t0:.0f}s)")
    finally:
        cm.fetch_ticker_data = real_fetch
    return pd.DataFrame(rows)


def attach_outcomes(preds, price_map):
    """실제 h거래일 수익률·방향, 직전 20일 추세, 20일 변동성을 붙인다. 아직 안 지난 건 NaN."""
    out = []
    for (b, h), g in preds.groupby(['benchmark', 'h']):
        c = price_map[b]['Close']
        fwd = c.shift(-h) / c - 1
        tr20 = c / c.shift(20) - 1
        vol20 = np.log(c).diff().rolling(20).std() * np.sqrt(252)
        g = g.copy()
        g['actual_ret'] = g['asof'].map(fwd)
        g['trend20'] = g['asof'].map(tr20)
        g['vol20'] = g['asof'].map(vol20)
        g['actual_up'] = (g['actual_ret'] > 0).astype(float).where(g['actual_ret'].notna())
        base = ex_ante_base_rates(c, h, list(g['asof']))
        g['ex_ante_base'] = g['asof'].map(base)
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else preds


def summarize_group(g, h, step):
    g = g[g['actual_up'].notna()].sort_values('asof')
    n = len(g)
    if n < 5:
        return {'n': n}
    y = g['actual_up'].values
    p = g['up_prob'].values / 100.0
    pb = g['ex_ante_base'].values
    hit = float(((p >= 0.5) == (y == 1)).mean())
    base_pred_hit = float(((pb >= 0.5) == (y == 1)).mean())
    up_rate = float(y.mean())
    span_days = n * step
    n_eff = int(max(1, min(n, round(span_days / max(h, 1)))))
    lo, hi = wilson(int(round(hit * n_eff)), n_eff)
    d = (p - y) ** 2 - (pb - y) ** 2
    mean_d, d_lo, d_hi = block_bootstrap_diff(d, block=math.ceil(max(h, 5) / max(step, 1)))
    verdict = ('모델이 기준선보다 유의하게 우수' if (np.isfinite(d_hi) and d_hi < 0) else
               '모델이 기준선보다 유의하게 나쁨' if (np.isfinite(d_lo) and d_lo > 0) else '기준선과 구분 불가')
    return {'n': n, 'n_eff': n_eff, 'hit': hit * 100, 'hit_ci': (lo, hi), 'naive_always_up': up_rate * 100,
            'naive_best': max(up_rate, 1 - up_rate) * 100, 'ex_ante_base_hit': base_pred_hit * 100,
            'brier_model': float(np.mean((p - y) ** 2)), 'brier_base': float(np.mean((pb - y) ** 2)),
            'brier_diff': mean_d, 'brier_diff_ci': (d_lo, d_hi), 'auc': auc_score(p, y),
            'ic_spearman': spearman(p, g['actual_ret']), 'avg_prob': float(p.mean() * 100),
            'pred_up_share': float((p >= 0.5).mean() * 100), 'verdict': verdict}


def regime_tables(preds):
    g = preds[preds['actual_up'].notna()].copy()
    if g.empty:
        return {}
    g['hit'] = ((g['up_prob'] >= 50) == (g['actual_up'] == 1)).astype(float)
    g['month'] = pd.to_datetime(g['asof']).dt.strftime('%Y-%m')
    g['trend'] = np.where(g['trend20'] > 0, '직전20일 상승', '직전20일 하락')
    if g['vol20'].notna().sum() >= 6:
        q = g['vol20'].quantile([1 / 3, 2 / 3]).values
        g['volreg'] = np.where(g['vol20'] <= q[0], '저변동', np.where(g['vol20'] <= q[1], '중변동', '고변동'))
    else:
        g['volreg'] = '전체'
    out = {}
    for name in ('month', 'trend', 'volreg'):
        t = g.groupby(['h', name]).agg(n=('hit', 'size'), hit=('hit', 'mean'), actual_up=('actual_up', 'mean'),
                                       avg_prob=('up_prob', 'mean')).reset_index()
        out[name] = t
    # 확률 구간별 실제 상승률(보정 점검)
    g['bin'] = pd.cut(g['up_prob'], [0, 40, 50, 60, 70, 101], right=False, labels=['<40', '40-50', '50-60', '60-70', '70+'])
    out['calib'] = g.groupby(['h', 'bin'], observed=True).agg(n=('hit', 'size'), actual_up=('actual_up', 'mean'),
                                                              avg_prob=('up_prob', 'mean')).reset_index()
    return out


def ytd_context(price_map, start):
    rows = []
    for b, df in price_map.items():
        c = df['Close'][df.index >= pd.Timestamp(start)]
        if len(c) < 5:
            continue
        r = np.log(c).diff().dropna()
        dd = (c / c.cummax() - 1).min()
        rows.append({'benchmark': b, 'ytd_return': float(c.iloc[-1] / c.iloc[0] - 1) * 100, 'up_day_share': float((r > 0).mean()) * 100,
                     'ann_vol': float(r.std() * np.sqrt(252)) * 100, 'max_drawdown': float(dd) * 100, 'days': int(len(c))})
    return pd.DataFrame(rows)


def build_report(summ, regimes, ctx, meta):
    L = []
    L.append(f"[백테스트 · 워크포워드] 기간 {meta['start']}~{meta['end']} · {meta['step']}거래일 간격 · 예측 {meta['n_pred']}건 (결과 확정 {meta['n_scored']}건)")
    L.append("※ 백테스트이며 실전 적중률이 아닙니다. 벤치마크 방향(QQQ/SMH/SPY) 기준, 겹치는 표본은 독립 관측 수(n_eff)로 판단.")
    L.append("")
    L.append("== 올해 시장 경향 ==")
    for _, r in ctx.iterrows():
        L.append(f"  {r['benchmark']}: 누적 {r['ytd_return']:+.1f}% · 상승일 비중 {r['up_day_share']:.0f}% · 연환산 변동성 {r['ann_vol']:.0f}% · 최대낙폭 {r['max_drawdown']:.0f}%")
    L.append("")
    L.append("== 벤치마크·호라이즌별 성적 ==")
    for (b, h), s in sorted(summ.items()):
        if s.get('n', 0) < 5:
            L.append(f"  {b} {h}일: 표본 {s.get('n', 0)}건 — 판정 불가")
            continue
        L.append(f"  {b} {h:>2}일 | 적중 {s['hit']:.0f}% (95%CI {s['hit_ci'][0]:.0f}~{s['hit_ci'][1]:.0f}, n_eff={s['n_eff']}) "
                 f"| 늘 상승 찍기 {s['naive_always_up']:.0f}% · 그날의 과거평균 따라가기 {s['ex_ante_base_hit']:.0f}% "
                 f"| Brier 차이 {s['brier_diff']:+.4f} (95%CI {s['brier_diff_ci'][0]:+.4f}~{s['brier_diff_ci'][1]:+.4f}) "
                 f"| AUC {s['auc']:.2f} · IC {s['ic_spearman']:+.2f} | 상승 예측 비중 {s['pred_up_share']:.0f}% → {s['verdict']}")
    L.append("")
    L.append("== 국면별(직전 20일 추세) ==")
    t = regimes.get('trend')
    if t is not None:
        for _, r in t.iterrows():
            L.append(f"  {int(r['h']):>2}일 · {r['trend']}: n={int(r['n'])} 적중 {r['hit']*100:.0f}% | 실제 상승 {r['actual_up']*100:.0f}% · 평균 예측확률 {r['avg_prob']:.0f}%")
    L.append("")
    L.append("== 국면별(변동성 3분위) ==")
    t = regimes.get('volreg')
    if t is not None:
        for _, r in t.iterrows():
            L.append(f"  {int(r['h']):>2}일 · {r['volreg']}: n={int(r['n'])} 적중 {r['hit']*100:.0f}% | 실제 상승 {r['actual_up']*100:.0f}% · 평균 예측확률 {r['avg_prob']:.0f}%")
    L.append("")
    L.append("== 월별 ==")
    t = regimes.get('month')
    if t is not None:
        for _, r in t.iterrows():
            L.append(f"  {r['month']} · {int(r['h']):>2}일: n={int(r['n'])} 적중 {r['hit']*100:.0f}% | 실제 상승 {r['actual_up']*100:.0f}% · 평균 예측확률 {r['avg_prob']:.0f}%")
    L.append("")
    L.append("== 확률 구간별 실제 상승률(보정 점검) ==")
    t = regimes.get('calib')
    if t is not None:
        for _, r in t.iterrows():
            L.append(f"  {int(r['h']):>2}일 · 예측 {r['bin']}: n={int(r['n'])} 평균 예측 {r['avg_prob']:.0f}% → 실제 상승 {r['actual_up']*100:.0f}%")
    return "\n".join(L)


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else '2026-01-01'
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    outdir = sys.argv[3] if len(sys.argv) > 3 else 'reports'
    max_sec = int(os.environ.get('BT_MAX_SECONDS', '3000'))
    os.makedirs(outdir, exist_ok=True)
    print(f"백테스트 시작: {start}~ · 간격 {step}거래일")
    price_map = {}
    for b in BENCHMARKS:
        df = cm.fetch_ticker_data(b)
        if df is not None and len(df):
            price_map[b] = df
    if not price_map:
        msg = "시세 조회 실패 — 백테스트 불가"
        open(os.path.join(outdir, 'backtest_2026.txt'), 'w', encoding='utf-8').write(msg)
        print(msg)
        return
    macro_df = cm.fetch_macro_data()
    extra = {}
    try:
        br = cm.fetch_constituent_breadth_features()
        extra['SMH'] = br if (br is not None and len(br)) else None
    except Exception:
        extra['SMH'] = None
    preds = run_backtest(price_map, macro_df, start, step, extra_map=extra, max_seconds=max_sec)
    if preds.empty:
        msg = "예측이 하나도 만들어지지 않음 — 학습 실패(원인은 실행 로그 참고)"
        open(os.path.join(outdir, 'backtest_2026.txt'), 'w', encoding='utf-8').write(msg)
        print(msg)
        return
    preds = attach_outcomes(preds, price_map)
    summ = {(b, h): summarize_group(g, h, step) for (b, h), g in preds.groupby(['benchmark', 'h'])}
    regimes = regime_tables(preds)
    ctx = ytd_context(price_map, start)
    meta = {'start': start, 'end': str(pd.Timestamp(preds['asof'].max()).date()), 'step': step,
            'n_pred': int(len(preds)), 'n_scored': int(preds['actual_up'].notna().sum())}
    text = build_report(summ, regimes, ctx, meta)
    print(text)
    open(os.path.join(outdir, 'backtest_2026.txt'), 'w', encoding='utf-8').write(text)
    preds.to_csv(os.path.join(outdir, 'backtest_2026_preds.tsv'), index=False, sep='\t')   # .csv 가 아닌 .tsv: 로그 병합기가 건드리지 않게
    ser = {f"{b}_{h}": {k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()} for (b, h), s in summ.items()}
    json.dump({'meta': meta, 'summary': ser}, open(os.path.join(outdir, 'backtest_2026.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=str)


if __name__ == '__main__':
    main()
