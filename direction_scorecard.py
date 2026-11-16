"""
direction_scorecard.py — 방향 예측 성적표 (v34, 읽기 전용: 로그·모델을 건드리지 않음 → 검증 시계 영향 없음)

왜 만들었나 (로드맵 3번 · round3 '다음 후보' 3번)
- 텔레그램의 적중률은 '맞힌 비율'만 보여준다. 모델이 거의 항상 '상승'만 찍으면 적중률 = 그 기간 상승 비율이라
  실력과 구분이 안 된다. 그래서 세 가지를 같이 본다.
    1) 항상 상승 기준선과의 차이          (적중률이 기준선과 같으면 정보 없음)
    2) Brier 점수 vs 상수 예측            (확률 숫자 자체에 정보가 있는가. 낮을수록 좋음)
    3) 독립 구간 수                       (겹치는 5일·20일 예측과 같이 움직이는 종목을 한 덩어리로 센 표본 수)
- 검증 시계: v31(2026-10-17, 학습 라벨 변경) 이전 행은 다른 라벨의 예측이라 '전체'와 'v31 이후'를 따로 낸다.

정의
- 독립 계열: 레버리지는 롱(TQQQ·SOXL·SPXL)만(인버스는 확률 반전한 거울), 개별주는 전부.
- 기준 상수 2종: 0.5(아무 정보 없음), 사후 상승비율(그 기간을 다 보고 고른 최적 상수 → 모델에 가장 불리한 엄격 기준).
- skill = 1 - Brier(모델)/Brier(기준). 0 이하면 상수 예측보다 못함.
- 독립 구간 수 = (정산된 서로 다른 기준일 수) // 호라이즌. 같은 날 여러 종목은 같은 충격을 공유하므로 1일로 센다.
- 비중첩 적중률: 기준일을 h거래일 간격으로 솎아 계산. 시작점(offset 0..h-1)에 따라 달라지므로 최소~최대를 같이 낸다.

v35 추가 (전부 읽기 전용 · 기존 predicted_up_prob 는 그대로 두고 '그림자'로만 비교)
- 확률 구간별 신뢰도표: 모델이 55~60%라고 한 날들이 실제로 몇 % 올랐나.
- 그림자 재보정: 로그에 이미 있는 값만으로 그날 알 수 있었던 대체 확률을 만들어 Brier 로 비교한다(룩어헤드 없음).
    half     = 항상 50%
    shrink50 = 모델 확률을 50% 쪽으로 절반 당김
    prior    = 그날까지 '정산이 끝난' 실전 상승비율(사전분포 Beta(10,10) = 50% 20건어치)
    base     = 그날 모델의 학습 기준선(logs/edge_history.csv 에 있을 때만)
  후보는 2026-11-16 에 사전등록했다. 그 전 행은 후보를 고를 때 이미 본 데이터라 '참고(표본 내)'이고,
  판정은 등록일 이후 행으로만 하며 독립 구간 60개 전에는 결론을 내지 않는다. 결론이 나도 모델 출력을 자동으로 바꾸지 않는다.
- 엣지 검증: 엣지(보정확률-기준선) 부호별 실제 상승비율. 엣지는 v35 부터 logs/edge_history.csv 에 쌓인다.

실행: python direction_scorecard.py [로그폴더=logs] [출력폴더=reports]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

LOGS = ['scalp', 'swing', 'position', 'stock_scalp', 'stock_swing', 'stock_position']
LONGS = ['TQQQ', 'SOXL', 'SPXL']
CLOCK_START = '2026-10-17'          # v31 라벨 통일 적용일(이전 행은 다른 라벨)
MIN_INDEP = 15                      # common.evaluate_signal 의 min_eff 와 같은 문턱
SHADOW_REG_DATE = '2026-11-16'      # 그림자 후보 사전등록일(이 날 이후 가격일만 판정에 사용)
SHADOW_MIN_INDEP = 60               # 그림자 후보 판정에 필요한 독립 구간 수(사전 고정)
PRIOR_A = PRIOR_B = 10.0            # prior 후보의 Beta 사전분포(50% 20건어치)
SHADOWS = ['half', 'shrink50', 'prior', 'base']
PROB_BINS = [(0, 50), (50, 55), (55, 60), (60, 65), (65, 100.0001)]


def _prep(df, name):
    """정산 완료 + 독립 계열 행만. 숫자가 깨진 행은 버린다. 필요한 컬럼이 없으면 빈 프레임."""
    need = ['ticker', 'horizon_days', 'predicted_label', 'predicted_up_prob', 'actual_label', 'correct']
    if df is None or len(df) == 0 or any(c not in df.columns for c in need):
        return pd.DataFrame()
    d = df.copy()
    for c in need[1:]:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna(subset=need)
    if not name.startswith('stock'):
        d = d[d['ticker'].isin(LONGS)]
    day = pd.to_datetime(d['price_date'], errors='coerce') if 'price_date' in d.columns else pd.Series(pd.NaT, index=d.index)
    fb = pd.to_datetime(d['run_date'], errors='coerce').dt.normalize() if 'run_date' in d.columns else day
    d['day'] = day.fillna(fb).dt.normalize()
    d = d.dropna(subset=['day'])
    # 같은 가격일·같은 종목 예측이 두 번 기록된 옛 행(v25 이전 토요일+월요일 실행 등)은 한 번만 센다(첫 기록 유지)
    d = d.sort_values('run_date', kind='stable').drop_duplicates(['day', 'ticker'], keep='first') if 'run_date' in d.columns else d.drop_duplicates(['day', 'ticker'], keep='first')
    d['p'] = (d['predicted_up_prob'] / 100.0).clip(0.0, 1.0)
    d['tday'] = pd.to_datetime(d['target_date'], errors='coerce').dt.normalize() if 'target_date' in d.columns else pd.NaT
    return d.sort_values('day', kind='stable')


def load_edge(log_dir):
    """logs/edge_history.csv → DataFrame(log, ticker, day, base_rate, edge). 없거나 깨졌으면 빈 프레임."""
    try:
        e = pd.read_csv(os.path.join(log_dir, 'edge_history.csv'))
        e['day'] = pd.to_datetime(e['price_date'], errors='coerce').dt.normalize()
        for c in ('base_rate', 'edge'):
            e[c] = pd.to_numeric(e[c], errors='coerce')
        e['ticker'] = e['ticker'].astype(str).str.strip().str.upper()
        return e.dropna(subset=['day', 'base_rate']).drop_duplicates(['log', 'ticker', 'day'], keep='first')[['log', 'ticker', 'day', 'base_rate', 'edge']]
    except Exception:
        return pd.DataFrame(columns=['log', 'ticker', 'day', 'base_rate', 'edge'])


def add_shadows(d, edge=None, name=None):
    """그림자 확률 열(sh_half, sh_shrink50, sh_prior, sh_base)과 edge 열을 붙인다. 각 행은 그 가격일에 이미 알 수 있던 정보만 쓴다."""
    d = d.copy()
    if len(d) == 0:
        return d
    d['sh_half'] = 0.5
    d['sh_shrink50'] = 0.5 + 0.5 * (d['p'] - 0.5)
    # prior: 목표일이 이 행의 가격일 '이전 또는 같은 날'인 행만 결과가 알려져 있다(같은 날 종가로 정산되므로 포함).
    days = d['day'].values
    td = d['tday'].values
    y = d['actual_label'].values.astype(float)
    known = ~pd.isna(td)
    order_t = np.argsort(np.where(known, td, np.datetime64('2262-01-01')), kind='stable')
    ts, ys = td[order_t], y[order_t]
    cum = np.concatenate([[0.0], np.cumsum(np.where(pd.isna(ts), 0.0, ys))])
    n_known = int(known.sum())
    pos = np.searchsorted(ts[:n_known], days, side='right')
    d['sh_prior'] = (PRIOR_A + cum[pos]) / (PRIOR_A + PRIOR_B + pos)
    d['sh_base'] = np.nan
    d['edge'] = np.nan
    if edge is not None and len(edge) and name is not None:
        e = edge[edge['log'] == name]
        if len(e):
            m = d[['ticker', 'day']].merge(e, on=['ticker', 'day'], how='left')
            d['sh_base'] = (m['base_rate'].values / 100.0).clip(0.0, 1.0)
            d['edge'] = m['edge'].values
    return d


def _block_ci(day_vals, block, nb=1000, seed=0):
    """기준일별 값의 평균에 대한 이동블록 부트스트랩 95% 구간. 기준일이 3블록 미만이면 (nan, nan)."""
    x = np.asarray(day_vals, dtype=float)
    n = len(x)
    if n < 3 * block:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    means = np.empty(nb)
    for b in range(nb):
        st = rng.integers(0, n - block + 1, k)
        means[b] = np.concatenate([x[s0:s0 + block] for s0 in st])[:n].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def shadow_scores(d):
    """{후보: {'n','brier','brier_model','diff','ci_low','ci_high'}} — diff = 모델 Brier - 후보 Brier (>0 이면 후보가 더 정확)."""
    out = {}
    if d is None or len(d) == 0 or 'sh_half' not in d.columns:
        return out
    h = max(int(d['horizon_days'].median()), 1)
    y = d['actual_label'].astype(float)
    bm = (d['p'] - y) ** 2
    for k in SHADOWS:
        q = d[f'sh_{k}']
        ok = q.notna()
        if ok.sum() == 0:
            continue
        bs = (q[ok] - y[ok]) ** 2
        diff = (bm[ok] - bs)
        by_day = diff.groupby(d.loc[ok, 'day']).mean().sort_index()
        lo, hi = _block_ci(by_day.values, max(5, 2 * h))
        out[k] = {'n': int(ok.sum()), 'n_days': int(len(by_day)), 'brier': float(bs.mean()), 'brier_model': float(bm[ok].mean()),
                  'diff': float(diff.mean()), 'ci_low': lo, 'ci_high': hi}
    return out


def reliability(d):
    """확률 구간별 [구간, 행 수, 평균 예측확률%, 실제 상승%]."""
    rows = []
    if d is None or len(d) == 0:
        return rows
    for lo, hi in PROB_BINS:
        g = d[(d['p'] * 100 >= lo) & (d['p'] * 100 < hi)]
        if len(g):
            rows.append([f"{lo:.0f}~{min(hi, 100):.0f}%", int(len(g)), float(g['p'].mean() * 100), float(g['actual_label'].mean() * 100), int(g['day'].nunique())])
    return rows


def edge_table(d):
    """엣지 부호·크기별 [이름, 행 수, 기준일 수, 실제 상승%, 평균 수익률%]. 엣지 이력이 붙은 행만."""
    rows = []
    if d is None or len(d) == 0 or 'edge' not in d.columns:
        return rows
    g = d[d['edge'].notna()]
    if len(g) == 0:
        return rows
    ret = pd.to_numeric(g['actual_return'], errors='coerce') if 'actual_return' in g.columns else pd.Series(np.nan, index=g.index)
    for name, m in (('엣지>0', g['edge'] > 0), ('엣지<=0', g['edge'] <= 0), ('엣지>=+3', g['edge'] >= 3), ('엣지<=-3', g['edge'] <= -3)):
        if m.sum():
            rows.append([name, int(m.sum()), int(g.loc[m, 'day'].nunique()), float(g.loc[m, 'actual_label'].mean() * 100), float(ret[m].mean() * 100)])
    return rows


def score(d):
    """한 묶음(이미 _prep 된 프레임)의 지표 dict. 비었으면 None."""
    if d is None or len(d) == 0:
        return None
    h = max(int(d['horizon_days'].median()), 1)
    y = d['actual_label'].values.astype(float)
    p = d['p'].values
    up = float(y.mean())
    acc = float(d['correct'].mean())
    brier = float(np.mean((p - y) ** 2))
    b_half = 0.25
    b_clim = float(up * (1 - up))
    days = np.sort(d['day'].unique())
    n_indep = int(len(days) // h)
    nonover = []
    for off in range(h):
        keep = set(days[off::h])
        g = d[d['day'].isin(keep)]
        if len(g):
            nonover.append(float(g['correct'].mean()))
    return {
        'horizon': h, 'n_rows': int(len(d)), 'n_days': int(len(days)), 'n_indep': n_indep,
        'accuracy': acc * 100, 'always_up': up * 100, 'edge_vs_always_up': (acc - up) * 100,
        'pred_up_share': float((d['predicted_label'] == 1).mean()) * 100,
        'mean_prob': float(p.mean()) * 100,
        'brier': brier, 'brier_half': b_half, 'brier_clim': b_clim,
        'skill_half': 1 - brier / b_half,
        'skill_clim': (1 - brier / b_clim) if b_clim > 0 else float('nan'),
        'nonoverlap_min': min(nonover) * 100 if nonover else float('nan'),
        'nonoverlap_max': max(nonover) * 100 if nonover else float('nan'),
        'verdict': verdict(n_indep, acc - up, brier, b_clim),
    }


def verdict(n_indep, edge, brier, b_clim):
    if n_indep < MIN_INDEP:
        return f"판정 불가(독립 {n_indep}구간 < {MIN_INDEP})"
    if brier < b_clim and edge > 0:
        return "상수 예측보다 나음(유의성은 별도 확인)"
    return "상수 예측을 못 넘음"


def run(log_dir="logs", out_dir=None, clock_start=CLOCK_START):
    lines = ["방향 예측 성적표 (읽기 전용 · 실전 로그 기준 · 독립 구간 15개 미만은 판정 불가)",
             "skill>0 이면 상수 예측보다 확률이 정확함. '사후 상승비율' 기준은 모델에 가장 불리한 엄격 기준."]
    out = {}
    edge = load_edge(log_dir)
    detail = ["", "=" * 60, "확률 신뢰도 · 그림자 재보정 · 엣지 검증 (v35, 읽기 전용)",
              f"그림자 후보 사전등록일 {SHADOW_REG_DATE} · 판정은 등록일 이후 독립 구간 {SHADOW_MIN_INDEP}개부터 · 모델 출력은 자동으로 바뀌지 않음",
              "차이 = 모델 Brier - 후보 Brier (양수면 후보가 더 정확). [ ] 는 기준일 블록 부트스트랩 95% 구간, 0을 포함하면 구분 불가."]
    for name in LOGS:
        path = os.path.join(log_dir, f"predictions_{name}.csv")
        try:
            d = _prep(pd.read_csv(path), name)
        except Exception as e:
            lines.append(f"\n[{name}] 읽기 실패(건너뜀): {e}")
            continue
        try:
            ds_ = add_shadows(d, edge, name)
            if len(ds_):
                h_ = max(int(ds_['horizon_days'].median()), 1)
                detail.append(f"\n[{name}] h={h_}")
                rel = reliability(ds_)
                if rel:
                    detail.append("  신뢰도표(전체): " + " | ".join(f"{r[0]} 행{r[1]}·일{r[4]} 예측{r[2]:.0f}%→실제{r[3]:.0f}%" for r in rel))
                reg = ds_[ds_['day'] >= pd.Timestamp(SHADOW_REG_DATE)]
                n_reg = int(reg['day'].nunique() // h_) if len(reg) else 0
                for label, part in (('표본 내 참고(등록 전 포함 전체)', ds_), (f'등록 이후({SHADOW_REG_DATE}~)', reg)):
                    sc = shadow_scores(part)
                    if not sc:
                        detail.append(f"  그림자 · {label}: 정산된 행 없음")
                        continue
                    detail.append(f"  그림자 · {label}: 모델 Brier {next(iter(sc.values()))['brier_model']:.4f}")
                    for k, v in sc.items():
                        ci = f"[{v['ci_low']:+.4f},{v['ci_high']:+.4f}]" if v['ci_low'] == v['ci_low'] else "[표본 부족]"
                        detail.append(f"    {k:8s} Brier {v['brier']:.4f} (행 {v['n']}) 차이 {v['diff']:+.4f} {ci}")
                    out[f"{name}|shadow|{label}"] = sc
                detail.append(f"  → 판정: " + (f"보류(등록 이후 독립 구간 {n_reg} < {SHADOW_MIN_INDEP})" if n_reg < SHADOW_MIN_INDEP
                                                else "등록 이후 표본 충족 — 위 '등록 이후' 구간의 95% 구간이 0을 넘는 후보만 검토 대상"))
                et = edge_table(ds_)
                if et:
                    detail.append("  엣지 검증: " + " | ".join(f"{r[0]} 행{r[1]}·일{r[2]} 실제상승 {r[3]:.0f}% 평균수익 {r[4]:+.2f}%" for r in et)
                                  + "  (기준일 60일 미만이면 우연과 구분 불가)")
                    out[f"{name}|edge"] = et
        except Exception as e:
            detail.append(f"\n[{name}] 그림자/엣지 계산 실패(건너뜀): {type(e).__name__}: {e}")
        for label, part in (('전체', d), (f'v31 이후({clock_start}~)', d[d['day'] >= pd.Timestamp(clock_start)] if len(d) else d)):
            s = score(part)
            if s is None:
                lines.append(f"\n[{name} · {label}] 정산된 행 없음")
                continue
            out[f"{name}|{label}"] = s
            lines.append(f"\n[{name} · {label}] h={s['horizon']}  행 {s['n_rows']} · 기준일 {s['n_days']} · 독립 구간 {s['n_indep']}")
            lines.append(f"  적중 {s['accuracy']:.1f}% vs 항상상승 {s['always_up']:.1f}% (차 {s['edge_vs_always_up']:+.1f}%p) · "
                         f"상승예측 비중 {s['pred_up_share']:.0f}% · 평균 예측확률 {s['mean_prob']:.1f}%")
            lines.append(f"  Brier {s['brier']:.4f} | 0.5 상수 {s['brier_half']:.4f} (skill {s['skill_half']:+.3f}) | "
                         f"사후 상승비율 상수 {s['brier_clim']:.4f} (skill {s['skill_clim']:+.3f})")
            lines.append(f"  비중첩 적중률 {s['nonoverlap_min']:.0f}~{s['nonoverlap_max']:.0f}% (시작점에 따라) → {s['verdict']}")
    text = "\n".join(lines + detail) + "\n"
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "direction_scorecard.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        with open(os.path.join(out_dir, "direction_scorecard.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
    return text, out


if __name__ == "__main__":
    a = sys.argv
    print(run(a[1] if len(a) > 1 else "logs", a[2] if len(a) > 2 else "reports")[0])
