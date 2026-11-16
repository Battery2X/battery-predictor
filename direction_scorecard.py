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
    d['p'] = (d['predicted_up_prob'] / 100.0).clip(0.0, 1.0)
    return d.sort_values('day')


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
    for name in LOGS:
        path = os.path.join(log_dir, f"predictions_{name}.csv")
        try:
            d = _prep(pd.read_csv(path), name)
        except Exception as e:
            lines.append(f"\n[{name}] 읽기 실패(건너뜀): {e}")
            continue
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
    text = "\n".join(lines) + "\n"
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
