"""
backtest_vol.py — 변동성 후보(EWMA·RV20·RV60·HAR·GJR·빠른 EWMA) 워크포워드 비교 (v32, 읽기 전용: 로그를 건드리지 않음)

왜 만들었나
- 실시간 변동성 로그는 정산이 끝난 표본이 적다(1일 n=48, 5일 n_eff≈2.4). 과거 가격으로 같은 후보를 같은 정의로 평가해
  '어느 후보가 낫다고 할 근거가 있는가'를 먼저 본다. 방향 모델·기존 로그·검증 시계에는 영향이 없다.

방법
- 기준일 T 마다 그날까지의 종가만 잘라 후보를 계산(룩어헤드 없음) → T 이후 h거래일 실현변동성(risk_model.realized_vol_after 와 동일 정의)과 비교.
- 지표: QLIKE(낮을수록 좋음), 편향 bias_var = sum(실현²)/sum(예측²) (1.0 이상적, >1 과소예측),
  EWMA 대비 QLIKE 차이의 이동블록 부트스트랩 95% CI (CI가 0을 포함하면 '구분 불가').
- 충격 후 조건부 편향: 충격일 = |r_t| / (전일 EWMA 일변동성) > SHOCK_Z (사전 고정, 사후에 고르지 않음).
  기준일 T 가 마지막 충격일로부터 며칠째인지로 구간을 나눠 bias_var 를 본다. 같은 충격이 5일 안에 이어지면 하나의 이벤트로 센다.

한계
- 표본이 겹친다(블록 부트스트랩으로 불확실성만 추정, 독립 표본이 늘어나는 것은 아님). 종목 간 상관이 커서 종목별 결과를 독립 증거로 세지 말 것.
- 후보 선택을 이 백테스트로 고르면 그 자체가 다중비교다 → 사전에 정한 후보만 비교하고, 선택은 홀드아웃(다른 종목·다른 기간)으로 확인한다.

실행: python backtest_vol.py [종목쉼표목록=TQQQ,SOXL,SPXL] [간격=2] [출력폴더=reports] [데이터시작=2019-01-01]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import risk_model as rm

HORIZONS = (1, 5, 20)
CANDS = ['ewma', 'rv20', 'rv60', 'har', 'gjr', 'ewmaf']
SHOCK_Z = 2.5                       # 사전 고정 임계값(사후 조정 금지)
BUCKETS = [(0, 1, '충격일~+1일'), (2, 5, '+2~5일'), (6, 10, '+6~10일'), (11, 20, '+11~20일')]
START_OBS = 300


def qlike(real, pred):
    q = (np.asarray(real) / np.asarray(pred)) ** 2
    return q - np.log(q) - 1


def block_boot_mean(x, block, nb=1000, seed=0):
    """이동블록 부트스트랩으로 평균의 95% CI. 표본이 블록보다 짧으면 (평균, nan, nan)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < max(2 * block, 10):
        return float(np.mean(x)) if n else np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    means = []
    for _ in range(nb):
        st = rng.integers(0, n - block + 1, k)
        idx = np.concatenate([np.arange(s, s + block) for s in st])[:n]
        means.append(x[idx].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def days_since_shock(close, z=SHOCK_Z):
    """각 날짜에서 '가장 최근 충격일로부터 며칠째'(충격일 당일=0). 충격 없으면 NaN. + 이벤트 날짜 목록."""
    s = rm._clean_close(close)
    r = np.log(s).diff()
    sig_prev = np.sqrt((r ** 2).ewm(alpha=0.06, adjust=False).mean().shift(1))     # 전일까지의 EWMA 일변동성
    shock = ((r.abs() / sig_prev) > z) & (r.abs() <= rm.SPLIT_SUSPECT)
    pos = np.arange(len(s))
    last = pd.Series(np.where(shock.values, pos, np.nan), index=s.index).ffill()
    ds = pd.Series(pos, index=s.index) - last
    events, prev = [], -10 ** 9
    for i in np.where(shock.values)[0]:
        if i - prev > 5:
            events.append(s.index[i])
        prev = i
    return ds, events


def walk(close, step=2, start=START_OBS, horizons=HORIZONS):
    """기준일별 후보 예측과 실현값. 후보는 기준일까지의 종가만 사용."""
    c = rm._clean_close(close)
    if c is None or len(c) < start + 30:
        return pd.DataFrame()
    ds, _ = days_since_shock(c)
    rows = []
    for T in range(start, len(c) - 1, step):
        cs = c.iloc[:T + 1]
        base = {'ewma': rm.ewma_vol(cs), 'rv20': rm.rolling_rv(cs, 20), 'rv60': rm.rolling_rv(cs, 60), 'ewmaf': rm.ewma_fast_vol(cs)}
        for h in horizons:
            real = rm.realized_vol_after(c, c.index[T], h)
            if real is None or not np.isfinite(real) or real <= 0:
                continue
            rows.append(dict(date=c.index[T], h=h, real=real, dss=ds.iloc[T], **base,
                             har=rm.har_vol(cs, h), gjr=rm.gjr_vol(cs, h)))
    return pd.DataFrame(rows)


def summarize_one(name, df, lines):
    for h, g in df.groupby('h'):
        g = g.dropna(subset=CANDS)
        if len(g) < 30:
            lines.append(f"[{name}] h={h}: 표본 부족(n={len(g)})")
            continue
        n_nonover = max(1, len(g) * 2 // h)
        lines.append(f"\n[{name}] h={h}  n={len(g)} (겹침 있음, 비중첩≈{n_nonover})")
        base_q = qlike(g.real.values, g.ewma.values)
        for k in CANDS:
            q = qlike(g.real.values, g[k].values)
            bv = float((g.real ** 2).sum() / (g[k] ** 2).sum())
            if k == 'ewma':
                lines.append(f"  {k:6s} QLIKE {q.mean():.3f}  bias_var {bv:.2f}")
            else:
                m, lo, hi = block_boot_mean(base_q - q, max(20, 2 * h))
                tag = "우수(CI>0)" if lo > 0 else ("열위(CI<0)" if hi < 0 else "구분 불가")
                lines.append(f"  {k:6s} QLIKE {q.mean():.3f}  bias_var {bv:.2f}  EWMA대비 {m:+.3f} [{lo:+.3f},{hi:+.3f}] {tag}")


def summarize_shock(all_df, lines):
    lines.append("\n=== 충격 후 조건부 편향 bias_var (전 종목 합산, 1.0 이상적 / >1 과소예측) ===")
    lines.append(f"충격일 정의: |수익률|/전일 EWMA 일변동성 > {SHOCK_Z} (사전 고정)")
    for h, g in all_df.groupby('h'):
        lines.append(f"\nh={h}")
        for lo, hi, lab in BUCKETS + [(21, 10 ** 6, '+21일 이후/충격 없음')]:
            if lab.startswith('+21'):
                gg = g[(g.dss > 20) | g.dss.isna()]
            else:
                gg = g[(g.dss >= lo) & (g.dss <= hi)]
            if len(gg) < 10:
                lines.append(f"  {lab:14s} n={len(gg)} (부족)")
                continue
            bv = {k: float((gg.real ** 2).sum() / (gg[k] ** 2).sum()) for k in ['ewma', 'gjr', 'har', 'ewmaf']}
            lines.append(f"  {lab:14s} n={len(gg):4d}  " + "  ".join(f"{k} {v:.2f}" for k, v in bv.items()))


def run(closes, step=2, out_dir=None):
    """closes: {티커: Close Series}. 결과 문자열(와 파일)을 돌려준다."""
    lines = ["변동성 후보 워크포워드 비교 (백테스트이며 실전 성적이 아님; 후보 선택은 홀드아웃 확인 전까지 보류)"]
    frames = []
    for t, c in closes.items():
        try:
            df = walk(c, step=step)
        except Exception as e:
            lines.append(f"[{t}] 실패(무시): {e}")
            continue
        if df.empty:
            lines.append(f"[{t}] 데이터 부족")
            continue
        df['ticker'] = t
        frames.append(df)
        summarize_one(t, df, lines)
    if frames:
        allf = pd.concat(frames, ignore_index=True)
        lines.append("\n=== 종목 간 일관성: EWMA 보다 QLIKE 가 낮은 종목 수 / 전체 종목 수 ===")
        for h, g in allf.groupby('h'):
            parts = []
            for k in ['har', 'gjr', 'ewmaf', 'rv20']:
                wins = 0
                tot = 0
                for t, gt in g.groupby('ticker'):
                    gt = gt.dropna(subset=CANDS)
                    if len(gt) < 30:
                        continue
                    tot += 1
                    wins += int(qlike(gt.real.values, gt[k].values).mean() < qlike(gt.real.values, gt.ewma.values).mean())
                parts.append(f"{k} {wins}/{tot}")
            lines.append(f"  h={h}: " + "  ".join(parts) + "   (종목이 서로 상관돼 있어 독립 증거로 세지 말 것)")
        summarize_shock(allf, lines)
    text = "\n".join(lines) + "\n"
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "backtest_vol.txt"), "w", encoding="utf-8") as f:
            f.write(text)
    return text


def main(argv):
    import common as cm
    tickers = (argv[1] if len(argv) > 1 else "TQQQ,SOXL,SPXL").split(",")
    step = int(argv[2]) if len(argv) > 2 else 2
    out_dir = argv[3] if len(argv) > 3 else "reports"
    start = argv[4] if len(argv) > 4 else "2019-01-01"
    closes = {}
    for t in tickers:
        try:
            d = cm.fetch_ticker_data(t.strip(), start_date=start)
            if d is not None and len(d):
                closes[t.strip()] = d['Close'].ffill().dropna()
        except Exception as e:
            print(f"⚠️ [{t}] 시세 조회 실패(건너뜀): {e}")
    print(run(closes, step=step, out_dir=out_dir))


if __name__ == "__main__":
    main(sys.argv)
