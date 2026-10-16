"""
btc_leadlag_test.py — "비트코인이 SOXL/반도체보다 먼저 움직이나?"를 데이터로 검정한다 (수동 실행용, 예측 모델과 무관).

방법
- BTC 일간 수익률의 k일 전 값(k=1..3)과 오늘의 반도체(SMH)·나스닥(QQQ) 수익률의 상관을 계산.
  (yfinance BTC-USD 일봉은 UTC 자정에 끝나므로 k>=1 이면 미국 개장 전에 확정된 값 → 진짜 '선행'만 본다.)
- 유의성: BTC 시계열을 원형 이동(circular shift, 최소 20일)시킨 순열검정. 자기상관·꼬리 구조를 보존해서
  단순 t-검정보다 보수적이다. 여러 lag를 동시에 보므로 '최대 |상관|' 기준 p값(다중비교 보정)도 같이 낸다.
- 해석 기준: 보정 p < 0.05 이고 상관의 부호·크기가 구간(전반/후반)별로 일관할 때만 '후보'. 그래도 실전 검증 전까진 신호 아님.

실행: python btc_leadlag_test.py [시작일 기본 2020-01-01]   (GitHub Actions 'tools' 워크플로우 또는 로컬 PC)
"""
import sys
import numpy as np
import pandas as pd


def leadlag_stats(btc_ret, tgt_ret, max_lag=3, n_perm=2000, seed=0, min_shift=20):
    """
    btc_ret, tgt_ret: 같은 인덱스(미국 거래일)의 일간 수익률 Series.
    반환 dict: lags=[{lag, corr, p_single}], p_max(다중비교 보정), n
    """
    df = pd.concat([btc_ret.rename('b'), tgt_ret.rename('t')], axis=1).dropna()
    n = len(df)
    if n < 100 or max_lag < 1:
        return None
    b = df['b'].to_numpy(); t = df['t'].to_numpy()

    def corrs(bv):
        out = []
        for k in range(1, max_lag + 1):
            x, y = bv[:-k], t[k:]
            sx, sy = x.std(), y.std()
            out.append(0.0 if sx == 0 or sy == 0 else float(np.corrcoef(x, y)[0, 1]))
        return np.array(out)

    obs = corrs(b)
    rng = np.random.default_rng(seed)
    perm = np.empty((n_perm, max_lag))
    for i in range(n_perm):
        s = int(rng.integers(min_shift, n - min_shift))
        perm[i] = corrs(np.roll(b, s))
    p_single = (1 + (np.abs(perm) >= np.abs(obs)).sum(axis=0)) / (n_perm + 1)
    p_max = (1 + (np.abs(perm).max(axis=1) >= np.abs(obs).max()).sum()) / (n_perm + 1)
    return {'n': n, 'p_max': float(p_max),
            'lags': [{'lag': k + 1, 'corr': float(obs[k]), 'p_single': float(p_single[k])}
                     for k in range(max_lag)]}


def half_split_consistent(btc_ret, tgt_ret, lag=1):
    """전반/후반 구간에서 lag 상관의 부호가 같은지 (일관성 점검)."""
    df = pd.concat([btc_ret.shift(lag).rename('b'), tgt_ret.rename('t')], axis=1).dropna()
    h = len(df) // 2
    c1 = df.iloc[:h].corr().iloc[0, 1]
    c2 = df.iloc[h:].corr().iloc[0, 1]
    return float(c1), float(c2), bool(np.sign(c1) == np.sign(c2))


def _load(ticker, start):
    import yfinance as yf
    d = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if d is None or d.empty:
        return None
    c = d['Close']
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    c.index = pd.to_datetime(c.index).tz_localize(None)
    return c.dropna()


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else '2020-01-01'
    btc, smh, qqq = _load('BTC-USD', start), _load('SMH', start), _load('QQQ', start)
    if btc is None or smh is None or qqq is None:
        print("데이터 조회 실패 — 검정 불가 (조용히 종료)")
        return
    # BTC 는 주말에도 거래 → 미국 거래일 기준으로 '직전 거래일 종가 대비' 수익률로 재계산하면 주말 변동이 월요일에 합산된다.
    btc_on_us = btc.reindex(smh.index.union(btc.index)).ffill().reindex(smh.index)
    r_btc = np.log(btc_on_us).diff()
    for name, s in (('SMH(반도체)', smh), ('QQQ(나스닥)', qqq)):
        r_t = np.log(s.reindex(smh.index)).diff()
        res = leadlag_stats(r_btc, r_t, max_lag=3)
        print(f"\n== BTC → {name} (표본 {res['n']}일) ==")
        for L in res['lags']:
            print(f"  BTC {L['lag']}일 전 ↔ 오늘 수익률: 상관 {L['corr']:+.3f}, p={L['p_single']:.3f}")
        c1, c2, same = half_split_consistent(r_btc, r_t, 1)
        print(f"  다중비교 보정 p(최대|상관| 기준) = {res['p_max']:.3f}")
        print(f"  1일 lag 전반/후반 상관: {c1:+.3f} / {c2:+.3f} → 부호 {'일관' if same else '불일관'}")
        verdict = '후보(추가 검증 필요)' if (res['p_max'] < 0.05 and same) else '선행성 근거 없음'
        print(f"  판정: {verdict}")
    print("\n※ 상관이 있어도 '공통 요인(금리·유동성)'이 둘을 같이 움직인 것일 수 있어 인과로 읽으면 안 됩니다.")


if __name__ == '__main__':
    main()
