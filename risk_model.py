"""
risk_model.py — 방향이 아니라 '얼마나 흔들릴지(변동성)'를 예측하고, 그 예측을 로그로 쌓아 검증한다. (v26)

왜 만들었나 (실제 로그로 확인한 한계)
- 방향 엣지 3%p를 통계적으로 확인하려면 독립 관측이 약 2,178건 필요하다(검정력 80%).
  5일 예측은 14~43년, 20일 예측은 58~173년이 걸린다 → 방향 적중률로는 사실상 검증 불가.
- 변동성은 연속값이고 지속성이 강해서 훨씬 적은 표본으로 검증된다(test_risk_model.py T7 시뮬레이션).

원칙
- 기존 방향 모델·피처·임계값·예측 로그(predictions_*.csv)는 건드리지 않는다 → 방향 모델의 검증 시계는 리셋되지 않는다.
- 변동성 예측은 새 로그(logs/vol_forecasts.csv)에 별도로 쌓는다. 이 모델의 검증 시계는 오늘부터 새로 시작한다.
- 레버리지 ETF 자신의 가격으로 변동성을 계산한다(지수 변동성 × 3 으로 다시 곱하는 이중 계산 방지).
- 포지션 규모·매매 추천은 하지 않는다. '예상 변동폭'과 '현재 상태 설명'만 제공한다.
- 모든 함수는 잘못된 입력·빈 데이터에서 예외 대신 None/빈 결과를 돌려준다(리포트가 죽지 않도록).
"""
import os

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

TD = 252
VOL_LOG_NAME = "vol_forecasts.csv"
VOL_LOG_COLUMNS = ['run_date', 'ticker', 'price_date', 'horizon_days',
                   'vol_ewma', 'vol_rv20', 'vol_rv60',
                   'target_date', 'vol_realized', 'settled']
CANDIDATES = ['vol_ewma', 'vol_rv20', 'vol_rv60']


# ---------------------------------------------------------------- 기초 계산
def _clean_close(close):
    try:
        s = pd.Series(close).astype(float)
    except Exception:
        return None
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    s = s[s > 0]
    return s if len(s) else None


def log_returns(close):
    s = _clean_close(close)
    if s is None or len(s) < 2:
        return None
    return np.log(s).diff().dropna()


def ewma_vol(close, lam=0.94, min_obs=30):
    """RiskMetrics식 EWMA 변동성(연율). 마지막 종가까지의 정보만 사용."""
    r = log_returns(close)
    if r is None or len(r) < min_obs:
        return None
    v = (r ** 2).ewm(alpha=1 - lam, adjust=False).mean().iloc[-1]
    return float(np.sqrt(v * TD))


def rolling_rv(close, window):
    """최근 window일 일수익률의 제곱평균 제곱근(연율)."""
    r = log_returns(close)
    if r is None or len(r) < window:
        return None
    return float(np.sqrt((r.iloc[-window:] ** 2).mean() * TD))


def forecast_vol(close):
    """후보 3개를 모두 계산해 돌려준다. 어느 것이 나은지는 로그 성적표(score_vol_log)가 판단한다."""
    return {'vol_ewma': ewma_vol(close), 'vol_rv20': rolling_rv(close, 20), 'vol_rv60': rolling_rv(close, 60)}


# ---------------------------------------------------------------- 현재 상태 설명(예측 아님)
def risk_snapshot(close):
    """리포트용: 예상 변동폭 + 현재 상태 설명. 데이터가 모자라면 None."""
    try:
        s = _clean_close(close)
        if s is None or len(s) < 61:
            return None
        f = forecast_vol(s)
        if f['vol_ewma'] is None:
            return None
        vol = f['vol_ewma']
        r = log_returns(s)
        rv20_hist = (r ** 2).rolling(20).mean().dropna()
        rv20_hist = np.sqrt(rv20_hist * TD).iloc[-750:]
        pct = float((rv20_hist <= rv20_hist.iloc[-1]).mean()) if len(rv20_hist) >= 60 else None
        high60 = float(s.iloc[-60:].max())
        dd = float(s.iloc[-1] / high60 - 1)
        above200 = bool(s.iloc[-1] > s.iloc[-200:].mean()) if len(s) >= 200 else None
        snap = {
            'price_date': pd.Timestamp(s.index[-1]) if isinstance(s.index, pd.DatetimeIndex) else None,
            'vol_ewma': vol, 'vol_rv20': f['vol_rv20'], 'vol_rv60': f['vol_rv60'],
            'move_1d': vol / np.sqrt(TD),              # 하루 1σ
            'move_5d': vol * np.sqrt(5 / TD),          # 5거래일 1σ
            'vol_percentile': pct, 'drawdown60': dd, 'above_sma200': above200,
        }
        snap['regime'] = _regime_label(pct, above200, dd)
        return snap
    except Exception as e:
        print(f"⚠️ risk_snapshot 실패(무시): {e}")
        return None


def _regime_label(vol_pct, above200, dd):
    """단순 규칙의 '현재 상태 설명'. 미래를 예측하는 라벨이 아니다."""
    if vol_pct is None:
        return None
    vol = "고변동" if vol_pct >= 0.7 else ("저변동" if vol_pct <= 0.3 else "중간변동")
    if above200 is None:
        trend = "추세 불명"
    elif above200 and dd > -0.10:
        trend = "200일선 위"
    elif above200:
        trend = "200일선 위·고점 대비 큰 낙폭"
    else:
        trend = "200일선 아래"
    return f"{vol} · {trend}"


# ---------------------------------------------------------------- 변동성 예측 로그
def _empty_log():
    return pd.DataFrame(columns=VOL_LOG_COLUMNS)


def load_vol_log(log_dir="logs"):
    path = os.path.join(log_dir, VOL_LOG_NAME)
    try:
        if not os.path.exists(path):
            return _empty_log()
        df = pd.read_csv(path)
        if len(df) > 0 and not {'ticker', 'price_date', 'horizon_days'}.issubset(df.columns):
            raise ValueError("필수 컬럼(ticker/price_date/horizon_days) 없음 — 변동성 로그가 아닌 파일")
        for c in VOL_LOG_COLUMNS:
            if c not in df.columns:
                df[c] = np.nan
        for c in ('run_date', 'price_date', 'target_date'):
            df[c] = pd.to_datetime(df[c], errors='coerce')
        return df[VOL_LOG_COLUMNS]
    except Exception as e:
        # 읽기에 실패한 원본을 덮어쓰지 않도록 먼저 보존한다(이후 저장이 기록을 날리는 사고 방지)
        try:
            import shutil
            from datetime import datetime
            shutil.copy2(path, f"{path}.corrupt_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        except Exception:
            pass
        print(f"⚠️ 변동성 로그 읽기 실패(원본 보존 후 빈 로그로 대체): {e}")
        return _empty_log()


def save_vol_log(df, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    df.to_csv(os.path.join(log_dir, VOL_LOG_NAME), index=False)


def append_vol_forecasts(df, run_date, ticker, close, horizons=(1, 5, 20)):
    """종목 1개의 오늘 예측을 호라이즌별로 추가. 같은 (가격일, 종목, 호라이즌)은 중복 추가하지 않는다."""
    s = _clean_close(close)
    if s is None or not isinstance(s.index, pd.DatetimeIndex):
        return df
    f = forecast_vol(s)
    if any(f[c] is None for c in CANDIDATES):
        return df
    pdate = pd.Timestamp(s.index[-1]).normalize()
    new = []
    for h in horizons:
        dup = ((df['ticker'] == ticker) & (df['price_date'] == pdate) & (df['horizon_days'] == h)).any() if len(df) else False
        if dup:
            continue
        new.append({'run_date': pd.Timestamp(run_date).normalize(), 'ticker': ticker, 'price_date': pdate,
                    'horizon_days': h, **{c: f[c] for c in CANDIDATES},
                    'target_date': pdate + BDay(h), 'vol_realized': np.nan, 'settled': 0})
    if not new:
        return df
    add = pd.DataFrame(new, columns=VOL_LOG_COLUMNS)
    return add if len(df) == 0 else pd.concat([df, add], ignore_index=True)


def realized_vol_after(close, price_date, horizon_days):
    """price_date 종가 이후 h거래일 동안의 실현변동성(연율, 제곱평균 제곱근). 아직 h일이 안 지났으면 None."""
    s = _clean_close(close)
    if s is None or not isinstance(s.index, pd.DatetimeIndex):
        return None
    pos = s.index.searchsorted(pd.Timestamp(price_date), side='right') - 1
    if pos < 0 or pos + int(horizon_days) > len(s) - 1:
        return None
    seg = np.log(s.iloc[pos:pos + int(horizon_days) + 1]).diff().dropna()
    if len(seg) < int(horizon_days):
        return None
    return float(np.sqrt((seg ** 2).mean() * TD))


def settle_vol_log(df, price_lookup):
    """미정산 행 중 호라이즌이 지난 것을 실현변동성으로 채운다. price_lookup(ticker) -> Close Series 또는 None."""
    if len(df) == 0:
        return df
    df = df.copy()
    todo = df['settled'].fillna(0).astype(int) == 0
    for t in df.loc[todo, 'ticker'].unique():
        try:
            close = price_lookup(t)
        except Exception as e:
            print(f"⚠️ [{t}] 시세 조회 실패(정산 보류): {e}")
            continue
        if close is None:
            continue
        for idx in df[todo & (df['ticker'] == t)].index:
            rv = realized_vol_after(close, df.at[idx, 'price_date'], df.at[idx, 'horizon_days'])
            if rv is not None:
                df.at[idx, 'vol_realized'] = rv
                df.at[idx, 'settled'] = 1
    return df


def score_vol_log(df):
    """
    후보별 성적표. QLIKE(낮을수록 좋음)와 편향(실현/예측 평균; 1.0이 이상적).
    n_eff = 서로 안 겹치는 구간 수(가격일 수 / 호라이즌)로 보수적으로 표기.
    """
    if len(df) == 0:
        return pd.DataFrame()
    s = df[df['settled'].fillna(0).astype(int) == 1].copy()
    out = []
    for h, g in s.groupby('horizon_days'):
        n_eff = g['price_date'].nunique() / float(h)
        for c in CANDIDATES:
            x = g[[c, 'vol_realized']].dropna()
            x = x[(x[c] > 0) & (x['vol_realized'] > 0)]
            if len(x) == 0:
                continue
            ratio = (x['vol_realized'] / x[c]) ** 2
            qlike = float((ratio - np.log(ratio) - 1).mean())
            out.append({'horizon': int(h), 'candidate': c, 'n': len(x), 'n_eff': round(n_eff, 1),
                        'qlike': qlike, 'bias': float((x['vol_realized'] / x[c]).mean())})
    return pd.DataFrame(out)
