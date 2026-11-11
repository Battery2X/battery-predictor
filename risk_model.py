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
# v32: 그림자(shadow) 후보 2개를 로그 맨 뒤에 추가. 기존 컬럼 순서·의미는 그대로(옛 로그는 새 컬럼이 NaN으로 채워져 읽힘).
VOL_LOG_COLUMNS = ['run_date', 'ticker', 'price_date', 'horizon_days',
                   'vol_ewma', 'vol_rv20', 'vol_rv60',
                   'target_date', 'vol_realized', 'settled',
                   'vol_har', 'vol_gjr', 'vol_ewmaf']
CANDIDATES = ['vol_ewma', 'vol_rv20', 'vol_rv60']          # 필수 후보(하나라도 계산 실패면 그날은 기록 안 함: v26 동작 유지)
SHADOW_CANDIDATES = ['vol_har', 'vol_gjr', 'vol_ewmaf']                  # 선택 후보(계산 실패 시 NaN으로 두고 기록은 계속)
ALL_CANDIDATES = CANDIDATES + SHADOW_CANDIDATES


# ---------------------------------------------------------------- 기초 계산
def _clean_close(close):
    try:
        s = pd.Series(close).astype(float)
    except Exception:
        return None
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    s = s[s > 0]
    if isinstance(s.index, pd.DatetimeIndex):      # v27: 정렬 안 됨/중복 날짜가 수익률을 엉뚱하게 만들지 않도록
        if not s.index.is_monotonic_increasing:
            s = s.sort_index()
        if s.index.has_duplicates:
            s = s[~s.index.duplicated(keep='last')]
    return s if len(s) else None


# 일 로그수익률 절댓값이 이 값을 넘으면(+100% / -50% 이상) 3배 ETF의 정상 움직임이 아니라
# 주식분할·역분할 미조정, 잘못된 시세로 의심한다(SOXL 역대 최악의 하루도 약 -0.4 수준).
SPLIT_SUSPECT = 0.7


def log_returns(close, keep_suspect=False):
    """일 로그수익률. 기본은 분할 의심 수익률을 제외한다(변동성이 100거래일 넘게 부풀려지는 것 방지)."""
    s = _clean_close(close)
    if s is None or len(s) < 2:
        return None
    r = np.log(s).diff().dropna()
    if not keep_suspect:
        r = r[r.abs() <= SPLIT_SUSPECT]
    return r


def suspect_returns(close, lookback=250):
    """최근 lookback개 수익률 중 분할 의심 건수와 가장 최근 의심 날짜."""
    r = log_returns(close, keep_suspect=True)
    if r is None:
        return 0, None
    r = r.iloc[-lookback:]
    bad = r[r.abs() > SPLIT_SUSPECT]
    if len(bad) == 0:
        return 0, None
    last = bad.index[-1]
    return int(len(bad)), (pd.Timestamp(last) if isinstance(last, pd.Timestamp) else None)


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


# ---------------------------------------------------------------- v32 그림자 후보 (호라이즌별 예측)
# 두 후보 모두 종가(Close)만 쓰고, 기준일까지의 수익률만 사용한다(룩어헤드 없음).
# 기존 방향 모델·기존 3개 후보의 값은 바뀌지 않는다 → 방향 모델 검증 시계 리셋 없음.
_HAR_MIN_OBS = 300
_HAR_FIT_WINDOW = 750
_VAR_FLOOR = 1e-10          # 일 수익률 0%인 날의 log(0) 방지(일 분산 하한)
# GJR-GARCH(1,1) '고정 파라미터'(적합 안 함 → 튜닝/스누핑 여지 없음). 주가지수에서 흔한 값을 사전 고정.
_GJR_ALPHA, _GJR_GAMMA, _GJR_BETA = 0.04, 0.08, 0.90
_GJR_PERSIST = _GJR_ALPHA + _GJR_GAMMA / 2.0 + _GJR_BETA


def har_vol(close, h, min_obs=_HAR_MIN_OBS, fit_window=_HAR_FIT_WINDOW):
    """
    log-HAR(일·주·월 평균 제곱수익률 3성분)로 향후 h거래일 평균 일분산을 예측 → 연율 변동성.
    - 종가만 있으므로 실현분산 대용치는 일 수익률 제곱(고빈도 RV보다 잡음 큼).
    - 회귀 목표는 'log(향후 h일 평균 r²)'. 로그 모형의 평균 편향은 Duan smearing으로 보정한다.
    - 학습 표본은 라벨이 기준일 이전에 이미 확정된 행만 쓴다(기준일 T 에서 t<=T-h).
    데이터 부족·특이행렬·비정상 값이면 None.
    """
    try:
        h = int(h)
        r = log_returns(close)
        if r is None or h < 1 or len(r) < min_obs:
            return None
        r2 = (r ** 2).clip(lower=_VAR_FLOOR)
        d = np.log(r2)
        w = np.log(r2.rolling(5).mean())
        m = np.log(r2.rolling(22).mean())
        y = np.log(r2.rolling(h).mean().shift(-h))
        X = pd.concat([d, w, m], axis=1)
        X.columns = ['d', 'w', 'm']
        x_now = X.iloc[-1].values.astype(float)
        tr = pd.concat([X, y.rename('y')], axis=1).dropna().iloc[-fit_window:]
        if len(tr) < 120 or not np.all(np.isfinite(x_now)):
            return None
        A = np.column_stack([np.ones(len(tr)), tr[['d', 'w', 'm']].values])
        beta, *_ = np.linalg.lstsq(A, tr['y'].values, rcond=None)
        resid = tr['y'].values - A @ beta
        smear = float(np.mean(np.exp(resid)))
        pred_var = float(np.exp(beta[0] + beta[1:] @ x_now) * smear)
        if not np.isfinite(pred_var) or pred_var <= 0:
            return None
        return float(np.sqrt(pred_var * TD))
    except Exception:
        return None


def gjr_vol(close, h, min_obs=120, window=750):
    """
    GJR-GARCH(1,1) 고정 파라미터 필터 + 분산 타기팅(장기분산 = 최근 window일 평균 r²). 향후 h일 평균 분산의 연율 변동성.
    하락일(r<0)에 충격 계수가 커지는 비대칭 반응을 반영한다. 파라미터를 적합하지 않으므로 최적화 과적합이 없다.
    """
    try:
        h = int(h)
        r = log_returns(close)
        if r is None or h < 1 or len(r) < min_obs:
            return None
        x = r.iloc[-window:].values.astype(float)
        lrv = float(np.mean(x ** 2))
        if not np.isfinite(lrv) or lrv <= 0:
            return None
        omega = (1.0 - _GJR_PERSIST) * lrv
        s2 = lrv
        for e in x:                                   # s2 = 다음 날 조건부 분산
            s2 = omega + (_GJR_ALPHA + (_GJR_GAMMA if e < 0 else 0.0)) * e * e + _GJR_BETA * s2
        ks = np.arange(h)                              # k=1..h → 지수 k-1
        path = lrv + (_GJR_PERSIST ** ks) * (s2 - lrv)
        avg = float(np.mean(path))
        if not np.isfinite(avg) or avg <= 0:
            return None
        return float(np.sqrt(avg * TD))
    except Exception:
        return None


def ewma_fast_vol(close, lam_slow=0.94, lam_fast=0.80, min_obs=30):
    """비대칭 속도 EWMA(경험적 휴리스틱): max(EWMA 0.94, EWMA 0.80). 충격 직후엔 빠른 쪽을 따라 오르고 평시엔 느린 쪽.
    문헌상 이 정확한 형태의 근거는 확인하지 못했다(후임 3종 모두 불확실) → 경험적 검증만 한다. 호라이즌과 무관한 값."""
    a, b = ewma_vol(close, lam_slow, min_obs), ewma_vol(close, lam_fast, min_obs)
    if a is None or b is None:
        return None
    return float(max(a, b))


def forecast_vol_shadow(close, h):
    """호라이즌 h 에서의 그림자 후보 값 {'vol_har','vol_gjr','vol_ewmaf'} (실패한 것은 None)."""
    return {'vol_har': har_vol(close, h), 'vol_gjr': gjr_vol(close, h), 'vol_ewmaf': ewma_fast_vol(close)}


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
        n_susp, susp_date = suspect_returns(s)
        raw_vol = None
        if n_susp:
            r_all = log_returns(s, keep_suspect=True)
            raw_vol = float(np.sqrt((r_all ** 2).ewm(alpha=1 - 0.94, adjust=False).mean().iloc[-1] * TD)) if r_all is not None else None
        snap = {
            'data_suspect': n_susp, 'data_suspect_date': susp_date,
            'move_5d_incl_suspect': (raw_vol * np.sqrt(5 / TD)) if raw_vol else None,
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
    path = os.path.join(log_dir, VOL_LOG_NAME)
    tmp = f"{path}.tmp_{os.getpid()}"
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)          # v27: 원자적 교체 — 쓰는 도중 끊겨도 원본이 잘리지 않음
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


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
        try:
            sh = forecast_vol_shadow(s, h)
        except Exception:
            sh = {}
        new.append({'run_date': pd.Timestamp(run_date).normalize(), 'ticker': ticker, 'price_date': pdate,
                    'horizon_days': h, **{c: f[c] for c in CANDIDATES},
                    'target_date': pdate + BDay(h), 'vol_realized': np.nan, 'settled': 0,
                    **{c: (np.nan if sh.get(c) is None else sh[c]) for c in SHADOW_CANDIDATES}})
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
    if (seg.abs() > SPLIT_SUSPECT).any():
        return np.nan          # 분할 의심 구간: 정산은 끝내되 값은 NaN으로 두어 성적표에서 제외
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
    후보별 성적표. QLIKE(낮을수록 좋음)와 편향 지표.
    - bias_var : sum(실현²) / sum(예측²)  (분산 단위, 1.0이 이상적, >1 과소예측). 호라이즌에 상관없이 '보정된 예측이면 ≈1'이 되는
                 올바른 편향 지표다(SOXL 5년 워크포워드: EWMA 1일 1.23·5일 1.07·20일 1.02).
    - bias     : (구) 실현/예측 비율의 산술평균(변동성 단위). 보정된 예측이어도 1.0이 아니고(1일은 하루 수익률 1개가 실현값이라
                 약 0.8, 기하평균은 약 0.5) 극단값에 끌려서 해석이 어렵다. 호환용으로만 유지한다 — 판단에는 bias_var 를 쓸 것.
    n_eff = 서로 안 겹치는 구간 수(가격일 수 / 호라이즌)로 보수적으로 표기(같은 날 여러 종목은 1건으로 센다).
    그림자 후보(vol_har/vol_gjr)는 값이 있는 행끼리만 채점한다(옛 행은 NaN이라 제외 → n 이 다를 수 있음).
    """
    if len(df) == 0:
        return pd.DataFrame()
    s = df[df['settled'].fillna(0).astype(int) == 1].copy()
    out = []
    for h, g in s.groupby('horizon_days'):
        for c in ALL_CANDIDATES:
            if c not in g.columns:
                continue
            x = g[[c, 'vol_realized', 'price_date']].dropna()
            x = x[(x[c] > 0) & (x['vol_realized'] > 0)]
            if len(x) == 0:
                continue
            n_eff = x['price_date'].nunique() / float(h)
            ratio_v = x['vol_realized'] / x[c]
            ratio = ratio_v ** 2
            qlike = float((ratio - np.log(ratio) - 1).mean())
            out.append({'horizon': int(h), 'candidate': c, 'n': len(x), 'n_eff': round(n_eff, 1),
                        'qlike': qlike,
                        'bias_var': float((x['vol_realized'] ** 2).sum() / (x[c] ** 2).sum()),
                        'bias': float(ratio_v.mean())})
    return pd.DataFrame(out)
