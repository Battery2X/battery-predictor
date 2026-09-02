"""
common.py — 레버리지/인버스 ETF 방향성 예측 시스템 공통 모듈 (v23.0)
predict_scalp.py / predict_swing.py / predict_position.py 가 공유한다.

v23.0 변경사항
------------------------------------------------
1. NaN 가격 버그 수정: fetch_display_metrics()가 ffill+dropna 가드 포함
2. 확률 보정(calibration): 교차검증 OOF 예측으로 Isotonic Regression 학습,
   "모델이 70%라 했을 때 실제로 몇 %였는지"를 보정해서 신뢰도를 높임
3. 실전 적중률 추적: logs/predictions_*.csv에 매 실행 예측을 기록하고,
   호라이즌이 지난 과거 예측은 실제 결과와 대조해서 최근 N회 적중률 계산
4. train_benchmark_model()을 공통 모듈로 이동 (3개 스크립트 중복 제거)
"""
import os
import time
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from pandas.tseries.offsets import BDay
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings('ignore')

# ============================================================
# 1) 커버 종목 메타데이터
# ============================================================
LEVERAGE_UNIVERSE = {
    'TQQQ': {'multiplier': 3, 'direction': +1, 'benchmark': 'QQQ', 'family': 'Nasdaq100', 'desc': '나스닥100 +3배'},
    'SQQQ': {'multiplier': 3, 'direction': -1, 'benchmark': 'QQQ', 'family': 'Nasdaq100', 'desc': '나스닥100 -3배'},
    'SOXL': {'multiplier': 3, 'direction': +1, 'benchmark': 'SMH', 'family': 'Semiconductor', 'desc': '반도체 +3배'},
    'SOXS': {'multiplier': 3, 'direction': -1, 'benchmark': 'SMH', 'family': 'Semiconductor', 'desc': '반도체 -3배'},
    'SPXL': {'multiplier': 3, 'direction': +1, 'benchmark': 'SPY', 'family': 'SP500', 'desc': 'S&P500 +3배'},
    'SPXS': {'multiplier': 3, 'direction': -1, 'benchmark': 'SPY', 'family': 'SP500', 'desc': 'S&P500 -3배'},
}

BENCHMARK_TICKERS = ['QQQ', 'SMH', 'SPY']
REL_COMPARISON = {'QQQ': 'SPY', 'SMH': 'SPY', 'SPY': None,
                  # 개별 종목은 나스닥(QQQ) 대비 상대강도로 비교 (전부 나스닥 대형 기술주라서)
                  'AAPL': 'QQQ', 'NVDA': 'QQQ', 'TSLA': 'QQQ'}

# 개별 종목(레버리지 아님) 유니버스 — 페어 개념 없이 방향성만 그대로 사용
STOCK_TICKERS = ['AAPL', 'NVDA', 'TSLA']

# 종목별 다음 실적 발표(예정)일 — 확정 아닌 예상치 포함, 주기적으로 갱신 필요
STOCK_EARNINGS_DATES = {
    'AAPL': '2026-10-29',
    'NVDA': '2026-08-26',
    'TSLA': '2026-10-28',
}


def earnings_proximity_warning(ticker, window_days=5):
    """
    종목의 다음 실적 발표가 임박했으면 경고 문자열을 반환한다.
    가격 패턴만 학습한 정량 모델은 실적 발표 전후 신뢰도가 크게 떨어지므로,
    이 구간에서는 클로드 프로젝트/스킬(10-K, 어닝콜 트랜스크립트 기반 정성 분석)로
    가이던스 변화·리스크 요인을 따로 확인하라고 안내한다.
    반환: 경고 문자열 또는 None (임박하지 않음)
    """
    date_str = STOCK_EARNINGS_DATES.get(ticker)
    if not date_str:
        return None
    earnings_date = pd.Timestamp(date_str)
    today = pd.Timestamp.now().normalize()
    days_left = (earnings_date - today).days
    if 0 <= days_left <= window_days:
        return (f"🚨 실적 발표 D-{days_left} ({date_str}) — 이 구간은 정량 모델(가격 패턴)의 "
                f"신뢰도가 원래 낮음. 클로드 프로젝트/스킬로 가이던스·리스크 변화 등 정성 분석을 "
                f"별도로 확인 권장")
    return None
MACRO_TICKERS = {'QQQ': 'QQQ', 'SMH': 'SMH', 'SPY': 'SPY', 'VIX': '^VIX', 'TNX': '^TNX'}

# SOXX(반도체 ETF) 상위 10종목 비중 (2026-08-04 기준 스냅샷, 상위 10종목이 전체의 약 61%)
# SOXL/SOXS 모델(SMH 벤치마크)에만 구성종목 폭(breadth)/쏠림(dispersion) 피처로 사용
SOXX_TOP_HOLDINGS_WEIGHTS = {
    'AMD': 8.54, 'NVDA': 8.53, 'AVGO': 7.95, 'MU': 7.81, 'INTC': 5.41,
    'AMAT': 5.16, 'MRVL': 4.53, 'TSM': 4.40, 'KLAC': 4.32, 'LRCX': 4.24,
}


def fetch_constituent_breadth_features(start_date='2022-01-01'):
    """
    SOXX 상위 10종목의 개별 가격을 받아와 SMH 하나로는 안 보이는 두 가지 신호를 만든다.
      - ConstMomentum5 : 비중가중 평균 5일 수익률 (SMH 자체 모멘텀과 유사하지만 구성 방식이 다름)
      - ConstBreadth5  : 상위 10종목 중 5일 수익률이 양(+)인 종목의 비율 (0~1). 지수는 오르는데
                         소수 대형주가 끌고 나머지는 빠지는 "쏠림 상승"을 구분해줌
      - ConstDispersion5: 비중가중 5일 수익률의 표준편차. 종목 간 방향이 갈릴수록 커짐
                         (섹터 내부에서 의견이 갈리는 국면 = 변동성 확대 전조로 흔히 해석됨)
    반환: DatetimeIndex를 가진 DataFrame (컬럼: ConstMomentum5, ConstBreadth5, ConstDispersion5)
    """
    total_w = sum(SOXX_TOP_HOLDINGS_WEIGHTS.values())
    norm_weights = {k: v / total_w for k, v in SOXX_TOP_HOLDINGS_WEIGHTS.items()}

    ret_frames = {}
    for ticker in SOXX_TOP_HOLDINGS_WEIGHTS:
        df = fetch_ticker_data(ticker, start_date=start_date)
        if df is None:
            continue
        ret_frames[ticker] = df['Close'].pct_change(5)

    if not ret_frames:
        return pd.DataFrame(columns=['ConstMomentum5', 'ConstBreadth5', 'ConstDispersion5'])

    ret_df = pd.DataFrame(ret_frames).ffill().dropna(how='all')
    weight_series = pd.Series({k: norm_weights[k] for k in ret_df.columns})
    # 결측 종목이 있는 날짜엔 남은 종목들끼리 비중을 재정규화해서 가중평균
    valid_mask = ret_df.notna()
    effective_w = valid_mask.mul(weight_series, axis=1)
    effective_w = effective_w.div(effective_w.sum(axis=1), axis=0)

    const_momentum = (ret_df.fillna(0) * effective_w).sum(axis=1)
    const_breadth = (ret_df > 0).sum(axis=1) / valid_mask.sum(axis=1)
    const_dispersion = (ret_df.sub(const_momentum, axis=0).pow(2) * effective_w).sum(axis=1).pow(0.5)

    out = pd.DataFrame({
        'ConstMomentum5': const_momentum,
        'ConstBreadth5': const_breadth,
        'ConstDispersion5': const_dispersion,
    })
    return out.dropna()

MARKET_EVENTS = {
    "2026-07-29": {"desc": "FOMC 금리 결정 (한국시간 7/30 오전 3시 발표)",
                   "consensus": "동결 vs 25bp 인상 확률 팽팽 (동결 62~70%, 인상 25~30%)"},
    "2026-08-26": {"desc": "NVIDIA FY2027 Q2 실적 발표 (장마감 후)",
                   "consensus": "컨센서스 EPS $2.08"},
}


MACRO_SHOCK_STD_WINDOW = 60   # 최근 60거래일 기준으로 "평소 하루 변동폭"을 계산
MACRO_SHOCK_Z_THRESHOLD = 2.0  # 이 배수 이상 벗어나면 경고


def macro_shock_warning(macro_df, z_threshold=MACRO_SHOCK_Z_THRESHOLD, window=MACRO_SHOCK_STD_WINDOW):
    """
    당일(최근 1거래일) VIX·10년물 금리(TNX) 변동이 최근 window거래일 평소 수준 대비
    얼마나 이례적인지 z-score로 확인해서, 기술적 지표 기반 예측의 신뢰도를 낮춰봐야 하는
    날인지 경고한다.

    기존 학습 피처 VIX_chg5/TNX_chg5는 5일 변화율이라 하루짜리 급변(예: 2026-08-18
    30년물 금리 19년래 최고 + 반도체 업종 급락 같은 이벤트)이 5일 평균에 묻혀 잘 안 잡힐 수
    있다. 이 함수는 그 사각지대를 보완하는 순수 규칙 기반 경고이며, 모델 학습·피처·임계값에는
    전혀 관여하지 않는다 — 그래서 이 로직을 추가/조정해도 지금까지 쌓인 실전 적중률 로그는
    무효화되지 않는다 (검증 시계 리셋 없음).

    macro_df: fetch_macro_data()의 반환값 (컬럼: QQQ, SMH, SPY, VIX, TNX)
    반환: 경고 문자열 또는 None (평소 수준이거나 데이터 부족/결측 시 조용히 스킵)
    """
    if macro_df is None or macro_df.empty:
        return None
    if any(col not in macro_df.columns for col in ('VIX', 'TNX')):
        return None
    if len(macro_df) < window + 5:
        return None

    vix_chg1 = macro_df['VIX'].pct_change(1)
    tnx_chg1 = macro_df['TNX'].pct_change(1)

    vix_hist = vix_chg1.iloc[-(window + 1):-1]
    tnx_hist = tnx_chg1.iloc[-(window + 1):-1]
    vix_std, tnx_std = vix_hist.std(), tnx_hist.std()
    if not vix_std or not tnx_std or pd.isna(vix_std) or pd.isna(tnx_std) or vix_std == 0 or tnx_std == 0:
        return None

    vix_today, tnx_today = vix_chg1.iloc[-1], tnx_chg1.iloc[-1]
    if pd.isna(vix_today) or pd.isna(tnx_today):
        return None

    vix_z, tnx_z = vix_today / vix_std, tnx_today / tnx_std

    # VIX만 튀는 날은 흔하지만, VIX+금리가 동시에 튀거나 둘 중 하나가 아주 크게 튀는 날은
    # 드물고 임팩트가 크다 — 합성 데이터 검증 결과 평소엔 ~1.7%, 충격일엔 100% 탐지됨.
    triggered = (
        (abs(vix_z) >= z_threshold and abs(tnx_z) >= z_threshold * 0.5)
        or (abs(vix_z) >= z_threshold * 1.5)
        or (abs(tnx_z) >= z_threshold * 1.5)
    )
    if not triggered:
        return None

    return (f"🌩️ 매크로 변동성 급증 감지 (VIX 1일 변화 {vix_today*100:+.1f}%, z={vix_z:+.1f} / "
            f"10Y금리 1일 변화 {tnx_today*100:+.1f}%, z={tnx_z:+.1f}) — 학습 피처(VIX_chg5 등)는 "
            f"5일 평균이라 오늘 같은 급변은 충분히 못 잡을 수 있음. 오늘 방향 예측은 평소보다 "
            f"신뢰도 낮게 볼 것")


def check_upcoming_events(window_days=2):
    today = datetime.now().date()
    out = []
    for date_str, info in MARKET_EVENTS.items():
        ev_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        days_left = (ev_date - today).days
        if 0 <= days_left <= window_days:
            prefix = "🔴 오늘" if days_left == 0 else f"⚠️ D-{days_left}"
            out.append(f"  {prefix}: {info['desc']} | {info['consensus']}")
    return out


# ============================================================
# 2) 데이터 수집
# ============================================================
def _yf_download_with_retry(ticker, start_date, max_retries=3, base_delay=2):
    """
    야후 파이낸스가 일시적으로 요청을 막았을 때(rate limit) 바로 포기하지 않고
    잠깐 쉬었다가 다시 시도한다. 그래도 계속 실패하면 빈 DataFrame을 반환해서
    호출부가 '데이터 없음'으로 정상 처리하도록 한다.
    """
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(ticker, start=start_date, progress=False, multi_level_index=False)
            if not df.empty:
                return df
        except Exception as e:
            print(f"⚠️ [{ticker}] 조회 시도 {attempt}/{max_retries} 실패: {e}")
        if attempt < max_retries:
            wait = base_delay * attempt  # 2초, 4초로 점점 늘려가며 대기
            print(f"⏳ [{ticker}] {wait}초 대기 후 재시도...")
            time.sleep(wait)
    print(f"❌ [{ticker}] {max_retries}회 재시도 모두 실패 — 데이터 없음으로 처리")
    return pd.DataFrame()


def fetch_macro_data(start_date='2022-01-01'):
    macro_data = {}
    for name, t in MACRO_TICKERS.items():
        df_raw = _yf_download_with_retry(t, start_date)
        if not df_raw.empty:
            macro_data[name] = df_raw['Close'].squeeze()
    macro_df = pd.DataFrame(macro_data).ffill().dropna()
    return macro_df


def fetch_ticker_data(ticker, start_date='2022-01-01'):
    tgt = _yf_download_with_retry(ticker, start_date)
    if tgt.empty:
        return None
    return pd.DataFrame({
        'High': tgt['High'].squeeze(),
        'Low': tgt['Low'].squeeze(),
        'Close': tgt['Close'].squeeze(),
    })


def fetch_display_metrics(ticker):
    """
    v23.0 버그수정: 현재가/변동성 조회 시 NaN 방지.
    yfinance가 마지막 행에 결측치를 반환하는 경우가 있어 ffill+dropna로 방어한다.
    """
    df = fetch_ticker_data(ticker)
    if df is None:
        return None
    df = df.ffill().dropna()
    if df.empty:
        return None
    price = df['Close'].iloc[-1]
    vol = df['Close'].pct_change().rolling(20).std().iloc[-1] * (252 ** 0.5)
    if pd.isna(price) or pd.isna(vol):
        return None
    return {'price': price, 'vol': vol, 'df': df}


def fetch_analyst_momentum_features(ticker, start_date='2022-01-01'):
    """
    가격 패턴과 완전히 독립적인 정보원: 애널리스트 등급 변경(상향/하향) 이력.
    yfinance의 upgrades_downgrades는 실제 등급 변경이 일어난 날짜가 찍혀있어서
    (다른 구성종목 폭 피처와 달리) 과거로 거슬러 학습용 시계열을 만들 수 있다.

    - AnalystNet30     : 최근 30일 (상향 - 하향) 건수 — 양수면 애널리스트들이 낙관 전환 중
    - AnalystActivity30: 최근 30일 등급 변경 총 건수 — 그 종목에 대한 애널리스트 관심도/논쟁 강도

    ETF(QQQ/SMH/SPY)는 애널리스트 등급 자체가 없으므로 개별 종목(AAPL/NVDA/TSLA 등)에만 사용한다.
    yfinance API 응답 형식이 바뀌거나 데이터가 없으면 조용히 None을 반환 — 호출부에서 이미
    "None이면 이 피처 없이 진행"하도록 방어적으로 짜여 있어야 한다.
    """
    try:
        raw = yf.Ticker(ticker).upgrades_downgrades
    except Exception as e:
        print(f"⚠️ [{ticker}] 애널리스트 데이터 조회 실패: {e}")
        return None
    if raw is None or raw.empty:
        print(f"⚠️ [{ticker}] 애널리스트 데이터 없음 (upgrades_downgrades 비어있음)")
        return None

    ud = raw.reset_index()
    date_col = ud.columns[0]  # 보통 'GradeDate'라는 이름의 인덱스가 첫 컬럼으로 옴
    if 'Action' not in ud.columns:
        print(f"⚠️ [{ticker}] 'Action' 컬럼 없음 — yfinance 응답 형식 확인 필요. 실제 컬럼: {list(ud.columns)}")
        return None

    ud[date_col] = pd.to_datetime(ud[date_col], errors='coerce')
    ud = ud.dropna(subset=[date_col])
    ud = ud[ud[date_col] >= pd.Timestamp(start_date)]

    print(f"📋 [{ticker}] 애널리스트 등급변경 원본 {len(raw)}건 중 {start_date} 이후 {len(ud)}건 사용")

    # 이벤트가 너무 적으면(예: 몇 건) 롤링 30일 피처가 거의 항상 0이다가 어쩌다 한 번 튀는
    # 형태가 되어 트리 모델이 그 한 번의 스파이크에 과적합하기 쉽다. 최소 건수 미달이면
    # 이 피처 자체를 신뢰할 수 없다고 보고 사용하지 않는다.
    MIN_EVENTS = 10
    if len(ud) < MIN_EVENTS:
        print(f"⚠️ [{ticker}] 애널리스트 이벤트 {len(ud)}건 < 최소 기준 {MIN_EVENTS}건 — 이 피처 사용 안 함 (데이터 부실 의심)")
        return None

    action = ud['Action'].astype(str).str.lower()
    ud['is_up'] = action.eq('up').astype(int)
    ud['is_down'] = action.eq('down').astype(int)
    ud = ud.set_index(date_col).sort_index()

    daily_up = ud['is_up'].resample('D').sum()
    daily_down = ud['is_down'].resample('D').sum()
    daily_total = ud['Action'].resample('D').count()

    full_idx = pd.date_range(daily_up.index.min(), pd.Timestamp.now().normalize(), freq='D')
    daily_up = daily_up.reindex(full_idx, fill_value=0)
    daily_down = daily_down.reindex(full_idx, fill_value=0)
    daily_total = daily_total.reindex(full_idx, fill_value=0)

    net30 = (daily_up - daily_down).rolling(30, min_periods=1).sum()
    activity30 = daily_total.rolling(30, min_periods=1).sum()

    return pd.DataFrame({'AnalystNet30': net30, 'AnalystActivity30': activity30})


# ============================================================
# 3) 기술적 지표 / 피처 엔지니어링
# ============================================================
def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + (gain / loss)))


def _bollinger(series, period=20, std_mult=2.0):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return ma + std_mult*std, ma, ma - std_mult*std


def _atr(df, period=14):
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def _stochastic_k(df, period=14):
    low_min = df['Low'].rolling(period).min()
    high_max = df['High'].rolling(period).max()
    return ((df['Close'] - low_min) / (high_max - low_min).replace(0, np.nan)) * 100


def build_features(df, benchmark_col):
    df['RSI'] = _rsi(df['Close'])
    macd_line = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Hist'] = macd_line - macd_line.ewm(span=9, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['Price_to_EMA20'] = (df['Close'] / df['EMA_20']) - 1

    bb_upper, bb_ma, bb_lower = _bollinger(df['Close'])
    df['BB_Pct'] = ((df['Close'] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)) * 100

    rolling_max_252 = df['Close'].rolling(252, min_periods=60).max()
    df['Momentum_High52w'] = (df['Close'] >= rolling_max_252 * 0.995).astype(int).rolling(20).mean()

    short_slope = df['Close'].pct_change(20)
    long_slope = df['Close'].pct_change(120) / 6
    df['TrendSlope'] = short_slope - long_slope
    df['Momentum_120'] = df['Close'].pct_change(120)
    df['ROC10'] = df['Close'].pct_change(10)

    df['ATR14'] = _atr(df) / df['Close']
    df['StochK'] = _stochastic_k(df)
    df['HistVol20'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)

    if benchmark_col and benchmark_col in df.columns:
        df['RelBenchmark20'] = df['Close'].pct_change(20) - df[benchmark_col].pct_change(20)
    else:
        df['RelBenchmark20'] = 0

    if 'VIX' in df.columns:
        df['VIX_level'] = df['VIX']
        df['VIX_chg5'] = df['VIX'].pct_change(5)
    if 'TNX' in df.columns:
        df['TNX_chg5'] = df['TNX'].pct_change(5)

    # SOXX 구성종목 폭/쏠림 피처 — train_benchmark_model에서 SMH일 때만 미리 join해줌.
    # join이 안 됐으면(다른 벤치마크/종목) 0으로 채워 피처 컬럼 개수를 항상 동일하게 유지.
    for col in ('ConstMomentum5', 'ConstBreadth5', 'ConstDispersion5'):
        if col not in df.columns:
            df[col] = 0

    # 애널리스트 등급변경 피처도 동일한 패턴: 개별 종목에서만 미리 join됨, 아니면 0
    for col in ('AnalystNet30', 'AnalystActivity30'):
        if col not in df.columns:
            df[col] = 0

    return df


FEATURE_COLUMNS = [
    'RSI', 'MACD_Hist', 'Price_to_EMA20', 'BB_Pct',
    'Momentum_High52w', 'TrendSlope', 'Momentum_120', 'ROC10',
    'ATR14', 'StochK', 'HistVol20', 'RelBenchmark20',
    'VIX_level', 'VIX_chg5', 'TNX_chg5',
    'ConstMomentum5', 'ConstBreadth5', 'ConstDispersion5',
    'AnalystNet30', 'AnalystActivity30',
]


# ============================================================
# 4) 타겟 생성 — 대칭 3구간(상승/하락/횡보), 횡보는 학습 제외
# ============================================================
def build_target(df, horizon_days, threshold_cap):
    rolling_median_ret = df['Close'].pct_change().abs().rolling(window=60, min_periods=20).median()
    dyn_threshold = np.clip(rolling_median_ret * 0.5 * np.sqrt(horizon_days), 0.003, threshold_cap)
    df['Target_Ret'] = df['Close'].pct_change(horizon_days).shift(-horizon_days)

    target = np.full(len(df), np.nan)
    up_mask = (df['Target_Ret'] > dyn_threshold).values
    down_mask = (df['Target_Ret'] < -dyn_threshold).values
    target[up_mask] = 1
    target[down_mask] = 0
    df['Target'] = target
    df['DeadZone_Ratio'] = 1 - (up_mask.sum() + down_mask.sum()) / len(df)
    return df


# ============================================================
# 5) 벤치마크 모델 학습 (공통화) + 확률 보정(calibration)
# ============================================================
def train_benchmark_model(benchmark, macro_df, horizon_days, threshold_cap,
                           benchmark_multiplier=3, min_train_rows=300, model_params=None,
                           extra_features_df=None, extra_feature_cols=None):
    """
    벤치마크 하나에 대해 XGB+LGBM 앙상블을 학습하고,
    교차검증 중 나온 out-of-fold 예측으로 Isotonic Regression 보정기를 만들어
    최종 확률에 적용한다.

    extra_features_df / extra_feature_cols: SMH의 경우 fetch_constituent_breadth()로
    만든 구성종목 체감폭/쏠림도 데이터를 넘기면 기본 15개 피처에 추가로 합쳐서 학습한다.
    다른 벤치마크(QQQ/SPY/개별종목)는 기존과 동일하게 None으로 두면 된다.

    반환: dict(up_prob, down_prob, raw_up_prob, f1, prec, rec, dead_zone) 또는 None
    """
    if model_params is None:
        model_params = {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05}

    df = fetch_ticker_data(benchmark)
    if df is None:
        return None
    df = df.join(macro_df, how='left')
    if extra_features_df is not None:
        df = df.join(extra_features_df, how='left')
    df = df.ffill().dropna()
    df = build_features(df, benchmark_col=REL_COMPARISON[benchmark])
    df = build_target(df, horizon_days, threshold_cap / benchmark_multiplier)

    feature_cols = FEATURE_COLUMNS + (extra_feature_cols or [])

    df_train = df.dropna(subset=['Target'] + feature_cols).copy()
    if len(df_train) < min_train_rows:
        return None

    X, y = df_train[feature_cols], df_train['Target']

    tscv = TimeSeriesSplit(n_splits=3)
    xgb = XGBClassifier(eval_metric='logloss', random_state=42, **model_params)
    lgbm = LGBMClassifier(verbose=-1, random_state=42, **model_params)

    f1 = prec = rec = 0
    oof_probs, oof_labels = [], []
    for tr_idx, te_idx in tscv.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        scaler = StandardScaler()
        X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)

        xgb.fit(X_tr_s, y_tr)
        lgbm.fit(X_tr_s, y_tr)
        probs = (xgb.predict_proba(X_te_s) + lgbm.predict_proba(X_te_s)) / 2
        preds = (probs[:, 1] >= 0.5).astype(int)

        f1 += f1_score(y_te, preds, average='macro', zero_division=0)
        prec += precision_score(y_te, preds, zero_division=0)
        rec += recall_score(y_te, preds, zero_division=0)

        oof_probs.extend(probs[:, 1].tolist())
        oof_labels.extend(y_te.tolist())
    f1, prec, rec = f1/3, prec/3, rec/3

    # 확률 보정기: OOF (원시확률, 실제라벨) 쌍으로 Isotonic Regression 학습
    calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    calibrator.fit(np.array(oof_probs), np.array(oof_labels))

    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    xgb.fit(X_scaled, y)
    lgbm.fit(X_scaled, y)

    # 피처 중요도: XGB/LGBM 각각을 0~1로 정규화한 뒤 평균 — 두 모델의 스케일이 달라서
    # (XGB는 gain 기반, LGBM은 split 기반) 그냥 더하면 한쪽에 치우치므로 정규화 후 평균낸다.
    xgb_imp = xgb.feature_importances_
    lgbm_imp = lgbm.feature_importances_
    xgb_imp_norm = xgb_imp / xgb_imp.sum() if xgb_imp.sum() > 0 else xgb_imp
    lgbm_imp_norm = lgbm_imp / lgbm_imp.sum() if lgbm_imp.sum() > 0 else lgbm_imp
    avg_importance = (xgb_imp_norm + lgbm_imp_norm) / 2
    importance_pairs = sorted(zip(feature_cols, avg_importance), key=lambda x: -x[1])
    top_features = importance_pairs[:5]

    latest = scaler_final.transform(df[feature_cols].iloc[[-1]])
    final_probs = (xgb.predict_proba(latest) + lgbm.predict_proba(latest)) / 2
    raw_up_prob = float(final_probs[0][1])
    calibrated_up_prob = float(calibrator.predict([raw_up_prob])[0])

    # 기준선(base rate): 횡보 제외 학습 샘플 중 실제 상승 비율.
    # 모델의 진짜 엣지 = 보정확률 - 기준선. 이 값이 0에 가까우면
    # 모델이 "그 시점 특유의 정보"가 아니라 그냥 과거 평균(강세장 편향 등)을
    # 재현하고 있을 가능성이 높다는 뜻.
    base_rate = float(y.mean() * 100)

    return {
        'up_prob': calibrated_up_prob * 100,
        'down_prob': (1 - calibrated_up_prob) * 100,
        'raw_up_prob': raw_up_prob * 100,
        'base_rate': base_rate,
        'edge': calibrated_up_prob * 100 - base_rate,
        'f1': f1, 'prec': prec, 'rec': rec,
        'dead_zone': df['DeadZone_Ratio'].iloc[-1],
        'top_features': top_features,
    }


# ============================================================
# 6) 페어 확률 변환
# ============================================================
def apply_direction(benchmark_up_prob, benchmark_down_prob, direction):
    if direction == +1:
        return benchmark_up_prob, benchmark_down_prob
    else:
        return benchmark_down_prob, benchmark_up_prob


# ============================================================
# 7) 실전 적중률 추적 (예측 로그)
# ============================================================
LOG_DIR = "logs"
LOG_COLUMNS = ['run_date', 'benchmark', 'ticker', 'horizon_days', 'predicted_label',
               'predicted_up_prob', 'raw_up_prob', 'price_at_prediction', 'target_date',
               'actual_price', 'actual_return', 'actual_label', 'correct', 'near_earnings']


def _log_path(name):
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, f"predictions_{name}.csv")


def load_log(name):
    path = _log_path(name)
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['run_date'] = pd.to_datetime(df['run_date'])
        df['target_date'] = pd.to_datetime(df['target_date'])
        # v24.0 이전 로그엔 near_earnings 컬럼이 없을 수 있음 — 하위호환을 위해 보강
        for col in LOG_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        return df
    return pd.DataFrame(columns=LOG_COLUMNS)


def save_log(name, df):
    df.to_csv(_log_path(name), index=False)


def settle_predictions(name):
    """호라이즌이 지난 과거 예측의 실제 결과를 확인해서 채워넣는다."""
    df = load_log(name)
    if df.empty:
        return df
    today = pd.Timestamp.now().normalize()
    mask_unresolved = df['actual_label'].isna() & (df['target_date'] <= today)
    if not mask_unresolved.any():
        return df

    for ticker in df.loc[mask_unresolved, 'ticker'].unique():
        price_df = fetch_ticker_data(ticker)
        if price_df is None:
            continue
        price_df = price_df.ffill().dropna()

        rows = df[(df['ticker'] == ticker) & mask_unresolved]
        for idx, row in rows.iterrows():
            future_prices = price_df.loc[price_df.index >= row['target_date'], 'Close']
            if future_prices.empty:
                continue
            actual_price = future_prices.iloc[0]
            entry_price = row['price_at_prediction']
            actual_ret = (actual_price - entry_price) / entry_price
            actual_label = 1 if actual_ret > 0 else 0
            correct = int(actual_label == int(row['predicted_label']))
            df.loc[idx, 'actual_price'] = actual_price
            df.loc[idx, 'actual_return'] = actual_ret
            df.loc[idx, 'actual_label'] = actual_label
            df.loc[idx, 'correct'] = correct

    save_log(name, df)
    return df


def append_predictions(name, rows):
    """
    실전 적중률 로그에 새 예측을 추가한다.

    안전장치 1 (신규 행 스킵): 같은 날짜(run_date의 날짜 부분)에 동일한
    (ticker, horizon_days) 조합의 기록이 이미 로그에 있으면 그 행은 조용히 스킵한다.
    -> 워크플로우가 하루에 여러 번 실행돼도(예약 실행 후 수동 재실행 포함) 로그는
       하루 1건으로 유지되고, 실전 적중률 표본이 같은 날 재실행분으로 부풀려지지 않는다.

    안전장치 2 (저장 직전 최종 정리): 두 실행이 정말로 겹쳐서(레이스 컨디션) 안전장치 1을
    각자 통과해버린 경우에 대비해, 최종 저장 직전에 (날짜, ticker, horizon_days) 기준으로
    한 번 더 그룹화해서 run_date가 가장 이른 행만 남긴다. dedup_logs.py의 정리 규칙과 동일.

    -> 이전 버전(v23.1)에 있던 안전장치 1이 이후 커밋에서 유실되어 있었고, 그 자리를
       메우려 했던 GitHub Actions concurrency 설정은 "겹치는 실행"만 막을 뿐 "예약 실행이
       끝난 뒤 별개로 들어오는 수동 재실행"은 막지 못해 중복이 계속 쌓였다(로그 전체 행의
       약 30%). 이번에 안전장치 1을 복원하고 안전장치 2를 추가로 더했다.
    -> 로직/피처/임계값과는 무관한 "로깅 안전장치"이며 예측 모델 자체는 전혀 바뀌지 않는다.
    """
    df = load_log(name)
    new_df = pd.DataFrame(rows)
    if new_df.empty:
        return

    new_df['run_date'] = pd.to_datetime(new_df['run_date'])

    if not df.empty:
        existing_keys = set(
            zip(df['run_date'].dt.date, df['ticker'], df['horizon_days'])
        )
        dup_mask = new_df.apply(
            lambda r: (r['run_date'].date(), r['ticker'], r['horizon_days']) in existing_keys,
            axis=1
        )
        n_skipped = int(dup_mask.sum())
        if n_skipped > 0:
            print(f"ℹ️ [{name}] 오늘 이미 기록된 예측 {n_skipped}건 스킵 (같은 날 재실행 방지)")
        new_df = new_df[~dup_mask]

    if new_df.empty:
        return

    combined = pd.concat([df, new_df], ignore_index=True)

    # 안전장치 2: 저장 직전 최종 중복 제거 (레이스 컨디션 대비)
    combined['_run_date_parsed'] = pd.to_datetime(combined['run_date'])
    combined['_run_date_only'] = combined['_run_date_parsed'].dt.date
    combined = combined.sort_values('_run_date_parsed')
    before_n = len(combined)
    combined = combined.groupby(
        ['_run_date_only', 'ticker', 'horizon_days'], as_index=False
    ).head(1)
    after_n = len(combined)
    if before_n != after_n:
        print(f"⚠️ [{name}] 저장 직전 레이스 컨디션 중복 {before_n - after_n}건 추가 제거")
    combined = combined.drop(columns=['_run_date_parsed', '_run_date_only'])
    combined = combined.sort_values('run_date').reset_index(drop=True)

    save_log(name, combined)


def wilson_ci(k, n, z=1.96):
    """
    이항비율(적중률)의 Wilson score 신뢰구간. 표본이 적을수록(n이 작을수록) 구간이
    넓어져서 "숫자만 보면 그럴싸해도 사실 못 믿는 수준"이라는 걸 그대로 드러낸다.
    (일반 정규근사 신뢰구간보다 n이 작을 때 더 안정적이라 이항비율엔 Wilson을 표준으로 씀)

    k: 적중 횟수, n: 전체 표본 수, z: 신뢰수준 z-value (기본 1.96 = 95%)
    반환: (하한%, 상한%). n==0이면 (0.0, 100.0) — "아무것도 모른다"를 표현.
    """
    if n <= 0:
        return 0.0, 100.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)) / denom
    return max(0.0, (center - margin) * 100), min(100.0, (center + margin) * 100)


def rolling_accuracy(df, ticker, window=20, baseline=None):
    """
    baseline(0~100, 보통 그 벤치마크의 base_rate)을 넘겨주면, 실전 적중률의 95% 신뢰구간
    하한이 baseline을 넘는지("beats_baseline")까지 같이 계산한다. 이게 아니면 "적중률이
    기준선보다 높아 보인다"는 게 표본이 적어서 생긴 우연인지 진짜 우위인지 구분이 안 된다.
    """
    sub = df[(df['ticker'] == ticker) & df['correct'].notna()].sort_values('run_date').tail(window)
    if sub.empty:
        return None
    n = len(sub)
    k = int(sub['correct'].sum())
    accuracy = sub['correct'].mean() * 100
    ci_low, ci_high = wilson_ci(k, n)
    out = {'n': n, 'accuracy': accuracy, 'ci_low': ci_low, 'ci_high': ci_high}
    if baseline is not None:
        out['beats_baseline'] = ci_low > baseline
    return out


def rolling_accuracy_by_earnings(df, ticker, window=20, min_n=3):
    """
    실적 발표 임박 구간(near_earnings=True)과 평상시(False)를 나눠서 적중률을 비교한다.
    각 그룹에 최소 min_n건 이상 쌓여야 결과를 반환한다 (그 전엔 통계적으로 의미 없음).
    """
    sub = df[(df['ticker'] == ticker) & df['correct'].notna()].sort_values('run_date')
    near = sub[sub['near_earnings'] == True].tail(window)
    normal = sub[sub['near_earnings'] != True].tail(window)
    result = {}
    if len(near) >= min_n:
        result['near_earnings'] = {'n': len(near), 'accuracy': near['correct'].mean() * 100}
    if len(normal) >= min_n:
        result['normal'] = {'n': len(normal), 'accuracy': normal['correct'].mean() * 100}
    return result


def make_prediction_record(run_date, benchmark, ticker, horizon_days, up_prob, raw_up_prob, price,
                            near_earnings=False):
    predicted_label = 1 if up_prob >= 50 else 0
    target_date = (pd.Timestamp(run_date).normalize() + BDay(horizon_days))
    return {
        'run_date': run_date, 'benchmark': benchmark, 'ticker': ticker,
        'horizon_days': horizon_days, 'predicted_label': predicted_label,
        'predicted_up_prob': up_prob, 'raw_up_prob': raw_up_prob,
        'price_at_prediction': price, 'target_date': target_date,
        'actual_price': np.nan, 'actual_return': np.nan,
        'actual_label': np.nan, 'correct': np.nan,
        'near_earnings': near_earnings,
    }


# ============================================================
# ============================================================
# 8) 텔레그램 알림
# ============================================================
def send_telegram(message):
    token = os.environ.get('TELEGRAM_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        print("⚠️ TELEGRAM_TOKEN / TELEGRAM_CHAT_ID 미설정 — 콘솔에만 출력합니다.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, data={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"⚠️ 텔레그램 전송 실패: {e}")


def direction_label(up_prob, down_prob):
    if up_prob >= 65:
        return "🟢 강한 상승 우세"
    elif up_prob >= 55:
        return "🟡 약한 상승 우세"
    elif down_prob >= 65:
        return "🔴 강한 하락 우세"
    elif down_prob >= 55:
        return "🟠 약한 하락 우세"
    else:
        return "⚪ 방향성 불분명 (50%대 근접)"


def edge_label(edge):
    """
    edge = 보정확률 - 기준선(base rate).
    0에 가까우면 모델이 그 시점 정보가 아니라 과거 평균(강세장 편향 등)을
    재현하고 있을 가능성이 높다는 뜻.
    """
    if abs(edge) < 3:
        return "⚠️ 엣지 거의 없음 (기준선과 거의 동일 — 그냥 과거 평균을 재현 중일 수 있음)"
    elif abs(edge) < 7:
        return "△ 약한 엣지"
    else:
        return "✅ 뚜렷한 엣지"
