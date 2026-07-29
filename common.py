"""
common.py — 레버리지/인버스 ETF 방향성 예측 시스템 공통 모듈 (v20.0)
predict_scalp.py / predict_swing.py / predict_position.py 가 공유한다.
"""
import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1) 커버 종목 메타데이터 — 지수 계열별 레버리지/인버스 3쌍
# ============================================================
# benchmark_ticker: yfinance에서 받아올 실제 벤치마크 시세 티커
LEVERAGE_UNIVERSE = {
    # --- 나스닥100 계열 ---
    'TQQQ': {'multiplier': 3, 'direction': +1, 'benchmark': 'QQQ', 'family': 'Nasdaq100', 'desc': '나스닥100 +3배'},
    'SQQQ': {'multiplier': 3, 'direction': -1, 'benchmark': 'QQQ', 'family': 'Nasdaq100', 'desc': '나스닥100 -3배'},
    # --- 반도체(SOX) 계열 ---
    'SOXL': {'multiplier': 3, 'direction': +1, 'benchmark': 'SMH', 'family': 'Semiconductor', 'desc': '반도체 +3배'},
    'SOXS': {'multiplier': 3, 'direction': -1, 'benchmark': 'SMH', 'family': 'Semiconductor', 'desc': '반도체 -3배'},
    # --- S&P500 계열 ---
    'SPXL': {'multiplier': 3, 'direction': +1, 'benchmark': 'SPY', 'family': 'SP500', 'desc': 'S&P500 +3배'},
    'SPXS': {'multiplier': 3, 'direction': -1, 'benchmark': 'SPY', 'family': 'SP500', 'desc': 'S&P500 -3배'},
}

MACRO_TICKERS = {'QQQ': 'QQQ', 'SMH': 'SMH', 'SPY': 'SPY', 'VIX': '^VIX', 'TNX': '^TNX'}

# 매크로 이벤트 캘린더 — 세 스크립트가 공통으로 참조
MARKET_EVENTS = {
    "2026-07-29": {"desc": "FOMC 금리 결정 (한국시간 7/30 오전 3시 발표)",
                   "consensus": "동결 vs 25bp 인상 확률 팽팽 (동결 62~70%, 인상 25~30%)"},
    "2026-08-26": {"desc": "NVIDIA FY2027 Q2 실적 발표 (장마감 후)",
                   "consensus": "컨센서스 EPS $2.08"},
}


def check_upcoming_events(window_days=2):
    from datetime import datetime
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
def fetch_macro_data(start_date='2022-01-01'):
    macro_data = {}
    for name, t in MACRO_TICKERS.items():
        df_raw = yf.download(t, start=start_date, progress=False, multi_level_index=False)
        if not df_raw.empty:
            macro_data[name] = df_raw['Close'].squeeze()
    macro_df = pd.DataFrame(macro_data).ffill().dropna()
    return macro_df


def fetch_ticker_data(ticker, start_date='2022-01-01'):
    tgt = yf.download(ticker, start=start_date, progress=False, multi_level_index=False)
    if tgt.empty:
        return None
    return pd.DataFrame({
        'High': tgt['High'].squeeze(),
        'Low': tgt['Low'].squeeze(),
        'Close': tgt['Close'].squeeze(),
    })


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
    """호라이즌에 무관하게 공통으로 쓰는 피처셋."""
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

    # 벤치마크 대비 20일 상대 모멘텀 (leakage 아님: 과거값만 사용)
    if benchmark_col in df.columns:
        df['RelBenchmark20'] = df['Close'].pct_change(20) - df[benchmark_col].pct_change(20)
    else:
        df['RelBenchmark20'] = 0

    # 매크로 레짐: VIX 수준/변화, 금리(TNX) 변화
    if 'VIX' in df.columns:
        df['VIX_level'] = df['VIX']
        df['VIX_chg5'] = df['VIX'].pct_change(5)
    if 'TNX' in df.columns:
        df['TNX_chg5'] = df['TNX'].pct_change(5)

    return df


FEATURE_COLUMNS = [
    'RSI', 'MACD_Hist', 'Price_to_EMA20', 'BB_Pct',
    'Momentum_High52w', 'TrendSlope', 'Momentum_120', 'ROC10',
    'ATR14', 'StochK', 'HistVol20', 'RelBenchmark20',
    'VIX_level', 'VIX_chg5', 'TNX_chg5',
]


# ============================================================
# 4) 타겟 생성 (호라이즌별로 파라미터만 다르게)
# ============================================================
def build_target(df, horizon_days, threshold_cap):
    """horizon_days 뒤 수익률이 동적 임계값을 넘으면 1 (상승), 아니면 0."""
    rolling_median_ret = df['Close'].pct_change().abs().rolling(window=60, min_periods=20).median()
    dyn_threshold = np.clip(rolling_median_ret * 0.5 * np.sqrt(horizon_days), 0.003, threshold_cap)
    df['Target_Ret'] = df['Close'].pct_change(horizon_days).shift(-horizon_days)
    df['Target'] = np.where(df['Target_Ret'] > dyn_threshold, 1, 0)
    return df


# ============================================================
# 5) 텔레그램 알림
# ============================================================
def send_telegram(message):
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        print("⚠️ TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 미설정 — 콘솔에만 출력합니다.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, data={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"⚠️ 텔레그램 전송 실패: {e}")


def direction_label(up_prob, down_prob):
    if up_prob >= 65:
        return "🟢 강한 상승 추세"
    elif up_prob >= 50:
        return "🟡 약한 상승 추세"
    elif down_prob >= 65:
        return "🔴 강한 하락 경계"
    else:
        return "🟠 하락 우세"
