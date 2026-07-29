import traceback
import requests
import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# v18.0 주요 개선 사항 (레버리지/인버스 ETF 대응)
# ============================================================
# [v17 유지] Lookahead Bias 제거, 반도체 특화 피처, TimeSeriesSplit, 앙상블
# [v18 신규 1] SQQQ / SOXS / SOXL 레버리지·인버스 ETF 지원 추가
# [v18 신규 2] 레버리지 상품은 벤치마크(SMH/Nasdaq) 대비 상대강도 피처를
#              leakage 방지를 위해 자동으로 제외 (자기 자신을 예측하는 꼴 방지)
# [v18 신규 3] 레버리지 상품 리포트에 decay 경고 및 배수/방향 정보 자동 표기
# ============================================================

MARKET_EVENTS = {
    "2026-06-24": {
        "desc": "Micron(MU) Q3 FY26 실적 발표 (장마감 후)",
        "ticker": "MU",
        "consensus": "EPS $19.72 / 매출 $34.5B 예상 | HBM 완판 및 한계 이익률 구조적 변화 확인"
    },
    "2026-07-29": {
        "desc": "FOMC 금리 결정 (한국시간 7/30 오전 3시 발표)",
        "ticker": None,
        "consensus": "동결 vs 25bp 인상 확률이 이례적으로 팽팽 (동결 62~70%, 인상 25~30%)"
    },
    "2026-08-26": {
        "desc": "NVIDIA FY2027 Q2 실적 발표 (장마감 후)",
        "ticker": "NVDA",
        "consensus": "컨센서스 EPS $2.08"
    }
}

# 레버리지/인버스 ETF 메타데이터: (배수, 벤치마크, 방향)
# 방향: +1 = 순방향(벤치마크와 같은 방향), -1 = 인버스(반대 방향)
LEVERAGED_ETFS = {
    'SQQQ': {'multiplier': 3, 'benchmark': 'Nasdaq', 'direction': -1, 'desc': '나스닥100 -3배 인버스'},
    'SOXS': {'multiplier': 3, 'benchmark': 'SMH',    'direction': -1, 'desc': '반도체 -3배 인버스'},
    'SOXL': {'multiplier': 3, 'benchmark': 'SMH',    'direction': +1, 'desc': '반도체 +3배 레버리지'},
}


def check_upcoming_events():
    from datetime import datetime
    today = datetime.now().date()
    out, out_detail = [], {}
    for date_str, info in MARKET_EVENTS.items():
        ev_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        days_left = (ev_date - today).days
        if 0 <= days_left <= 2:
            prefix = "🔴 오늘" if days_left == 0 else f"⚠️ D-{days_left}"
            out.append(f"  {prefix}: {info['desc']}")
            if info['ticker']:
                out_detail[info['ticker']] = {
                    'days_left': days_left,
                    'consensus': info['consensus']
                }
    return out, out_detail


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + (gain / loss)))


def calculate_bollinger(series, period=20, std_mult=2.0):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return ma + std_mult*std, ma, ma - std_mult*std


def calculate_advanced_features(df, ticker):
    """v18.0 피처: 레버리지 상품 leakage 방지 로직 포함"""
    # 1. 52주 신고가 근접 빈도 (레짐 감지)
    rolling_max_252 = df['Close'].rolling(252, min_periods=60).max()
    df['Momentum_High52w'] = (df['Close'] >= rolling_max_252 * 0.995).astype(int).rolling(20).mean()

    # 단/장기 추세 기울기 차이
    short_slope = df['Close'].pct_change(20)
    long_slope = df['Close'].pct_change(120) / 6
    df['TrendSlope'] = short_slope - long_slope
    df['Momentum_120'] = df['Close'].pct_change(120)

    # 2. 벤치마크 대비 상대강도 — 레버리지 상품은 제외(leakage 방지)
    if ticker in LEVERAGED_ETFS:
        df['RelStrength_SMH'] = 0  # 자기 자신(혹은 역방향)을 예측에 쓰지 않도록 무력화
    elif 'SMH' in df.columns:
        df['RelStrength_SMH'] = df['Close'].pct_change(20) - df['SMH'].pct_change(20)
    else:
        df['RelStrength_SMH'] = 0

    # 3. 옵션 시장 내재 변동성 대용치 (Historical Volatility 연율화)
    df['HistVol20'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)

    return df


try:
    print("🚀 v18.0 US AI·반도체 + 레버리지 ETF 방향성 통제소 가동...")
    event_warnings, event_detail = check_upcoming_events()

    start_date = '2022-01-01'
    macro_tickers = {'Nasdaq': '^IXIC', 'SMH': 'SMH', 'VIX': '^VIX', 'TNX': '^TNX'}
    macro_data = {}
    for name, t in macro_tickers.items():
        df_raw = yf.download(t, start=start_date, progress=False, multi_level_index=False)
        if not df_raw.empty:
            macro_data[name] = df_raw['Close'].squeeze()

    macro_df = pd.DataFrame(macro_data).ffill().dropna()
    macro_df['Nasdaq_ret'] = macro_df['Nasdaq'].pct_change()
    macro_df['Nasdaq_accel'] = macro_df['Nasdaq_ret'].pct_change()

    # 기존 개별주 + 레버리지/인버스 ETF 추가
    targets = ['NVDA', 'MU', 'AVGO', 'XOVL', 'SQQQ', 'SOXS', 'SOXL']
    final_report = "🤖\n" + "=" * 40 + "\n"

    if event_warnings:
        final_report += "📅 [이벤트 경고]\n"
        for w in event_warnings:
            final_report += w + "\n"
        final_report += "=" * 40 + "\n"

    for ticker in targets:
        tgt_data = yf.download(ticker, start=start_date, progress=False, multi_level_index=False)
        if tgt_data.empty: continue

        df = pd.DataFrame({
            'High':  tgt_data['High'].squeeze(),
            'Low':   tgt_data['Low'].squeeze(),
            'Close': tgt_data['Close'].squeeze()
        })
        df = df.join(macro_df, how='left').ffill().dropna()

        # 기술적 지표 생성
        df['RSI'] = calculate_rsi(df['Close'])
        macd_line = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_Hist'] = macd_line - macd_line.ewm(span=9, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['Price_to_EMA20'] = (df['Close'] / df['EMA_20']) - 1

        bb_upper, bb_ma, bb_lower = calculate_bollinger(df['Close'])
        df['BB_Pct'] = ((df['Close']-bb_lower)/(bb_upper-bb_lower).replace(0, np.nan))*100

        # 특화 피처 생성 (레버리지 여부 전달)
        df = calculate_advanced_features(df, ticker)

        # 미래 참조 오류 제거: 과거 데이터만으로 동적 임계값 생성
        # 레버리지 상품은 변동폭이 커서 임계값 상한을 더 넓게 허용
        cap = 0.05 if ticker in LEVERAGED_ETFS else 0.025
        rolling_median_ret = df['Close'].pct_change().abs().rolling(window=60, min_periods=20).median()
        dyn_threshold_series = np.clip(rolling_median_ret * 0.5, 0.003, cap)

        df['Target_Ret'] = df['Close'].pct_change().shift(-1)
        df['Target'] = np.where(df['Target_Ret'] > dyn_threshold_series, 1, 0)

        features = [
            'RSI', 'MACD_Hist', 'Price_to_EMA20', 'BB_Pct',
            'Momentum_High52w', 'TrendSlope', 'Momentum_120',
            'RelStrength_SMH', 'HistVol20'
        ]

        df_train = df.dropna(subset=['Target'] + features).copy()

        if len(df_train) < 300:
            continue

        X = df_train[features]
        y = df_train['Target']

        tscv = TimeSeriesSplit(n_splits=3)
        rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=5, class_weight='balanced')
        hgb = HistGradientBoostingClassifier(max_iter=200, random_state=42, max_depth=4, learning_rate=0.05)

        avg_f1, avg_prec, avg_rec = 0, 0, 0

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            rf.fit(X_train_s, y_train)
            hgb.fit(X_train_s, y_train)

            preds_prob = (rf.predict_proba(X_test_s) + hgb.predict_proba(X_test_s)) / 2
            preds = (preds_prob[:, 1] >= 0.5).astype(int)

            avg_f1 += f1_score(y_test, preds, average='macro', zero_division=0)
            avg_prec += precision_score(y_test, preds, zero_division=0)
            avg_rec += recall_score(y_test, preds, zero_division=0)

        avg_f1 /= 3
        avg_prec /= 3
        avg_rec /= 3

        scaler_final = StandardScaler()
        X_scaled = scaler_final.fit_transform(X)
        rf.fit(X_scaled, y)
        hgb.fit(X_scaled, y)

        latest_features = df[features].iloc[-1].values.reshape(1, -1)
        X_latest = scaler_final.transform(latest_features)

        final_probs = (rf.predict_proba(X_latest) + hgb.predict_proba(X_latest)) / 2
        up_prob, down_prob = final_probs[0][1]*100, final_probs[0][0]*100

        if up_prob >= 65: direction = "🟢 강한 상승 추세"
        elif up_prob >= 50: direction = "🟡 약한 상승 추세"
        elif down_prob >= 65: direction = "🔴 강한 하락 경계"
        else: direction = "🟠 하락 우세"

        current_price = df['Close'].iloc[-1]
        volatility = df['HistVol20'].iloc[-1]
        rel_strength = df['RelStrength_SMH'].iloc[-1]

        final_report += f"📌 {ticker} 분석 결과\n"
        if ticker in LEVERAGED_ETFS:
            info = LEVERAGED_ETFS[ticker]
            final_report += f"  * ⚠️ 레버리지/인버스 상품: {info['desc']} (벤치마크: {info['benchmark']})\n"
            final_report += f"  * ⚠️ 일별 리밸런싱 decay 존재 — 횡보장 장기보유 시 가치 잠식 가능\n"
        if ticker in event_detail:
             final_report += f"  * 📊 [단기 실적 이벤트] {event_detail[ticker]['consensus']}\n"
        final_report += f"  * 방향성: {direction} (상승확률 {up_prob:.1f}%)\n"
        final_report += f"  * 현재가: ${current_price:.2f} | 20일 변동성(연율): {volatility*100:.1f}%\n"
        if ticker not in LEVERAGED_ETFS:
            final_report += f"  * 섹터(SMH) 대비 초과 수익률: {rel_strength*100:+.2f}%\n"
        final_report += f"  * 교차검증 평가지표: F1 {avg_f1*100:.1f}% | P {avg_prec*100:.0f}% | R {avg_rec*100:.0f}%\n"
        final_report += "-"*40 + "\n"

    print("\n" + final_report)

except Exception as e:
    error_msg = f"🚨 에러:\n{traceback.format_exc()[:800]}"
    print(error_msg)
