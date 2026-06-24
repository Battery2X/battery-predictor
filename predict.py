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
# v17.0 주요 개선 사항 (AI/반도체 슈퍼사이클 맞춤형)
# ============================================================
# [수정 1] 타겟 생성 시 Lookahead Bias(미래 참조) 완전 제거
# [수정 2] 반도체 특화 피처 추가 (SMH 대비 상대강도, 20일 역사적 변동성)
# [수정 3] 시계열 교차 검증(TimeSeriesSplit) 도입으로 과적합 방지
# [수정 4] 노이즈에 강한 HistGradientBoosting 모델로 앙상블 업그레이드
# ============================================================

MARKET_EVENTS = {
    "2026-06-24": {
        "desc": "Micron(MU) Q3 FY26 실적 발표 (장마감 후)",
        "ticker": "MU",
        "consensus": "EPS $19.72 / 매출 $34.5B 예상 | HBM 완판 및 한계 이익률 구조적 변화 확인"
    }
}

def check_upcoming_events():
    from datetime import datetime
    today = datetime.now().date()
    out, out_detail =, {}
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

def calculate_advanced_features(df):
    """v17.0 신규 피처: 구조적 재평가 장세 맞춤형"""
    # 1. 기존 레짐 감지 피처
    rolling_max_252 = df['Close'].rolling(252, min_periods=60).max()
    df = (df['Close'] >= rolling_max_252 * 0.995).astype(int).rolling(20).mean()
    
    short_slope = df['Close'].pct_change(20)
    long_slope = df['Close'].pct_change(120) / 6
    df = short_slope - long_slope
    df = df['Close'].pct_change(120)

    # 2. [신규] 반도체 섹터(SMH) 대비 상대 강도 (Relative Strength)
    if 'SMH' in df.columns:
        df = df['Close'].pct_change(20) - df.pct_change(20)
    else:
        df = 0

    # 3. [신규] 옵션 시장 내재 변동성 대용치 (Historical Volatility 연율화)
    df = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
    
    return df

try:
    print("🚀 v17.0 US AI·반도체 올인 방향성 통제소 가동...")
    event_warnings, event_detail = check_upcoming_events()
    
    start_date = '2022-01-01'
    macro_tickers = {'Nasdaq': '^IXIC', 'SMH': 'SMH', 'VIX': '^VIX', 'TNX': '^TNX'}
    macro_data = {}
    for name, t in macro_tickers.items():
        df_raw = yf.download(t, start=start_date, progress=False, multi_level_index=False)
        if not df_raw.empty:
            macro_data[name] = df_raw['Close'].squeeze()
            
    macro_df = pd.DataFrame(macro_data).ffill().dropna()
    macro_df = macro_df['Nasdaq'].pct_change()
    macro_df = macro_df.pct_change()

    targets =
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
        df = calculate_rsi(df['Close'])
        macd_line = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df = macd_line - macd_line.ewm(span=9, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['Price_to_EMA20'] = (df['Close'] / df['EMA_20']) - 1
        
        bb_upper, bb_ma, bb_lower = calculate_bollinger(df['Close'])
        df = ((df['Close']-bb_lower)/(bb_upper-bb_lower).replace(0, np.nan))*100

        # 특화 피처 생성
        df = calculate_advanced_features(df)

        # [수정 1] 미래 참조 오류 제거: 과거 데이터만으로 동적 임계값 생성
        rolling_median_ret = df['Close'].pct_change().abs().rolling(window=60, min_periods=20).median()
        dyn_threshold_series = np.clip(rolling_median_ret * 0.5, 0.003, 0.025)
        
        df = df['Close'].pct_change().shift(-1)
        df = np.where(df > dyn_threshold_series, 1, 0)

        # 훈련 데이터 준비
        features =
        
        df_train = df.dropna(subset= + features).copy()
        
        if len(df_train) < 300:
            continue

        X = df_train[features]
        y = df_train
        
        # [수정 3] 시계열 교차 검증 (TimeSeriesSplit) 및 모델 훈련
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

        # 최종 예측 (전체 데이터로 마지막 학습 후 가장 최근 데이터 예측)
        scaler_final = StandardScaler()
        X_scaled = scaler_final.fit_transform(X)
        rf.fit(X_scaled, y)
        hgb.fit(X_scaled, y)
        
        latest_features = df[features].iloc[-1].values.reshape(1, -1)
        X_latest = scaler_final.transform(latest_features)
        
        final_probs = (rf.predict_proba(X_latest) + hgb.predict_proba(X_latest)) / 2
        up_prob, down_prob = final_probs*100, final_probs*100

        if up_prob >= 65: direction = "🟢 강한 상승 추세"
        elif up_prob >= 50: direction = "🟡 약한 상승 추세"
        elif down_prob >= 65: direction = "🔴 강한 하락 경계"
        else: direction = "🟠 하락 우세"

        current_price = df['Close'].iloc[-1]
        volatility = df.iloc[-1]
        rel_strength = df.iloc[-1]

        final_report += f"📌 {ticker} 분석 결과\n"
        if ticker in event_detail:
             final_report += f"  * 📊 [단기 실적 이벤트] {event_detail[ticker]['consensus']}\n"
        final_report += f"  * 방향성: {direction} (상승확률 {up_prob:.1f}%)\n"
        final_report += f"  * 현재가: ${current_price:.2f} | 20일 변동성(연율): {volatility*100:.1f}%\n"
        final_report += f"  * 섹터(SMH) 대비 초과 수익률: {rel_strength*100:+.2f}%\n"
        final_report += f"  * 교차검증 평가지표: F1 {avg_f1*100:.1f}% | P {avg_prec*100:.0f}% | R {avg_rec*100:.0f}%\n"
        final_report += "-"*40 + "\n"

    print("\n" + final_report)

except Exception as e:
    error_msg = f"🚨 에러:\n{traceback.format_exc()[:800]}"
    print(error_msg)
