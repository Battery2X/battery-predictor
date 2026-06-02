import traceback
import requests
import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

def send_telegram_message(text):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={'chat_id': chat_id, 'text': text}
            )
        except Exception as e:
            print("텔레그램 전송 실패:", e)

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1 / period, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - (100 / (1 + (gain / loss)))

try:
    print("🔮 양자컴퓨팅 4개사 3개월(60영업일) 중기 대세 분석 스크립트 가동...")
    start_date = '2021-01-01'
    
    # 매크로 유동성 데이터 수집
    macro_tickers = {'Nasdaq': '^IXIC', 'VIX': '^VIX', 'TNX': '^TNX'}
    macro_data = {}
    for name, t in macro_tickers.items():
        df_raw = yf.download(t, start=start_date, progress=False)
        if not df_raw.empty:
            series = df_raw['Close'].squeeze()
            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)
            macro_data[name] = series
    macro_df = pd.DataFrame(macro_data).ffill().dropna()

    # 양자컴퓨팅 타겟 종목 (4개사)
    quantum_targets = ['IONQ', 'INFQ', 'QBTS', 'RGTI']
    
    final_report = "⚛️ [QUANTUM 양자컴퓨팅 중기 대세 통제소 v1.0]\n"
    final_report += "📅 분석 기준: 향후 3개월(60영업일 뒤) 대세 상승 확률\n"
    final_report += "=" * 40 + "\n"

    for ticker in quantum_targets:
        tgt_data = yf.download(ticker, start=start_date, progress=False)
        if tgt_data.empty:
            final_report += f"⚠️ {ticker}: 데이터 수집 불가\n"
            continue
            
        df = pd.DataFrame({
            'High': tgt_data['High'].squeeze(),
            'Low': tgt_data['Low'].squeeze(),
            'Close': tgt_data['Close'].squeeze()
        })
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        df = df.join(macro_df, how='left').ffill().dropna()
        
        # 3개월 예측용 중기 모멘텀 피처 생성
        df['RSI_Medium'] = calculate_rsi(df['Close'], period=42) # 2달 기준 과열도
        df['MA_60'] = df['Close'].rolling(60).mean()
        df['MA_120'] = df['Close'].rolling(120).mean()
        df['Disparity_60'] = (df['Close'] / df['MA_60']) - 1
        
        # 3개월 변동성 및 타겟 라벨링 (60일 뒤 종가가 변동성 폭 이상 상승할 확률)
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Vol_60'] = df['Log_Ret'].rolling(60).std() * np.sqrt(252)
        df['Target_Threshold'] = df['Vol_60'] * 0.4 * (60 / 252)
        
        # 미래 60영업일 뒤의 종가 비교
        df['Future_60D_Close'] = df['Close'].shift(-60)
        
        features = ['Nasdaq', 'VIX', 'TNX', 'RSI_Medium', 'Disparity_60', 'Vol_60']
        latest_features = df[features].iloc[-1]
        current_price = df['Close'].iloc[-1]
        latest_rsi = df['RSI_Medium'].iloc[-1]
        disp_60 = df['Disparity_60'].iloc[-1]
        
        df_train = df.dropna(subset=['Future_60D_Close'] + features).copy()
        df_train['Target'] = np.where((df_train['Future_60D_Close'] / df_train['Close'] - 1) > df_train['Target_Threshold'], 1, 0)
        
        if len(df_train) < 300:
            final_report += f"🔍 {ticker}\n  * ⚠️ 학습 데이터 부족\n"
            continue
            
        # ML 학습
        X = df_train[features]
        y = df_train['Target']
        split = int(len(X) * 0.8)
        
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        rf = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=5, class_weight='balanced')
        gb = GradientBoostingClassifier(n_estimators=300, random_state=42, max_depth=3, learning_rate=0.05)
        
        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        
        # 평가 및 앙상블
        rf_probs = rf.predict_proba(X_test)[:, 1]
        gb_probs = gb.predict_proba(X_test)[:, 1]
        ensemble_test_probs = (rf_probs + gb_probs) / 2
        preds = (ensemble_test_probs >= 0.5).astype(int)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        
        # 신뢰도 필터 검증
        if f1 < 0.50:
            final_report += f"📌 {ticker}\n  * ⚠️ 매크로 관망 (모델 신뢰도 F1 {f1*100:.1f}% 미달)\n"
            continue
            
        # 최신 데이터 예측
        rf_latest = rf.predict_proba(latest_features.values.reshape(1, -1))[0][1]
        gb_latest = gb.predict_proba(latest_features.values.reshape(1, -1))[0][1]
        up_prob = ((rf_latest + gb_latest) / 2) * 100
        
        if up_prob >= 60:
            decision = "🟢 [대세 상승 진입] 3개월 투자 매력도 높음"
        elif up_prob >= 40:
            decision = "🟡 [박스권 횡보] 추가 매수 자제 및 보유 유지"
        else:
            decision = "🔴 [리스크 관리] 3개월 내 조정 확률 높음, 비중 축소"
            
        final_report += f"📌 {ticker} 분석 결과\n"
        final_report += f"  * 🎯 결론: {decision}\n"
        final_report += f"  * 3개월 뒤 상승 확률: {up_prob:.1f}%\n"
        final_report += f"  * 상태: 신뢰도 F1 {f1*100:.1f}% | 60일 이격도 {disp_60*100:.1f}%\n"
        final_report += "-" * 40 + "\n"

    print(final_report)
    send_telegram_message(final_report)

except Exception as e:
    error_msg = f"🚨 양자 모델 에러:\n{traceback.format_exc()[:500]}"
    send_telegram_message(error_msg)
