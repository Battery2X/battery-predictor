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
    print("🔮 양자컴퓨팅 4개사 3개월 중기 대세 분석 스크립트 v1.1 가동...")
    start_date = '2021-01-01'
    
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

    # [개선] 3개월(60일) 예측 주기에 맞춘 매크로 트렌드 가공
    macro_df['Nasdaq_60D_Ret'] = macro_df['Nasdaq'].pct_change(60) # 3개월 누적 수익률
    macro_df['TNX_60D_Diff'] = macro_df['TNX'].diff(60)           # 3개월 금리 변동 폭
    macro_df['VIX_60D_Mean'] = macro_df['VIX'].rolling(60).mean()   # 3개월 평균 공포지수
    macro_df = macro_df.dropna()

    quantum_targets = ['IONQ', 'INFQ', 'QBTS', 'RGTI']
    
    final_report = "⚛️ [QUANTUM 양자컴퓨팅 중기 대세 통제소 v1.1]\n"
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
        
        # 중기 모멘텀 지표
        df['RSI_Medium'] = calculate_rsi(df['Close'], period=42) 
        df['MA_60'] = df['Close'].rolling(60).mean()
        df['Disparity_60'] = (df['Close'] / df['MA_60']) - 1
        
        # 변동성 및 타겟 라벨링
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Vol_60'] = df['Log_Ret'].rolling(60).std() * np.sqrt(252)
        df['Target_Threshold'] = df['Vol_60'] * 0.3 * (60 / 252) # 타겟 문턱값 최적화
        
        df['Future_60D_Close'] = df['Close'].shift(-60)
        
        # [개선] 정제된 트렌드 피처셋 사용
        features = ['Nasdaq_60D_Ret', 'TNX_60D_Diff', 'VIX_60D_Mean', 'RSI_Medium', 'Disparity_60', 'Vol_60']
        
        latest_features = df[features].iloc[-1]
        current_price = df['Close'].iloc[-1]
        disp_60 = df['Disparity_60'].iloc[-1]
        
        df_train = df.dropna(subset=['Future_60D_Close'] + features).copy()
        df_train['Target'] = np.where((df_train['Future_60D_Close'] / df_train['Close'] - 1) > df_train['Target_Threshold'], 1, 0)
        
        # [개선] 신생 종목을 위한 최소 행 기준 완화 (300 -> 120)
        MIN_LONG_ROWS = 120
        if len(df_train) < MIN_LONG_ROWS:
            final_report += f"📌 {ticker}\n"
            final_report += f"  * ⚠️ 학습 데이터 부족 (현재 {len(df_train)}행 / 최소 {MIN_LONG_ROWS}행 필요)\n"
            final_report += f"  * 현재가: ${current_price:.2f}\n"
            final_report += "-" * 40 + "\n"
            continue
            
        X = df_train[features]
        y = df_train['Target']
        split = int(len(X) * 0.8)
        
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        rf = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=4, class_weight='balanced')
        gb = GradientBoostingClassifier(n_estimators=300, random_state=42, max_depth=3, learning_rate=0.03)
        
        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        
        rf_probs = rf.predict_proba(X_test)[:, 1]
        gb_probs = gb.predict_proba(X_test)[:, 1]
        ensemble_test_probs = (rf_probs + gb_probs) / 2
        preds = (ensemble_test_probs >= 0.5).astype(int)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        
        # 최신 데이터 예측
        rf_latest = rf.predict_proba(latest_features.values.reshape(1, -1))[0][1]
        gb_latest = gb.predict_proba(latest_features.values.reshape(1, -1))[0][1]
        up_prob = ((rf_latest + gb_latest) / 2) * 100
        
        # [개선] 신뢰도 미달 시 차단 대신 경고 후 확률 오픈
        status_str = f"신뢰도 F1 {f1*100:.1f}%"
        if f1 < 0.48:
            status_str = f"⚠️ 신뢰도 낮음 ({status_str})"
        
        if up_prob >= 62:
            decision = "🟢 [중기 우상향] 3개월 투자 매력도 높음"
        elif up_prob >= 40:
            decision = "🟡 [중기 횡보] 추가 매수 보류 및 보유 유지"
        else:
            decision = "🔴 [중기 위험] 3개월 내 하방 압력 우세"
            
        final_report += f"📌 {ticker} 분석 결과\n"
        final_report += f"  * 🎯 결론: {decision}\n"
        final_report += f"  * 3개월 뒤 상승 확률: {up_prob:.1f}%\n"
        final_report += f"  * 상태: {status_str} | 60일 이격도 {disp_60*100:.1f}%\n"
        final_report += "-" * 40 + "\n"

    print(final_report)
    send_telegram_message(final_report)

except Exception as e:
    error_msg = f"🚨 양자 모델 에러:\n{traceback.format_exc()[:500]}"
    print(error_msg)
    send_telegram_message(error_msg)
