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
    print("🔮 [v1.2] 양자컴퓨팅 4개사 듀얼 타임프레임 리스크 관리 엔진 가동...")
    start_date = '2021-01-01'
    
    macro_tickers = {'Nasdaq': '^IXIC', 'VIX': '^VIX', 'TNX': '^TNX'}
    macro_raw = {}
    for name, t in macro_tickers.items():
        df_raw = yf.download(t, start=start_date, progress=False)
        if not df_raw.empty:
            series = df_raw['Close'].squeeze()
            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)
            macro_raw[name] = series
    macro_base = pd.DataFrame(macro_raw).ffill().dropna()

    quantum_targets = ['IONQ', 'INFQ', 'QBTS', 'RGTI']
    
    final_report = "⚛️ [QUANTUM 양자컴퓨팅 리스크 관리 통제소 v1.2]\n"
    final_report += "=" * 40 + "\n"

    for ticker in quantum_targets:
        tgt_data = yf.download(ticker, start=start_date, progress=False)
        if tgt_data.empty:
            final_report += f"📌 {ticker}\n  * ⚠️ 데이터 수집 불가\n"
            final_report += "-" * 40 + "\n"
            continue
            
        # --------------------------------------------------------
        # [디벨롭 1] 듀얼 타임프레임 엔진 (데이터 길이에 따른 자동 전환)
        # --------------------------------------------------------
        raw_len = len(tgt_data)
        if raw_len >= 250:
            lookahead = 60
            window_name = "3개월 (60영업일 뒤)"
            min_rows = 120
        else:
            lookahead = 20
            window_name = "1개월 (20영업일 뒤)"
            min_rows = 40  # 데이터가 극도로 적은 신생 종목용 커트라인

        # 해당 주기에 맞춤형 매크로 트렌드 가공
        macro_df = macro_base.copy()
        macro_df['Macro_Ret'] = macro_df['Nasdaq'].pct_change(lookahead)
        macro_df['TNX_Diff'] = macro_df['TNX'].diff(lookahead)
        macro_df['VIX_Mean'] = macro_df['VIX'].rolling(lookahead).mean()
        macro_df = macro_df.dropna()

        df = pd.DataFrame({
            'High': tgt_data['High'].squeeze(),
            'Low': tgt_data['Low'].squeeze(),
            'Close': tgt_data['Close'].squeeze()
        })
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        df = df.join(macro_df, how='left').ffill().dropna()
        
        # 타임프레임 최적화 피처 엔지니어링
        df['RSI_Target'] = calculate_rsi(df['Close'], period=int(lookahead * 0.7)) 
        df['MA_Target'] = df['Close'].rolling(lookahead).mean()
        df['Disparity_Target'] = (df['Close'] / df['MA_Target']) - 1
        
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Vol_Target'] = df['Log_Ret'].rolling(lookahead).std() * np.sqrt(252)
        df['Target_Threshold'] = df['Vol_Target'] * 0.3 * (lookahead / 252)
        
        # 미래 타겟 매칭
        df['Future_Close'] = df['Close'].shift(-lookahead)
        
        features = ['Macro_Ret', 'TNX_Diff', 'VIX_Mean', 'RSI_Target', 'Disparity_Target', 'Vol_Target']
        
        latest_features = df[features].iloc[-1]
        current_price = df['Close'].iloc[-1]
        disp_target = df['Disparity_Target'].iloc[-1]
        ma_target_val = df['MA_Target'].iloc[-1]
        
        df_train = df.dropna(subset=['Future_Close'] + features).copy()
        df_train['Target'] = np.where((df_train['Future_Close'] / df_train['Close'] - 1) > df_train['Target_Threshold'], 1, 0)
        
        if len(df_train) < min_rows:
            final_report += f"📌 {ticker} ({window_name})\n"
            final_report += f"  * ⚠️ 학습 데이터 부족 (현재 {len(df_train)}행)\n"
            final_report += "-" * 40 + "\n"
            continue
            
        # 머신러닝 연산
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
        
        # 오늘 기준 상승 확률 계산
        rf_latest = rf.predict_proba(latest_features.values.reshape(1, -1))[0][1]
        gb_latest = gb.predict_proba(latest_features.values.reshape(1, -1))[0][1]
        up_prob = ((rf_latest + gb_latest) / 2) * 100
        
        # --------------------------------------------------------
        # [디벨롭 2] 가짜 신호 강제 차단 필터 (F1 50% 미만 덮어쓰기)
        # --------------------------------------------------------
        if f1 < 0.50:
            decision = "🔴 [매매 보류] 모델 신뢰도 미달로 예측 무효화"
            prob_str = "계산 불가 (영역 불확실)"
        else:
            prob_str = f"{up_prob:.1f}%"
            if up_prob >= 62:
                decision = "🟢 [중기 우상향] 적극적 모멘텀 유효"
            elif up_prob >= 40:
                decision = "🟡 [중기 횡보] 추가 매수 금지 및 비중 유지"
            else:
                decision = "🔴 [중기 위험] 하방 압력 우세, 리스크 관리"
                
        # --------------------------------------------------------
        # [디벨롭 3] 이격도 연동형 안전 지정가 거미줄 계산기
        # --------------------------------------------------------
        if disp_target > 0.15:  # 이격도가 15% 이상 벌어진 과열 상태일 때
            safe_lower = ma_target_val * 0.95
            safe_upper = ma_target_val * 1.02
            target_band_str = f"${safe_lower:.2f} ~ ${safe_upper:.2f} (과열 진정 타점)"
        else:
            target_band_str = "현재가 인근 분할 대응 가능"

        final_report += f"📌 {ticker} 분석 결과 [{window_name}]\n"
        final_report += f"  * 🎯 결론: {decision}\n"
        final_report += f"  * {window_name} 상승 확률: {prob_str}\n"
        final_report += f"  * 안전 매수 지정가 밴드: {target_band_str}\n"
        final_report += f"  * 상태: 분석 신뢰도 F1 {f1*100:.1f}% | 이격도 {disp_target*100:.1f}%\n"
        final_report += "-" * 40 + "\n"

    print(final_report)
    send_telegram_message(final_report)

except Exception as e:
    error_msg = f"🚨 양자 제어 엔진 v1.2 에러:\n{traceback.format_exc()[:500]}"
    print(error_msg)
    send_telegram_message(error_msg)
