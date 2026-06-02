import traceback
import requests
import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
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
    print("🚀 v17.0 US AI·반도체 올인 전략 '장기 매크로 통제소' 가동 (20영업일 예측)...")

    # ==========================================
    # [1] 장기 매크로 및 유동성 데이터 수집
    # ==========================================
    start_date = '2021-01-01'  # 장기 학습을 위해 데이터 기간 확장
    
    macro_tickers = {
        'Nasdaq': '^IXIC', 
        'SMH': 'SMH',      
        'VIX': '^VIX',     
        'TNX': '^TNX',    # 미 10년물 국채금리
        'IRX': '^IRX'     # 미 13주(3개월) 단기 국채금리 (장단기 금리차 계산용)
    }
    
    macro_data = {}
    for name, t in macro_tickers.items():
        df_raw = yf.download(t, start=start_date, progress=False)
        if not df_raw.empty:
            series = df_raw['Close'].squeeze()
            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)
            macro_data[name] = series
            
    macro_df = pd.DataFrame(macro_data)
    macro_df = macro_df.ffill().dropna()

    # 장기 매크로 피처 설계
    macro_df['Yield_Curve'] = macro_df['TNX'] - macro_df['IRX']  # 장단기 금리차 (리세션 감지)
    macro_df['Nasdaq_3M_Ret'] = macro_df['Nasdaq'].pct_change(60) # 나스닥 3달 누적 수익률
    macro_df['VIX_EMA20'] = macro_df['VIX'].ewm(span=20, adjust=False).mean()

    # ==========================================
    # [2] 타겟 종목 리스트
    # ==========================================
    targets = ['XOVR', 'SOXL', 'NVDL', 'DXYZ', 'TECL']
    
    final_report = "🦅 [US AI·반도체 올인 '장기' 매크로 통제소 v17.0]\n"
    final_report += "📅 분석 기준: 향후 20영업일(한 달 뒤) 대세 상승 확률\n"
    final_report += "=" * 40 + "\n"

    for ticker in targets:
        print(f"진행 중: {ticker} 장기 추세 분석...")
        tgt_data = yf.download(ticker, start=start_date, progress=False)
        
        if tgt_data.empty:
            final_report += f"⚠️ {ticker}: 데이터 수집 불가\n"
            final_report += "-" * 40 + "\n"
            continue
            
        df = pd.DataFrame({
            'High': tgt_data['High'].squeeze(),
            'Low': tgt_data['Low'].squeeze(),
            'Close': tgt_data['Close'].squeeze()
        })
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        df = df.join(macro_df, how='left').ffill().dropna()
        
        # ==========================================
        # [3] 장기 모멘텀 피처 엔지니어링
        # ==========================================
        df['RSI_Weekly'] = calculate_rsi(df['Close'], period=70) # 주간 단위 스케일의 rsi
        df['MA_60'] = df['Close'].rolling(60).mean()
        df['MA_120'] = df['Close'].rolling(120).mean()
        df['Disparity_120'] = (df['Close'] / df['MA_120']) - 1 # 120일선 이격도 (역사적 고점/저점 판독)
        
        # 20영업일(한 달) 변동성 지표
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Vol_20'] = df['Log_Ret'].rolling(20).std() * np.sqrt(252)

        # [장기 타겟 설계] 향후 20영업일 동안의 '평균 가격'이 현재 가격보다 한 달 변동성 폭의 0.5배 이상 상승할지 여부
        df['Long_Threshold'] = df['Vol_20'] * 0.5 * (20 / 252)
        df['Future_20D_Avg'] = df['Close'].shift(-20).rolling(20).mean() # 미래 20일간의 평균가
        
        features = ['Yield_Curve', 'Nasdaq_3M_Ret', 'VIX_EMA20', 'RSI_Weekly', 'Disparity_120', 'Vol_20']
        
        # 최신 상태 추출 (예측 타겟용 오늘 자 데이터)
        latest_features = df[features].iloc[-1]
        current_price = df['Close'].iloc[-1]
        latest_rsi_w = df['RSI_Weekly'].iloc[-1]
        disp_120 = df['Disparity_120'].iloc[-1]
        
        # 데이터 누수 없는 장기 학습 데이터셋 생성
        df_train = df.dropna(subset=['Future_20D_Avg'] + features).copy()
        df_train['Target'] = np.where((df_train['Future_20D_Avg'] / df_train['Close'] - 1) > df_train['Long_Threshold'], 1, 0)
        
        # 장기 모델인 만큼 최소 400영업일(약 1년 8개월) 이상의 데이터 요구
        MIN_LONG_ROWS = 400
        if len(df_train) < MIN_LONG_ROWS:
            final_report += f"🔍 {ticker}\n"
            final_report += f"  * ⚠️ 장기 학습 데이터 부족 (현재 {len(df_train)}행 / 최소 {MIN_LONG_ROWS}행 필요)\n"
            final_report += f"  * 현재가: ${current_price:.2f} | 120일 이격도: {disp_120*100:.1f}%\n"
            final_report += "-" * 40 + "\n"
            continue
            
        # ==========================================
        # [4] 장기 ML 모델 학습 (투 트랙 전략 담당)
        # ==========================================
        X = df_train[features]
        y = df_train['Target']
        split = int(len(X) * 0.8)
        
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # 클래스 균형 가중치 수동 계산 (GB용)
        pos_ratio = y_train.mean()
        neg_ratio = 1 - pos_ratio
        sample_weights = np.where(y_train == 1, 1.0 / (2 * pos_ratio + 1e-9), 1.0 / (2 * neg_ratio + 1e-9))
        
        rf = RandomForestClassifier(n_estimators=500, random_state=42, max_depth=6, class_weight='balanced')
        gb = GradientBoostingClassifier(n_estimators=500, random_state=42, max_depth=4, learning_rate=0.03, subsample=0.8)
        
        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train, sample_weight=sample_weights)
        
        # 검증셋 평가
        rf_probs = rf.predict_proba(X_test)
        gb_probs = gb.predict_proba(X_test)
        ensemble_probs = (rf_probs + gb_probs) / 2
        preds = (ensemble_probs[:, 1] >= 0.5).astype(int)
        
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        
        # 장기 거시 추세 모델 신뢰도 커트라인
        MIN_F1_LONG = 0.52
        if f1 < MIN_F1_LONG:
            final_report += f"📌 {ticker}\n"
            final_report += f"  * ⚠️ 매크로 관망 (장기 신뢰도 F1 {f1*100:.1f}% 미달)\n"
            final_report += f"  * 현재가: ${current_price:.2f} | 120일 이격도: {disp_120*100:.1f}%\n"
            final_report += "-" * 40 + "\n"
            continue
            
        # 오늘 기준 한 달 뒤 최종 예측
        rf_latest = rf.predict_proba(latest_features.values.reshape(1, -1))
        gb_latest = gb.predict_proba(latest_features.values.reshape(1, -1))
        final_probs = (rf_latest + gb_latest) / 2
        up_prob = final_probs[0][1] * 100
        
        # 장기 전략 판단 마스터 로직
        if up_prob >= 60:
            strategy = "🟢 [전략 승인] 한 달간 대세 상승 주기 진입 유력"
        elif up_prob >= 45:
            strategy = "🟡 [전략 보류] 추세 방향성 모호, 분할 진입만 유효"
        else:
            strategy = "🔴 [전략 거부] 한 달 내 하방 압력 우세, 현금 확보 권장"
            
        final_report += f"📌 {ticker} 장기 추세 결과\n"
        final_report += f"  * 🎯 전략: {strategy}\n"
        final_report += f"  * 한 달 뒤 상승 확률: {up_prob:.1f}%\n"
        final_report += f"  * 상태: 장기 F1 {f1*100:.1f}% | 주간 RSI {latest_rsi_w:.1f} | 120일 이격도 {disp_120*100:.1f}%\n"
        final_report += "-" * 40 + "\n"

    print("\n" + final_report)
    send_telegram_message(final_report)
    print("✅ v17.0 장기 매크로 리포트 전송 완료")

except Exception as e:
    error_msg = f"🚨 장기 모델 에러 발생:\n{traceback.format_exc()[:800]}"
    print(error_msg)
    send_telegram_message(error_msg)
