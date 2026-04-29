import traceback
import requests
import os
import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import numpy as np
# 💡 앙상블 모델을 위한 라이브러리 추가
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def send_telegram_message(text):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("❌ 환경변수 오류")
        return
    requests.post(f"https://api.telegram.org/bot{token}/sendMessage", data={'chat_id': chat_id, 'text': text})

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + (gain/loss)))

try:
    print("🚀 v9.0 앙상블 정밀 분석 모델 가동...")
    start_date = '2024-01-01'

    # [1] 데이터 수집
    tickers = {'TIGER 레버리지': '412570', '삼성SDI': '006400', 'LG엔솔': '373220', 'POSCO홀딩스': '005490'}
    raw_data = {name: fdr.DataReader(t, start_date) for name, t in tickers.items()}
    
    usdkrw = fdr.DataReader('USD/KRW', start_date)['Close'].rename('USD_KRW')
    tsla = yf.download('TSLA', start=start_date, progress=False)['Close'].squeeze().shift(1).rename('TSLA_Close')
    vix = yf.download('^VIX', start=start_date, progress=False)['Close'].squeeze().shift(1).rename('VIX_Close')

    df = pd.concat([
        raw_data['TIGER 레버리지']['Close'].rename('LEV_Close'),
        raw_data['삼성SDI']['Close'].rename('SDI_Close'),
        raw_data['LG엔솔']['Close'].rename('LG_Close'),
        raw_data['POSCO홀딩스']['Close'].rename('POSCO_Close'),
        usdkrw, tsla, vix
    ], axis=1).ffill().dropna()

    # [2] 피처 엔지니어링 (MACD 추가)
    df['LEV_Return'] = df['LEV_Close'].pct_change()
    df['LEV_RSI'] = calculate_rsi(df['LEV_Close'])
    df['SDI_RSI'] = calculate_rsi(df['SDI_Close'])
    df['LG_RSI'] = calculate_rsi(df['LG_Close'])
    df['POSCO_RSI'] = calculate_rsi(df['POSCO_Close'])
    df['Sector_Heat'] = (df['SDI_RSI'] + df['LG_RSI'] + df['POSCO_RSI']) / 3
    df['LEV_Vol_5d'] = df['LEV_Return'].rolling(5).std()

    # MACD 산출 (12일 EMA - 26일 EMA)
    exp1 = df['LEV_Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['LEV_Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal'] # MACD 모멘텀

    # 🚨 수정 완료: TSLA, 환율, MACD를 모두 학습 변수에 투입!
    features = ['LEV_Return', 'SDI_RSI', 'Sector_Heat', 'LEV_RSI', 'VIX_Close', 'TSLA_Close', 'USD_KRW', 'MACD_Hist']
    latest_features = df[features].iloc[-1]

    # [3] ML 모델 학습 (다중 앙상블 모델로 교체)
    TARGET_THRESHOLD = 0.005 
    df['Target_Next_Day'] = np.where(df['LEV_Return'].shift(-1) >= TARGET_THRESHOLD, 1, 0)
    df_clean = df.dropna()
    X, y = df_clean[features], df_clean['Target_Next_Day']

    split = int(len(df_clean) * 0.8)
    
    # Random Forest와 Gradient Boosting을 결합한 Soft Voting 앙상블
    clf1 = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=5, class_weight='balanced')
    clf2 = GradientBoostingClassifier(n_estimators=200, random_state=42, max_depth=5)
    model = VotingClassifier(estimators=[('rf', clf1), ('gb', clf2)], voting='soft')
    
    model.fit(X.iloc[:split], y.iloc[:split])
    
    accuracy = accuracy_score(y.iloc[split:], model.predict(X.iloc[split:]))
    probs = model.predict_proba(latest_features.values.reshape(1, -1))[0]
    up_prob, down_prob = probs[1] * 100, probs[0] * 100

    # [4] 종목별 예상 변동 범위 계산
    individual_ranges = ""
    for name, ticker_code in tickers.items():
        price_series = raw_data[name]['Close']
        curr_p = price_series.iloc[-1]
        vol = price_series.pct_change().tail(20).std()
        individual_ranges += f"* {name}: {int(curr_p*(1-vol)):,}원 ~ {int(curr_p*(1+vol)):,}원 (±{vol*100:.2f}%)\n"

    # [5] 위험 점수 (MACD 하락 다이버전스 패널티 추가)
    sell_risk_score = 0
    if latest_features['LEV_RSI'] > 75: sell_risk_score += 30
    if latest_features['Sector_Heat'] > 75: sell_risk_score += 20
    if latest_features['VIX_Close'] > 22: sell_risk_score += 20
    if latest_features['MACD_Hist'] < 0: sell_risk_score += 30 # MACD 모멘텀 꺾임

    recent_high = raw_data['TIGER 레버리지']['Close'].tail(5).max()
    stop_loss = int(recent_high * (1 - (df_clean['LEV_Return'].tail(5).std() * 1.5)))
    target_weight = 30 if sell_risk_score >= 60 else (0 if sell_risk_score >= 80 else 100)

    # [6] 리포트 생성
    res_msg = "상승 우세 📈" if up_prob > down_prob else "조정/하락 우세 📉"
    
    final_report = f"""
🤖 [레버리지 정밀 분석 v9.0 앙상블]

🎯 내일 방향성 예측
* 메인 시나리오: {res_msg}
* 앙상블 확신도: {max(up_prob, down_prob):.1f}% (상승 {up_prob:.1f}% / 하락 {down_prob:.1f}%)

📊 종목별 예상 변동 범위 (1-Sigma)
{individual_ranges}
⚠️ 섹터 과열 및 모멘텀
* 레버리지 RSI: {latest_features['LEV_RSI']:.1f}
* 빅3 평균 RSI: {latest_features['Sector_Heat']:.1f}
* MACD 모멘텀: {"강세 (상승 추세)" if latest_features['MACD_Hist'] > 0 else "약세 (꺾임/조정)"}

⚖️ 권장 비중: 레버리지 {target_weight}% / 현금 {100-target_weight}%
🛡️ 기계적 스탑로스: {stop_loss:,}원

※ 모델: RF + GB 앙상블 | 승률: {accuracy*100:.2f}%
    """
    send_telegram_message(final_report)
    print("✅ v9.0 앙상블 리포트 전송 완료")

except Exception as e:
    send_telegram_message(f"🚨 에러: {traceback.format_exc()[:300]}")
