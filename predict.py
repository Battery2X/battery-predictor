import traceback
import requests
import os
import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 1. 텔레그램 전송 함수
def send_telegram_message(text):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("❌ 환경변수 설정 오류")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': text}
    requests.post(url, data=payload)

# 2. RSI 계산 함수 (EMA 방식)
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + (gain/loss)))

# ==========================================
# 메인 분석 로직 시작
# ==========================================
try:
    print("🚀 v8.2 개별 종목 정밀 분석 모델 가동...")
    start_date = '2024-01-01'

    # [1] 데이터 수집 및 개별 데이터프레임 관리
    tickers = {
        'TIGER 레버리지': '412570', 
        '삼성SDI': '006400', 
        'LG엔솔': '373220', 
        'POSCO홀딩스': '005490'
    }
    
    raw_data = {}
    for name, t in tickers.items():
        raw_data[name] = fdr.DataReader(t, start_date)
    
    # 매크로 지표
    usdkrw = fdr.DataReader('USD/KRW', start_date)['Close'].rename('USD_KRW')
    tsla = yf.download('TSLA', start=start_date, progress=False)['Close'].squeeze().shift(1).rename('TSLA_Close')
    vix = yf.download('^VIX', start=start_date, progress=False)['Close'].squeeze().shift(1).rename('VIX_Close')

    # 학습용 통합 DF 생성
    df = pd.concat([
        raw_data['TIGER 레버리지']['Close'].rename('LEV_Close'),
        raw_data['삼성SDI']['Close'].rename('SDI_Close'),
        raw_data['LG엔솔']['Close'].rename('LG_Close'),
        raw_data['POSCO홀딩스']['Close'].rename('POSCO_Close'),
        usdkrw, tsla, vix
    ], axis=1).ffill().dropna()

    # [2] 피처 엔지니어링
    df['LEV_Return'] = df['LEV_Close'].pct_change()
    df['LEV_RSI'] = calculate_rsi(df['LEV_Close'])
    df['SDI_RSI'] = calculate_rsi(df['SDI_Close'])
    df['LG_RSI'] = calculate_rsi(df['LG_Close'])
    df['POSCO_RSI'] = calculate_rsi(df['POSCO_Close'])
    df['Sector_Heat'] = (df['SDI_RSI'] + df['LG_RSI'] + df['POSCO_RSI']) / 3
    df['LEV_Vol_5d'] = df['LEV_Return'].rolling(5).std()

    features = ['LEV_Return', 'SDI_RSI', 'LG_RSI', 'POSCO_RSI', 'Sector_Heat', 'LEV_RSI', 'VIX_Close']
    latest_features = df[features].iloc[-1]

    # [3] ML 모델 학습 및 확률 추출
    TARGET_THRESHOLD = 0.005 
    df['Target_Next_Day'] = np.where(df['LEV_Return'].shift(-1) >= TARGET_THRESHOLD, 1, 0)
    df_clean = df.dropna()
    X = df_clean[features]
    y = df_clean['Target_Next_Day']

    split = int(len(df_clean) * 0.8)
    model = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=5, class_weight='balanced')
    model.fit(X.iloc[:split], y.iloc[:split])
    
    accuracy = accuracy_score(y.iloc[split:], model.predict(X.iloc[split:]))
    probs = model.predict_proba(latest_features.values.reshape(1, -1))[0]
    up_prob, down_prob = probs[1] * 100, probs[0] * 100

    # [4] 개별 종목별 예상 변동 범위 계산 (핵심 업데이트)
    individual_ranges = ""
    for name, ticker_code in tickers.items():
        price_series = raw_data[name]['Close']
        curr_p = price_series.iloc[-1]
        vol = price_series.pct_change().tail(20).std() # 20일 변동성
        low_p = int(curr_p * (1 - vol))
        high_p = int(curr_p * (1 + vol))
        individual_ranges += f"* {name}: {low_p:,}원 ~ {high_p:,}원 (±{vol*100:.2f}%)\n"

    # [5] 위험 점수 및 비중 계산
    sell_risk_score = 0
    if latest_features['LEV_RSI'] > 75: sell_risk_score += 40
    if latest_features['Sector_Heat'] > 70: sell_risk_score += 20
    if latest_features['VIX_Close'] > 22: sell_risk_score += 20
    if latest_features['LEV_RSI'] > 80: sell_risk_score += 20

    recent_high = raw_data['TIGER 레버리지']['Close'].tail(5).max()
    stop_loss = int(recent_high * (1 - (df_clean['LEV_Return'].tail(5).std() * 1.5)))
    target_weight = 30 if sell_risk_score >= 60 else (0 if sell_risk_score >= 80 else 100)

    # [6] 리포트 생성
    res_msg = "상승 우세 📈" if up_prob > down_prob else "조정/하락 우세 📉"
    
    final_report = f"""
🤖 [레버리지 정밀 분석 v8.2]

🎯 내일 방향성 예측
* 메인 시나리오: {res_msg}
* 확신도: {max(up_prob, down_prob):.1f}% (상승 {up_prob:.1f}% / 하락 {down_prob:.1f}%)

📊 종목별 예상 변동 범위 (1-Sigma)
{individual_ranges}
⚠️ 섹터 과열 상태
* 레버리지 RSI: {latest_features['LEV_RSI']:.1f}
* 빅3 평균 RSI: {latest_features['Sector_Heat']:.1f}
  - 삼성SDI: {latest_features['SDI_RSI']:.1f}
  - LG엔솔: {latest_features['LG_RSI']:.1f}
  - POSCO홀딩스: {latest_features['POSCO_RSI']:.1f}

⚖️ 권장 비중: 레버리지 {target_weight}% / 현금 {100-target_weight}%
🛡️ 기계적 스탑로스: {stop_loss:,}원

※ 모델 승률: {accuracy*100:.2f}% | EMA 기준
    """
    send_telegram_message(final_report)
    print("✅ 개별 종목 범위 포함 리포트 전송 완료")

except Exception as e:
    send_telegram_message(f"🚨 에러: {traceback.format_exc()[:300]}")
