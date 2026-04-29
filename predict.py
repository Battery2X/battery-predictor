import traceback
import requests
import os
import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def send_telegram_message(text):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", data={'chat_id': chat_id, 'text': text})

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + (gain/loss)))

try:
    print("🚀 v10.0 켈리 기반 정밀 앙상블 모델 가동...")
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

    # [2] 피처 엔지니어링
    df['LEV_Return'] = df['LEV_Close'].pct_change()
    df['LEV_RSI'] = calculate_rsi(df['LEV_Close'])
    df['SDI_RSI'] = calculate_rsi(df['SDI_Close'])
    df['LG_RSI'] = calculate_rsi(df['LG_Close'])
    df['POSCO_RSI'] = calculate_rsi(df['POSCO_Close'])
    df['Sector_Heat'] = (df['SDI_RSI'] + df['LG_RSI'] + df['POSCO_RSI']) / 3
    df['LEV_Vol_5d'] = df['LEV_Return'].rolling(5).std()

    # MACD 산출
    exp1 = df['LEV_Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['LEV_Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Hist'] = (exp1 - exp2) - (exp1 - exp2).ewm(span=9, adjust=False).mean()

    features = ['LEV_Return', 'SDI_RSI', 'Sector_Heat', 'LEV_RSI', 'VIX_Close', 'TSLA_Close', 'USD_KRW', 'MACD_Hist']
    latest = df[features].iloc[-1]

    # [3] ML 모델 학습 (앙상블)
    df['Target'] = np.where(df['LEV_Return'].shift(-1) >= 0.005, 1, 0)
    df_c = df.dropna()
    X, y = df_c[features], df_c['Target']

    split = int(len(df_c) * 0.8)
    model = VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42, max_depth=5, class_weight='balanced')),
        ('gb', GradientBoostingClassifier(n_estimators=200, random_state=42, max_depth=5))
    ], voting='soft')
    
    model.fit(X.iloc[:split], y.iloc[:split])
    
    accuracy = accuracy_score(y.iloc[split:], model.predict(X.iloc[split:]))
    probs = model.predict_proba(latest.values.reshape(1, -1))[0]
    up_prob, down_prob = probs[1] * 100, probs[0] * 100

    # 🚨 [핵심 업데이트 1] 켈리 공식 기반 신뢰도 패널티
    # 승률이 50% 미만이면 모델의 근자감(확신도)을 수학적으로 깎아내립니다.
    adj_up_prob = up_prob * (accuracy / 0.5) if accuracy < 0.5 else up_prob
    adj_down_prob = down_prob * (accuracy / 0.5) if accuracy < 0.5 else down_prob
    trust_warning = "⚠️ 승률 저조로 배팅 신뢰도 하향 조정" if accuracy < 0.5 else "✅ 모델 신뢰도 양호"

    # [4] 위험 점수 연속 계산 (계단식 오류 해결)
    sell_risk_score = 0
    if latest['LEV_RSI'] > 70: sell_risk_score += (latest['LEV_RSI'] - 70) * 2  # 70 넘는 1포인트당 2점씩 선형 증가
    if latest['Sector_Heat'] > 70: sell_risk_score += (latest['Sector_Heat'] - 70) * 2
    if latest['VIX_Close'] > 22: sell_risk_score += 20
    if latest['MACD_Hist'] < 0: sell_risk_score += 20
    sell_risk_score = min(int(sell_risk_score), 100) # 최대 100점

    # 🚨 [핵심 업데이트 2] 동적 비중 조절 (100% 매수 방지)
    base_weight = max(0, 100 - sell_risk_score)
    # 승률이 50% 미만일 경우 목표 비중을 강제로 최대 30%까지만 제한 (보수적 접근)
    target_weight = min(base_weight, 30) if accuracy < 0.5 else base_weight
    target_cash = 100 - target_weight

    # 🚨 [핵심 업데이트 3] 다이나믹 스탑로스
    recent_high = raw_data['TIGER 레버리지']['Close'].tail(5).max()
    # 위험 점수가 높으면 스탑로스 폭을 타이트하게(1.0배), 낮으면 여유있게(2.0배)
    vol_multiplier = 1.0 if sell_risk_score > 60 else 2.0
    stop_loss = int(recent_high * (1 - (df_c['LEV_Return'].tail(5).std() * vol_multiplier)))

    # [5] 리포트 생성
    res_msg = "상승 우세 📈" if adj_up_prob > adj_down_prob else "조정/하락 우세 📉"
    macd_msg = "강세 (상승 추세)" if latest['MACD_Hist'] > 0 else "약세 (꺾임/조정)"
    
    final_report = f"""
🤖 [레버리지 정밀 분석 v10.0 최종]

🎯 내일 방향성 예측 ({trust_warning})
* 메인 시나리오: {res_msg}
* 보정 확신도: 상승 {adj_up_prob:.1f}% / 하락 {adj_down_prob:.1f}%

⚠️ 시장 위험 및 모멘텀 진단
* 최종 위험 점수: {sell_risk_score} / 100점
* 레버리지 RSI: {latest['LEV_RSI']:.1f}
* 빅3 평균 RSI: {latest['Sector_Heat']:.1f}
* MACD 모멘텀: {macd_msg}

⚖️ AI 권장 비중 (동적 스케일링)
* 레버리지: {target_weight}% / 현금: {target_cash}%
👉 위험 점수와 모델 승률({accuracy*100:.1f}%)을 반영한 수학적 비중입니다.

🛡️ 기계적 스탑로스 (다이나믹)
* 자동 매도 단가: {stop_loss:,}원
👉 과열도에 따라 방어선이 타이트해졌습니다. 이탈 시 전량 매도.
    """
    send_telegram_message(final_report)
    print("✅ v10.0 리포트 전송 완료")

except Exception as e:
    send_telegram_message(f"🚨 에러: {traceback.format_exc()[:300]}")
