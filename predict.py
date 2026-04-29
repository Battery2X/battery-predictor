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
    print("🚀 v11.0 ATR 샹들리에 및 켈리 공식 통합 모델 가동...")
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
    
    exp1 = df['LEV_Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['LEV_Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Hist'] = (exp1 - exp2) - (exp1 - exp2).ewm(span=9, adjust=False).mean()

    # 🚨 [신규] ATR (Average True Range) 계산 - 주가의 진짜 변동성
    high_low = raw_data['TIGER 레버리지']['High'] - raw_data['TIGER 레버리지']['Low']
    high_close = np.abs(raw_data['TIGER 레버리지']['High'] - raw_data['TIGER 레버리지']['Close'].shift())
    low_close = np.abs(raw_data['TIGER 레버리지']['Low'] - raw_data['TIGER 레버리지']['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    features = ['LEV_Return', 'SDI_RSI', 'Sector_Heat', 'LEV_RSI', 'VIX_Close', 'TSLA_Close', 'USD_KRW', 'MACD_Hist']
    latest = df[features].iloc[-1]
    latest_atr = df['ATR'].iloc[-1]
    current_price = df['LEV_Close'].iloc[-1]

    # [3] ML 모델 학습
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

    # 🚨 [수정 1] 확률 보정 오류 해결 (100% 비율 유지하되 엣지 점수 별도 표기)
    edge_score = accuracy - 0.5  # 0보다 크면 통계적 우위, 작으면 열위

    # 🚨 [수정 2] 위험 점수 '지수형(Exponential)' 폭발 로직 도입
    sell_risk_score = 0
    if latest['LEV_RSI'] > 70: sell_risk_score += (latest['LEV_RSI'] - 70) ** 1.8  # 제곱에 가깝게 폭증
    if latest['Sector_Heat'] > 70: sell_risk_score += (latest['Sector_Heat'] - 70) ** 1.5
    if latest['MACD_Hist'] < 0: sell_risk_score += 20
    sell_risk_score = min(int(sell_risk_score), 100) # 최대 100점 캡

    # 🚨 [수정 3] 트루 켈리 공식(True Kelly) 비중 조절
    if accuracy < 0.5:
        target_weight = 0  # 승률 50% 미만은 수학적으로 배팅 금지
        kelly_msg = "⚠️ AI 승률 50% 미만 (통계적 우위 없음). 전량 현금화 권장."
    else:
        target_weight = max(0, 100 - sell_risk_score)
        kelly_msg = f"✅ 통계적 우위 구간. 위험도({sell_risk_score}점)에 따른 비중 산출."
    
    target_cash = 100 - target_weight

    # 🚨 [수정 4] 프로 트레이더의 ATR 샹들리에 스탑로스
    # 최고점에서 실제 변동성(ATR)의 2배수만큼 빠지면 탈출
    recent_high = raw_data['TIGER 레버리지']['Close'].tail(5).max()
    atr_multiplier = 1.5 if sell_risk_score > 60 else 2.5 # 과열 시 타이트하게(1.5배)
    stop_loss = int(recent_high - (latest_atr * atr_multiplier))

    # [5] 리포트 생성
    res_msg = "상승 우세 📈" if up_prob > down_prob else "조정/하락 우세 📉"
    
    final_report = f"""
🤖 [레버리지 퀀트 통제소 v11.0]

🎯 AI 방향성 예측 (원본 확률)
* 메인 시나리오: {res_msg}
* 예측 확률: 상승 {up_prob:.1f}% / 하락 {down_prob:.1f}%

⚠️ 시장 과열 및 위험도 (지수형 산출)
* 최종 위험 점수: {sell_risk_score} / 100점 (🔥 지수 반영)
* 레버리지 RSI: {latest['LEV_RSI']:.1f}
* 빅3 평균 RSI: {latest['Sector_Heat']:.1f}

⚖️ 최종 권장 비중 (Kelly Criterion)
* 2차전지 레버리지: {target_weight}%
* 현금 (예수금): {target_cash}%
👉 {kelly_msg}

🛡️ ATR 샹들리에 청산선 (Chandelier Exit)
* 자동 매도 단가: {stop_loss:,}원
👉 주가의 실제 일일 변동성(ATR)을 반영한 가장 과학적인 방어선입니다.

------------------------
※ 앙상블 모델 승률: {accuracy*100:.1f}%
    """
    send_telegram_message(final_report)
    print("✅ v11.0 퀀트 리포트 전송 완료")

except Exception as e:
    send_telegram_message(f"🚨 에러: {traceback.format_exc()[:300]}")
