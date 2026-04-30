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
    print("🚀 v13.0 기관급 하이브리드 추세-앙상블 모델 가동...")
    start_date = '2022-01-01'

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
    df['TSLA_Return'] = df['TSLA_Close'].pct_change()
    df['USD_KRW_Return'] = df['USD_KRW'].pct_change()
    
    df['LEV_RSI'] = calculate_rsi(df['LEV_Close'])
    df['SDI_RSI'] = calculate_rsi(df['SDI_Close'])
    df['Sector_Heat'] = (df['SDI_RSI'] + calculate_rsi(df['LG_Close']) + calculate_rsi(df['POSCO_Close'])) / 3
    
    # 🚨 [신규] 대세 추세 판단을 위한 20일 지수이동평균선 (생명선)
    df['EMA_20'] = df['LEV_Close'].ewm(span=20, adjust=False).mean()
    df['Price_to_EMA20'] = (df['LEV_Close'] / df['EMA_20']) - 1 # 이격도
    
    exp1 = df['LEV_Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['LEV_Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Hist'] = (exp1 - exp2) - (exp1 - exp2).ewm(span=9, adjust=False).mean()

    # ATR 계산
    high_low = raw_data['TIGER 레버리지']['High'] - raw_data['TIGER 레버리지']['Low']
    high_close = np.abs(raw_data['TIGER 레버리지']['High'] - raw_data['TIGER 레버리지']['Close'].shift())
    low_close = np.abs(raw_data['TIGER 레버리지']['Low'] - raw_data['TIGER 레버리지']['Close'].shift())
    df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

    # 🚨 학습 피처에 '20일선 이격도(Price_to_EMA20)' 추가 (추세 추종)
    features = ['LEV_Return', 'Sector_Heat', 'LEV_RSI', 'VIX_Close', 'TSLA_Return', 'USD_KRW_Return', 'MACD_Hist', 'Price_to_EMA20']
    
    df_clean = df.dropna()
    latest = df_clean[features].iloc[-1]
    latest_atr = df_clean['ATR'].iloc[-1]
    current_price = df_clean['LEV_Close'].iloc[-1]
    is_uptrend = current_price > df_clean['EMA_20'].iloc[-1] # 대세 상승장 여부

    # [3] ML 모델 학습 
    # 타겟을 0.2% 이상 상승으로 낮춰 노이즈 필터링 강화
    df_clean['Target'] = np.where(df_clean['LEV_Return'].shift(-1) > 0.005, 1, 0)
    X_train, y_train = df_clean[features].iloc[:-1], df_clean['Target'].iloc[:-1]

    split = int(len(X_train) * 0.8)
    model = VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(n_estimators=300, random_state=42, max_depth=5, class_weight='balanced')),
        ('gb', GradientBoostingClassifier(n_estimators=300, random_state=42, max_depth=5))
    ], voting='soft')
    
    model.fit(X_train.iloc[:split], y_train.iloc[:split])
    
    accuracy = accuracy_score(y_train.iloc[split:], model.predict(X_train.iloc[split:]))
    probs = model.predict_proba(latest.values.reshape(1, -1))[0]
    up_prob, down_prob = probs[1] * 100, probs[0] * 100

    # [4] 지수형 위험 점수
    sell_risk_score = 0
    if latest['LEV_RSI'] > 70: sell_risk_score += (latest['LEV_RSI'] - 70) ** 1.8
    if latest['Sector_Heat'] > 70: sell_risk_score += (latest['Sector_Heat'] - 70) ** 1.5
    if latest['MACD_Hist'] < 0: sell_risk_score += 20
    if latest['VIX_Close'] > 22: sell_risk_score += 15
    sell_risk_score = min(int(sell_risk_score), 100) 

    # [5] 🚨 [핵심] 하이브리드 비중 조절 로직 (수익 극대화)
    base_weight = max(0, 100 - sell_risk_score)
    
    if accuracy < 0.5:
        if is_uptrend:
            # 대세 상승장인데 AI가 헷갈려함 -> 잦은 매매 방지를 위해 '추적 비중(20%)'만 홀딩
            target_weight = min(base_weight, 20)
            strategy_msg = "⚠️ AI 승률 저조. 단, 대세 상승장(20일선 위)이므로 최소 추적 비중 20% 유지."
        else:
            # 하락장인데 AI도 헷갈려함 -> 완벽한 0% 현금화
            target_weight = 0
            strategy_msg = "🚨 하락 추세 + AI 승률 저조. 확률적 엣지 소멸. 전량 현금화 대피."
    else:
        target_weight = base_weight
        strategy_msg = f"✅ 통계적 엣지 확보. 과열도({sell_risk_score}점)를 차감한 최적 비중 적용."

    target_cash = 100 - target_weight

    # [6] 상시 출력 ATR 스탑로스
    atr_multiplier = max(1.0, 3.0 - (sell_risk_score / 40)) # 과열될수록 1.0에 수렴
    recent_high = raw_data['TIGER 레버리지']['Close'].tail(5).max()
    stop_loss = int(recent_high - (latest_atr * atr_multiplier))
    stop_loss = max(stop_loss, int(recent_high * 0.90)) # 최대 -10% 하드 캡

    # [7] 리포트 포매팅
    res_msg = "상승 돌파 📈" if up_prob > down_prob else "조정/하락 경보 📉"
    trend_msg = "🟢 대세 상승장 (주가 > 20일선)" if is_uptrend else "🔴 하락/역배열 (주가 < 20일선)"
    
    final_report = f"""
🤖 [레버리지 하이브리드 통제소 v13.0]

📊 딥러닝 시장 분석
* 매크로 추세: {trend_msg}
* 내일 AI 예측: {res_msg} (상승 {up_prob:.1f}% / 하락 {down_prob:.1f}%)
* 모델 신뢰도: {accuracy*100:.1f}%

⚠️ 시장 과열 및 위험도 (0~100)
* 붕괴 위험 점수: {sell_risk_score}점
* 레버리지 RSI: {latest['LEV_RSI']:.1f} | 빅3 평균: {latest['Sector_Heat']:.1f}

⚖️ 최적 포트폴리오 비중
* 2차전지 레버리지: {target_weight}%
* 원화 예수금(현금): {target_cash}%
👉 {strategy_msg}

🛡️ 기관급 ATR 스탑로스 (상시 가동)
* 자동 매도 단가: {stop_loss:,}원
👉 비중 0% 지시가 나오더라도, 개인 판단으로 잔여 물량 홀딩 시 반드시 위 가격을 마지노선으로 설정하세요.
    """
    send_telegram_message(final_report)
    print("✅ v13.0 수익/손실 최적화 리포트 전송 완료")

except Exception as e:
    send_telegram_message(f"🚨 에러: {traceback.format_exc()[:300]}")
