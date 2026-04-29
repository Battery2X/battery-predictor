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

# 2. 기술적 지표 계산 함수들
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, short=12, long=26):
    exp1 = series.ewm(span=short, adjust=False).mean()
    exp2 = series.ewm(span=long, adjust=False).mean()
    return exp1 - exp2

# ==========================================
# 메인 로직 시작
# ==========================================
try:
    print("🚀 예측 및 매도 분석 시스템 가동...")
    start_date = '2024-01-01'

    # [1] 데이터 수집 (Volume 및 VIX 추가)
    # 국장 데이터
    etf_raw = fdr.DataReader('305540', start_date)
    etf = etf_raw['Close'].rename('ETF_Close')
    etf_vol = etf_raw['Volume'].rename('ETF_Volume') # 거래량 중요
    sdi = fdr.DataReader('006400', start_date)['Close'].rename('SDI_Close')
    lg = fdr.DataReader('373220', start_date)['Close'].rename('LG_Close')
    posco = fdr.DataReader('005490', start_date)['Close'].rename('POSCO_Close')
    usdkrw = fdr.DataReader('USD/KRW', start_date)['Close'].rename('USD_KRW')

    # 미국 데이터 (시차 교정)
    tsla = yf.download('TSLA', start=start_date)['Close'].squeeze().shift(1).rename('TSLA_Close')
    lit = yf.download('LIT', start=start_date)['Close'].squeeze().shift(1).rename('LIT_Close')
    vix = yf.download('^VIX', start=start_date)['Close'].squeeze().shift(1).rename('VIX_Close')

    # 데이터 병합
    df = pd.concat([etf, etf_vol, sdi, lg, posco, usdkrw, tsla, lit, vix], axis=1)
    df = df.fillna(method='ffill').dropna()

    # [2] 피처 엔지니어링 (매도 지표 포함)
    for col in ['ETF_Close', 'SDI_Close', 'LG_Close', 'POSCO_Close', 'USD_KRW', 'TSLA_Close', 'LIT_Close']:
        df[col.replace('Close', 'Return').replace('USD_KRW', 'FX_Return')] = df[col].pct_change()

    # 매도 판단을 위한 지표들
    df['ETF_RSI'] = calculate_rsi(df['ETF_Close'], period=14)
    df['ETF_MACD'] = calculate_macd(df['ETF_Close'])
    df['ETF_MA5_Ratio'] = df['ETF_Close'] / df['ETF_Close'].rolling(window=5).mean()
    df['Vol_Change'] = df['ETF_Volume'].pct_change() # 거래량 변화율
    # 볼린저 밴드 상단 (20일 기준)
    df['BB_Upper'] = df['ETF_Close'].rolling(20).mean() + (df['ETF_Close'].rolling(20).std() * 2)

    # [3] ML 모델 학습 (방향성 예측)
    TARGET_THRESHOLD = 0.003
    df['Target_Next_Day'] = np.where(df['ETF_Return'].shift(-1) >= TARGET_THRESHOLD, 1, 0)
    df = df.dropna()

    features = ['ETF_Return', 'SDI_Return', 'LG_Return', 'POSCO_Return', 
                'FX_Return', 'TSLA_Return', 'LIT_Return', 'ETF_RSI', 
                'ETF_MA5_Ratio', 'ETF_MACD']

    X = df[features]
    y = df['Target_Next_Day']

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=5, class_weight='balanced')
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    # [4] 실전 예측 및 매도 위험 점수 계산
    latest = df.iloc[-1]
    prediction = model.predict(latest[features].values.reshape(1, -1))
    prob = model.predict_proba(latest[features].values.reshape(1, -1))[0]

    # --- 매도 위험 점수 산출 로직 ---
    sell_risk_score = 0
    if latest['ETF_RSI'] > 70: sell_risk_score += 30 # 과매수 상태
    # 주가는 올랐는데 거래량이 전날보다 줄었다면 (상승 동력 약화)
    if latest['ETF_Return'] > 0 and latest['Vol_Change'] < 0: sell_risk_score += 30 
    if latest['ETF_Close'] >= latest['BB_Upper']: sell_risk_score += 20 # 밴드 상단 터치
    if latest['VIX_Close'] > 20: sell_risk_score += 20 # 시장 공포 지수 상승

    # [5] 메시지 구성
    result_msg = "📈 확실한 상승 예상" if prediction[0] == 1 else "📉 하락 또는 횡보 예상"
    
    risk_comment = ""
    if sell_risk_score >= 70:
        risk_comment = "🚨 [강력 매도 권고] 지표가 매우 과열되었습니다. 오늘 시초가에 수익 실현을 강력 추천합니다!"
    elif sell_risk_score >= 40:
        risk_comment = "⚠️ [주의] 상승세가 둔화되고 있습니다. 이익 보존을 위해 분할 매도를 고려하세요."
    else:
        risk_comment = "✅ [보유 유지] 과열 신호가 없습니다. 추세를 좀 더 즐기셔도 좋습니다."

    final_message = f"""
🤖 [TIGER 2차전지 TOP10 AI 리포트]

🎯 내일 방향: {result_msg} (확률: {max(prob)*100:.1f}%)
📊 매도 위험 점수: {sell_risk_score}점 / 100
💡 전략 가이드: {risk_comment}

------------------------
* RSI: {latest['ETF_RSI']:.1f} (70이상 과매수)
* VIX(공포지수): {latest['VIX_Close']:.1f}
* 모델 승률: {accuracy * 100:.2f}%
------------------------
※ 08:40 분석 데이터 기준
    """

    send_telegram_message(final_message)
    print("✅ 분석 완료 및 전송 성공")

except Exception as e:
    error_msg = traceback.format_exc()
    print(error_msg)
    send_telegram_message(f"🚨 [에러 알림]\n\n{error_msg[:1000]}")
