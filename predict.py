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

# 1. 텔레그램 전송 함수 (GitHub Secrets 환경변수 사용)
def send_telegram_message(text):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("❌ 환경변수 설정 오류: 텔레그램 토큰 또는 ID가 없습니다.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': text}
    requests.post(url, data=payload)

# 2. 기술적 지표 계산 함수
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ==========================================
# 메인 분석 로직 시작
# ==========================================
try:
    print("🚀 레버리지 전용 전략 통제소 v7.0 가동...")
    start_date = '2024-01-01'

    # [1] 데이터 수집 (레버리지 특화)
    lev_raw = fdr.DataReader('306530', start_date) 
    lev_close = lev_raw['Close'].rename('LEV_Close')
    lev_vol = lev_raw['Volume'].rename('LEV_Volume')
    
    sdi = fdr.DataReader('006400', start_date)['Close'].rename('SDI_Close')
    lg = fdr.DataReader('373220', start_date)['Close'].rename('LG_Close')
    posco = fdr.DataReader('005490', start_date)['Close'].rename('POSCO_Close')
    
    usdkrw = fdr.DataReader('USD/KRW', start_date)['Close'].rename('USD_KRW')
    tsla = yf.download('TSLA', start=start_date, progress=False)['Close'].squeeze().shift(1).rename('TSLA_Close')
    vix = yf.download('^VIX', start=start_date, progress=False)['Close'].squeeze().shift(1).rename('VIX_Close')

    df = pd.concat([lev_close, lev_vol, sdi, lg, posco, usdkrw, tsla, vix], axis=1)
    df = df.ffill().dropna()

    # [2] 피처 엔지니어링 및 섹터 과열도 분석
    df['LEV_Return'] = df['LEV_Close'].pct_change()
    df['LEV_RSI'] = calculate_rsi(df['LEV_Close'], period=14)
    df['LEV_Vol_5d'] = df['LEV_Return'].rolling(5).std()
    
    df['SDI_RSI'] = calculate_rsi(df['SDI_Close'], period=14)
    df['LG_RSI'] = calculate_rsi(df['LG_Close'], period=14)
    df['POSCO_RSI'] = calculate_rsi(df['POSCO_Close'], period=14)
    df['Sector_Heat'] = (df['SDI_RSI'] + df['LG_RSI'] + df['POSCO_RSI']) / 3

    # [3] AI 방향성 예측 모델 (타겟: 0.5% 이상 상승)
    TARGET_THRESHOLD = 0.005 
    df['Target_Next_Day'] = np.where(df['LEV_Return'].shift(-1) >= TARGET_THRESHOLD, 1, 0)
    
    features = ['LEV_Return', 'SDI_RSI', 'LG_RSI', 'POSCO_RSI', 'Sector_Heat', 'LEV_RSI', 'VIX_Close']
    df_clean = df.dropna()
    X = df_clean[features]
    y = df_clean['Target_Next_Day']

    split = int(len(df_clean) * 0.8)
    model = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=5, class_weight='balanced')
    model.fit(X.iloc[:split], y.iloc[:split])

    latest = df_clean.iloc[-1]
    prediction = model.predict(latest[features].values.reshape(1, -1))
    
    # [4] 매도 위험 점수 (Risk Score) 산출
    sell_risk_score = 0
    if latest['LEV_RSI'] > 75: sell_risk_score += 40
    if latest['Sector_Heat'] > 70: sell_risk_score += 20
    if latest['LEV_Vol_5d'] > df_clean['LEV_Vol_5d'].tail(20).mean() * 1.3: sell_risk_score += 20
    if latest['VIX_Close'] > 22: sell_risk_score += 20

    # [5] 자산 배분 및 탈출 로직 (v7.0 핵심)
    recent_high = df_clean['LEV_Close'].tail(5).max()
    vol_drop_rate = latest['LEV_Vol_5d'] * 1.5
    stop_loss_price = int(recent_high * (1 - vol_drop_rate))
    
    target_stock_weight = max(0, 100 - sell_risk_score)
    if sell_risk_score >= 80: 
        target_stock_weight = 0
    elif sell_risk_score >= 60: 
        target_stock_weight = min(target_stock_weight, 30)
    target_cash_weight = 100 - target_stock_weight

    result_msg = "상승 돌파 📈" if prediction[0] == 1 else "하락/조정 경보 📉"
    action_guide = ""
    if sell_risk_score >= 80: action_guide = "🚨 [전량 매도] 시장이 광기에 달했습니다. 즉시 현금화하세요."
    elif sell_risk_score >= 60: action_guide = "⚠️ [비중 축소] 초과열 상태입니다. 목표 비중에 맞춰 절반 이상 매수하세요."
    else: action_guide = "✅ [추세 홀딩] 제시된 익절가(스탑로스)만 걸어두고 상승을 즐기세요."

    final_message = f"""
🤖 [레버리지 자산 통제소 v7.0]

📊 위험 점수: {sell_risk_score}점 / 100
({result_msg})

⚖️ [AI 권장 포트폴리오 비중]
* 2차전지 레버리지: {target_stock_weight}% 
* 원화 예금(현금): {target_cash_weight}%
👉 {action_guide}

🛡️ [기계적 탈출선 (트레일링 스톱)]
* 자동 매도 단가: {stop_loss_price:,}원
👉 HTS/MTS에 위 가격으로 자동 감시 주문을 설정하세요.

------------------------
* 현재가 기준 RSI: {latest['LEV_RSI']:.1f}
* 섹터 전체 과열도: {latest['Sector_Heat']:.1f}
------------------------
※ 08:40 KST 시스템 분석 완료
    """

    send_telegram_message(final_message)
    print("✅ 자산 통제 리포트 전송 완료!")

except Exception as e:
    error_msg = traceback.format_exc()
    print(error_msg)
    send_telegram_message(f"🚨 [시스템 에러]\n\n{error_msg[:1000]}")
