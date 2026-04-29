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

# 2. RSI 계산 함수 (증권사 HTS 표준 EMA 방식 적용)
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ==========================================
# 메인 분석 로직 시작
# ==========================================
try:
    print("🚀 레버리지 정밀 통제소 v8.0 가동 (티커: 412570)...")
    start_date = '2024-01-01'

    # [1] 데이터 수집 (레버리지 특화: 412570)
    lev_raw = fdr.DataReader('412570', start_date) 
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

    features = ['LEV_Return', 'SDI_RSI', 'LG_RSI', 'POSCO_RSI', 'Sector_Heat', 'LEV_RSI', 'VIX_Close']

    # 🚨 미래 참조 버그 방지: '오늘' 데이터를 결측치 제거 전에 미리 추출
    latest_features = df[features].iloc[-1]
    recent_high = df['LEV_Close'].tail(5).max()
    latest_vol = df['LEV_Vol_5d'].iloc[-1]
    current_price = df['LEV_Close'].iloc[-1]

    # [3] AI 예측 모델 학습 (타겟: 0.5% 이상 상승)
    TARGET_THRESHOLD = 0.005 
    df['Target_Next_Day'] = np.where(df['LEV_Return'].shift(-1) >= TARGET_THRESHOLD, 1, 0)
    
    df_clean = df.dropna() 
    X = df_clean[features]
    y = df_clean['Target_Next_Day']

    split = int(len(df_clean) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=5, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # 승률 계산
    accuracy = accuracy_score(y_test, model.predict(X_test))
    
    # [4] 정밀 확률 및 예상 주가 변동 범위 계산 (v8.0 핵심)
    probs = model.predict_proba(latest_features.values.reshape(1, -1))[0]
    up_prob = probs[1] * 100
    down_prob = probs[0] * 100
    
    daily_std = df_clean['LEV_Return'].tail(20).std()
    expected_plus_price = int(current_price * (1 + daily_std))
    expected_minus_price = int(current_price * (1 - daily_std))
    expected_range_pct = daily_std * 100

    # [5] 매도 위험 점수 계산
    sell_risk_score = 0
    if latest_features['LEV_RSI'] > 75: sell_risk_score += 40
    if latest_features['Sector_Heat'] > 70: sell_risk_score += 20
    if latest_vol > df_clean['LEV_Vol_5d'].tail(20).mean() * 1.3: sell_risk_score += 20
    if latest_features['VIX_Close'] > 22: sell_risk_score += 20

    # [6] 자산 배분 및 기계적 스탑로스 계산
    vol_drop_rate = latest_vol * 1.5
    stop_loss_price = int(recent_high * (1 - vol_drop_rate))
    
    target_stock_weight = max(0, 100 - sell_risk_score)
    if sell_risk_score >= 80: target_stock_weight = 0
    elif sell_risk_score >= 60: target_stock_weight = min(target_stock_weight, 30)
    target_cash_weight = 100 - target_stock_weight

    # [7] 메시지 포매팅
    prediction_result = "상승 돌파 📈" if up_prob > down_prob else "하락/조정 경보 📉"
    conf_level = max(up_prob, down_prob)
    
    action_guide = ""
    if sell_risk_score >= 80: action_guide = "🚨 [전량 매도] 시장이 광기에 달했습니다. 장 마감 전 전량 현금화하세요."
    elif sell_risk_score >= 60: action_guide = "⚠️ [비중 축소] 초과열 상태. 장 마감 전 목표 비중에 맞춰 절반 이상 매도하세요."
    else: action_guide = "✅ [추세 홀딩] 종가 기준 홀딩 유지. 스탑로스만 체크하세요."

    final_message = f"""
🤖 [레버리지 정밀 통제소 v8.0]

🎯 내일 방향성 예측
* 메인 시나리오: {prediction_result}
* 통계적 확신도: {conf_level:.2f}% (상승 {up_prob:.1f}% vs 하락 {down_prob:.1f}%)

📊 예상 주가 변동 범위 (1-Sigma)
* 예상 범위: {expected_minus_price:,}원 ~ {expected_plus_price:,}원
* 변동폭 기준: ±{expected_range_pct:.2f}% 내외

⚠️ 매도 위험 점수: {sell_risk_score} / 100
* 레버리지 RSI: {latest_features['LEV_RSI']:.2f}
* 섹터 과열도: {latest_features['Sector_Heat']:.2f}

⚖️ AI 권장 포트폴리오 비중
* 레버리지: {target_stock_weight}% | 현금: {target_cash_weight}%
👉 {action_guide}

🛡️ 기계적 탈출선 (Trailing Stop)
* 자동 매도 단가: {stop_loss_price:,}원
👉 주가가 이 선을 깨면 즉각 엑시트하세요.

------------------------
※ AI 모델 승률: {accuracy * 100:.2f}% | EMA 기준
    """
    send_telegram_message(final_message)
    print("✅ 정밀 자산 통제 리포트 전송 완료!")

except Exception as e:
    error_msg = traceback.format_exc()
    print(error_msg)
    send_telegram_message(f"🚨 [시스템 에러]\n\n{error_msg[:1000]}")
