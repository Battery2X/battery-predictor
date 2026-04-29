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
    print("🚀 v12.0 기관급 손익비 최적화 모델 가동...")
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

    # [2] 피처 엔지니어링 (🚨 핵심: 머신러닝 맞춤형 등락률 변환)
    df['LEV_Return'] = df['LEV_Close'].pct_change()
    df['TSLA_Return'] = df['TSLA_Close'].pct_change() # 가격 대신 등락률 사용
    df['USD_KRW_Return'] = df['USD_KRW'].pct_change() # 가격 대신 등락률 사용
    
    df['LEV_RSI'] = calculate_rsi(df['LEV_Close'])
    df['SDI_RSI'] = calculate_rsi(df['SDI_Close'])
    df['LG_RSI'] = calculate_rsi(df['LG_Close'])
    df['POSCO_RSI'] = calculate_rsi(df['POSCO_Close'])
    df['Sector_Heat'] = (df['SDI_RSI'] + df['LG_RSI'] + df['POSCO_RSI']) / 3
    
    exp1 = df['LEV_Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['LEV_Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Hist'] = (exp1 - exp2) - (exp1 - exp2).ewm(span=9, adjust=False).mean()

    # ATR 계산
    high_low = raw_data['TIGER 레버리지']['High'] - raw_data['TIGER 레버리지']['Low']
    high_close = np.abs(raw_data['TIGER 레버리지']['High'] - raw_data['TIGER 레버리지']['Close'].shift())
    low_close = np.abs(raw_data['TIGER 레버리지']['Low'] - raw_data['TIGER 레버리지']['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    # 🚨 학습 피처 라인업 변경 (절대 가격 아웃, 모멘텀 인)
    features = ['LEV_Return', 'SDI_RSI', 'Sector_Heat', 'LEV_RSI', 'VIX_Close', 'TSLA_Return', 'USD_KRW_Return', 'MACD_Hist']
    
    df_clean = df.dropna()
    latest = df_clean[features].iloc[-1]
    latest_atr = df_clean['ATR'].iloc[-1]
    current_price = df_clean['LEV_Close'].iloc[-1]

    # [3] ML 모델 학습 및 검증
    df_clean['Target'] = np.where(df_clean['LEV_Return'].shift(-1) >= 0.005, 1, 0)
    # 마지막 날(내일의 정답이 없는 날)은 학습에서 제외
    X_train = df_clean[features].iloc[:-1]
    y_train = df_clean['Target'].iloc[:-1]

    split = int(len(X_train) * 0.8)
    model = VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(n_estimators=300, random_state=42, max_depth=5, class_weight='balanced')),
        ('gb', GradientBoostingClassifier(n_estimators=300, random_state=42, max_depth=5))
    ], voting='soft')
    
    model.fit(X_train.iloc[:split], y_train.iloc[:split])
    
    accuracy = accuracy_score(y_train.iloc[split:], model.predict(X_train.iloc[split:]))
    probs = model.predict_proba(latest.values.reshape(1, -1))[0]
    up_prob, down_prob = probs[1] * 100, probs[0] * 100

    # [4] 위험 점수 계산 (지수형 폭발 로직 유지)
    sell_risk_score = 0
    if latest['LEV_RSI'] > 70: sell_risk_score += (latest['LEV_RSI'] - 70) ** 1.8
    if latest['Sector_Heat'] > 70: sell_risk_score += (latest['Sector_Heat'] - 70) ** 1.5
    if latest['MACD_Hist'] < 0: sell_risk_score += 20
    sell_risk_score = min(int(sell_risk_score), 100) 

    # [5] 🚨 [핵심 업데이트] 동적 ATR 곱셈기 (Dynamic Multiplier)
    # 60점 절벽을 없애고, 위험 점수가 높을수록 곱셈기가 스무스하게 1.0배까지 낮아짐
    # 점수가 0이면 3.0배(넉넉함), 점수가 100이면 1.0배(매우 타이트함)
    atr_multiplier = max(1.0, 3.0 - (sell_risk_score / 50))
    recent_high = raw_data['TIGER 레버리지']['Close'].tail(5).max()
    stop_loss = int(recent_high - (latest_atr * atr_multiplier))
    
    # 손실 방지 안전장치: 아무리 넉넉해도 고점 대비 -7% 이상은 못 떨어지게 캡(Cap) 씌움
    min_allowable_stop = int(recent_high * 0.93) 
    stop_loss = max(stop_loss, min_allowable_stop)

    # [6] 켈리 비중 산출 및 UI 논리 제어
    if accuracy < 0.5:
        target_weight = 0
        kelly_msg = "⚠️ AI 승률 50% 미만. 확률적 엣지가 소멸하여 배팅을 전면 중단합니다."
        stop_loss_ui = f"🚫 [즉시 전량 매도] 비중이 0%이므로 스탑로스가 불필요합니다. 현재가({current_price:,}원) 부근 시장가 매도."
    else:
        target_weight = max(0, 100 - sell_risk_score)
        kelly_msg = f"✅ 통계적 우위 구간. 포트폴리오의 {target_weight}% 비중을 투입합니다."
        stop_loss_ui = f"🛡️ [자동 매도 단가] {stop_loss:,}원 (고점 대비 ATR {atr_multiplier:.1f}배수 방어선)"

    target_cash = 100 - target_weight

    # [7] 리포트 포매팅
    res_msg = "상승 돌파 📈" if up_prob > down_prob else "조정/하락 경보 📉"
    
    final_report = f"""
🤖 [레버리지 퀀트 통제소 v12.0 Final]

🎯 AI 방향성 예측 (등락률 정규화 완료)
* 메인 시나리오: {res_msg}
* 예측 확률: 상승 {up_prob:.1f}% / 하락 {down_prob:.1f}%
* AI 검증 승률: {accuracy*100:.1f}%

⚠️ 시장 과열 및 위험도 측정
* 최종 위험 점수: {sell_risk_score} / 100점
* 레버리지 RSI: {latest['LEV_RSI']:.1f}
* 빅3 평균 RSI: {latest['Sector_Heat']:.1f}

⚖️ 최적 포트폴리오 비중 (Kelly)
* 2차전지 레버리지: {target_weight}%
* 원화 예수금: {target_cash}%
👉 {kelly_msg}

{stop_loss_ui}
    """
    send_telegram_message(final_report)
    print("✅ v12.0 퍼포먼스 최적화 리포트 전송 완료")

except Exception as e:
    send_telegram_message(f"🚨 에러: {traceback.format_exc()[:300]}")
