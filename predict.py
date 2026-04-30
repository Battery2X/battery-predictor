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
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={'chat_id': chat_id, 'text': text}
        )


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1 / period, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - (100 / (1 + (gain / loss)))


try:
    print("🚀 v14.0 하이브리드 추세-앙상블 모델 가동...")

    # ==========================================
    # [1] 데이터 수집 (2022년부터 — 하락장 사이클 포함)
    # ==========================================
    start_date = '2022-01-01'

    tickers = {
        'TIGER 레버리지': '412570',
        '삼성SDI': '006400',
        'LG엔솔': '373220',
        'POSCO홀딩스': '005490'
    }
    raw_data = {name: fdr.DataReader(t, start_date) for name, t in tickers.items()}

    usdkrw = fdr.DataReader('USD/KRW', start_date)['Close'].rename('USD_KRW')
    tsla = yf.download('TSLA', start=start_date, progress=False)['Close'].squeeze().shift(1).rename('TSLA_Close')
    vix = yf.download('^VIX', start=start_date, progress=False)['Close'].squeeze().shift(1).rename('VIX_Close')

    df = pd.concat([
        raw_data['TIGER 레버리지']['Close'].rename('LEV_Close'),
        raw_data['TIGER 레버리지']['Volume'].rename('LEV_Volume'),
        raw_data['삼성SDI']['Close'].rename('SDI_Close'),
        raw_data['LG엔솔']['Close'].rename('LG_Close'),
        raw_data['POSCO홀딩스']['Close'].rename('POSCO_Close'),
        usdkrw, tsla, vix
    ], axis=1).ffill().dropna()

    # ==========================================
    # [2] 피처 엔지니어링
    # ==========================================
    df['LEV_Return'] = df['LEV_Close'].pct_change()
    df['TSLA_Return'] = df['TSLA_Close'].pct_change()
    df['USD_KRW_Return'] = df['USD_KRW'].pct_change()

    # RSI (임계값 65로 민감도 강화)
    df['LEV_RSI'] = calculate_rsi(df['LEV_Close'])
    df['Sector_Heat'] = (
        calculate_rsi(df['SDI_Close']) +
        calculate_rsi(df['LG_Close']) +
        calculate_rsi(df['POSCO_Close'])
    ) / 3

    # 거래량 비율 (20일 평균 대비)
    df['Volume_Ratio'] = df['LEV_Volume'] / df['LEV_Volume'].rolling(20).mean()

    # 20일 EMA 추세
    df['EMA_20'] = df['LEV_Close'].ewm(span=20, adjust=False).mean()
    df['Price_to_EMA20'] = (df['LEV_Close'] / df['EMA_20']) - 1

    # MACD 히스토그램
    exp1 = df['LEV_Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['LEV_Close'].ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = macd_line - signal_line

    # ATR 계산
    lev_high = raw_data['TIGER 레버리지']['High']
    lev_low = raw_data['TIGER 레버리지']['Low']
    lev_close = raw_data['TIGER 레버리지']['Close']
    high_low = lev_high - lev_low
    high_close = np.abs(lev_high - lev_close.shift())
    low_close = np.abs(lev_low - lev_close.shift())
    df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

    # ==========================================
    # [3] ML 모델 학습
    # ==========================================
    features = [
        'LEV_Return', 'Sector_Heat', 'LEV_RSI',
        'VIX_Close', 'TSLA_Return', 'USD_KRW_Return',
        'MACD_Hist', 'Price_to_EMA20', 'Volume_Ratio'
    ]

    # 타겟: 다음 날 +0.5% 이상 상승 = 1
    df['Target'] = np.where(df['LEV_Return'].shift(-1) > 0.005, 1, 0)

    df_clean = df.dropna()
    X = df_clean[features].iloc[:-1]
    y = df_clean['Target'].iloc[:-1]

    split = int(len(X) * 0.8)

    model = VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(
            n_estimators=300, random_state=42, max_depth=5, class_weight='balanced'
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=300, random_state=42, max_depth=5
        ))
    ], voting='soft')

    model.fit(X.iloc[:split], y.iloc[:split])

    accuracy = accuracy_score(y.iloc[split:], model.predict(X.iloc[split:]))
    latest = df_clean[features].iloc[-1]
    probs = model.predict_proba(latest.values.reshape(1, -1))[0]
    up_prob, down_prob = probs[1] * 100, probs[0] * 100

    latest_atr = df_clean['ATR'].iloc[-1]
    current_price = df_clean['LEV_Close'].iloc[-1]
    is_uptrend = current_price > df_clean['EMA_20'].iloc[-1]

    # ==========================================
    # [4] 위험 점수 (RSI 임계값 65로 강화)
    # ==========================================
    sell_risk_score = 0

    if latest['LEV_RSI'] > 65:
        sell_risk_score += (latest['LEV_RSI'] - 65) ** 1.8
    if latest['Sector_Heat'] > 65:
        sell_risk_score += (latest['Sector_Heat'] - 65) ** 1.5
    if latest['MACD_Hist'] < 0:
        sell_risk_score += 20
    if latest['VIX_Close'] > 22:
        sell_risk_score += 15
    if latest['Volume_Ratio'] > 2.0:
        sell_risk_score += 15  # 거래량 급증 = 과열 신호

    sell_risk_score = min(int(sell_risk_score), 100)

    # ==========================================
    # [5] 하이브리드 비중 조절 (ML 예측 방향 반영)
    # ==========================================
    base_weight = max(0, 100 - sell_risk_score)

    if accuracy < 0.5:
        if is_uptrend:
            target_weight = min(base_weight, 20)
            strategy_msg = "⚠️ AI 승률 저조. 대세 상승장이므로 최소 20% 유지."
        else:
            target_weight = 0
            strategy_msg = "🚨 하락 추세 + AI 승률 저조. 전량 현금화."
    else:
        # ML 예측 확률로 비중 ±30% 조정
        # up_prob=20% → -18%, up_prob=80% → +18%
        prob_factor = (up_prob - 50) / 50
        prob_adjustment = prob_factor * 30
        target_weight = int(max(0, min(100, base_weight + prob_adjustment)))

        direction = "상승" if up_prob >= 50 else "하락"
        strategy_msg = (
            f"✅ 엣지 확보. 기본비중 {base_weight}%에서 "
            f"AI {direction}신호({up_prob:.0f}%)로 {prob_adjustment:+.0f}% 조정 → {target_weight}%"
        )

    target_cash = 100 - target_weight

    # ==========================================
    # [6] ATR 스탑로스
    # ==========================================
    atr_multiplier = max(1.0, 3.0 - (sell_risk_score / 40))
    recent_high = raw_data['TIGER 레버리지']['Close'].tail(5).max()
    stop_loss = int(recent_high - (latest_atr * atr_multiplier))
    stop_loss = max(stop_loss, int(recent_high * 0.90))  # 최대 -10% 하드캡

    # ==========================================
    # [7] 리포트 출력
    # ==========================================
    res_msg = "상승 돌파 📈" if up_prob > down_prob else "조정/하락 경보 📉"
    trend_msg = "🟢 대세 상승장 (주가 > 20일선)" if is_uptrend else "🔴 하락/역배열 (주가 < 20일선)"

    final_report = f"""
🤖 [레버리지 하이브리드 통제소 v14.0]

📊 시장 분석
* 매크로 추세: {trend_msg}
* 현재가: {int(current_price):,}원 | 스탑로스: {stop_loss:,}원
* 내일 AI 예측: {res_msg} (상승 {up_prob:.1f}% / 하락 {down_prob:.1f}%)
* 모델 신뢰도: {accuracy * 100:.1f}%

⚠️ 시장 과열 및 위험도 (0~100)
* 붕괴 위험 점수: {sell_risk_score}점
* 레버리지 RSI: {latest['LEV_RSI']:.1f} | 빅3 평균: {latest['Sector_Heat']:.1f}
* 거래량 비율: {latest['Volume_Ratio']:.2f}배 (1.0 = 평균)

⚖️ 최적 포트폴리오 비중
* 2차전지 레버리지: {target_weight}%
* 원화 예수금(현금): {target_cash}%
👉 {strategy_msg}

🛡️ ATR 스탑로스 (상시 가동)
* 자동 매도 단가: {stop_loss:,}원
* 손실 허용폭: 현재가 대비 {((stop_loss / current_price) - 1) * 100:.1f}%
👉 잔여 물량 홀딩 시 반드시 위 가격을 마지노선으로 설정하세요.
    """.strip()

    print(final_report)
    send_telegram_message(final_report)
    print("\n✅ v14.0 리포트 전송 완료")

except Exception as e:
    error_msg = f"🚨 에러 발생:\n{traceback.format_exc()[:500]}"
    print(error_msg)
    send_telegram_message(error_msg)
