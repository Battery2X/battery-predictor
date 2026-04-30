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
    print("🚀 v15.0 방향성 집중 모델 가동...")

    # ==========================================
    # [1] 데이터 수집 (2022년부터 — 하락장 사이클 포함)
    # ==========================================
    start_date = '2022-01-01'

    tickers = {
        'TIGER 레버리지': '412570',
        '삼성SDI':        '006400',
        'LG엔솔':         '373220',
        'POSCO홀딩스':    '005490'
    }
    raw_data = {name: fdr.DataReader(t, start_date) for name, t in tickers.items()}

    usdkrw = fdr.DataReader('USD/KRW', start_date)['Close'].rename('USD_KRW')

    # 미국장은 한국장보다 늦게 끝나므로 전날 종가 사용 (shift 정상)
    tsla = yf.download('TSLA', start=start_date, progress=False)['Close'].squeeze().shift(1).rename('TSLA_Close')
    vix  = yf.download('^VIX', start=start_date, progress=False)['Close'].squeeze().shift(1).rename('VIX_Close')

    df = pd.concat([
        raw_data['TIGER 레버리지']['Close'].rename('LEV_Close'),
        raw_data['삼성SDI']['Close'].rename('SDI_Close'),
        raw_data['LG엔솔']['Close'].rename('LG_Close'),
        raw_data['POSCO홀딩스']['Close'].rename('POSCO_Close'),
        usdkrw, tsla, vix
    ], axis=1).ffill().dropna()

    # ==========================================
    # [2] 피처 엔지니어링 (방향 예측 핵심 7개)
    # ==========================================

    # 환율 변화 (외국인 수급 proxy — 원화 강세면 외국인 유입)
    df['USD_KRW_Return'] = df['USD_KRW'].pct_change()

    # 미국 전날 밤 분위기
    df['TSLA_Return'] = df['TSLA_Close'].pct_change()

    # RSI — 단기 과매수/과매도
    df['LEV_RSI']     = calculate_rsi(df['LEV_Close'])
    df['Sector_Heat'] = (
        calculate_rsi(df['SDI_Close']) +
        calculate_rsi(df['LG_Close'])  +
        calculate_rsi(df['POSCO_Close'])
    ) / 3

    # MACD 히스토그램 — 모멘텀 방향
    exp1 = df['LEV_Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['LEV_Close'].ewm(span=26, adjust=False).mean()
    macd_line   = exp1 - exp2
    df['MACD_Hist'] = macd_line - macd_line.ewm(span=9, adjust=False).mean()

    # 20일선 이격도 — 추세 위치
    df['EMA_20']         = df['LEV_Close'].ewm(span=20, adjust=False).mean()
    df['Price_to_EMA20'] = (df['LEV_Close'] / df['EMA_20']) - 1

    # ATR — 스탑로스 계산용 (피처 아님)
    lev_high  = raw_data['TIGER 레버리지']['High']
    lev_low   = raw_data['TIGER 레버리지']['Low']
    lev_close = raw_data['TIGER 레버리지']['Close']
    df['ATR'] = pd.concat([
        lev_high - lev_low,
        np.abs(lev_high - lev_close.shift()),
        np.abs(lev_low  - lev_close.shift())
    ], axis=1).max(axis=1).rolling(14).mean()

    # ==========================================
    # [3] ML 모델 학습
    # 목표: 내일 +0.5% 이상 상승 여부 (방향성)
    # ==========================================
    features = [
        'TSLA_Return',     # 미국 전날 밤
        'VIX_Close',       # 공포 지수
        'USD_KRW_Return',  # 환율
        'LEV_RSI',         # 과매수/과매도
        'Sector_Heat',     # 섹터 온도
        'MACD_Hist',       # 모멘텀
        'Price_to_EMA20'   # 추세 위치
    ]

    df['Target'] = np.where(
        df['LEV_Close'].pct_change().shift(-1) > 0.005, 1, 0
    )

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

    accuracy           = accuracy_score(y.iloc[split:], model.predict(X.iloc[split:]))
    latest             = df_clean[features].iloc[-1]
    probs              = model.predict_proba(latest.values.reshape(1, -1))[0]
    up_prob, down_prob = probs[1] * 100, probs[0] * 100

    current_price = df_clean['LEV_Close'].iloc[-1]
    is_uptrend    = current_price > df_clean['EMA_20'].iloc[-1]
    latest_atr    = df_clean['ATR'].iloc[-1]

    # ==========================================
    # [4] 방향 판단 + 행동 지침
    # ==========================================
    if up_prob >= 65:
        direction     = "🟢 강한 상승 예측"
        action        = "✅ 매수 or 홀딩 유지"
        weight_guide  = "레버리지 비중 유지 또는 확대"
    elif up_prob >= 50:
        direction     = "🟡 약한 상승 예측"
        action        = "✅ 홀딩 유지 (관망)"
        weight_guide  = "현 비중 유지"
    elif down_prob >= 65:
        direction     = "🔴 강한 하락 예측"
        action        = "🚨 비중 절반 이상 축소 고려"
        weight_guide  = "레버리지 비중 축소"
    else:
        direction     = "🟠 약한 하락 예측"
        action        = "⚠️ 신규 매수 자제, 홀딩"
        weight_guide  = "현 비중 유지, 추가 매수 금지"

    # 신뢰도 낮을 때 경고
    confidence_msg = ""
    if accuracy < 0.55:
        confidence_msg = "\n⚠️ 모델 신뢰도 낮음 — 신호 강도 낮춰서 해석하세요."

    # ==========================================
    # [5] ATR 스탑로스
    # ==========================================
    # RSI 기반 간이 위험점수 (스탑 계산용으로만 사용)
    rsi_risk = max(0, latest['LEV_RSI'] - 68)
    atr_multiplier = max(1.0, 3.0 - (rsi_risk / 10))
    recent_high    = df_clean['LEV_Close'].tail(5).max()
    stop_loss      = int(recent_high - (latest_atr * atr_multiplier))
    stop_loss      = max(stop_loss, int(recent_high * 0.90))  # 최대 -10% 하드캡

    # ==========================================
    # [6] 리포트 출력
    # ==========================================
    trend_msg = "🟢 20일선 위 (상승 추세)" if is_uptrend else "🔴 20일선 아래 (하락 추세)"

    final_report = f"""
🤖 [레버리지 방향성 통제소 v15.0]

📊 내일 방향 예측
* {direction}
* 상승 확률: {up_prob:.1f}% | 하락 확률: {down_prob:.1f}%
* 모델 신뢰도: {accuracy * 100:.1f}%{confidence_msg}

📈 시장 상태
* 추세: {trend_msg}
* RSI: {latest['LEV_RSI']:.1f} | 섹터 온도: {latest['Sector_Heat']:.1f}
* VIX: {latest['VIX_Close']:.1f} | 전날 TSLA: {latest['TSLA_Return']*100:.1f}%

💡 행동 지침
* {action}
* {weight_guide}

🛡️ 스탑로스 (항상 유지)
* 현재가: {int(current_price):,}원
* 매도 단가: {stop_loss:,}원 ({((stop_loss/current_price)-1)*100:.1f}%)
    """.strip()

    print(final_report)
    send_telegram_message(final_report)
    print("\n✅ v15.0 리포트 전송 완료")

except Exception as e:
    error_msg = f"🚨 에러 발생:\n{traceback.format_exc()[:500]}"
    print(error_msg)
    send_telegram_message(error_msg)
