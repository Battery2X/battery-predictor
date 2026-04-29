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

# 텔레그램 전송 함수
def send_telegram_message(text):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        print("❌ 환경변수에 TELEGRAM_TOKEN 또는 TELEGRAM_CHAT_ID가 없습니다.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': text}
    requests.post(url, data=payload)

# 기술적 지표 계산 함수
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
# 메인 로직 시작 (에러 감지기 작동)
# ==========================================
try:
    print("🚀 예측 모델 실행 시작...")
    start_date = '2024-01-01'

    # 1. 데이터 수집
    etf = fdr.DataReader('305540', start_date)['Close'].rename('ETF_Close')
    sdi = fdr.DataReader('006400', start_date)['Close'].rename('SDI_Close')
    lg = fdr.DataReader('373220', start_date)['Close'].rename('LG_Close')
    posco = fdr.DataReader('005490', start_date)['Close'].rename('POSCO_Close')
    usdkrw = fdr.DataReader('USD/KRW', start_date)['Close'].rename('USD_KRW')

    # 미국 데이터 (GitHub Actions의 UTC 시차 방어를 위해 날짜 인덱스를 KST에 맞게 조정)
    tsla = yf.download('TSLA', start=start_date)['Close'].squeeze().shift(1).rename('TSLA_Close')
    lit = yf.download('LIT', start=start_date)['Close'].squeeze().shift(1).rename('LIT_Close')

    # 데이터 병합 (결측치 채우기를 더 안전하게 처리)
    df = pd.concat([etf, sdi, lg, posco, usdkrw, tsla, lit], axis=1)
    df = df.fillna(method='ffill').dropna()

    # 데이터가 너무 적게 남았을 경우 강제 에러 발생
    if len(df) < 50:
        raise ValueError(f"데이터가 충분하지 않습니다. 현재 데이터 길이: {len(df)}")

    # 2. 피처 엔지니어링
    for col in df.columns:
        if 'Close' in col or 'KRW' in col:
            new_col = col.replace('Close', 'Return').replace('USD_KRW', 'FX_Return')
            df[new_col] = df[col].pct_change()

    df['ETF_RSI'] = calculate_rsi(df['ETF_Close'], period=14)
    df['ETF_MACD'] = calculate_macd(df['ETF_Close'])
    df['ETF_MA5_Ratio'] = df['ETF_Close'] / df['ETF_Close'].rolling(window=5).mean()
    df['ETF_Vol_5d'] = df['ETF_Return'].rolling(window=5).std()

    # 3. 정답지(Label) 생성
    TARGET_THRESHOLD = 0.003
    df['Target_Next_Day'] = np.where(df['ETF_Return'].shift(-1) >= TARGET_THRESHOLD, 1, 0)
    df = df.dropna()

    features = ['ETF_Return', 'SDI_Return', 'LG_Return', 'POSCO_Return', 
                'FX_Return', 'TSLA_Return', 'LIT_Return', 
                'ETF_RSI', 'ETF_MACD', 'ETF_MA5_Ratio', 'ETF_Vol_5d']

    X = df[features]
    y = df['Target_Next_Day']

    # 4. 모델 학습
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=5, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 5. 실전 예측 및 텔레그램 메시지 생성
    latest_data = X.iloc[-1].values.reshape(1, -1)
    prediction = model.predict(latest_data)
    prob = model.predict_proba(latest_data)[0]

    # 메시지 생성 로직 복구 (에러 원인 해결)
    if prediction[0] == 1:
        result_msg = f"📈 상승 예상 (+0.3% 돌파 확률: {prob[1]*100:.1f}%)"
    else:
        result_msg = f"📉 하락 또는 횡보 예상 (확률: {prob[0]*100:.1f}%)"

    final_message = f"🤖 [TIGER 2차전지 TOP10 AI 예측]\n\n{result_msg}\n\n* 모델 승률: {accuracy * 100:.2f}%\n* 분석 완료 시점 기준"

    # 정상 완료 시 텔레그램 전송
    send_telegram_message(final_message)
    print("✅ 실행 및 텔레그램 전송 완료!")

except Exception as e:
    # 에러 발생 시 상세 로그 전송
    error_msg = traceback.format_exc()
    print("❌ 치명적인 에러 발생:\n", error_msg)
    send_telegram_message(f"🚨 [긴급] 예측 봇 실행 실패!\n\n{error_msg[:1000]}")
