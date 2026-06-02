import traceback
import requests
import os
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
        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={'chat_id': chat_id, 'text': text}
            )
        except Exception as e:
            print("텔레그램 전송 실패:", e)

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1 / period, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - (100 / (1 + (gain / loss)))

try:
    print("🚀 v16.1 US AI·반도체 올인 전략 완전체 모델 가동...")

    # ==========================================
    # [1] 미국 매크로 데이터 수집 (시장의 방향성 감지)
    # ==========================================
    start_date = '2022-01-01'
    
    macro_tickers = {
        'Nasdaq': '^IXIC', 
        'SMH': 'SMH',      
        'VIX': '^VIX',     
        'TNX': '^TNX'      
    }
    
    macro_data = {}
    for name, t in macro_tickers.items():
        df_raw = yf.download(t, start=start_date, progress=False)
        if not df_raw.empty:
            series = df_raw['Close'].squeeze()
            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)
            macro_data[name] = series
            
    macro_df = pd.DataFrame(macro_data)
    macro_df = macro_df.ffill().dropna()
    
    if macro_df.empty:
        raise ValueError("매크로 데이터를 정상적으로 합치지 못했습니다.")

    macro_df['Nasdaq_Ret'] = macro_df['Nasdaq'].pct_change()
    macro_df['SMH_Ret'] = macro_df['SMH'].pct_change()

    # ==========================================
    # [2] 타겟 종목 리스트 (야수 올인 시나리오)
    # ==========================================
    targets = ['XOVR', 'SOXL', 'NVDL', 'DXYZ', 'TECL']
    
    final_report = "🤖 [US AI·반도체 올인 방향성 통제소 v16.1]\n"
    final_report += "=" * 40 + "\n"

    for ticker in targets:
        print(f"진행 중: {ticker} 분석...")
        tgt_data = yf.download(ticker, start=start_date, progress=False)
        
        if tgt_data.empty:
            final_report += f"⚠️ {ticker}: 현재 데이터를 불러올 수 없습니다.\n"
            final_report += "-" * 40 + "\n"
            continue
            
        df = pd.DataFrame({
            'High': tgt_data['High'].squeeze(),
            'Low': tgt_data['Low'].squeeze(),
            'Close': tgt_data['Close'].squeeze()
        })
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        df = df.join(macro_df, how='left').ffill().dropna()
        
        # ==========================================
        # [3] 피처 엔지니어링 (기술적 지표 계산)
        # ==========================================
        df['RSI'] = calculate_rsi(df['Close'])
        
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        df['MACD_Hist'] = macd_line - macd_line.ewm(span=9, adjust=False).mean()
        
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['Price_to_EMA20'] = (df['Close'] / df['EMA_20']) - 1
        
        df['ATR'] = pd.concat([
            df['High'] - df['Low'],
            np.abs(df['High'] - df['Close'].shift()),
            np.abs(df['Low'] - df['Close'].shift())
        ], axis=1).max(axis=1).rolling(14).mean()
        
        df['Target'] = np.where(df['Close'].pct_change().shift(-1) > 0.005, 1, 0)
        df_clean = df.dropna()
        
        if df_clean.empty:
            continue

        current_price = df_clean['Close'].iloc[-1]
        is_uptrend = current_price > df_clean['EMA_20'].iloc[-1]
        latest_atr = df_clean['ATR'].iloc[-1]
        latest_rsi = df_clean['RSI'].iloc[-1]
        
        # ATR 변동성 기반 스탑로스 계산 (최대 -10% 하드캡 보호)
        rsi_risk = max(0, latest_rsi - 68)
        atr_multiplier = max(1.0, 3.0 - (rsi_risk / 10))
        recent_high = df_clean['Close'].tail(5).max()
        stop_loss = recent_high - (latest_atr * atr_multiplier)
        stop_loss = max(stop_loss, recent_high * 0.90) 
        
        # 상장 초기 종목(DXYZ, XOVR 등) 데이터 부족 시 예외 처리
        if len(df_clean) < 100:
            final_report += f"🔍 {ticker} (상장 초기 데이터 부족으로 기술적 지표만 출력)\n"
            final_report += f"  * 현재가: ${current_price:.2f} | 매도 Limit: ${stop_loss:.2f}\n"
            final_report += f"  * RSI: {latest_rsi:.1f} | 추세: {'🟢 상승' if is_uptrend else '🔴 하락'}\n"
            final_report += "-" * 40 + "\n"
            continue
            
        # ==========================================
        # [4] ML 모델 학습 및 예측
        # ==========================================
        features = ['Nasdaq_Ret', 'SMH_Ret', 'VIX', 'TNX', 'RSI', 'MACD_Hist', 'Price_to_EMA20']
        
        X = df_clean[features].iloc[:-1]
        y = df_clean['Target'].iloc[:-1]
        split = int(len(X) * 0.8)
        
        model = VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(n_estimators=300, random_state=42, max_depth=5, class_weight='balanced')),
            ('gb', GradientBoostingClassifier(n_estimators=300, random_state=42, max_depth=5))
        ], voting='soft')
        
        model.fit(X.iloc[:split], y.iloc[:split])
        accuracy = accuracy_score(y.iloc[split:], model.predict(X.iloc[split:]))
        
        latest = df_clean[features].iloc[-1]
        probs = model.predict_proba(latest.values.reshape(1, -1))[0]
        up_prob, down_prob = probs[1] * 100, probs[0] * 100
        
        if up_prob >= 65:
            direction = "🟢 강한 상승"
        elif up_prob >= 50:
            direction = "🟡 약한 상승"
        elif down_prob >= 65:
            direction = "🔴 강한 하락"
        else:
            direction = "🟠 약한 하락"
            
        final_report += f"📌 {ticker} 예측 결과\n"
        final_report += f"  * {direction} (상승 {up_prob:.1f}% / 하락 {down_prob:.1f}%)\n"
        final_report += f"  * 현재가: ${current_price:.2f} | 매도 Limit: ${stop_loss:.2f}\n"
        final_report += f"  * 상태: 신뢰도 {accuracy*100:.1f}% | RSI {latest_rsi:.1f}\n"
        final_report += "-" * 40 + "\n"

    print("\n" + final_report)
    send_telegram_message(final_report)
    print("✅ v16.1 텔레그램 리포트 전송 완료")

except Exception as e:
    error_msg = f"🚨 에러 발생:\n{traceback.format_exc()[:500]}"
    print(error_msg)
    send_telegram_message(error_msg)
