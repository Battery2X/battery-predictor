# ... (기존 import 및 함수 동일) ...

def calculate_indicators(df):
    # RSI
    df['ETF_RSI'] = calculate_rsi(df['ETF_Close'], period=14)
    # MACD
    df['ETF_MACD'] = calculate_macd(df['ETF_Close'])
    # 거래량 변화율 (추가: 거래량 없는 상승 필터링)
    df['Vol_Change'] = df['ETF_Volume'].pct_change()
    # 볼린저 밴드 상단 (추가: 고점 돌파 확인)
    df['BB_Upper'] = df['ETF_Close'].rolling(20).mean() + (df['ETF_Close'].rolling(20).std() * 2)
    return df

try:
    # 1. 데이터 수집 (Volume 추가)
    etf_raw = fdr.DataReader('305540', start_date)
    etf = etf_raw['Close'].rename('ETF_Close')
    etf_vol = etf_raw['Volume'].rename('ETF_Volume') # 거래량 추가
    
    # 미국 공포지수 VIX 추가 (yfinance)
    vix = yf.download('^VIX', start=start_date)['Close'].squeeze().shift(1).rename('VIX_Close')

    # 데이터 병합
    df = pd.concat([etf, etf_vol, sdi, lg, posco, usdkrw, tsla, lit, vix], axis=1)
    df = df.ffill().dropna()

    # 2. 피처 엔지니어링 및 위험 점수 산출
    df = calculate_indicators(df)
    
    # [매도 위험 점수 로직]
    # 1. RSI가 70 이상인가? (과매수)
    # 2. 주가는 올랐는데 거래량은 줄었는가? (상승 동력 상실)
    # 3. MACD가 시그널선을 하향 돌파하려 하는가?
    # 4. 볼린저 밴드 상단에 닿았는가?
    
    latest = df.iloc[-1]
    sell_risk_score = 0
    if latest['ETF_RSI'] > 70: sell_risk_score += 30
    if latest['ETF_Close'] > df.iloc[-2]['ETF_Close'] and latest['Vol_Change'] < 0: sell_risk_score += 30
    if latest['ETF_Close'] >= latest['BB_Upper']: sell_risk_score += 20
    if latest['VIX_Close'] > 20: sell_risk_score += 20

    # 3. 모델 학습 및 예측 (기존 로직 동일)
    # ...
    
    # 4. 텔레그램 메시지 고도화
    risk_comment = ""
    if sell_risk_score >= 70: risk_comment = "🚨 [강력 매도 권고] 지표가 과열되었습니다. 수익 실현을 고려하세요!"
    elif sell_risk_score >= 40: risk_comment = "⚠️ [주의] 주가 상승세가 둔화되고 있습니다. 분할 매도를 추천합니다."
    else: risk_comment = "✅ [보유 가능] 아직 과열 신호가 없으며 추세가 유효합니다."

    final_message = f"""
🤖 [TIGER 2차전지 TOP10 AI 리포트]

🎯 내일 방향: {result_msg}
📊 매도 위험 점수: {sell_risk_score}점 / 100
💡 전략 가이드: {risk_comment}

* RSI: {latest['ETF_RSI']:.1f} (70이상 과매수)
* 모델 승률: {accuracy * 100:.2f}%
    """
    send_telegram_message(final_message)

except Exception as e:
    # ... 에러 전송 로직 ...
