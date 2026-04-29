import os
import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
import warnings
warnings.filterwarnings('ignore')

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
USE_TELEGRAM     = True

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_macd(series, short=12, long=26, signal=9):
    exp1 = series.ewm(span=short, adjust=False).mean()
    exp2 = series.ewm(span=long, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger(series, period=20):
    ma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    position = (series - lower) / (upper - lower + 1e-10)
    return position.clip(0, 1)

def send_telegram(token, chat_id, message):
    import requests
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            print("✅ Telegram 전송 성공!")
        else:
            print(f"⚠️ Telegram 전송 실패: {resp.text}")
    except Exception as e:
        print(f"⚠️ Telegram 오류: {e}")

print("📡 데이터를 불러오는 중...")
start_date = '2022-01-01'

etf_raw = fdr.DataReader('305540', start_date)
etf     = etf_raw['Close'].rename('ETF_Close')

try:
    etf_vol    = etf_raw['Volume'].rename('ETF_Volume')
    has_volume = True
except KeyError:
    has_volume = False

sdi    = fdr.DataReader('006400', start_date)['Close'].rename('SDI_Close')
lg     = fdr.DataReader('373220', start_date)['Close'].rename('LG_Close')
posco  = fdr.DataReader('005490', start_date)['Close'].rename('POSCO_Close')
usdkrw = fdr.DataReader('USD/KRW', start_date)['Close'].rename('USD_KRW')

tsla = yf.download('TSLA', start=start_date, progress=False)['Close'].squeeze().shift(1).rename('TSLA_Close')
lit  = yf.download('LIT',  start=start_date, progress=False)['Close'].squeeze().shift(1).rename('LIT_Close')

try:
    vix = yf.download('^VIX', start=start_date, progress=False)['Close'].squeeze().shift(1).rename('VIX')
    has_vix = True
    print("✅ VIX 로드 성공")
except:
    has_vix = False

cols = [etf, sdi, lg, posco, usdkrw, tsla, lit]
if has_volume: cols.append(etf_vol)
if has_vix:   cols.append(vix)

df = pd.concat(cols, axis=1).ffill().dropna()
print(f"✅ 데이터 로드 완료: {len(df)}거래일")

for col in ['ETF_Close','SDI_Close','LG_Close','POSCO_Close','USD_KRW','TSLA_Close','LIT_Close']:
    new_col = col.replace('Close','Return').replace('USD_KRW','FX_Return')
    df[new_col] = df[col].pct_change()

macd_line, signal_line, histogram = calculate_macd(df['ETF_Close'])
df['ETF_MACD_Line'] = macd_line
df['ETF_MACD_Sig']  = signal_line
df['ETF_MACD_Hist'] = histogram
df['ETF_RSI']         = calculate_rsi(df['ETF_Close'], period=14)
df['ETF_RSI_Short']   = calculate_rsi(df['ETF_Close'], period=7)
df['ETF_BB_Position'] = calculate_bollinger(df['ETF_Close'])
df['ETF_MA5_Ratio']   = df['ETF_Close'] / df['ETF_Close'].rolling(5).mean()
df['ETF_MA20_Ratio']  = df['ETF_Close'] / df['ETF_Close'].rolling(20).mean()
df['ETF_Vol_5d']      = df['ETF_Return'].rolling(5).std()
df['ETF_Vol_20d']     = df['ETF_Return'].rolling(20).std()
df['ETF_Return_Lag1']  = df['ETF_Return'].shift(1)
df['ETF_Return_Lag2']  = df['ETF_Return'].shift(2)
df['TSLA_Return_Lag1'] = df['TSLA_Return'].shift(1)

if has_volume:
    df['ETF_Vol_Ratio'] = df['ETF_Volume'] / df['ETF_Volume'].rolling(20).mean()
if has_vix and 'VIX' in df.columns:
    df['VIX_Change'] = df['VIX'].pct_change()
    df['VIX_Level']  = df['VIX']

TARGET_THRESHOLD = 0.003
df['Target_Next_Day'] = np.where(df['ETF_Return'].shift(-1) >= TARGET_THRESHOLD, 1, 0)
df = df.dropna()

features = [
    'ETF_Return','SDI_Return','LG_Return','POSCO_Return',
    'FX_Return','TSLA_Return','LIT_Return',
    'ETF_RSI','ETF_RSI_Short',
    'ETF_MACD_Line','ETF_MACD_Sig','ETF_MACD_Hist',
    'ETF_BB_Position','ETF_MA5_Ratio','ETF_MA20_Ratio',
    'ETF_Vol_5d','ETF_Vol_20d',
    'ETF_Return_Lag1','ETF_Return_Lag2','TSLA_Return_Lag1',
]
if has_volume: features.append('ETF_Vol_Ratio')
if has_vix and 'VIX' in df.columns: features += ['VIX_Change','VIX_Level']

X = df[features]
y = df['Target_Next_Day']

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

rf = RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_leaf=5,
                             class_weight='balanced', random_state=42, n_jobs=-1)
gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, random_state=42)
try:
    from lightgbm import LGBMClassifier
    lgbm = LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                           class_weight='balanced', random_state=42, verbose=-1)
    ensemble = VotingClassifier(estimators=[('rf',rf),('gb',gb),('lgbm',lgbm)], voting='soft')
    print("⚡ RF + GBM + LGBM 3중 앙상블")
except ImportError:
    ensemble = VotingClassifier(estimators=[('rf',rf),('gb',gb)], voting='soft')
    print("⚡ RF + GBM 2중 앙상블")

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

proba_test = ensemble.predict_proba(X_test)[:, 1]
best_thresh, best_precision = 0.5, 0
for thresh in np.arange(0.40, 0.75, 0.05):
    pred_t = (proba_test >= thresh).astype(int)
    if pred_t.sum() < 5: continue
    prec = precision_score(y_test, pred_t, zero_division=0)
    if prec > best_precision:
        best_precision = prec
        best_thresh = thresh

latest_data  = X.iloc[-1].values.reshape(1, -1)
prob_up      = ensemble.predict_proba(latest_data)[0][1]
signal       = "📈 상승 예상" if prob_up >= best_thresh else "📉 하락/횡보 예상"
action       = "✅ 매수 신호" if prob_up >= best_thresh else "⛔ 관망/홀드"
today_str    = df.index[-1].strftime('%Y-%m-%d')
etf_latest   = df['ETF_Close'].iloc[-1]
rsi_latest   = df['ETF_RSI'].iloc[-1]
macd_hist_latest = df['ETF_MACD_Hist'].iloc[-1]
bb_pos_latest    = df['ETF_BB_Position'].iloc[-1]

print(f"""
╔══════════════════════════════════════════╗
║  TIGER 2차전지 TOP10 실전 예측 v4.0     ║
╠══════════════════════════════════════════╣
  기준일     : {today_str}
  현재 NAV   : {etf_latest:,.0f}원
  RSI(14)    : {rsi_latest:.1f}
  MACD Hist  : {macd_hist_latest:+.2f}
  BB 위치    : {bb_pos_latest:.2f}
──────────────────────────────────────────
  상승 확률  : {prob_up*100:.1f}%
  임계값     : {best_thresh:.2f}
  예측 결과  : {signal}
  투자 판단  : {action}
╚══════════════════════════════════════════╝
""")

if USE_TELEGRAM:
    msg = f"""
*📊 TIGER 2차전지 TOP10 예측 ({today_str})*

현재 NAV: {etf_latest:,.0f}원
RSI(14): {rsi_latest:.1f} | MACD Hist: {macd_hist_latest:+.2f}
BB 위치: {bb_pos_latest:.2f}

*상승 확률: {prob_up*100:.1f}%*
*→ {signal} ({action})*

_(임계값 {best_thresh:.2f} 기준 | 모델 v4.0)_
"""
    send_telegram(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, msg.strip())
