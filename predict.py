import traceback
import requests
import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# v16.6 변경사항
# ============================================================
# [종목 교체] SOXL, TECL 제거 → AVGO, MU 추가
#   - SOXL/TECL: 만성적 P/R 미달, 3배 레버리지 변동성 감쇄 가치 낮음
#   - AVGO: 모멘텀 프레임워크 1위 — AI 반도체 매출 가속, 추정치 개정 최강
#   - MU: 모멘텀 프레임워크 2위 — HBM 슈퍼사이클, 단 실적 변동성 큼
# [유지] XOVR(이벤트드리븐), NVDL(실보유), DXYZ(8월재판단)
# 기존 v16.5 개선사항(P/R필터, 신호반전감지, 이벤트캘린더) 그대로 유지
# ============================================================

MARKET_EVENTS = {
    "2026-06-18": "FOMC 금리 결정 (새벽 3:00 KST)",
    "2026-06-20": "미국 6월 CPI 발표",
    "2026-06-25": "Micron(MU) 실적 발표 예정 — 변동성 17.6% 가격책정",
    "2026-07-29": "FOMC 7월 회의",
    "2026-08-25": "NVDA Q2 실적 발표 (예정)",
    "2026-08-27": "FOMC 잭슨홀 미팅",
    "2026-09-16": "FOMC 9월 회의",
}

def check_upcoming_events():
    from datetime import datetime
    today = datetime.now().date()
    out = []
    for date_str, event in MARKET_EVENTS.items():
        ev_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        days_left = (ev_date - today).days
        if 0 <= days_left <= 2:
            prefix = "🔴 오늘" if days_left == 0 else f"⚠️ D-{days_left}"
            out.append(f"  {prefix}: {event}")
    return out

SIGNAL_HISTORY_FILE = "signal_history.json"

def load_signal_history():
    try:
        if os.path.exists(SIGNAL_HISTORY_FILE):
            with open(SIGNAL_HISTORY_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return {}

def save_signal_history(history):
    try:
        with open(SIGNAL_HISTORY_FILE, 'w') as f:
            json.dump(history, f, ensure_ascii=False)
    except:
        pass

def detect_signal_flip(ticker, today_sig, prev_history):
    if ticker not in prev_history:
        return None
    prev_sig = prev_history[ticker].get('signal', '')
    up = ['강한상승', '약한상승', '우상향']
    dn = ['강한하락', '약한하락', '하락우세']
    prev_dir = 'up' if any(s in prev_sig for s in up) else ('dn' if any(s in prev_sig for s in dn) else 'neutral')
    now_dir  = 'up' if any(s in today_sig for s in up) else ('dn' if any(s in today_sig for s in dn) else 'neutral')
    if prev_dir != now_dir and prev_dir != 'neutral' and now_dir != 'neutral':
        return f"🔄 신호 반전: {prev_sig} → {today_sig}"
    return None

def send_telegram_message(text):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={'chat_id': chat_id, 'text': text})
        except Exception as e:
            print("텔레그램 전송 실패:", e)

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + (gain / loss)))

def calculate_bollinger(series, period=20, std_mult=2.0):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return ma + std_mult*std, ma, ma - std_mult*std

def get_dynamic_threshold(close_series):
    daily_ret = close_series.pct_change().abs()
    median_move = daily_ret.median()
    return float(np.clip(median_move * 0.5, 0.003, 0.025))


try:
    print("🚀 v16.6 US AI·반도체 올인 방향성 통제소 가동...")
    print("   [종목 교체: SOXL/TECL 제거 → AVGO/MU 추가]\n")

    event_warnings = check_upcoming_events()
    prev_history = load_signal_history()
    today_history = {}

    start_date = '2022-01-01'
    macro_tickers = {'Nasdaq': '^IXIC', 'SMH': 'SMH', 'VIX': '^VIX', 'TNX': '^TNX'}
    macro_data = {}
    for name, t in macro_tickers.items():
        df_raw = yf.download(t, start=start_date, progress=False)
        if not df_raw.empty:
            series = df_raw['Close'].squeeze()
            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)
            macro_data[name] = series
    macro_df = pd.DataFrame(macro_data).ffill().dropna()
    macro_df['Nasdaq_Ret'] = macro_df['Nasdaq'].pct_change()
    macro_df['SMH_Ret'] = macro_df['SMH'].pct_change()
    nasdaq_ma200 = macro_df['Nasdaq'].rolling(200).mean()
    nasdaq_ma200_gap = ((macro_df['Nasdaq'] / nasdaq_ma200) - 1).iloc[-1] * 100

    # [종목 교체] SOXL, TECL 제거 → AVGO, MU 추가
    targets = ['XOVR', 'AVGO', 'NVDL', 'DXYZ', 'MU']

    final_report = "🤖 [US AI·반도체 올인 방향성 통제소 v16.6]\n"
    final_report += "=" * 40 + "\n"

    if event_warnings:
        final_report += "📅 [이벤트 경고]\n"
        for w in event_warnings:
            final_report += w + "\n"
        final_report += "=" * 40 + "\n"

    for ticker in targets:
        tgt_data = yf.download(ticker, start=start_date, progress=False)
        if tgt_data.empty:
            final_report += f"⚠️ {ticker}: 데이터 수집 불가\n" + "-"*40 + "\n"
            continue

        df = pd.DataFrame({
            'High':  tgt_data['High'].squeeze(),
            'Low':   tgt_data['Low'].squeeze(),
            'Close': tgt_data['Close'].squeeze()
        })
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.join(macro_df, how='left').ffill().dropna()

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

        bb_upper, bb_ma, bb_lower = calculate_bollinger(df['Close'])
        df['BB_Upper'] = bb_upper
        df['BB_Lower'] = bb_lower
        df['BB_Pct'] = ((df['Close']-bb_lower)/(bb_upper-bb_lower).replace(0,np.nan))*100

        current_price = df['Close'].iloc[-1]
        is_uptrend = current_price > df['EMA_20'].iloc[-1]
        latest_atr = df['ATR'].iloc[-1]
        latest_rsi = df['RSI'].iloc[-1]
        bb_lower_val = df['BB_Lower'].iloc[-1]
        bb_ma_val = bb_ma.iloc[-1]
        bb_pct = df['BB_Pct'].iloc[-1]

        rsi_risk = max(0, latest_rsi - 68)
        atr_mult = max(1.0, 3.0 - (rsi_risk/10))
        stop_loss = max(current_price - (latest_atr*atr_mult), current_price*0.90)

        dyn_threshold = get_dynamic_threshold(df['Close'])
        features = ['Nasdaq_Ret','SMH_Ret','VIX','TNX','RSI','MACD_Hist','Price_to_EMA20']
        df['Next_Ret'] = df['Close'].pct_change().shift(-1)
        latest_features = df[features].iloc[-1]
        df_train = df.dropna(subset=['Next_Ret']+features).copy()
        df_train['Target'] = np.where(df_train['Next_Ret'] > dyn_threshold, 1, 0)

        MIN_TRAIN_ROWS = 300
        if len(df_train) < MIN_TRAIN_ROWS:
            final_report += f"🔍 {ticker} (데이터 부족, {len(df_train)}행)\n"
            final_report += f"  * 현재가: ${current_price:.2f} | 스탑로스: ${stop_loss:.2f}\n"
            final_report += f"  * BB 매수구간: ${bb_lower_val:.2f} ~ ${bb_ma_val:.2f}\n"
            final_report += f"  * RSI: {latest_rsi:.1f} | BB%: {bb_pct:.0f}%\n"
            final_report += "-"*40 + "\n"
            continue

        X = df_train[features]
        y = df_train['Target']
        split = int(len(X)*0.8)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X.iloc[:split])
        X_test_s  = scaler.transform(X.iloc[split:])
        X_latest  = scaler.transform(latest_features.values.reshape(1,-1))

        pos_ratio = y.iloc[:split].mean()
        neg_ratio = 1 - pos_ratio
        sw = np.where(y.iloc[:split]==1, 1.0/(2*pos_ratio+1e-9), 1.0/(2*neg_ratio+1e-9))

        rf = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=5, class_weight='balanced')
        gb = GradientBoostingClassifier(n_estimators=300, random_state=42, max_depth=4, learning_rate=0.05, subsample=0.8)
        rf.fit(X_train_s, y.iloc[:split])
        gb.fit(X_train_s, y.iloc[:split], sample_weight=sw)

        preds_prob = (rf.predict_proba(X_test_s)+gb.predict_proba(X_test_s))/2
        preds = (preds_prob[:,1]>=0.5).astype(int)
        f1 = f1_score(y.iloc[split:], preds, average='macro', zero_division=0)
        prec = precision_score(y.iloc[split:], preds, zero_division=0)
        rec = recall_score(y.iloc[split:], preds, zero_division=0)

        MIN_F1, MIN_PR = 0.50, 0.48
        pr_pass = (prec>=MIN_PR and rec>=MIN_PR)
        f1_pass = (f1>=MIN_F1)

        if not f1_pass or not pr_pass:
            fail = []
            if not f1_pass: fail.append(f"F1 {f1*100:.1f}%")
            if not pr_pass: fail.append(f"P{prec*100:.0f}%/R{rec*100:.0f}% 미달")
            today_history[ticker] = {'signal':'관망','price':float(current_price),'rsi':float(latest_rsi)}
            flip_msg = detect_signal_flip(ticker, '관망', prev_history)
            final_report += f"📌 {ticker}\n"
            final_report += f"  * ⚠️ 관망 ({' | '.join(fail)})\n"
            if flip_msg: final_report += f"  * {flip_msg}\n"
            final_report += f"  * 현재가: ${current_price:.2f} | 스탑로스: ${stop_loss:.2f}\n"
            final_report += f"  * BB 매수구간: ${bb_lower_val:.2f} ~ ${bb_ma_val:.2f}\n"
            final_report += f"  * RSI: {latest_rsi:.1f} | BB%: {bb_pct:.0f}%\n"
            final_report += "-"*40 + "\n"
            continue

        final_probs = (rf.predict_proba(X_latest)+gb.predict_proba(X_latest))/2
        up_prob, down_prob = final_probs[0][1]*100, final_probs[0][0]*100

        if up_prob>=65: direction="🟢 강한 상승"
        elif up_prob>=50: direction="🟡 약한 상승"
        elif down_prob>=65: direction="🔴 강한 하락"
        else: direction="🟠 약한 하락"

        entry_target = (bb_lower_val+bb_ma_val)/2
        today_history[ticker] = {'signal':direction,'price':float(current_price),'rsi':float(latest_rsi)}
        flip_msg = detect_signal_flip(ticker, direction, prev_history)

        final_report += f"📌 {ticker}\n"
        final_report += f"  * {direction} (상승 {up_prob:.1f}% / 하락 {down_prob:.1f}%)\n"
        if flip_msg: final_report += f"  * {flip_msg}\n"
        final_report += f"  * 현재가: ${current_price:.2f} | 스탑로스: ${stop_loss:.2f}\n"
        final_report += f"  * 📍매수 진입 타겟: ${entry_target:.2f} (BB하단 ${bb_lower_val:.2f})\n"
        final_report += f"  * F1 {f1*100:.1f}% | P {prec*100:.0f}% | R {rec*100:.0f}%\n"
        final_report += f"  * RSI: {latest_rsi:.1f} | BB%: {bb_pct:.0f}% | 추세: {'🟢' if is_uptrend else '🔴'}\n"
        final_report += "-"*40 + "\n"

    final_report += f"\n📊 나스닥 MA200 이격: {nasdaq_ma200_gap:+.1f}% (시장 국면 참고용)\n"
    save_signal_history(today_history)

    print("\n"+final_report)
    send_telegram_message(final_report)
    print("✅ v16.6 리포트 전송 완료")

except Exception as e:
    error_msg = f"🚨 에러:\n{traceback.format_exc()[:800]}"
    print(error_msg)
    send_telegram_message(error_msg)
