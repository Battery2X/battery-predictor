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
# v16.7 변경사항 — 4가지 구조적 개선
# ============================================================
# [개선 1] 레짐 전환 감지 피처 추가
#   - 52주 신고가 갱신 빈도(최근 20일 중 신고가 비율)
#   - 단기추세(20일 기울기) vs 장기추세(120일 기울기) 차이
#   - MU처럼 구조적 재평가 중인 종목을 모델이 별도 인식하게 함
#
# [개선 2] BB% 해석 이원화 (종목 카테고리별 차등)
#   - GROWTH 카테고리(최근 120일 +50% 이상 상승): BB%>100을
#     "과매수 경고"가 아니라 "강세 추세 확인"으로 재해석
#   - STANDARD 카테고리: 기존 방식(평균회귀 가정) 유지
#
# [개선 3] 실적 발표 이벤트 오버라이드
#   - 이벤트 캘린더에 종목별 실적 발표일 + 컨센서스 방향성 추가
#   - D-2 이내 실적 종목은 ML 신호와 별개로 펀더멘털 컨센서스 섹션 출력
#
# [개선 4] P/R 필터 카테고리별 동적 임계값
#   - GROWTH 종목: MIN_PR 0.40 (변동성 높아 기준 완화)
#   - LEVERAGE 종목: MIN_PR 0.50 (변동성 가장 높아 기준 강화)
#   - STANDARD 종목: MIN_PR 0.48 (기존 유지)
# ============================================================


# ============================================================
# [개선 3] 이벤트 캘린더 + 컨센서스 방향성
# ============================================================
MARKET_EVENTS = {
    "2026-06-18": {"desc": "FOMC 금리 결정 (새벽 3:00 KST)", "ticker": None, "consensus": None},
    "2026-06-20": {"desc": "미국 6월 CPI 발표", "ticker": None, "consensus": None},
    "2026-06-24": {
        "desc": "Micron(MU) Q3 FY26 실적 발표 (장마감 후)",
        "ticker": "MU",
        "consensus": "컨센서스: EPS $19.72~19.95 / 매출 $34.4~34.8B (전년比 +270%) | HBM 2026 완판·풀백로그 | 옵션시장 변동성 ±17% 가격책정"
    },
    "2026-07-29": {"desc": "FOMC 7월 회의", "ticker": None, "consensus": None},
    "2026-08-25": {
        "desc": "NVDA Q2 실적 발표 (예정)",
        "ticker": "NVDA",
        "consensus": "중국 매출 가이던스 제외 지속 여부 핵심 관전포인트"
    },
    "2026-08-27": {"desc": "FOMC 잭슨홀 미팅", "ticker": None, "consensus": None},
    "2026-09-16": {"desc": "FOMC 9월 회의", "ticker": None, "consensus": None},
}

def check_upcoming_events():
    from datetime import datetime
    today = datetime.now().date()
    out = []
    out_detail = {}
    for date_str, info in MARKET_EVENTS.items():
        ev_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        days_left = (ev_date - today).days
        if 0 <= days_left <= 2:
            prefix = "🔴 오늘" if days_left == 0 else f"⚠️ D-{days_left}"
            out.append(f"  {prefix}: {info['desc']}")
            if info['ticker']:
                out_detail[info['ticker']] = {
                    'days_left': days_left,
                    'consensus': info['consensus']
                }
    return out, out_detail


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


# ============================================================
# [개선 1] 레짐 전환 감지 피처
# ============================================================
def calculate_regime_features(df):
    """
    52주 신고가 갱신 빈도 + 단기/장기 추세 기울기 차이
    구조적 재평가 중인 종목(MU, AVGO형)을 식별하는 핵심 피처
    """
    # 52주(약 252거래일) 신고가 갱신 빈도 — 최근 20일 중 신고가 비율
    rolling_max_252 = df['Close'].rolling(252, min_periods=60).max()
    is_new_high = (df['Close'] >= rolling_max_252 * 0.995).astype(int)
    df['NewHigh_Freq_20D'] = is_new_high.rolling(20).mean()

    # 단기추세(20일) vs 장기추세(120일) 기울기 — 선형회귀 대용으로 퍼센트 변화 사용
    short_slope = df['Close'].pct_change(20)
    long_slope = df['Close'].pct_change(120) / 6  # 120일을 20일 단위로 정규화
    df['Trend_Accel'] = short_slope - long_slope  # 양수면 가속(레짐 전환), 음수면 둔화

    # 최근 120일 누적 수익률 — GROWTH 카테고리 판별용
    df['Ret_120D'] = df['Close'].pct_change(120)

    return df


def classify_category(ret_120d, ticker, leverage_tickers):
    """
    [개선 2, 4] 종목 카테고리 분류
    - LEVERAGE: 레버리지 ETF (NVDL 등)
    - GROWTH: 최근 120일 +50% 이상 상승한 구조적 재평가 종목
    - STANDARD: 그 외 일반 종목
    """
    if ticker in leverage_tickers:
        return 'LEVERAGE'
    if pd.notna(ret_120d) and ret_120d >= 0.50:
        return 'GROWTH'
    return 'STANDARD'


# 카테고리별 P/R 임계값 [개선 4]
PR_THRESHOLDS = {
    'GROWTH':   0.40,
    'STANDARD': 0.48,
    'LEVERAGE': 0.50,
}

LEVERAGE_TICKERS = {'NVDL'}


try:
    print("🚀 v16.7 US AI·반도체 올인 방향성 통제소 가동...")
    print("   [개선: 레짐감지 | BB%이원화 | 실적이벤트오버라이드 | P/R카테고리별]\n")

    event_warnings, event_detail = check_upcoming_events()
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

    targets = ['XOVR', 'AVGO', 'NVDL', 'DXYZ', 'MU']

    final_report = "🤖 [US AI·반도체 올인 방향성 통제소 v16.7]\n"
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
        df['BB_Pct'] = ((df['Close']-bb_lower)/(bb_upper-bb_lower).replace(0, np.nan))*100

        # [개선 1] 레짐 전환 감지 피처
        df = calculate_regime_features(df)

        current_price = df['Close'].iloc[-1]
        is_uptrend = current_price > df['EMA_20'].iloc[-1]
        latest_atr = df['ATR'].iloc[-1]
        latest_rsi = df['RSI'].iloc[-1]
        bb_lower_val = df['BB_Lower'].iloc[-1]
        bb_ma_val = bb_ma.iloc[-1]
        bb_pct = df['BB_Pct'].iloc[-1]
        new_high_freq = df['NewHigh_Freq_20D'].iloc[-1]
        trend_accel = df['Trend_Accel'].iloc[-1]
        ret_120d = df['Ret_120D'].iloc[-1]

        # [개선 2, 4] 카테고리 분류
        category = classify_category(ret_120d, ticker, LEVERAGE_TICKERS)
        min_pr = PR_THRESHOLDS[category]

        rsi_risk = max(0, latest_rsi - 68)
        atr_mult = max(1.0, 3.0 - (rsi_risk/10))
        stop_loss = max(current_price - (latest_atr*atr_mult), current_price*0.90)

        dyn_threshold = get_dynamic_threshold(df['Close'])
        features = ['Nasdaq_Ret', 'SMH_Ret', 'VIX', 'TNX', 'RSI', 'MACD_Hist',
                    'Price_to_EMA20', 'NewHigh_Freq_20D', 'Trend_Accel']
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
        X_latest  = scaler.transform(latest_features.values.reshape(1, -1))

        pos_ratio = y.iloc[:split].mean()
        neg_ratio = 1 - pos_ratio
        sw = np.where(y.iloc[:split]==1, 1.0/(2*pos_ratio+1e-9), 1.0/(2*neg_ratio+1e-9))

        rf = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=5, class_weight='balanced')
        gb = GradientBoostingClassifier(n_estimators=300, random_state=42, max_depth=4, learning_rate=0.05, subsample=0.8)
        rf.fit(X_train_s, y.iloc[:split])
        gb.fit(X_train_s, y.iloc[:split], sample_weight=sw)

        preds_prob = (rf.predict_proba(X_test_s)+gb.predict_proba(X_test_s))/2
        preds = (preds_prob[:,1]>=0.5).astype(int)
        f1   = f1_score(y.iloc[split:], preds, average='macro', zero_division=0)
        prec = precision_score(y.iloc[split:], preds, zero_division=0)
        rec  = recall_score(y.iloc[split:], preds, zero_division=0)

        # [개선 4] 카테고리별 동적 P/R 임계값 적용
        MIN_F1 = 0.50
        pr_pass = (prec >= min_pr and rec >= min_pr)
        f1_pass = (f1 >= MIN_F1)

        # [개선 2] BB% 이원화 해석
        if category == 'GROWTH' and bb_pct > 100:
            bb_interpretation = "🚀 강세 추세 확인 (구조적 재평가 중 — 평균회귀 가정 미적용)"
        elif bb_pct > 100:
            bb_interpretation = "⚠️ 밴드 상단 돌파 (과매수 경계)"
        elif bb_pct < 0:
            bb_interpretation = "✅ 밴드 하단 이탈 (과매도 기회)"
        else:
            bb_interpretation = None

        cat_label = {'GROWTH': '🚀구조적성장', 'STANDARD': '표준', 'LEVERAGE': '⚡레버리지'}[category]

        # [개선 3] 실적 이벤트 오버라이드 — ML 판정 전에 먼저 출력
        earnings_override = ""
        if ticker in event_detail:
            d_left = event_detail[ticker]['days_left']
            consensus = event_detail[ticker]['consensus']
            earnings_override = f"  * 📊 [실적 D-{d_left}] {consensus}\n"

        if not f1_pass or not pr_pass:
            fail = []
            if not f1_pass: fail.append(f"F1 {f1*100:.1f}%")
            if not pr_pass: fail.append(f"P{prec*100:.0f}%/R{rec*100:.0f}% 미달(기준{min_pr*100:.0f}%)")
            today_history[ticker] = {'signal':'관망','price':float(current_price),'rsi':float(latest_rsi)}
            flip_msg = detect_signal_flip(ticker, '관망', prev_history)

            final_report += f"📌 {ticker} [{cat_label}]\n"
            if earnings_override:
                final_report += earnings_override
            final_report += f"  * ⚠️ 관망 ({' | '.join(fail)})\n"
            if flip_msg:
                final_report += f"  * {flip_msg}\n"
            final_report += f"  * 현재가: ${current_price:.2f} | 스탑로스: ${stop_loss:.2f}\n"
            final_report += f"  * BB 매수구간: ${bb_lower_val:.2f} ~ ${bb_ma_val:.2f}\n"
            final_report += f"  * RSI: {latest_rsi:.1f} | BB%: {bb_pct:.0f}%"
            if bb_interpretation:
                final_report += f" → {bb_interpretation}"
            final_report += "\n"
            final_report += f"  * 신고가빈도(20D): {new_high_freq*100:.0f}% | 추세가속도: {trend_accel*100:+.1f}%p\n"
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

        final_report += f"📌 {ticker} [{cat_label}]\n"
        if earnings_override:
            final_report += earnings_override
        final_report += f"  * {direction} (상승 {up_prob:.1f}% / 하락 {down_prob:.1f}%)\n"
        if flip_msg:
            final_report += f"  * {flip_msg}\n"
        final_report += f"  * 현재가: ${current_price:.2f} | 스탑로스: ${stop_loss:.2f}\n"
        final_report += f"  * 📍매수 진입 타겟: ${entry_target:.2f} (BB하단 ${bb_lower_val:.2f})\n"
        final_report += f"  * F1 {f1*100:.1f}% | P {prec*100:.0f}% | R {rec*100:.0f}% (기준P/R {min_pr*100:.0f}%)\n"
        final_report += f"  * RSI: {latest_rsi:.1f} | BB%: {bb_pct:.0f}%"
        if bb_interpretation:
            final_report += f" → {bb_interpretation}"
        final_report += f" | 추세: {'🟢' if is_uptrend else '🔴'}\n"
        final_report += f"  * 신고가빈도(20D): {new_high_freq*100:.0f}% | 추세가속도: {trend_accel*100:+.1f}%p\n"
        final_report += "-"*40 + "\n"

    final_report += f"\n📊 나스닥 MA200 이격: {nasdaq_ma200_gap:+.1f}% (시장 국면 참고용)\n"
    save_signal_history(today_history)

    print("\n" + final_report)
    send_telegram_message(final_report)
    print("✅ v16.7 리포트 전송 완료")

except Exception as e:
    error_msg = f"🚨 에러:\n{traceback.format_exc()[:800]}"
    print(error_msg)
    send_telegram_message(error_msg)
