import traceback
import requests
import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# v1.4 변경사항
# [종목 제거] QBTS, RGTI 제거
#   - 만성적 매매보류(F1 30~45%), 이격도 과열기준 자체를 못 채움
#   - 한 번도 유효 신호를 낸 적 없어 모델 가치 낮음
# [유지] IONQ(유일한 강신호), INFQ(8월 재판단 대기)
# ============================================================

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
    gain = delta.where(delta>0,0).ewm(alpha=1/period,adjust=False).mean()
    loss = -delta.where(delta<0,0).ewm(alpha=1/period,adjust=False).mean()
    return 100 - (100/(1+(gain/loss)))

def get_dynamic_threshold(future_ret_series, target_ratio=0.30):
    percentile = (1-target_ratio)*100
    threshold = np.percentile(future_ret_series.dropna(), percentile)
    return float(np.clip(threshold, 0.03, 0.40))

def get_disparity_threshold(disparity_series):
    return float(np.percentile(disparity_series.dropna().abs(), 75))

try:
    print("🔮 [v1.4] 양자컴퓨팅 리스크 관리 엔진 가동...")
    print("  \n")

    start_date = '2021-01-01'
    macro_tickers = {'Nasdaq':'^IXIC','VIX':'^VIX','TNX':'^TNX'}
    macro_raw = {}
    for name, t in macro_tickers.items():
        df_raw = yf.download(t, start=start_date, progress=False, multi_level_index=False)
        if not df_raw.empty:
            series = df_raw['Close'].squeeze()
            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)
            macro_raw[name] = series
    macro_base = pd.DataFrame(macro_raw).ffill().dropna()

    # [종목 제거] QBTS, RGTI 제거
    quantum_targets = ['IONQ', 'INFQ']

    final_report = "⚛️\n"
    final_report += "=" * 40 + "\n"

    for ticker in quantum_targets:
        tgt_data = yf.download(ticker, start=start_date, progress=False, multi_level_index=False)
        if tgt_data.empty:
            final_report += f"📌 {ticker}\n  * ⚠️ 데이터 수집 불가\n" + "-"*40 + "\n"
            continue

        raw_len = len(tgt_data)
        if raw_len >= 250:
            lookahead, window_name, min_rows = 60, "3개월 (60영업일 뒤)", 150
        else:
            lookahead, window_name, min_rows = 20, "1개월 (20영업일 뒤)", 60

        macro_df = macro_base.copy()
        macro_df = macro_df['Nasdaq'].pct_change(lookahead)
        macro_df = macro_df.diff(lookahead)
        macro_df['VIX_Mean'] = macro_df['VIX'].rolling(lookahead).mean()
        nasdaq_ma200 = macro_df['Nasdaq'].rolling(200).mean()
        macro_df['Nasdaq_MA200_Gap'] = (macro_df['Nasdaq']/nasdaq_ma200) - 1
        macro_df = macro_df.dropna()

        df = pd.DataFrame({
            'High': tgt_data['High'].squeeze(),
            'Low': tgt_data['Low'].squeeze(),
            'Close': tgt_data['Close'].squeeze()
        })
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.join(macro_df, how='left').ffill().dropna()

        rsi_period = max(14, int(lookahead*0.7))
        df = calculate_rsi(df['Close'], period=rsi_period)
        df = df['Close'].rolling(lookahead).mean()
        df = (df['Close']/df) - 1
        df = np.log(df['Close']/df['Close'].shift(1))
        df = df.rolling(lookahead).std()*np.sqrt(252)
        df['Future_Close'] = df['Close'].shift(-lookahead)
        df = (df['Future_Close']/df['Close']) - 1

        features =
        latest_features = df[features].iloc[-1]
        current_price = df['Close'].iloc[-1]
        disp_target = df.iloc[-1]
        ma_target_val = df.iloc[-1]

        df_train = df.dropna(subset=+features).copy()
        if len(df_train) < min_rows:
            final_report += f"📌 {ticker} ({window_name})\n"
            final_report += f"  * ⚠️ 학습 데이터 부족 ({len(df_train)}행 / 최소 {min_rows}행)\n"
            final_report += "-"*40 + "\n"
            continue

        dyn_threshold = get_dynamic_threshold(df_train, target_ratio=0.30)
        df_train = np.where(df_train>dyn_threshold, 1, 0)
        class_ratio = df_train.mean()
        disp_threshold = get_disparity_threshold(df_train)

        X = df_train[features]; y = df_train
        split = int(len(X)*0.8)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X.iloc[:split])
        X_test_s = scaler.transform(X.iloc[split:])
        X_latest = scaler.transform(latest_features.values.reshape(1,-1))

        pos_ratio = y.iloc[:split].mean(); neg_ratio = 1-pos_ratio
        sw = np.where(y.iloc[:split]==1, 1.0/(2*pos_ratio+1e-9), 1.0/(2*neg_ratio+1e-9))

        rf = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=4, class_weight='balanced')
        gb = GradientBoostingClassifier(n_estimators=300, random_state=42, max_depth=3, learning_rate=0.03, subsample=0.8)
        rf.fit(X_train_s, y.iloc[:split])
        gb.fit(X_train_s, y.iloc[:split], sample_weight=sw)

        rf_probs_te = rf.predict_proba(X_test_s)[:,1]
        gb_probs_te = gb.predict_proba(X_test_s)[:,1]
        ens_probs_te = (rf_probs_te+gb_probs_te)/2
        preds = (ens_probs_te>=0.5).astype(int)
        f1 = f1_score(y.iloc[split:], preds, average='macro', zero_division=0)
        prec = precision_score(y.iloc[split:], preds, zero_division=0)
        rec = recall_score(y.iloc[split:], preds, zero_division=0)

        rf_up = rf.predict_proba(X_latest)
        gb_up = gb.predict_proba(X_latest)
        up_prob = ((rf_up+gb_up)/2)*100

        MIN_F1 = 0.50
        if f1 < MIN_F1:
            decision = "🔴 [매매 보류] 모델 신뢰도 미달 — 예측 무효"
            prob_str = "계산 불가"; ml_signal = "neutral"
        else:
            prob_str = f"{up_prob:.1f}%"
            if up_prob>=62: decision="🟢 [중기 우상향] 모멘텀 유효"; ml_signal="up"
            elif up_prob>=40: decision="🟡 [중기 횡보] 비중 유지"; ml_signal="neutral"
            else: decision="🔴 [중기 위험] 하방 압력 우세"; ml_signal="down"

        is_overheated = abs(disp_target) > disp_threshold
        if is_overheated:
            target_band_str = f"${ma_target_val*0.92:.2f} ~ ${ma_target_val*1.05:.2f} (과열 진정 타점)"
            heat_signal = "overheated"
        else:
            target_band_str = "현재가 인근 분할 대응 가능"
            heat_signal = "normal"

        conflict_warning = ""
        if ml_signal=="up" and heat_signal=="overheated":
            conflict_warning = f"  * ⚠️ [신호 충돌] ML=상승({up_prob:.0f}%) vs 이격도=과열({disp_target*100:.0f}%)\n     → 추격 매수 금지. 과열 진정 후 타점 진입 권장\n"
        elif ml_signal=="down" and heat_signal=="normal":
            conflict_warning = f"  * ℹ️ [참고] ML=하락 신호이나 이격도는 정상권\n     → 급락 시 분할 매수 기회 탐색 가능\n"

        final_report += f"📌 {ticker} [{window_name}]\n"
        final_report += f"  * 🎯 결론: {decision}\n"
        if conflict_warning: final_report += conflict_warning
        final_report += f"  * {window_name} 상승 확률: {prob_str}\n"
        final_report += f"  * 매수 타점: {target_band_str}\n"
        final_report += f"  * F1 {f1*100:.1f}% | P {prec*100:.0f}% | R {rec*100:.0f}%\n"
        final_report += f"  * 이격도 {disp_target*100:.1f}% (과열기준 >{disp_threshold*100:.0f}%)\n"
        final_report += f"  * threshold: {dyn_threshold*100:.1f}% | Target=1 비율: {class_ratio:.1%}\n"
        final_report += "-"*40 + "\n"

    print("\n"+final_report)
    send_telegram_message(final_report)
    print("✅ v1.4 양자 리스크 리포트 전송 완료")

except Exception as e:
    error_msg = f"🚨 양자 엔진 v1.4 에러:\n{traceback.format_exc()[:800]}"
    print(error_msg)
    send_telegram_message(error_msg)
