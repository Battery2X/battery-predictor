import traceback
import requests
import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# v17.2 변경사항
# ============================================================
# [Gemini 제안 1 수정 반영] Nasdaq 200일선 국면 피처
#   - Gemini 방식(0/1 이진값) → 이격률 연속값으로 개선
#   - 이유: MA200 근처에서 0↔1 노이즈 방지, 강도까지 학습 가능
#
# [Gemini 제안 2 절충 반영] 훈련/테스트 분할 개선
#   - 95/5 분할 거부: 테스트셋 ~60행 → F1 신뢰도 붕괴
#   - 대신: Walk-Forward 방식 (슬라이딩 윈도우 3회 평균 F1)
#   - 효과: 최신 데이터 학습 + 안정적인 신뢰도 평가 동시 달성
#   - 연산 부담 최소화: fold=3 고정
# ============================================================


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


def train_and_evaluate(X_tr, y_tr, X_te, y_te):
    """RF + GB 앙상블 학습 및 macro F1 반환"""
    pos = y_tr.mean()
    neg = 1 - pos
    sw  = np.where(y_tr == 1,
                   1.0 / (2 * pos + 1e-9),
                   1.0 / (2 * neg + 1e-9))

    rf = RandomForestClassifier(
        n_estimators=500, random_state=42,
        max_depth=6, class_weight='balanced'
    )
    gb = GradientBoostingClassifier(
        n_estimators=500, random_state=42,
        max_depth=4, learning_rate=0.03, subsample=0.8
    )
    rf.fit(X_tr, y_tr)
    gb.fit(X_tr, y_tr, sample_weight=sw)

    probs = (rf.predict_proba(X_te) + gb.predict_proba(X_te)) / 2
    preds = (probs[:, 1] >= 0.5).astype(int)

    f1   = f1_score(y_te, preds, average='macro', zero_division=0)
    prec = precision_score(y_te, preds, zero_division=0)
    rec  = recall_score(y_te, preds, zero_division=0)
    return rf, gb, f1, prec, rec


LEVERAGE_META = {
    'SOXL': {'lev': 3, 'threshold': 0.08, 'label': '반도체 3x'},
    'TECL': {'lev': 3, 'threshold': 0.08, 'label': '테크 3x'},
    'NVDL': {'lev': 2, 'threshold': 0.06, 'label': '엔비디아 2x'},
    'XOVR': {'lev': 1, 'threshold': 0.03, 'label': '스페이스X ETF'},
    'DXYZ': {'lev': 1, 'threshold': 0.03, 'label': 'AI 유니콘 펀드'},
}


try:
    print("🚀 v17.2 US AI·반도체 올인 '장기 매크로 통제소' 가동...")
    print("   [추가: MA200 국면 이격률 + Walk-Forward 3-Fold 검증]\n")

    start_date = '2021-01-01'

    macro_tickers = {
        'Nasdaq': '^IXIC',
        'SMH':    'SMH',
        'VIX':    '^VIX',
        'TNX':    '^TNX',
        'IRX':    '^IRX'
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

    macro_df['Yield_Curve']       = macro_df['TNX'] - macro_df['IRX']
    macro_df['Nasdaq_3M_Ret']     = macro_df['Nasdaq'].pct_change(60)
    macro_df['VIX_EMA20']         = macro_df['VIX'].ewm(span=20, adjust=False).mean()
    macro_df['VIX_Level']         = macro_df['VIX']

    # [Gemini 제안 1 개선판] 0/1 이진값 → 연속 이격률
    nasdaq_ma200                  = macro_df['Nasdaq'].rolling(200).mean()
    macro_df['Nasdaq_MA200_Gap']  = (macro_df['Nasdaq'] / nasdaq_ma200) - 1
    # 예) +0.10 = MA200 대비 10% 위(강세장), -0.05 = 5% 아래(약세장)

    targets      = ['XOVR', 'SOXL', 'NVDL', 'DXYZ', 'TECL']
    final_report = "🦅 [US AI·반도체 올인 '장기' 매크로 통제소 v17.2]\n"
    final_report += "📅 분석 기준: 향후 20영업일 대세 상승 확률\n"
    final_report += "=" * 40 + "\n"

    for ticker in targets:
        meta      = LEVERAGE_META.get(ticker, {'lev': 1, 'threshold': 0.03, 'label': ticker})
        threshold = meta['threshold']
        lev_label = meta['label']

        print(f"진행 중: {ticker} ({lev_label})...")
        tgt_data = yf.download(ticker, start=start_date, progress=False)

        if tgt_data.empty:
            final_report += f"⚠️ {ticker}: 데이터 수집 불가\n" + "-" * 40 + "\n"
            continue

        df = pd.DataFrame({
            'High':  tgt_data['High'].squeeze(),
            'Low':   tgt_data['Low'].squeeze(),
            'Close': tgt_data['Close'].squeeze()
        })
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.join(macro_df, how='left').ffill().dropna()

        # 피처 엔지니어링
        df['RSI_Weekly'] = calculate_rsi(df['Close'], period=70)
        df['Log_Ret']    = np.log(df['Close'] / df['Close'].shift(1))
        df['Ret_60D']    = np.log(df['Close'] / df['Close'].shift(60))
        df['Ret_60D_Z']  = (
            (df['Ret_60D'] - df['Ret_60D'].rolling(120).mean())
            / (df['Ret_60D'].rolling(120).std() + 1e-9)
        )
        df['Vol_20']     = df['Log_Ret'].rolling(20).std() * np.sqrt(252)

        df['Future_20D'] = df['Close'].shift(-20)
        df['Future_Ret'] = (df['Future_20D'] / df['Close']) - 1

        features = [
            'Yield_Curve', 'Nasdaq_3M_Ret', 'VIX_EMA20', 'VIX_Level',
            'RSI_Weekly', 'Ret_60D_Z', 'Vol_20',
            'Nasdaq_MA200_Gap'   # ← v17.2 신규 피처
        ]

        latest_features = df[features].iloc[-1]
        current_price   = df['Close'].iloc[-1]
        latest_rsi_w    = df['RSI_Weekly'].iloc[-1]
        ret_60d_z       = df['Ret_60D_Z'].iloc[-1]
        current_vol     = df['Vol_20'].iloc[-1]
        ma200_gap       = df['Nasdaq_MA200_Gap'].iloc[-1]

        df_train = df.dropna(subset=['Future_Ret'] + features).copy()
        df_train['Target'] = np.where(df_train['Future_Ret'] > threshold, 1, 0)

        class_ratio = df_train['Target'].mean()
        print(f"  Target=1 비율: {class_ratio:.1%} | 데이터: {len(df_train)}행")

        MIN_LONG_ROWS = 400
        if len(df_train) < MIN_LONG_ROWS:
            final_report += f"🔍 {ticker}: 학습 데이터 부족 ({len(df_train)}행)\n" + "-" * 40 + "\n"
            continue

        X = df_train[features].values
        y = df_train['Target'].values

        # [Gemini 제안 2 절충] Walk-Forward 3-Fold
        # 고정 80/20 대신 슬라이딩 윈도우 3회 → 평균 F1로 신뢰도 판단
        # 마지막 fold는 가장 최신 데이터를 포함하므로 최신 트렌드 반영
        scaler   = StandardScaler()
        tscv     = TimeSeriesSplit(n_splits=3)
        f1_folds, prec_folds, rec_folds = [], [], []
        last_rf, last_gb = None, None

        for fold_idx, (tr_idx, te_idx) in enumerate(tscv.split(X)):
            X_tr = scaler.fit_transform(X[tr_idx])
            X_te = scaler.transform(X[te_idx])
            y_tr = y[tr_idx]
            y_te = y[te_idx]

            rf_f, gb_f, f1_f, prec_f, rec_f = train_and_evaluate(
                X_tr, y_tr, X_te, y_te
            )
            f1_folds.append(f1_f)
            prec_folds.append(prec_f)
            rec_folds.append(rec_f)

            # 마지막 fold의 모델 + scaler 보관 (예측에 사용)
            if fold_idx == 2:
                last_rf = rf_f
                last_gb = gb_f
                last_scaler = scaler

        avg_f1   = np.mean(f1_folds)
        avg_prec = np.mean(prec_folds)
        avg_rec  = np.mean(rec_folds)

        print(f"  Fold F1: {[f'{f:.2f}' for f in f1_folds]} → 평균 {avg_f1:.2f}")

        MIN_F1_LONG = 0.52
        if avg_f1 < MIN_F1_LONG:
            final_report += f"📌 {ticker} ({lev_label})\n"
            final_report += f"  * ⚠️ 매크로 관망 (3-Fold 평균 F1 {avg_f1*100:.1f}% 미달)\n"
            final_report += f"  * Fold별: {' / '.join([f'{f*100:.0f}%' for f in f1_folds])}\n"
            final_report += f"  * 현재가: ${current_price:.2f} | MA200 이격: {ma200_gap*100:+.1f}%\n"
            final_report += "-" * 40 + "\n"
            continue

        # 최종 예측 (마지막 fold 모델 사용)
        X_latest    = last_scaler.transform(latest_features.values.reshape(1, -1))
        final_probs = (last_rf.predict_proba(X_latest) + last_gb.predict_proba(X_latest)) / 2
        up_prob     = final_probs[0][1] * 100

        # 국면 해석
        if ma200_gap > 0.05:
            regime = f"🟢 강세장 (MA200 대비 {ma200_gap*100:+.1f}%)"
        elif ma200_gap < -0.05:
            regime = f"🔴 약세장 (MA200 대비 {ma200_gap*100:+.1f}%)"
        else:
            regime = f"🟡 중립구간 (MA200 대비 {ma200_gap*100:+.1f}%)"

        # 60일 모멘텀 해석
        if ret_60d_z > 2.0:
            z_comment = f"⚠️ 과열 (Z={ret_60d_z:.2f})"
        elif ret_60d_z < -1.5:
            z_comment = f"✅ 과매도 (Z={ret_60d_z:.2f})"
        else:
            z_comment = f"중립 (Z={ret_60d_z:.2f})"

        # 전략 판단
        if up_prob >= 60:
            strategy = "🟢 [전략 승인] 한 달 대세 상승 유력"
        elif up_prob >= 45:
            strategy = "🟡 [전략 보류] 분할 진입만 유효"
        else:
            strategy = "🔴 [전략 거부] 현금 유지 권장"

        final_report += f"📌 {ticker} ({lev_label})\n"
        final_report += f"  * 🎯 {strategy}\n"
        final_report += f"  * 한 달 뒤 상승 확률: {up_prob:.1f}% (기준: +{threshold*100:.0f}%)\n"
        final_report += f"  * 시장 국면: {regime}\n"
        final_report += f"  * 3-Fold F1 {avg_f1*100:.1f}% | P {avg_prec*100:.0f}% | R {avg_rec*100:.0f}%\n"
        final_report += f"  * 주간RSI {latest_rsi_w:.1f} | 모멘텀: {z_comment} | 변동성: {current_vol*100:.0f}%\n"
        final_report += "-" * 40 + "\n"

    print("\n" + final_report)
    send_telegram_message(final_report)
    print("✅ v17.2 장기 매크로 리포트 전송 완료")

except Exception as e:
    error_msg = f"🚨 에러:\n{traceback.format_exc()[:800]}"
    print(error_msg)
    send_telegram_message(error_msg)
