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
# v17.3 변경사항
# [종목 교체] SOXL, TECL 제거 → AVGO, MU 추가
#   - 만성적 매크로 관망(F1 45~48%) 종목 제거
#   - AVGO/MU는 1x 종목이므로 레버리지 메타 threshold 별도 적용
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

# [종목 교체] LEVERAGE_META에서 SOXL/TECL 제거, AVGO/MU 추가 (1x 일반주 취급)
LEVERAGE_META = {
    'NVDL': {'lev': 2, 'threshold': 0.06, 'label': '엔비디아 2x'},
    'XOVR': {'lev': 1, 'threshold': 0.03, 'label': '스페이스X 노출 ETF'},
    'DXYZ': {'lev': 1, 'threshold': 0.03, 'label': 'AI 유니콘 펀드'},
    'AVGO': {'lev': 1, 'threshold': 0.04, 'label': 'AI 커스텀칩(ASIC)'},
    'MU':   {'lev': 1, 'threshold': 0.05, 'label': 'HBM 메모리'},
}

try:
    print("🚀 v17.3 US AI·반도체 올인 '장기 매크로 통제소' 가동...")
    print("   [종목 교체: SOXL/TECL 제거 → AVGO/MU 추가]\n")

    start_date = '2021-01-01'
    macro_tickers = {'Nasdaq':'^IXIC','SMH':'SMH','VIX':'^VIX','TNX':'^TNX','IRX':'^IRX'}
    macro_data = {}
    for name, t in macro_tickers.items():
        df_raw = yf.download(t, start=start_date, progress=False)
        if not df_raw.empty:
            series = df_raw['Close'].squeeze()
            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)
            macro_data[name] = series
    macro_df = pd.DataFrame(macro_data).ffill().dropna()
    macro_df['Yield_Curve'] = macro_df['TNX'] - macro_df['IRX']
    macro_df['Nasdaq_3M_Ret'] = macro_df['Nasdaq'].pct_change(60)
    macro_df['VIX_EMA20'] = macro_df['VIX'].ewm(span=20,adjust=False).mean()
    macro_df['VIX_Level'] = macro_df['VIX']
    nasdaq_ma200 = macro_df['Nasdaq'].rolling(200).mean()
    macro_df['Nasdaq_MA200_Gap'] = (macro_df['Nasdaq']/nasdaq_ma200) - 1

    targets = ['XOVR', 'AVGO', 'NVDL', 'DXYZ', 'MU']
    final_report = "🦅 [US AI·반도체 올인 '장기' 매크로 통제소 v17.3]\n"
    final_report += "📅 분석 기준: 향후 20영업일 대세 상승 확률\n"
    final_report += "=" * 40 + "\n"

    for ticker in targets:
        meta = LEVERAGE_META.get(ticker, {'lev':1,'threshold':0.03,'label':ticker})
        threshold = meta['threshold']
        lev_label = meta['label']

        tgt_data = yf.download(ticker, start=start_date, progress=False)
        if tgt_data.empty:
            final_report += f"⚠️ {ticker}: 데이터 수집 불가\n" + "-"*40 + "\n"
            continue

        df = pd.DataFrame({
            'High': tgt_data['High'].squeeze(),
            'Low': tgt_data['Low'].squeeze(),
            'Close': tgt_data['Close'].squeeze()
        })
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.join(macro_df, how='left').ffill().dropna()

        df['RSI_Weekly'] = calculate_rsi(df['Close'], period=70)
        df['Log_Ret'] = np.log(df['Close']/df['Close'].shift(1))
        df['Ret_60D'] = np.log(df['Close']/df['Close'].shift(60))
        df['Ret_60D_Z'] = (df['Ret_60D']-df['Ret_60D'].rolling(120).mean())/(df['Ret_60D'].rolling(120).std()+1e-9)
        df['Vol_20'] = df['Log_Ret'].rolling(20).std()*np.sqrt(252)
        df['Future_20D'] = df['Close'].shift(-20)
        df['Future_Ret'] = (df['Future_20D']/df['Close']) - 1

        features = ['Yield_Curve','Nasdaq_3M_Ret','VIX_EMA20','VIX_Level','RSI_Weekly','Ret_60D_Z','Vol_20','Nasdaq_MA200_Gap']
        latest_features = df[features].iloc[-1]
        current_price = df['Close'].iloc[-1]
        latest_rsi_w = df['RSI_Weekly'].iloc[-1]
        ret_60d_z = df['Ret_60D_Z'].iloc[-1]
        current_vol = df['Vol_20'].iloc[-1]

        df_train = df.dropna(subset=['Future_Ret']+features).copy()
        df_train['Target'] = np.where(df_train['Future_Ret']>threshold, 1, 0)

        MIN_LONG_ROWS = 400
        if len(df_train) < MIN_LONG_ROWS:
            final_report += f"🔍 {ticker} ({lev_label}): 학습 데이터 부족 ({len(df_train)}행)\n" + "-"*40 + "\n"
            continue

        X = df_train[features].values
        y = df_train['Target'].values
        scaler = StandardScaler()
        tscv = TimeSeriesSplit(n_splits=3)
        f1_folds, prec_folds, rec_folds = [], [], []
        last_rf = last_gb = last_scaler = None

        for fold_idx, (tr_idx, te_idx) in enumerate(tscv.split(X)):
            X_tr = scaler.fit_transform(X[tr_idx]); X_te = scaler.transform(X[te_idx])
            y_tr, y_te = y[tr_idx], y[te_idx]
            pos = y_tr.mean(); neg = 1-pos
            sw = np.where(y_tr==1, 1.0/(2*pos+1e-9), 1.0/(2*neg+1e-9))
            rf = RandomForestClassifier(n_estimators=500, random_state=42, max_depth=6, class_weight='balanced')
            gb = GradientBoostingClassifier(n_estimators=500, random_state=42, max_depth=4, learning_rate=0.03, subsample=0.8)
            rf.fit(X_tr, y_tr); gb.fit(X_tr, y_tr, sample_weight=sw)
            probs = (rf.predict_proba(X_te)+gb.predict_proba(X_te))/2
            preds = (probs[:,1]>=0.5).astype(int)
            f1_folds.append(f1_score(y_te, preds, average='macro', zero_division=0))
            prec_folds.append(precision_score(y_te, preds, zero_division=0))
            rec_folds.append(recall_score(y_te, preds, zero_division=0))
            if fold_idx==2:
                last_rf, last_gb, last_scaler = rf, gb, scaler

        avg_f1 = np.mean(f1_folds); avg_prec = np.mean(prec_folds); avg_rec = np.mean(rec_folds)

        MIN_F1_LONG = 0.52
        if avg_f1 < MIN_F1_LONG:
            final_report += f"📌 {ticker} ({lev_label})\n"
            final_report += f"  * ⚠️ 매크로 관망 (3-Fold 평균 F1 {avg_f1*100:.1f}% 미달)\n"
            final_report += f"  * Fold별: {' / '.join([f'{f*100:.0f}%' for f in f1_folds])}\n"
            final_report += f"  * 현재가: ${current_price:.2f} | MA200 이격: {macro_df['Nasdaq_MA200_Gap'].iloc[-1]*100:+.1f}%\n"
            final_report += "-"*40 + "\n"
            continue

        X_latest = last_scaler.transform(latest_features.values.reshape(1,-1))
        final_probs = (last_rf.predict_proba(X_latest)+last_gb.predict_proba(X_latest))/2
        up_prob = final_probs[0][1]*100

        ma200_gap = macro_df['Nasdaq_MA200_Gap'].iloc[-1]
        regime = f"🟢 강세장 ({ma200_gap*100:+.1f}%)" if ma200_gap>0.05 else (f"🔴 약세장 ({ma200_gap*100:+.1f}%)" if ma200_gap<-0.05 else f"🟡 중립 ({ma200_gap*100:+.1f}%)")
        z_comment = f"⚠️ 과열 (Z={ret_60d_z:.2f})" if ret_60d_z>2.0 else (f"✅ 과매도 (Z={ret_60d_z:.2f})" if ret_60d_z<-1.5 else f"중립 (Z={ret_60d_z:.2f})")

        if up_prob>=60: strategy="🟢 [전략 승인] 한 달 대세 상승 유력"
        elif up_prob>=45: strategy="🟡 [전략 보류] 분할 진입만 유효"
        else: strategy="🔴 [전략 거부] 현금 유지 권장"

        final_report += f"📌 {ticker} ({lev_label})\n"
        final_report += f"  * 🎯 {strategy}\n"
        final_report += f"  * 한 달 뒤 상승 확률: {up_prob:.1f}% (기준: +{threshold*100:.0f}%)\n"
        final_report += f"  * 시장 국면: {regime}\n"
        final_report += f"  * 3-Fold F1 {avg_f1*100:.1f}% | P {avg_prec*100:.0f}% | R {avg_rec*100:.0f}%\n"
        final_report += f"  * 주간RSI {latest_rsi_w:.1f} | 모멘텀: {z_comment} | 변동성: {current_vol*100:.0f}%\n"
        final_report += "-"*40 + "\n"

    print("\n"+final_report)
    send_telegram_message(final_report)
    print("✅ v17.3 장기 매크로 리포트 전송 완료")

except Exception as e:
    error_msg = f"🚨 장기 모델 에러:\n{traceback.format_exc()[:800]}"
    print(error_msg)
    send_telegram_message(error_msg)
