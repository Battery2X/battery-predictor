"""
predict_position.py — 포지션(1개월 내외 방향성) 예측 (v20.0)
호라이즌: 20거래일 / 대상: 나스닥·반도체·S&P 레버리지·인버스 6종목
"""
import traceback
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from common import (
    LEVERAGE_UNIVERSE, fetch_macro_data, fetch_ticker_data,
    build_features, FEATURE_COLUMNS, build_target,
    check_upcoming_events, send_telegram, direction_label,
)

HORIZON_DAYS = 20
THRESHOLD_CAP = 0.15            # 포지션: 한 달간 15% 변동을 유의미한 신호로 간주
MIN_TRAIN_ROWS = 400


def run():
    print("🚀 [포지션] 20일 호라이즌 방향성 스캔 시작...")
    event_lines = check_upcoming_events(window_days=30)

    macro_df = fetch_macro_data()
    report = "🤖 [포지션 · 20일 호라이즌]\n" + "=" * 40 + "\n"
    if event_lines:
        report += "📅 [1개월 내 이벤트]\n" + "\n".join(event_lines) + "\n" + "=" * 40 + "\n"

    for ticker, meta in LEVERAGE_UNIVERSE.items():
        df = fetch_ticker_data(ticker)
        if df is None:
            continue
        df = df.join(macro_df, how='left').ffill().dropna()

        df = build_features(df, benchmark_col=meta['benchmark'])
        df = build_target(df, HORIZON_DAYS, THRESHOLD_CAP)

        df_train = df.dropna(subset=['Target'] + FEATURE_COLUMNS).copy()
        if len(df_train) < MIN_TRAIN_ROWS:
            continue

        X, y = df_train[FEATURE_COLUMNS], df_train['Target']

        tscv = TimeSeriesSplit(n_splits=3)
        xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42)
        lgbm = LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.03, verbose=-1, random_state=42)

        f1 = prec = rec = 0
        for tr_idx, te_idx in tscv.split(X):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
            scaler = StandardScaler()
            X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)

            xgb.fit(X_tr_s, y_tr)
            lgbm.fit(X_tr_s, y_tr)
            probs = (xgb.predict_proba(X_te_s) + lgbm.predict_proba(X_te_s)) / 2
            preds = (probs[:, 1] >= 0.5).astype(int)

            f1 += f1_score(y_te, preds, average='macro', zero_division=0)
            prec += precision_score(y_te, preds, zero_division=0)
            rec += recall_score(y_te, preds, zero_division=0)
        f1, prec, rec = f1/3, prec/3, rec/3

        scaler_final = StandardScaler()
        X_scaled = scaler_final.fit_transform(X)
        xgb.fit(X_scaled, y)
        lgbm.fit(X_scaled, y)

        latest = scaler_final.transform(df[FEATURE_COLUMNS].iloc[[-1]])
        final_probs = (xgb.predict_proba(latest) + lgbm.predict_proba(latest)) / 2
        up_prob, down_prob = final_probs[0][1]*100, final_probs[0][0]*100

        report += f"📌 {ticker} ({meta['desc']}, {meta['family']})\n"
        report += f"  * 🚨 3배 레버리지 장기보유 — decay 누적 위험 매우 큼, 포지션 목적엔 원래 부적합한 상품군\n"
        report += f"  * 방향성: {direction_label(up_prob, down_prob)} (상승확률 {up_prob:.1f}%)\n"
        report += f"  * 현재가: ${df['Close'].iloc[-1]:.2f} | 20일 변동성(연율): {df['HistVol20'].iloc[-1]*100:.1f}%\n"
        report += f"  * 교차검증: F1 {f1*100:.1f}% | P {prec*100:.0f}% | R {rec*100:.0f}%\n"
        report += "-" * 40 + "\n"

    print("\n" + report)
    send_telegram(report)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        err = f"🚨 [포지션] 에러:\n{traceback.format_exc()[:800]}"
        print(err)
        send_telegram(err)
