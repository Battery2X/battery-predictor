"""
predict_scalp.py — 단타(익일 방향성) 예측 (v22.0)
호라이즌: 1거래일
구조: QQQ/SMH/SPY 벤치마크 방향을 각 1회 예측 -> 롱/인버스 페어에 적용
"""
import traceback
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from common import (
    LEVERAGE_UNIVERSE, BENCHMARK_TICKERS, REL_COMPARISON,
    fetch_macro_data, fetch_ticker_data,
    build_features, FEATURE_COLUMNS, build_target,
    apply_direction, check_upcoming_events, send_telegram, direction_label,
)

HORIZON_DAYS = 1
LEVERAGED_THRESHOLD_CAP = 0.04     # 레버리지 상품(3배) 기준 하루 4%
BENCHMARK_MULTIPLIER = 3           # 벤치마크는 그 1/3만 움직이므로 임계값도 나눠줌
MIN_TRAIN_ROWS = 300


def train_benchmark_model(benchmark, macro_df):
    """벤치마크 하나에 대해 모델을 학습하고 (up_prob, down_prob, f1, prec, rec, df) 반환."""
    df = fetch_ticker_data(benchmark)
    if df is None:
        return None
    df = df.join(macro_df, how='left').ffill().dropna()
    df = build_features(df, benchmark_col=REL_COMPARISON[benchmark])
    df = build_target(df, HORIZON_DAYS, LEVERAGED_THRESHOLD_CAP / BENCHMARK_MULTIPLIER)

    df_train = df.dropna(subset=['Target'] + FEATURE_COLUMNS).copy()
    if len(df_train) < MIN_TRAIN_ROWS:
        return None

    X, y = df_train[FEATURE_COLUMNS], df_train['Target']

    tscv = TimeSeriesSplit(n_splits=3)
    xgb = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.08,
                         subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42)
    lgbm = LGBMClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, verbose=-1, random_state=42)

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

    return {'up_prob': up_prob, 'down_prob': down_prob, 'f1': f1, 'prec': prec, 'rec': rec,
            'dead_zone': df['DeadZone_Ratio'].iloc[-1]}


def run():
    print("🚀 [단타] 1일 호라이즌 방향성 스캔 시작 (벤치마크 중심 구조)...")
    event_lines = check_upcoming_events(window_days=1)

    macro_df = fetch_macro_data()
    report = "🤖 [단타 · 1일 호라이즌]\n" + "=" * 40 + "\n"
    if event_lines:
        report += "📅 [임박 이벤트]\n" + "\n".join(event_lines) + "\n" + "=" * 40 + "\n"

    for benchmark in BENCHMARK_TICKERS:
        result = train_benchmark_model(benchmark, macro_df)
        if result is None:
            continue

        report += f"■ 벤치마크: {benchmark} (방향성: {direction_label(result['up_prob'], result['down_prob'])}, " \
                  f"상승 {result['up_prob']:.1f}% / 하락 {result['down_prob']:.1f}%)\n"
        report += f"   교차검증: F1 {result['f1']*100:.1f}% | P {result['prec']*100:.0f}% | R {result['rec']*100:.0f}% " \
                  f"| 횡보비중 {result['dead_zone']*100:.0f}%\n"

        for ticker, meta in LEVERAGE_UNIVERSE.items():
            if meta['benchmark'] != benchmark:
                continue
            etf_df = fetch_ticker_data(ticker)
            if etf_df is None:
                continue
            up_prob, down_prob = apply_direction(result['up_prob'], result['down_prob'], meta['direction'])
            current_price = etf_df['Close'].iloc[-1]
            etf_vol = etf_df['Close'].pct_change().rolling(20).std().iloc[-1] * (252 ** 0.5)

            report += f"  📌 {ticker} ({meta['desc']})\n"
            report += f"    * ⚠️ 3배 레버리지 — 일별 리밸런싱 decay 존재, 단타 전용 상품\n"
            report += f"    * 방향성: {direction_label(up_prob, down_prob)} (상승 {up_prob:.1f}% / 하락 {down_prob:.1f}%) " \
                      f"— {benchmark} 모델 기준({'그대로' if meta['direction']==1 else '반전'})\n"
            report += f"    * 현재가: ${current_price:.2f} | 20일 변동성(연율): {etf_vol*100:.1f}%\n"
        report += "-" * 40 + "\n"

    print("\n" + report)
    send_telegram(report)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        err = f"🚨 [단타] 에러:\n{traceback.format_exc()[:800]}"
        print(err)
        send_telegram(err)
