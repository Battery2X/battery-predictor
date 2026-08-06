"""
predict_stock_scalp.py — 개별 종목(AAPL/NVDA/TSLA) 단타(익일 방향성) 예측
호라이즌: 1거래일
레버리지 ETF와 달리 페어(롱/인버스) 개념이 없어서 종목당 모델 하나씩 직접 사용.
"""
import traceback
import pandas as pd

from common import (
    STOCK_TICKERS, fetch_macro_data, fetch_display_metrics, train_benchmark_model,
    fetch_analyst_momentum_features,
    check_upcoming_events, send_telegram, direction_label, edge_label,
    settle_predictions, append_predictions, rolling_accuracy, rolling_accuracy_by_earnings, make_prediction_record,
    earnings_proximity_warning,
)

LOG_NAME = "stock_scalp"
HORIZON_DAYS = 1
THRESHOLD_CAP = 0.03      # 개별 종목(레버리지 아님) 기준 하루 3%
MODEL_PARAMS = {'n_estimators': 150, 'max_depth': 3, 'learning_rate': 0.08}


def run():
    print("🚀 [종목-단타] 1일 호라이즌 방향성 스캔 시작 (AAPL/NVDA/TSLA)...")
    run_date = pd.Timestamp.now()
    event_lines = check_upcoming_events(window_days=1)

    settled_df = settle_predictions(LOG_NAME)

    macro_df = fetch_macro_data()
    report = "🤖 [종목-단타 · 1일 호라이즌]\n" + "=" * 40 + "\n"
    if event_lines:
        report += "📅 [임박 이벤트]\n" + "\n".join(event_lines) + "\n" + "=" * 40 + "\n"

    new_rows = []
    for ticker in STOCK_TICKERS:
        analyst_df = fetch_analyst_momentum_features(ticker)
        if analyst_df is not None and not analyst_df.empty:
            result = train_benchmark_model(ticker, macro_df, HORIZON_DAYS, THRESHOLD_CAP,
                                            benchmark_multiplier=1, min_train_rows=300, model_params=MODEL_PARAMS,
                                            extra_features_df=analyst_df)
        else:
            result = train_benchmark_model(ticker, macro_df, HORIZON_DAYS, THRESHOLD_CAP,
                                            benchmark_multiplier=1, min_train_rows=300, model_params=MODEL_PARAMS)
        if result is None:
            report += f"📌 {ticker}: ⚠️ 데이터 부족 — 이번 회차 스킵\n" + "-"*40 + "\n"
            continue

        metrics = fetch_display_metrics(ticker)
        if metrics is None:
            report += f"📌 {ticker}: ⚠️ 시세 조회 실패 — 이번 회차 스킵\n" + "-"*40 + "\n"
            continue

        acc = rolling_accuracy(settled_df, ticker, window=20)
        acc_str = f"최근 {acc['n']}회 적중률 {acc['accuracy']:.0f}%" if acc else "적중률 데이터 축적 중"
        earn_warn = earnings_proximity_warning(ticker, window_days=5)
        is_near_earnings = earn_warn is not None

        report += f"📌 {ticker}\n"
        report += f"  * 방향성: {direction_label(result['up_prob'], result['down_prob'])} " \
                  f"(보정후 상승 {result['up_prob']:.1f}% / 하락 {result['down_prob']:.1f}%, 원시확률 {result['raw_up_prob']:.1f}%)\n"
        report += f"  * {edge_label(result['edge'])} (기준선 {result['base_rate']:.1f}% | 엣지 {result['edge']:+.1f}%p)\n"
        top_feat_str = ", ".join([f"{name}({imp*100:.0f}%)" for name, imp in result['top_features']])
        report += f"  * 🔍 상위 5개 피처: {top_feat_str}\n"
        if analyst_df is not None and not analyst_df.empty:
            last_a = analyst_df.iloc[-1]
            report += f"  * 🧑‍💼 애널리스트 최근 30일: 순상향 {last_a['AnalystNet30']:+.0f}건 | " \
                      f"총 등급변경 {last_a['AnalystActivity30']:.0f}건\n"
        report += f"  * 교차검증: F1 {result['f1']*100:.1f}% | P {result['prec']*100:.0f}% | R {result['rec']*100:.0f}% " \
                  f"| 횡보비중 {result['dead_zone']*100:.0f}%\n"
        report += f"  * 📊 실전 적중률: {acc_str}\n"
        acc_split = rolling_accuracy_by_earnings(settled_df, ticker, window=20)
        if 'near_earnings' in acc_split and 'normal' in acc_split:
            report += f"  * 📊 실적임박 구간 적중률 {acc_split['near_earnings']['accuracy']:.0f}%(n={acc_split['near_earnings']['n']}) " \
                      f"vs 평상시 {acc_split['normal']['accuracy']:.0f}%(n={acc_split['normal']['n']})\n"
        report += f"  * 현재가: ${metrics['price']:.2f} | 20일 변동성(연율): {metrics['vol']*100:.1f}%\n"
        if earn_warn:
            report += f"  * {earn_warn}\n"
        report += "-" * 40 + "\n"

        new_rows.append(make_prediction_record(run_date, ticker, ticker, HORIZON_DAYS,
                                                 result['up_prob'], result['raw_up_prob'], metrics['price'],
                                                 near_earnings=is_near_earnings))

    if new_rows:
        append_predictions(LOG_NAME, new_rows)

    print("\n" + report)
    send_telegram(report)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        err = f"🚨 [종목-단타] 에러:\n{traceback.format_exc()[:800]}"
        print(err)
        send_telegram(err)
