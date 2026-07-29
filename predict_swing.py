"""
predict_swing.py — 스윙(1~2주 방향성) 예측 (v23.0)
호라이즌: 5거래일
"""
import traceback
import pandas as pd

from common import (
    LEVERAGE_UNIVERSE, BENCHMARK_TICKERS,
    fetch_macro_data, fetch_display_metrics, train_benchmark_model,
    apply_direction, check_upcoming_events, send_telegram, direction_label,
    settle_predictions, append_predictions, rolling_accuracy, make_prediction_record,
)

LOG_NAME = "swing"
HORIZON_DAYS = 5
LEVERAGED_THRESHOLD_CAP = 0.08
MODEL_PARAMS = {'n_estimators': 250, 'max_depth': 4, 'learning_rate': 0.05}


def run():
    print("🚀 [스윙] 5일 호라이즌 방향성 스캔 시작 (v23.0: 보정+적중률추적)...")
    run_date = pd.Timestamp.now()
    event_lines = check_upcoming_events(window_days=7)

    # 1) 과거 예측 결과 정산 (호라이즌 지난 건들 실제 결과 확인) — 한 번만 호출
    settled_df = settle_predictions(LOG_NAME)

    macro_df = fetch_macro_data()
    report = "🤖 [스윙 · 5일 호라이즌]\n" + "=" * 40 + "\n"
    if event_lines:
        report += "📅 [1주일 내 이벤트]\n" + "\n".join(event_lines) + "\n" + "=" * 40 + "\n"

    new_rows = []
    for benchmark in BENCHMARK_TICKERS:
        result = train_benchmark_model(benchmark, macro_df, HORIZON_DAYS, LEVERAGED_THRESHOLD_CAP,
                                        min_train_rows=300, model_params=MODEL_PARAMS)
        if result is None:
            continue

        report += f"■ 벤치마크: {benchmark} (방향성: {direction_label(result['up_prob'], result['down_prob'])}, " \
                  f"보정후 상승 {result['up_prob']:.1f}% / 하락 {result['down_prob']:.1f}%, 원시확률 {result['raw_up_prob']:.1f}%)\n"
        report += f"   교차검증: F1 {result['f1']*100:.1f}% | P {result['prec']*100:.0f}% | R {result['rec']*100:.0f}% " \
                  f"| 횡보비중 {result['dead_zone']*100:.0f}%\n"

        for ticker, meta in LEVERAGE_UNIVERSE.items():
            if meta['benchmark'] != benchmark:
                continue
            metrics = fetch_display_metrics(ticker)
            if metrics is None:
                report += f"  📌 {ticker}: ⚠️ 시세 조회 실패 — 이번 회차 스킵\n"
                continue

            up_prob, down_prob = apply_direction(result['up_prob'], result['down_prob'], meta['direction'])
            raw_up, raw_down = apply_direction(result['raw_up_prob'], 100-result['raw_up_prob'], meta['direction'])

            acc = rolling_accuracy(settled_df, ticker, window=20)
            acc_str = f"최근 {acc['n']}회 적중률 {acc['accuracy']:.0f}%" if acc else "적중률 데이터 축적 중"

            report += f"  📌 {ticker} ({meta['desc']})\n"
            report += f"    * ⚠️ 3배 레버리지 — 수일 보유 시에도 decay 누적, 손절선 필수\n"
            report += f"    * 방향성: {direction_label(up_prob, down_prob)} (상승 {up_prob:.1f}% / 하락 {down_prob:.1f}%) " \
                      f"— {benchmark} 모델 기준({'그대로' if meta['direction']==1 else '반전'})\n"
            report += f"    * 📊 실전 적중률: {acc_str}\n"
            report += f"    * 현재가: ${metrics['price']:.2f} | 20일 변동성(연율): {metrics['vol']*100:.1f}%\n"

            new_rows.append(make_prediction_record(run_date, benchmark, ticker, HORIZON_DAYS,
                                                     up_prob, raw_up, metrics['price']))
        report += "-" * 40 + "\n"

    if new_rows:
        append_predictions(LOG_NAME, new_rows)

    print("\n" + report)
    send_telegram(report)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        err = f"🚨 [스윙] 에러:\n{traceback.format_exc()[:800]}"
        print(err)
        send_telegram(err)
