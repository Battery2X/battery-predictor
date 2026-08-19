"""
predict_swing.py — 스윙(1~2주 방향성) 예측 (v23.0)
호라이즌: 5거래일
"""
import traceback
import pandas as pd

from common import (
    LEVERAGE_UNIVERSE, BENCHMARK_TICKERS,
    fetch_macro_data, fetch_display_metrics, train_benchmark_model, fetch_constituent_breadth_features,
    apply_direction, check_upcoming_events, send_telegram, direction_label,
    settle_predictions, append_predictions, rolling_accuracy, make_prediction_record, edge_label,
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
    breadth_df = fetch_constituent_breadth_features()  # SOXX 상위10 종목 체감폭/쏠림도 — SMH 모델 전용 피처
    report = "🤖 [스윙 · 5일 호라이즌]\n" + "=" * 40 + "\n"
    if event_lines:
        report += "📅 [1주일 내 이벤트]\n" + "\n".join(event_lines) + "\n" + "=" * 40 + "\n"

    new_rows = []
    for benchmark in BENCHMARK_TICKERS:
        if benchmark == 'SMH' and not breadth_df.empty:
            # FEATURE_COLUMNS에 ConstMomentum5 등이 이미 포함돼 있으므로
            # extra_features_df만 join하면 됨 (extra_feature_cols를 또 넘기면 컬럼 중복 버그 발생)
            result = train_benchmark_model(benchmark, macro_df, HORIZON_DAYS, LEVERAGED_THRESHOLD_CAP,
                                            min_train_rows=300, model_params=MODEL_PARAMS,
                                            extra_features_df=breadth_df)
        else:
            result = train_benchmark_model(benchmark, macro_df, HORIZON_DAYS, LEVERAGED_THRESHOLD_CAP,
                                            min_train_rows=300, model_params=MODEL_PARAMS)
        if result is None:
            report += f"■ 벤치마크: {benchmark} — ❌ 데이터 조회 실패(야후 파이낸스 응답 없음/rate limit 의심) — 이번 회차 스킵\n" + "-"*40 + "\n"
            continue

        report += f"■ 벤치마크: {benchmark} (방향성: {direction_label(result['up_prob'], result['down_prob'])}, " \
                  f"보정후 상승 {result['up_prob']:.1f}% / 하락 {result['down_prob']:.1f}%, 원시확률 {result['raw_up_prob']:.1f}%)\n"
        report += f"   교차검증: F1 {result['f1']*100:.1f}% | P {result['prec']*100:.0f}% | R {result['rec']*100:.0f}% " \
                  f"| 횡보비중 {result['dead_zone']*100:.0f}%\n"
        report += f"   기준선(과거 상승비율): {result['base_rate']:.1f}% | 모델 엣지: {result['edge']:+.1f}%p " \
                  f"({edge_label(result['edge'])})\n"
        top_feat_str = ", ".join([f"{name}({imp*100:.0f}%)" for name, imp in result['top_features']])
        report += f"   🔍 상위 5개 피처: {top_feat_str}\n"
        if benchmark == 'SMH' and not breadth_df.empty:
            last = breadth_df.iloc[-1]
            report += f"   📦 SOXX 상위10 구성종목 폭: 상승비율 {last['ConstBreadth5']*100:.0f}% | " \
                      f"가중모멘텀(5일) {last['ConstMomentum5']*100:+.1f}% | 쏠림도 {last['ConstDispersion5']*100:.1f}%p\n"

        for ticker, meta in LEVERAGE_UNIVERSE.items():
            if meta['benchmark'] != benchmark:
                continue
            metrics = fetch_display_metrics(ticker)
            if metrics is None:
                report += f"  📌 {ticker}: ⚠️ 시세 조회 실패 — 이번 회차 스킵\n"
                continue

            up_prob, down_prob = apply_direction(result['up_prob'], result['down_prob'], meta['direction'])
            raw_up, raw_down = apply_direction(result['raw_up_prob'], 100-result['raw_up_prob'], meta['direction'])

            acc = rolling_accuracy(settled_df, ticker, window=20, baseline=result['base_rate'])
            if acc:
                sig = "✅ 기준선보다 유의미하게 우수" if acc.get('beats_baseline') else "⚠️ 아직 통계적으로 유의미한 우위 아님"
                acc_str = f"최근 {acc['n']}회 적중률 {acc['accuracy']:.0f}% (95% CI {acc['ci_low']:.0f}~{acc['ci_high']:.0f}%) — {sig}"
            else:
                acc_str = "적중률 데이터 축적 중"

            report += f"  📌 {ticker} ({meta['desc']})\n"
            report += f"    * ⚠️ 3배 레버리지 — 수일 보유 시에도 decay 누적, 손절선 필수\n"
            report += f"    * 방향성: {direction_label(up_prob, down_prob)} (상승 {up_prob:.1f}% / 하락 {down_prob:.1f}%) " \
                      f"— {benchmark} 모델 기준({'그대로' if meta['direction']==1 else '반전'})\n"
            report += f"    * {edge_label(result['edge'])} (엣지 {result['edge']:+.1f}%p) — 확률이 높아도 엣지가 약하면 사실상 과거 평균 수준\n"
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
