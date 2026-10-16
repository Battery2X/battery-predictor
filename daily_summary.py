"""
daily_summary.py — 6개 예측 스크립트가 끝난 뒤 실행되어 '쉬운 요약 리포트' 1건을 텔레그램으로 보낸다.

- 예측 모델·로그는 읽기만 한다(수정 없음). 검증 시계에 영향 없음.
- BTC / WSJ 헤드라인 / 핫이슈 중 일부가 실패해도 나머지 섹션은 정상 출력한다.
- 상세 리포트는 reports/detail_*.txt 에 저장돼 있다. 텔레그램으로 상세도 받으려면
  워크플로우 env 에 SEND_DETAIL=1 을 추가한다.
"""
import traceback
from datetime import datetime

from common import MARKET_EVENTS, STOCK_EARNINGS_DATES, send_telegram
from easy_report import (
    KST, load_summaries, load_hot_issues, upcoming_events, fetch_btc_context,
    fetch_news_headlines, build_sections, chunk_sections,
)


def collect_risk(tickers, now_kst, log_dir=None, report_dir=None, fetcher=None):
    """
    시세를 받아 (1) 리포트용 위험 스냅샷을 만들고 (2) 변동성 예측 로그를 정산·추가·저장한다(v26).
    방향 예측 로그(predictions_*.csv)는 읽지도 쓰지도 않는다. 어떤 단계가 실패해도 {} 또는 일부만 돌려주고 끝낸다.
    """
    risk = {}
    try:
        import os
        import risk_model as rm
        import common
        log_dir = log_dir or common.LOG_DIR
        report_dir = report_dir or common.REPORT_DIR
        fetch = fetcher or common.fetch_ticker_data
        cache = {}

        def lookup(t):
            if t not in cache:
                df = fetch(t)
                cache[t] = None if df is None or len(df) == 0 else df['Close'].ffill().dropna()
            return cache[t]

        log = rm.load_vol_log(log_dir)
        log = rm.settle_vol_log(log, lookup)
        for t in tickers:
            try:
                close = lookup(t)
                if close is None:
                    continue
                snap = rm.risk_snapshot(close)
                if snap:
                    risk[t] = snap
                log = rm.append_vol_forecasts(log, now_kst.date(), t, close)
            except Exception as e:
                print(f"⚠️ [{t}] 위험 스냅샷 실패(무시): {e}")
        rm.save_vol_log(log, log_dir)
        score = rm.score_vol_log(log)
        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, "vol_scorecard.txt"), "w", encoding="utf-8") as f:
            if score.empty:
                f.write("변동성 예측 성적표: 아직 정산된 행이 없습니다(예측 후 호라이즌 일수가 지나야 채점됩니다).\n")
            else:
                f.write("변동성 예측 성적표 (QLIKE 낮을수록 좋음 · bias 1.0이 이상적 · n_eff<80이면 아직 못 믿음)\n")
                f.write(score.to_string(index=False, float_format=lambda x: f"{x:.3f}") + "\n")
    except Exception as e:
        print(f"⚠️ 위험 모델 단계 실패(무시, 나머지 리포트는 계속): {e}")
    return risk


def run():
    now_kst = datetime.now(KST)
    summaries, stale = load_summaries()
    hot, hot_age = load_hot_issues()
    events = upcoming_events(MARKET_EVENTS, STOCK_EARNINGS_DATES, today=now_kst.date(), window=14)
    btc = fetch_btc_context()
    headlines = fetch_news_headlines()

    from easy_report import LEVERAGED_LONGS
    risk = collect_risk(LEVERAGED_LONGS, now_kst)

    sections = build_sections(summaries, stale, hot, hot_age, events, btc, headlines, now_kst, risk=risk)
    messages = chunk_sections(sections)
    for m in messages:
        print(m + "\n" + "=" * 40)
        send_telegram(m)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        err = f"🚨 [요약 리포트] 에러:\n{traceback.format_exc()[:800]}"
        print(err)
        send_telegram(err)
