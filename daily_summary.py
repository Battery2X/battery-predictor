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


def run():
    now_kst = datetime.now(KST)
    summaries, stale = load_summaries()
    hot, hot_age = load_hot_issues()
    events = upcoming_events(MARKET_EVENTS, STOCK_EARNINGS_DATES, today=now_kst.date(), window=14)
    btc = fetch_btc_context()
    headlines = fetch_news_headlines()

    sections = build_sections(summaries, stale, hot, hot_age, events, btc, headlines, now_kst)
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
