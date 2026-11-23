"""
daily_summary.py — 6개 예측 스크립트가 끝난 뒤 실행되어 '쉬운 요약 리포트' 1건을 텔레그램으로 보낸다.

- 예측 모델·로그는 읽기만 한다(수정 없음). 검증 시계에 영향 없음.
- BTC / WSJ 헤드라인 / 핫이슈 중 일부가 실패해도 나머지 섹션은 정상 출력한다.
- 상세 리포트는 reports/detail_*.txt 에 저장돼 있다. 텔레그램으로 상세도 받으려면
  워크플로우 env 에 SEND_DETAIL=1 을 추가한다.
"""
import os
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
        try:      # v33: 리포트의 '예상 흔들림' 옆에 실전 과소/과대예측 정도를 같이 보여주기 위한 요약(표시 전용)
            vb = {}
            for _, row in score[score['candidate'] == 'vol_ewma'].iterrows():
                bv = float(row['bias_var'])
                if bv > 0 and bv == bv:
                    vb[int(row['horizon'])] = {'ratio': bv ** 0.5, 'n_eff': float(row['n_eff']), 'n': int(row['n'])}
            if vb and risk:      # 종목 스냅샷이 하나도 없으면 기존처럼 {} 유지
                risk['_vol_bias'] = vb
        except Exception:
            pass
        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, "vol_scorecard.txt"), "w", encoding="utf-8") as f:
            if score.empty:
                f.write("변동성 예측 성적표: 아직 정산된 행이 없습니다(예측 후 호라이즌 일수가 지나야 채점됩니다).\n")
            else:
                f.write("변동성 예측 성적표 (QLIKE 낮을수록 좋음 · bias_var(분산단위 실현²합/예측²합) 1.0이 이상적·>1이면 과소예측 · bias는 구 지표(해석 주의) · vol_har/vol_gjr/vol_ewmaf는 그림자 후보 · n_eff<80이면 아직 못 믿음)\n")
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

    try:
        from easy_report import calendar_staleness
        cw = calendar_staleness(MARKET_EVENTS, STOCK_EARNINGS_DATES, today=now_kst.date())
        if cw:
            risk['_calendar_warn'] = cw
    except Exception as e:
        print(f"⚠️ 달력 점검 실패(무시): {e}")

    try:      # v35: 시장 맥락(공포탐욕지수 · Alpha Vantage 교차확인 · 엣지/맥락 로그). 모델 입력 아님, 실패해도 무시
        import common as _cm
        import market_context as _mc
        ctx = _mc.collect(summaries, _cm.LOG_DIR, earnings_manual=STOCK_EARNINGS_DATES, today=now_kst.date(), report_dir=_cm.REPORT_DIR)
        if isinstance(risk, dict):
            risk['_context'] = ctx
            if ctx.get('earnings_notes'):
                risk['_calendar_warn'] = list(risk.get('_calendar_warn') or []) + list(ctx['earnings_notes'])
    except Exception as e:
        print(f"⚠️ 시장 맥락 수집 실패(무시): {e}")

    try:      # v36: ETF 속 들여다보기(구성종목 기여도 · 실적 노출 · 재무). 모델 입력 아님, 실패해도 무시
        import common as _cm
        import etf_drivers as _ed
        if isinstance(risk, dict):
            risk['_drivers'] = _ed.collect(_cm.LOG_DIR, _cm.REPORT_DIR, today=now_kst.date())
    except Exception as e:
        print(f"⚠️ ETF 드라이버 수집 실패(무시): {e}")

    try:      # v34: 방향 성적표(항상상승 기준선·Brier·독립 구간 수)를 reports/ 에 저장. 로그는 읽기만 함(실패해도 무시)
        import common as _cm
        import direction_scorecard as _ds
        _ds.run(_cm.LOG_DIR, _cm.REPORT_DIR)
    except Exception as e:
        print(f"⚠️ 방향 성적표 저장 실패(무시): {e}")

    sections = build_sections(summaries, stale, hot, hot_age, events, btc, headlines, now_kst, risk=risk)
    if os.environ.get('REPORT_STYLE') != 'classic':      # v37: 한눈 요약 + 이벤트 카드 + 접어 두기 + 헤드라인 번역 + 그림 한 장(실패하면 기존 형식)
        try:
            import common as _cm
            import report_layout as _rl
            png = _rl.dashboard_png(summaries, risk, now_kst)
            if png:      # 저장소에는 넣지 않는다(매일 130KB 이진 파일이 쌓이면 저장소가 불어남) — 텔레그램으로만 보냄
                _rl.send_photo(png, caption=f"{now_kst.month}/{now_kst.day} 위험 상황 그림 (낙폭 · 공포탐욕 · 반도체지수 기여 · 실적 일정)")
            sections = _rl.restyle(sections, summaries, risk, now_kst, headlines=headlines, stale=stale)
        except Exception as e:
            print(f"⚠️ 리포트 재배치 단계 실패(기존 형식으로 전송): {e}")
    try:      # v39: 확장 종목 규칙 상태(표시 전용, 실패하면 조용히 생략)
        import universe_tracker as _ut
        _sec = _ut.status_section()
        if _sec:
            sections.append(_sec)
        _xs = os.path.join('reports', 'xs_status.txt')   # v40: 교차종목 예측(표시 전용, predict.yml 이 앞 단계에서 작성)
        if os.path.exists(_xs):
            _t = open(_xs, encoding='utf-8').read().strip()
            if _t:
                sections.append(_t)
    except Exception as e:
        print(f"⚠️ 규칙 상태 섹션 생략(무시): {e}")
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
