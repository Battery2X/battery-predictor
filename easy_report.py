"""
easy_report.py — 하루 1개짜리 '쉬운 요약 리포트' 생성기 (v25, 표시 전용)

예측 모델·학습·로그 기록은 전혀 건드리지 않는다. 6개 예측 스크립트가 남긴
reports/summary_*.json 과 hot_issues.json 을 읽어서 텔레그램용 메시지를 만든다.

원칙
- 매수/매도 추천 없음. "무엇을 봐야 하는지"와 "얼마나 믿을 수 있는지"만 보여준다.
- 외부 데이터(BTC, WSJ 헤드라인)는 실패하면 해당 섹션만 조용히 빠진다(리포트 전체는 유지).
- 텔레그램 parse_mode=HTML 이므로 외부 문자열은 html.escape 후 사용한다.
"""
import html
import json
import os
import re
import xml.etree.ElementTree as ET
import urllib.request
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from zoneinfo import ZoneInfo

KST = ZoneInfo('Asia/Seoul')
WEEKDAY_KO = ['월', '화', '수', '목', '금', '토', '일']
HELD_TICKERS = ['SOXL']            # 실제 보유 종목(표시 순서·강조용). 필요하면 수정.
HOT_ISSUES_PATH = 'hot_issues.json'
REPORT_DIR = 'reports'
SUMMARY_LOGS = ['scalp', 'swing', 'position', 'stock_scalp', 'stock_swing', 'stock_position']
LEVERAGED_LONGS = ['TQQQ', 'SOXL', 'SPXL']
BENCH_LABEL = {'QQQ': '나스닥100(QQQ)', 'SMH': '반도체(SMH)', 'SPY': 'S&P500(SPY)',
               'AAPL': 'AAPL', 'NVDA': 'NVDA', 'TSLA': 'TSLA'}
WSJ_MARKETS_RSS = 'https://feeds.content.dowjones.io/public/rss/RSSMarketsMain'
TELEGRAM_LIMIT = 3800   # 4096 한도보다 여유


# ------------------------------------------------------------------
# 1) 핫이슈 메모 (만료일이 지나면 자동으로 사라지고, 남은 일수를 보여준다)
# ------------------------------------------------------------------
def load_hot_issues(path=HOT_ISSUES_PATH, today=None, max_items=5):
    """
    hot_issues.json 형식:
      {"updated": "YYYY-MM-DD",
       "issues": [{"title": "...", "detail": "...", "until": "YYYY-MM-DD", "priority": 1}, ...]}
    파일이 없거나 깨졌거나 형식이 틀리면 ([], None) 을 반환 — 리포트는 그 섹션만 생략한다.
    반환: (활성 이슈 리스트[days_left 포함], 메모 갱신 후 경과일 또는 None)
    """
    today = pd.Timestamp(today or datetime.now(KST).date()).normalize()
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        raw = data.get('issues', [])
        updated = pd.Timestamp(data['updated']).normalize() if data.get('updated') else None
    except Exception:
        return [], None
    active = []
    for it in raw if isinstance(raw, list) else []:
        try:
            until = pd.Timestamp(it['until']).normalize()
            title = str(it['title']).strip()
            if not title or until < today:
                continue
            active.append({'title': title, 'detail': str(it.get('detail', '')).strip(),
                           'until': until, 'days_left': int((until - today).days),
                           'priority': int(it.get('priority', 9))})
        except Exception:
            continue          # 항목 하나가 깨져도 나머지는 계속
    active.sort(key=lambda x: (x['priority'], x['days_left']))
    age = int((today - updated).days) if updated is not None else None
    return active[:max_items], age


# ------------------------------------------------------------------
# 2) 앞으로의 일정 (MARKET_EVENTS + 종목 실적일)
# ------------------------------------------------------------------
def upcoming_events(market_events, earnings_dates, today=None, window=14, max_items=7):
    today = pd.Timestamp(today or datetime.now(KST).date()).normalize()
    rows = []
    for ds, info in market_events.items():
        try:
            d = pd.Timestamp(ds).normalize()
        except Exception:
            continue
        if 0 <= (d - today).days <= window:
            rows.append((d, info.get('desc', '')))
    for tk, ds in earnings_dates.items():
        try:
            d = pd.Timestamp(ds).normalize()
        except Exception:
            continue
        if 0 <= (d - today).days <= window:
            rows.append((d, f"{tk} 실적 발표 (미국 마감 후 · 한국시간 다음날 새벽)"))
    rows.sort(key=lambda x: x[0])
    return [{'date': d, 'days_left': int((d - today).days), 'desc': t} for d, t in rows[:max_items]]


# ------------------------------------------------------------------
# 3) 외부 참고 정보 (실패해도 조용히 생략)
# ------------------------------------------------------------------
def fetch_btc_context(downloader=None):
    """BTC 최근가·1일·7일 변화·30일 고점 대비. 모델 입력이 아닌 '참고 표시'. 실패 시 None."""
    try:
        if downloader is None:
            from common import _yf_download_with_retry
            start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            df = _yf_download_with_retry('BTC-USD', start, max_retries=2)
        else:
            df = downloader()
        if df is None or len(df) < 9:
            return None
        close = df['Close'].squeeze().dropna()
        if len(close) < 9:
            return None
        last = float(close.iloc[-1])
        return {'price': last,
                'ret1': last / float(close.iloc[-2]) - 1,
                'ret7': last / float(close.iloc[-8]) - 1,
                'from_high30': last / float(close.tail(30).max()) - 1}
    except Exception:
        return None


def fetch_news_headlines(max_items=3, timeout=8, opener=None):
    """
    WSJ(Dow Jones) 마켓 RSS 의 '제목'만 가져온다(요약·본문은 쓰지 않음, 개인 열람용 표시).
    네트워크 실패·형식 변경·빈 응답이면 [] — 리포트의 다른 부분엔 영향 없음.
    """
    try:
        if opener is None:
            req = urllib.request.Request(WSJ_MARKETS_RSS, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
        else:
            body = opener()
        root = ET.fromstring(body)
        titles = []
        for item in root.iter('item'):
            t = (item.findtext('title') or '').strip()
            if t:
                titles.append(t)
            if len(titles) >= max_items:
                break
        return titles
    except Exception:
        return []


# ------------------------------------------------------------------
# 4) 계산 보조: 회복 필요 상승률, 3배 ETF 변동성 잠식 추정
# ------------------------------------------------------------------
def recovery_needed(drawdown):
    """고점 대비 낙폭(음수)에서 본전까지 필요한 상승률. 예: -0.35 -> 0.538"""
    if drawdown is None or pd.isna(drawdown) or drawdown >= 0 or drawdown <= -0.999:
        return None
    return 1.0 / (1.0 + drawdown) - 1.0


def decay_estimate_monthly(vol_annual, leverage=3, trading_days=21):
    """
    일간 리밸런싱 레버리지 ETF의 '변동성 잠식' 추정(한 달).
    기초지수 변동성 s = vol/L 일 때, L배 ETF의 로그수익률은 L×(지수 로그수익률) 보다
    평균적으로 (L^2 - L)/2 × s^2 × t 만큼 낮다. (지수가 제자리여도 이만큼 깎이는 경향)
    변동성이 앞으로도 같다는 가정의 대략치일 뿐, 예측이 아니다.
    """
    if vol_annual is None or pd.isna(vol_annual) or vol_annual <= 0:
        return None
    s = vol_annual / leverage
    return (leverage ** 2 - leverage) / 2 * s ** 2 * (trading_days / 252)


# ------------------------------------------------------------------
# 5) 요약 데이터 읽기
# ------------------------------------------------------------------
def load_summaries(report_dir=REPORT_DIR, now=None, max_age_hours=20):
    """reports/summary_*.json 6개를 읽는다. 없거나 오래된 파일은 stale 목록에 넣는다."""
    now = now or datetime.now()
    out, stale = {}, []
    for name in SUMMARY_LOGS:
        path = os.path.join(report_dir, f'summary_{name}.json')
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            gen = datetime.fromisoformat(data['generated_at'])
            if (now - gen) > timedelta(hours=max_age_hours) or not data.get('items'):
                stale.append(name)
                if not data.get('items'):
                    continue
            out[name] = data
        except Exception:
            stale.append(name)
    return out, stale


def _series_items(summaries):
    """검증 대상 '독립 시리즈'만 추린다: 레버리지는 롱(direction=+1)만(인버스는 거울), 종목은 전부."""
    rows = []
    for name, data in summaries.items():
        h = data.get('horizon_days')
        for it in data.get('items', []):
            if name.startswith('stock') or it.get('direction') == 1:
                rows.append({'log': name, 'horizon': h, **it})
    return rows


# ------------------------------------------------------------------
# 6) 오늘의 한 줄 공부 (퀴즈에서 틀린 개념 위주로 돌려가며)
# ------------------------------------------------------------------
def lesson_of_the_day(today, board_items):
    lessons = []
    lessons.append("겹치는 표본: 20일 예측은 어제·오늘 예측이 거의 같은 20일 구간을 공유합니다. "
                   "20번 연속 맞혀도 '독립적으로 1~2번 맞힌' 정도로 봐야 해서, 이 리포트는 "
                   "독립 관측 수(n)로 판정합니다.")
    lessons.append("변동성 잠식: 지수가 +10% 후 -10%면 2일 누적 -1%지만, 3배 ETF는 "
                   "+30% 후 -30% = 1.3×0.7 = 0.91, 즉 -9%입니다. 출렁일수록 3배 ETF가 더 깎입니다.")
    lessons.append("물가와 주가: CPI가 예상보다 높으면 → 연준이 금리를 오래 높게 둘(또는 올릴) 거란 기대가 커짐 → "
                   "먼 미래 이익의 현재가치가 줄어 성장주·반도체가 더 크게 흔들립니다. "
                   "그래서 CPI·FOMC 전후엔 변동성이 커지는 경향이 있습니다.")
    lessons.append("롱/인버스는 정반대가 아닙니다: SOXL·SOXS는 '하루' 수익률만 ±3배입니다. "
                   "기간이 길고 출렁이면 둘 다 내릴 수 있습니다(변동성 잠식).")
    rec = None
    for it in board_items:
        if it.get('ticker') in HELD_TICKERS or rec is None:
            r = recovery_needed(it.get('drawdown'))
            if r is not None:
                rec = (it.get('ticker'), it.get('drawdown'), r)
    if rec:
        lessons.append(f"본전 계산: {rec[0]}은 고점 대비 {rec[1]*100:.0f}% → 본전까지 +{rec[2]*100:.0f}% 필요합니다. "
                       f"낙폭이 커질수록 회복에 필요한 상승률은 더 빨리 커집니다(-50%면 +100%).")
    else:
        lessons.append("본전 계산: -35% 하락하면 본전까지 +54%, -50%면 +100%가 필요합니다. "
                       "낙폭이 클수록 회복에 필요한 상승률은 훨씬 커집니다.")
    return lessons[pd.Timestamp(today).toordinal() % len(lessons)]


# ------------------------------------------------------------------
# 7) 메시지 조립
# ------------------------------------------------------------------
def _e(x):
    return html.escape(str(x), quote=False)


def _pct(x, digits=0, sign=False):
    if x is None or pd.isna(x):
        return '-'
    return f"{x*100:+.{digits}f}%" if sign else f"{x*100:.{digits}f}%"


def _edge_txt(edge):
    """엣지 표시. 반올림해서 0이면 '-0' 대신 '0'."""
    r = round(float(edge))
    return f"{0 if r == 0 else r:+d}" if r != 0 else "0"


def _fmt_date(d):
    d = pd.Timestamp(d)
    return f"{d.month}/{d.day}({WEEKDAY_KO[d.weekday()]})"


def _target_phrase(price_date, today_kst):
    nxt = pd.Timestamp(price_date) + BDay(1)
    if nxt.normalize() == pd.Timestamp(today_kst).normalize():
        return "오늘 밤 미국장(1일) · 5거래일 후 · 20거래일 후"
    return f"{_fmt_date(nxt)} 미국장(1일) · 5거래일 후 · 20거래일 후"


STATUS_ICON = {'pass': '✅', 'fail': '❌', 'insufficient': '⏳', None: '·'}


def build_sections(summaries, stale, hot, hot_age, events, btc, headlines, now_kst, risk=None):
    """텔레그램에 보낼 섹션 문자열 리스트(각각 독립 블록)를 만든다.
    risk: {티커: risk_model.risk_snapshot 결과} (없거나 일부만 있어도 됨)"""
    sections = []
    series = _series_items(summaries)

    # ---- 헤더 ----
    pdates = [pd.Timestamp(s['price_date']) for s in series if s.get('price_date')]
    head = f"📊 <b>{_fmt_date(now_kst)} 브리핑</b>"
    if pdates:
        pdate = max(pdates)
        head += f"\n기준: {_fmt_date(pdate)} 미국 종가 → 예측 대상: {_target_phrase(pdate, now_kst.date())}"
    if stale:
        head += f"\n⚠️ 오늘 갱신 안 된 항목: {', '.join(stale)} (이전 데이터일 수 있음)"
    sections.append(head)

    # ---- 1. 결론 ----
    status = [(s.get('signal') or {}).get('status') for s in series]
    n_total = len(series)
    n_pass = status.count('pass')
    n_fail = status.count('fail')
    n_wait = status.count('insufficient') + status.count(None)
    if n_total:
        if n_pass == 0:
            concl = (f"🔴 <b>1. 오늘의 결론</b>\n신호 신뢰도: 낮음 — 검증 통과 {n_pass}/{n_total}\n"
                     f"(표본 부족 ⏳{n_wait} · 통과 실패 ❌{n_fail})\n"
                     "쉬운 말: 모델이 '오른다'고 해도 \"평소 오르던 비율\"과 비슷하거나, 맞았는지 확인할 "
                     "표본이 아직 모자랍니다. 방향 판단의 근거로 쓰기엔 이릅니다.")
        else:
            names = [f"{s['ticker']}·{s['horizon']}일" for s, st in zip(series, status) if st == 'pass']
            concl = (f"🟡 <b>1. 오늘의 결론</b>\n검증 통과 {n_pass}/{n_total}: {', '.join(names)}\n"
                     f"(표본 부족 ⏳{n_wait} · 통과 실패 ❌{n_fail}) — 통과도 '지금까지' 기준이며 "
                     "여러 개를 동시에 보면 우연히 통과하는 경우가 섞입니다.")
        sections.append(concl)

    # ---- 2. 위험 상황판 ----
    board = [s for s in series if s['log'] == 'scalp' and s['ticker'] in LEVERAGED_LONGS]
    board.sort(key=lambda s: (s['ticker'] not in HELD_TICKERS, s['ticker']))
    if board:
        lines = ["⚠️ <b>2. 내 위험 상황판</b> (예측이 아니라 '지금 상태')"]
        for s in board:
            tag = " (보유)" if s['ticker'] in HELD_TICKERS else ""
            lines.append(f"<b>{s['ticker']}{tag}</b> ${s['price']:.2f} · 5일 {_pct(s.get('ret5'), 0, True)}"
                         f" · 고점대비 {_pct(s.get('drawdown'), 1, True)}")
            r = recovery_needed(s.get('drawdown'))
            d = decay_estimate_monthly(s.get('vol'))
            sub = []
            if r is not None and r > 0.05:
                sub.append(f"본전까지 +{r*100:.0f}% 필요")
            if s.get('vol') is not None and not pd.isna(s.get('vol')):
                txt = f"변동성 {s['vol']*100:.0f}%"
                if d is not None:
                    txt += f" → 지수가 제자리여도 한 달 약 -{d*100:.1f}% 깎이는 구조(추정)"
                sub.append(txt)
            if sub:
                lines.append("  └ " + " · ".join(sub))
            rk = (risk or {}).get(s['ticker'])
            _fin = lambda v: v is not None and not pd.isna(v) and np.isfinite(v)
            if rk and _fin(rk.get('move_1d')) and _fin(rk.get('move_5d')):
                line = (f"  └ 예상 흔들림(1σ): 하루 ±{rk['move_1d']*100:.1f}% · 5일 ±{rk['move_5d']*100:.1f}%"
                        " (3번 중 2번은 이 안, 방향 예측 아님)")
                if rk.get('regime'):
                    line += f"\n  └ 지금 상태: {_e(rk['regime'])}"
                if rk.get('data_suspect'):
                    alt = rk.get('move_5d_incl_suspect')
                    line += (f"\n  └ ⚠️ 데이터 이상 의심: 최근 1년에 하루 +100%/-50% 이상 움직임 {int(rk['data_suspect'])}건"
                             " — 분할 미조정 가능성이라 위 흔들림 계산에서는 뺐습니다"
                             + (f" (포함하면 5일 ±{alt*100:.1f}%, 실제 폭락·폭등이었다면 이쪽이 맞음)" if _fin(alt) else ""))
                lines.append(line)
        if btc:
            lines.append(f"참고 BTC ${btc['price']:,.0f} · 1일 {_pct(btc['ret1'],1,True)} · 7일 {_pct(btc['ret7'],1,True)}"
                         f" · 30일고점대비 {_pct(btc['from_high30'],1,True)} (주식 선행 여부는 아직 검증 전)")
        sections.append("\n".join(lines))

    # ---- 3. 일정 ----
    if events:
        lines = ["📅 <b>3. 앞으로 14일 일정</b>"]
        for ev in events:
            dd = "오늘" if ev['days_left'] == 0 else f"D-{ev['days_left']}"
            lines.append(f"{dd} {_fmt_date(ev['date'])} {_e(ev['desc'])}")
        sections.append("\n".join(lines))

    # ---- 4. 핫이슈 ----
    if hot:
        lines = ["🔥 <b>4. 핫이슈 (유지 중)</b>"]
        if hot_age is not None and hot_age >= 4:
            lines.append(f"⚠️ 메모가 {hot_age}일 전 갱신본 — 최신이 아닐 수 있음")
        for it in hot:
            tail = f" [~{_fmt_date(it['until'])}, D-{it['days_left']}]"
            lines.append(f"• <b>{_e(it['title'])}</b>{tail}")
            if it['detail']:
                lines.append(f"  {_e(it['detail'])}")
        sections.append("\n".join(lines))

    # ---- 5. 헤드라인 ----
    if headlines:
        lines = ["📰 <b>5. WSJ 마켓 헤드라인</b>"] + [f"• {_e(t)}" for t in headlines]
        sections.append("\n".join(lines))

    # ---- 6. 방향 신호 (참고) ----
    by_bench = {}
    for s in series:
        by_bench.setdefault(s['benchmark'], {})[s['horizon']] = s
    if by_bench:
        lines = ["🧭 <b>6. 방향 신호 (참고용 · 검증 전)</b>",
                 "상승확률 1일 / 5일 / 20일 (괄호: 평소 대비 차이, ±3%p 안이면 '평소와 같음')"]
        for b in ['QQQ', 'SMH', 'SPY', 'AAPL', 'NVDA', 'TSLA']:
            hs = by_bench.get(b)
            if not hs:
                continue
            cells, icons = [], []
            for h in (1, 5, 20):
                s = hs.get(h)
                if s is None:
                    cells.append('-')
                    icons.append('·')
                else:
                    cells.append(f"{s['up_prob']:.0f}%({_edge_txt(s['edge'])})")
                    icons.append(STATUS_ICON[(s.get('signal') or {}).get('status')])
            lines.append(f"{_e(BENCH_LABEL[b])}: {' / '.join(cells)}  검증 {''.join(icons)}")
            sm = next((v for v in hs.values() if v.get('breadth')), None)
            if sm:
                br = sm['breadth']
                lines.append(f"  └ 반도체 구성종목 상승비율 {br['up_share']*100:.0f}% · 5일 모멘텀 "
                             f"{br['momentum5']*100:+.1f}%")
        lines.append("검증: ✅통과 ❌못함 ⏳표본부족(독립 관측 15건 미만)")
        sections.append("\n".join(lines))

    # ---- 7. 한 줄 공부 ----
    sections.append("🎓 <b>오늘의 한 줄 공부</b>\n" + _e(lesson_of_the_day(now_kst.date(), board)))
    # 일부 섹션(뉴스·일정 등)이 조용히 생략돼도 번호가 건너뛰지 않도록 1부터 다시 매긴다
    counter = 0
    for i, sec in enumerate(sections):
        if re.search(r'<b>\d+\. ', sec):
            counter += 1
            sections[i] = re.sub(r'<b>\d+\. ', f'<b>{counter}. ', sec, count=1)
    return sections


def chunk_sections(sections, limit=TELEGRAM_LIMIT):
    """섹션 경계에서만 자르며 텔레그램 길이 한도 안으로 나눈다."""
    msgs, cur = [], ''
    for sec in sections:
        add = sec if not cur else '\n\n' + sec
        if cur and len(cur) + len(add) > limit:
            msgs.append(cur)
            cur = sec
        else:
            cur += add
    if cur:
        msgs.append(cur)
    # 한 섹션이 한도를 넘는 극단적 경우: 강제 분할(줄 단위)
    out = []
    for m in msgs:
        while len(m) > limit:
            cut = m.rfind('\n', 0, limit)
            cut = cut if cut > 0 else limit
            out.append(m[:cut])
            m = m[cut:].lstrip('\n')
        out.append(m)
    return _balance_tags(out)


def _balance_tags(chunks):
    """강제 분할이 <b>/<i>/<code> 태그 중간을 잘라도 각 조각이 독립적으로 유효한 HTML이 되도록 닫고/다시 연다."""
    out, carry = [], []
    for part in chunks:
        stack = list(carry)
        for mt in re.finditer(r'<(/?)(b|i|u|s|code|pre)>', part):
            if mt.group(1):
                if stack and stack[-1] == mt.group(2):
                    stack.pop()
            else:
                stack.append(mt.group(2))
        out.append(''.join(f'<{t}>' for t in carry) + part + ''.join(f'</{t}>' for t in reversed(stack)))
        carry = stack
    return out
