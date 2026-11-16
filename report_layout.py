"""
report_layout.py — 요약 리포트 다시 짜기 (v37): 한눈 요약 · 이벤트 카드 · 접어 두기 · 헤드라인 번역 · 그림 한 장

왜: 기존 리포트는 8개 섹션 80줄이 한 덩어리로 와서 "오늘 뭘 봐야 하는지"가 안 보였다.
  1) 맨 위 '한눈 요약' 6줄 안팎: 신호를 쓸 수 있나 / 내 보유 상태 / 심리 / 어제 무엇이 움직였나 / 임박 이벤트 / 데이터 이상
  2) '이벤트 카드': 비중 큰 종목의 실적 같은 큰 일정이 3거래일 안에 있으면, 내 ETF 로 환산한 크기와 확인 순서를 한 장으로
  3) 나머지 상세 섹션은 텔레그램 '펼치기 인용'으로 접는다(눌러야 펼쳐짐) → 화면 한 장에 요약이 다 들어옴
  4) WSJ 헤드라인은 한국어로 번역해 짧게
  5) 그림 한 장(PNG): 낙폭·예상 흔들림 / 공포탐욕 추이 / 반도체지수 기여 종목 / 실적 일정 — 글자는 영문·숫자만(한글 글꼴 의존 없음)
원칙: 표시만 바꾼다. 예측·로그·모델과 무관. 어느 단계가 실패해도 기존 리포트(build_sections 결과)를 그대로 보낸다.
매수/매도 판단은 넣지 않는다. 이벤트 카드는 '크기 환산'과 '미리 정해둘 질문'만 보여준다.
"""
import html
import io
import json
import os
import re
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

LONGS = ("SOXL", "TQQQ", "SPXL")
HELD = ("SOXL",)
INDEX_OF = {"SOXL": "SOXX", "TQQQ": "QQQ", "SPXL": "SPY"}
INDEX_NAME = {"SOXX": "반도체지수", "QQQ": "나스닥100", "SPY": "S&P500"}
EVENT_MIN_WEIGHT = 0.04        # 어느 한 지수에서 비중 4% 이상인 종목의 실적만 카드로
EVENT_WINDOW_BDAYS = 3
COLLAPSE_MAX = 3000            # 이보다 긴 섹션은 접지 않는다(강제 분할 시 태그가 깨질 수 있어서)
TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"
_E = lambda x: html.escape(str(x), quote=False)


def _fin(v):
    try:
        return v is not None and np.isfinite(float(v))
    except Exception:
        return False


def _md(d):
    d = pd.Timestamp(d)
    return f"{d.month}/{d.day}({'월화수목금토일'[d.weekday()]})"


def _scalp_items(summaries):
    try:
        return {it["ticker"]: it for it in ((summaries or {}).get("scalp") or {}).get("items", [])}
    except Exception:
        return {}


def _signal_counts(summaries):
    """(통과, 전체) — 독립 계열만(레버리지 롱 3 × 호라이즌 3 + 개별주)."""
    n_pass = n_tot = 0
    for name, data in (summaries or {}).items():
        for it in data.get("items", []):
            if it.get("direction") == -1:
                continue
            n_tot += 1
            if ((it.get("signal") or {}).get("status")) == "pass":
                n_pass += 1
    return n_pass, n_tot


# ------------------------------------------------------------------ 1) 임박 이벤트 찾기
def find_event(drivers, today, window=EVENT_WINDOW_BDAYS, min_weight=EVENT_MIN_WEIGHT):
    """
    3거래일 안에 실적을 내는 종목 중 어느 지수에서든 비중이 가장 큰 것 하나.
    반환 {'symbol','date','weights':{지수:비중},'comove':{지수:{beta,direct,n}}} 또는 None.
    """
    try:
        td = pd.Timestamp(today).normalize()
        end = td + pd.offsets.BDay(window)
        best = {}
        for idx, r in ((drivers or {}).get("indexes") or {}).items():
            for sym, d, w in ((r or {}).get("earn") or {}).get("names", []):
                dd = pd.Timestamp(d).normalize()
                if not (td <= dd <= end):
                    continue
                e = best.setdefault(sym, {"symbol": sym, "date": dd, "weights": {}, "comove": {}})
                e["weights"][idx] = float(w)
                cm = ((r or {}).get("comove") or {}).get(sym)
                if cm:
                    e["comove"][idx] = cm
        best = [e for e in best.values() if max(e["weights"].values()) >= min_weight]
        if not best:
            return None
        best.sort(key=lambda e: (e["date"], -max(e["weights"].values())))
        return best[0]
    except Exception:
        return None


def event_card(event, summaries, risk, today):
    """이벤트 카드 문자열(HTML) 또는 None. 방향 의견 없음 — 크기 환산과 확인 순서만."""
    try:
        if not event:
            return None
        sym, dd = event["symbol"], pd.Timestamp(event["date"])
        dday = int(np.busday_count(pd.Timestamp(today).date(), dd.date()))
        when = "오늘" if dday <= 0 else f"D-{dday}"
        lines = [f"🎯 <b>이벤트 카드 — {_E(sym)} 실적 {_md(dd)} · {when}</b>",
                 "미국 장 마감 후 발표면 한국시간 다음날 새벽에 결과가 나옵니다."]
        ws = " · ".join(f"{INDEX_NAME.get(i, i)} {w * 100:.1f}%" for i, w in sorted(event["weights"].items(), key=lambda kv: -kv[1]))
        lines.append(f"<b>① 얼마나 걸려 있나</b>  {_E(ws)}")
        conv = []
        for etf in LONGS:
            idx = INDEX_OF[etf]
            w = event["weights"].get(idx)
            if w is None:
                continue
            cm = (event.get("comove") or {}).get(idx)
            direct = 10 * w * 3
            if cm and _fin(cm.get("beta")) and cm["beta"] > 0:
                conv.append(f"{etf} ±{direct:.1f}%(직접) ~ ±{10 * cm['beta'] * 3:.0f}%(같이 움직인 과거 평균)")
            else:
                conv.append(f"{etf} ±{direct:.1f}%(직접 효과만)")
        if conv:
            lines.append(f"<b>② {_E(sym)}가 ±10% 움직이면</b>")
            lines += [f"  • {_E(c)}" for c in conv]
            lines.append("  '직접'은 비중×3배만 계산한 최소치, '같이 움직인 평균'은 최근 2달 동안 다른 종목까지 함께 움직인 폭입니다(인과 아님).")
        items = _scalp_items(summaries)
        mv = []
        for etf in LONGS:
            rk = (risk or {}).get(etf) if isinstance(risk, dict) else None
            if rk and _fin(rk.get("move_1d")):
                mv.append(f"{etf} ±{rk['move_1d'] * 100:.0f}%")
        if mv:
            lines.append(f"<b>③ 평소 하루 흔들림(1σ)</b>  {_E(' · '.join(mv))} — 이벤트 날은 이보다 클 수 있음")
        n_pass, n_tot = _signal_counts(summaries)
        lines.append(f"<b>④ 시스템이 아는 것</b>  방향은 모릅니다(검증 통과 {n_pass}/{n_tot}). 크기와 일정만 압니다.")
        lines.append("<b>⑤ 발표 전에 내가 정해둘 것</b>  감당할 하루 손실폭 · 발표를 안고 갈 비중 · 결과를 다시 볼 시각")
        lines.append("<b>⑥ 발표 후 확인 순서</b>  회사 공시 원문 → 다음 분기 가이던스 → 시간외 반응 → 며칠 뒤 EPS 추정치 변화(체력 줄)")
        return "\n".join(lines)
    except Exception:
        return None


# ------------------------------------------------------------------ 2) 한눈 요약
def glance(summaries, risk, now_kst, drivers=None, ctx=None, event=None, stale=None):
    """맨 위 요약 카드(HTML). 만들 수 없으면 None."""
    try:
        items = _scalp_items(summaries)
        if not items:
            return None
        pds = [pd.Timestamp(it["price_date"]) for it in items.values() if it.get("price_date")]
        head = f"📊 <b>{_md(now_kst)} 한눈 요약</b>" + (f"  <i>기준 {_md(max(pds))} 미국 종가</i>" if pds else "")
        lines = [head]
        n_pass, n_tot = _signal_counts(summaries)
        if n_tot:
            lines.append(("🔴 <b>방향 신호</b>  아직 쓸 수 없음 " if n_pass == 0 else "🟡 <b>방향 신호</b>  일부만 통과 ") + f"(검증 통과 {n_pass}/{n_tot})")
        for etf in sorted(LONGS, key=lambda t: (t not in HELD, t)):
            it = items.get(etf)
            if not it or not _fin(it.get("price")):
                continue
            rk = (risk or {}).get(etf) if isinstance(risk, dict) else None
            bits = [f"${it['price']:.2f}"]
            if _fin(it.get("ret5")):
                bits.append(f"5일 {it['ret5'] * 100:+.0f}%")
            if _fin(it.get("drawdown")):
                bits.append(f"고점대비 {it['drawdown'] * 100:.0f}%")
            if rk and _fin(rk.get("move_1d")):
                bits.append(f"하루 ±{rk['move_1d'] * 100:.0f}%")
            icon = "💼" if etf in HELD else "▫️"
            lines.append(f"{icon} <b>{etf}{'(보유)' if etf in HELD else ''}</b>  " + _E(" · ".join(bits)))
        fg = (ctx or {}).get("fg") if isinstance(ctx, dict) else None
        if fg and _fin(fg.get("score")):
            wk = f" ← 1주 전 {fg['prev_week']:.0f}" if _fin(fg.get("prev_week")) else ""
            lines.append(f"🌡️ <b>심리</b>  {_E(fg.get('bucket') or '')} {fg['score']:.0f}{wk}")
        sx = ((drivers or {}).get("indexes") or {}).get("SOXX") if isinstance(drivers, dict) else None
        a = (sx or {}).get("attr")
        if a and 1 in a:
            c = a[1]["contrib"]
            est = a[1]["est"]
            movers = sorted(c.items(), key=lambda kv: kv[1])[:3] if est < 0 else sorted(c.items(), key=lambda kv: -kv[1])[:3]
            verb = "끌어내림" if est < 0 else "끌어올림"
            lines.append(f"🧩 <b>반도체지수</b>  {_md(a['asof'])} {est * 100:+.1f}% — {_E('·'.join(k for k, _ in movers))} {verb}")
        if event:
            mw = max(event["weights"].items(), key=lambda kv: kv[1])
            dday = int(np.busday_count(pd.Timestamp(now_kst).date(), pd.Timestamp(event["date"]).date()))
            lines.append(f"🎯 <b>임박</b>  {_E(event['symbol'])} 실적 {_md(event['date'])} {'오늘' if dday <= 0 else f'D-{dday}'} — {INDEX_NAME.get(mw[0], mw[0])}의 {mw[1] * 100:.0f}%")
        warn = []
        if stale:
            warn.append(f"갱신 안 됨 {len(stale)}건")
        pw = (ctx or {}).get("price_warn") if isinstance(ctx, dict) else None
        if pw:
            warn.append(f"시세 불일치 {len(pw)}건")
        if (ctx or {}).get("env_changes") if isinstance(ctx, dict) else None:
            warn.append("라이브러리 버전 변경")
        if warn:
            lines.append("⚠️ <b>데이터 점검</b>  " + _E(" · ".join(warn)))
        elif isinstance(ctx, dict) and ctx.get("price_checked"):
            lines.append(f"✅ <b>데이터 점검</b>  시세 {ctx['price_checked']}종목 교차확인 일치")
        lines.append("<i>아래 상세는 눌러서 펼칩니다.</i>")
        return "\n".join(lines)
    except Exception:
        return None


# ------------------------------------------------------------------ 3) 접어 두기
def collapse(section):
    """'제목 줄 + 본문'인 섹션의 본문을 펼치기 인용으로 감싼다. 한 줄짜리·너무 긴 섹션·이미 인용이 있는 섹션은 그대로."""
    try:
        if not isinstance(section, str) or "<blockquote" in section or len(section) > COLLAPSE_MAX:
            return section
        head, sep, body = section.partition("\n")
        if not sep or not body.strip() or body.count("\n") < 3:      # 본문 3줄 이하는 접어도 줄지 않는다
            return section
        return f"{head}\n<blockquote expandable>{body}</blockquote>"
    except Exception:
        return section


# ------------------------------------------------------------------ 4) 헤드라인 번역
def translate_ko(texts, getter=None, timeout=8):
    """영문 제목 리스트 → 한국어 리스트. 구글 번역의 공개 엔드포인트(키 없음, 비공식)를 한 번 호출. 실패·개수 불일치면 원문을 그대로 돌려준다."""
    texts = [str(t) for t in (texts or []) if str(t).strip()]
    if not texts:
        return []
    try:
        q = "\n".join(t.replace("\n", " ") for t in texts)
        url = TRANSLATE_URL + "?" + urllib.parse.urlencode({"client": "gtx", "sl": "en", "tl": "ko", "dt": "t", "q": q})
        if getter is not None:
            raw = getter(url)
        else:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode("utf-8")
        data = json.loads(raw)
        out = "".join(seg[0] for seg in data[0] if seg and seg[0])
        parts = [p.strip() for p in out.split("\n") if p.strip()]
        if len(parts) != len(texts) or any(len(p) > 200 for p in parts):
            return texts
        return parts
    except Exception:
        return texts


def tidy_headline(t):
    """'Stock Market Today: ' 같은 머리말을 떼고 길이를 줄인다."""
    t = re.sub(r"^(Stock Market Today|Markets? Wrap|Live Updates?)\s*[:\-–—]\s*", "", str(t).strip(), flags=re.I)
    return t if len(t) <= 110 else t[:107].rstrip() + "…"


def headline_section(headlines, getter=None):
    try:
        hs = [tidy_headline(h) for h in (headlines or []) if str(h).strip()][:3]
        if not hs:
            return None
        ko = translate_ko(hs, getter=getter)
        return "📰 <b>5. 미국 시장 헤드라인 (WSJ · 자동 번역)</b>\n" + "\n".join(f"• {_E(k)}" for k in ko)
    except Exception:
        return None


# ------------------------------------------------------------------ 5) 조립
def restyle(sections, summaries, risk, now_kst, headlines=None, stale=None, translate_getter=None):
    """build_sections 결과를 받아 새 배치로 돌려준다. 실패하면 원본 그대로."""
    try:
        drivers = (risk or {}).get("_drivers") if isinstance(risk, dict) else None
        ctx = (risk or {}).get("_context") if isinstance(risk, dict) else None
        event = find_event(drivers, pd.Timestamp(now_kst).date())
        top = glance(summaries, risk, now_kst, drivers, ctx, event, stale)
        if not top:
            return sections
        out = [top]
        card = event_card(event, summaries, risk, pd.Timestamp(now_kst).date())
        if card:
            out.append(card)
        for sec in sections[1:]:                      # sections[0] = 기존 머리말(한눈 요약이 대신함)
            if "WSJ 마켓 헤드라인" in sec:
                hs = headline_section(headlines, getter=translate_getter)
                sec = hs or sec
            if "오늘의 결론" in sec and "신호 신뢰도: 낮음" in sec:
                continue                                  # 통과 0건이면 한눈 요약의 '방향 신호' 줄과 같은 말이라 생략(통과가 생기면 다시 보임)
            if "오늘의 한 줄 공부" in sec:
                out.append(sec)
                continue
            out.append(collapse(sec))
        counter = 0
        for i, sec in enumerate(out):
            if re.search(r"<b>\d+\. ", sec):
                counter += 1
                out[i] = re.sub(r"<b>\d+\. ", f"<b>{counter}. ", sec, count=1)
        return out
    except Exception as e:
        print(f"⚠️ 리포트 재배치 실패 — 기존 형식으로 전송: {type(e).__name__}: {e}")
        return sections


# ------------------------------------------------------------------ 6) 그림 한 장
def dashboard_png(summaries, risk, now_kst):
    """PNG bytes 또는 None. matplotlib 이 없거나 그릴 데이터가 모자라면 None(리포트는 글로만 나감)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    try:
        items = _scalp_items(summaries)
        drivers = (risk or {}).get("_drivers") if isinstance(risk, dict) else None
        ctx = (risk or {}).get("_context") if isinstance(risk, dict) else None
        etfs = [t for t in LONGS if t in items and _fin(items[t].get("drawdown"))]
        if not etfs:
            return None
        INK, MUTE, RED, GRN, BLU, AMB = "#1f2933", "#7b8794", "#d64545", "#2f9e6b", "#3b6fd4", "#e0a030"
        plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#cbd2d9", "axes.labelcolor": MUTE, "xtick.color": MUTE, "ytick.color": MUTE,
                             "axes.spines.top": False, "axes.spines.right": False})
        fig, ax = plt.subplots(2, 2, figsize=(10, 7.2), dpi=130)
        fig.patch.set_facecolor("white")
        pds = [pd.Timestamp(it["price_date"]) for it in items.values() if it.get("price_date")]
        fig.suptitle(f"Daily risk view  |  as of US close {max(pds).date() if pds else ''}", x=0.02, ha="left", fontsize=13, color=INK, fontweight="bold")

        # (a) 고점 대비 낙폭 + 하루 1σ
        a = ax[0, 0]
        dd = [items[t]["drawdown"] * 100 for t in etfs]
        ypos = np.arange(len(etfs))[::-1]
        a.barh(ypos, dd, color=[RED if t in HELD else "#e8a5a5" for t in etfs], height=0.55)
        for y, t, v in zip(ypos, etfs, dd):
            rk = (risk or {}).get(t) if isinstance(risk, dict) else None
            mv = f"   1-day move ±{rk['move_1d'] * 100:.0f}%" if rk and _fin(rk.get("move_1d")) else ""
            a.text(min(v, 0) - 1, y, f"{v:.0f}%", va="center", ha="right", color=INK, fontsize=10, fontweight="bold")
            a.text(1, y, f"{t}{' (held)' if t in HELD else ''}  ${items[t]['price']:.2f}{mv}", va="center", ha="left", color=INK, fontsize=9)
        a.set_xlim(min(dd) * 1.35 - 5, 60)
        a.set_yticks([])
        a.axvline(0, color="#cbd2d9", lw=1)
        a.set_title("Drawdown from 60-day high", loc="left", color=INK, fontsize=11)
        a.set_xticks([])
        for sp in ("left", "bottom"):
            a.spines[sp].set_visible(False)

        # (b) 공포탐욕 추이
        b = ax[0, 1]
        hist = (ctx or {}).get("fg_hist") if isinstance(ctx, dict) else None
        if hist and len(hist) >= 5:
            hs = pd.Series(hist)
            hs.index = pd.to_datetime(hs.index)
            hs = hs.sort_index().tail(60)
            for lo, hi, col in ((0, 25, "#f8d7d7"), (25, 45, "#fdeeda"), (45, 55, "#f1f3f5"), (55, 75, "#e3f4ea"), (75, 100, "#c9ebd6")):
                b.axhspan(lo, hi, color=col, lw=0)
            b.plot(hs.index, hs.values, color=INK, lw=1.8)
            b.scatter([hs.index[-1]], [hs.values[-1]], color=INK, s=28, zorder=3)
            b.text(hs.index[-1], hs.values[-1] + 5, f"{hs.values[-1]:.0f}", ha="right", color=INK, fontweight="bold")
            b.set_ylim(0, 100)
            b.set_yticks([12, 35, 50, 65, 88])
            b.set_yticklabels(["Extreme fear", "Fear", "Neutral", "Greed", "Extreme greed"], fontsize=8)
            b.tick_params(axis="x", labelsize=8, rotation=0)
            import matplotlib.dates as mdates
            b.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
            b.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        else:
            b.text(0.5, 0.5, "Fear & Greed: no data", ha="center", va="center", color=MUTE, transform=b.transAxes)
            b.set_xticks([]); b.set_yticks([])
        b.set_title("CNN Fear & Greed (last 60 days)", loc="left", color=INK, fontsize=11)

        # (c) 반도체지수 기여 종목
        c = ax[1, 0]
        sx = ((drivers or {}).get("indexes") or {}).get("SOXX") if isinstance(drivers, dict) else None
        at = (sx or {}).get("attr")
        if at and 1 in at:
            con = pd.Series(at[1]["contrib"]) * 100
            pick = pd.concat([con.nsmallest(5), con.nlargest(3)]).drop_duplicates()
            pick = pick[pick.abs() >= 0.01].sort_values()
            c.barh(np.arange(len(pick)), pick.values, color=[RED if v < 0 else GRN for v in pick.values], height=0.6)
            c.set_yticks(np.arange(len(pick)))
            c.set_yticklabels(pick.index, fontsize=9, color=INK)
            for y, v in enumerate(pick.values):
                c.text(v + (0.01 if v >= 0 else -0.01), y, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK)
            m = max(0.05, float(pick.abs().max())) * 1.35
            c.set_xlim(-m, m)
            c.axvline(0, color="#cbd2d9", lw=1)
            c.set_title(f"Semiconductor index {at[1]['est'] * 100:+.1f}% ({pd.Timestamp(at['asof']).strftime('%m/%d')}): who moved it (%p)", loc="left", color=INK, fontsize=11)
        else:
            c.text(0.5, 0.5, "Contribution: no data", ha="center", va="center", color=MUTE, transform=c.transAxes)
            c.set_xticks([]); c.set_yticks([])
            c.set_title("Semiconductor index: who moved it", loc="left", color=INK, fontsize=11)

        # (d) 10거래일 실적 노출
        d = ax[1, 1]
        rows = []
        for idx, col in (("SOXX", RED), ("QQQ", BLU), ("SPY", AMB)):
            for sym, dt, w in ((((drivers or {}).get("indexes") or {}).get(idx) or {}).get("earn") or {}).get("names", [])[:6]:
                rows.append((pd.Timestamp(dt), sym, idx, w * 100, col))
        if rows:
            df = pd.DataFrame(rows, columns=["d", "sym", "idx", "w", "col"])
            top = df.groupby("sym")["w"].max().nlargest(6).index
            df = df[df["sym"].isin(top)]
            order = df.groupby("sym")["d"].min().sort_values(ascending=False).index.tolist()
            off = {"SOXX": 0.25, "QQQ": 0.0, "SPY": -0.25}
            for _, r in df.iterrows():
                y = order.index(r["sym"]) + off[r["idx"]]
                d.barh(y, r["w"], color=r["col"], height=0.22)
            d.set_yticks(range(len(order)))
            d.set_yticklabels([f"{s}  {df[df['sym'] == s]['d'].min().strftime('%m/%d')}" for s in order], fontsize=9, color=INK)
            from matplotlib.patches import Patch
            d.legend(handles=[Patch(color=RED, label="Semis (SOXL)"), Patch(color=BLU, label="Nasdaq100 (TQQQ)"), Patch(color=AMB, label="S&P500 (SPXL)")],
                     frameon=False, fontsize=8, loc="center right")
            d.set_xlabel("index weight (%)", fontsize=8)
        else:
            d.text(0.5, 0.5, "No large earnings in next 10 trading days", ha="center", va="center", color=MUTE, transform=d.transAxes)
            d.set_xticks([]); d.set_yticks([])
        d.set_title("Earnings in next 10 trading days: weight at stake", loc="left", color=INK, fontsize=11)

        fig.tight_layout(rect=(0, 0, 1, 0.95))
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="white")
        plt.close(fig)
        return buf.getvalue()
    except Exception as e:
        print(f"⚠️ 그림 생성 실패(무시): {type(e).__name__}: {e}")
        try:
            plt.close("all")
        except Exception:
            pass
        return None


def send_photo(png, caption="", poster=None):
    """텔레그램 sendPhoto. 토큰이 없거나 실패하면 False(글 리포트는 별도로 나간다)."""
    try:
        token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
        if not png or not token or not chat:
            return False
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        if poster is None:
            import requests
            poster = lambda u, data, files: requests.post(u, data=data, files=files, timeout=20)
        r = poster(url, {"chat_id": chat, "caption": caption[:1000]}, {"photo": ("dashboard.png", png, "image/png")})
        return getattr(r, "status_code", 200) == 200
    except Exception as e:
        print(f"⚠️ 그림 전송 실패(무시): {type(e).__name__}")
        return False
