"""
xs_events.py — 정성/이벤트 데이터 피처 (v42): 실적 서프라이즈 · 애널리스트 의견 변경.

데이터(야후, yfinance): Ticker.get_earnings_dates(limit=100) → 발표일·EPS 추정·실제·서프라이즈(%)
                         Ticker.upgrades_downgrades → 날짜·증권사·등급 변경(up/down/init/main/reit)
미래정보 차단: t 일 피처에는 '발표/변경 날짜 < t'(엄격히 이전)인 이벤트만 쓴다.
  (장 마감 후 발표가 같은 날짜로 찍혀도 다음 날부터 반영되도록 하루를 비운다.)
'다음 실적일까지 남은 날' 같은 미래 일정 피처는 쓰지 않는다(과거 데이터에선 일정 공지 시점을 알 수 없어서).
실패·빈 데이터: 그 종목은 이벤트 피처가 NaN 으로 남고(LightGBM 은 NaN 처리 가능) 조용히 진행.
"""
import numpy as np
import pandas as pd

EVENT_FEATURES = ["ev_days_since_earn", "ev_last_surprise", "ev_surprise_avg4", "ev_surprise_trend",
                  "ev_net_upgrades_30d", "ev_actions_30d", "ev_net_upgrades_90d"]


def fetch_events(ticker):
    """(earnings_df, grades_df) — 실패하면 (None, None)."""
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
    except Exception as e:
        print(f"⚠️ [{ticker}] yfinance 준비 실패({e})")
        return None, None
    earn = grades = None
    try:
        e = tk.get_earnings_dates(limit=100)
        if e is not None and len(e):
            e = e.copy()
            e.index = pd.to_datetime(e.index).tz_localize(None).normalize()
            col = [c for c in e.columns if "urprise" in c]
            e = pd.DataFrame({"surprise": pd.to_numeric(e[col[0]], errors="coerce") if col else np.nan}, index=e.index)
            earn = e.dropna().sort_index()
    except Exception as ex:
        print(f"⚠️ [{ticker}] 실적 이력 실패({ex}) — 생략")
    try:
        g = tk.upgrades_downgrades
        if g is not None and len(g):
            g = g.copy()
            g.index = pd.to_datetime(g.index).tz_localize(None).normalize()
            act = g["Action"].astype(str).str.lower() if "Action" in g else pd.Series("", index=g.index)
            grades = pd.DataFrame({"up": (act == "up").astype(int), "down": (act == "down").astype(int)}, index=g.index).sort_index()
    except Exception as ex:
        print(f"⚠️ [{ticker}] 의견 변경 이력 실패({ex}) — 생략")
    return earn, grades


def event_features(dates, earn, grades):
    """dates(DatetimeIndex, 그 종목의 거래일) → 이벤트 피처 DataFrame (index=dates). t 이전 이벤트만."""
    d = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    out = pd.DataFrame(index=d, columns=EVENT_FEATURES, dtype=float)
    if earn is not None and len(earn):
        e = earn.sort_index()
        ed, es = e.index.values, e["surprise"].values
        pos = np.searchsorted(ed, d.values, side="left") - 1           # 날짜 < t 인 마지막 발표
        valid = pos >= 0
        idx = np.where(valid, pos, 0)
        last_date = np.where(valid, ed[idx], np.datetime64("NaT"))
        out["ev_days_since_earn"] = np.where(valid, (d.values - last_date) / np.timedelta64(1, "D"), np.nan)
        out["ev_last_surprise"] = np.where(valid, es[idx], np.nan)
        s = pd.Series(es)
        avg4 = s.rolling(4, min_periods=1).mean().values
        trend = (s - s.shift(1)).values
        out["ev_surprise_avg4"] = np.where(valid, avg4[idx], np.nan)
        out["ev_surprise_trend"] = np.where(valid, trend[idx], np.nan)
        out.loc[out["ev_days_since_earn"] > 200, ["ev_last_surprise", "ev_surprise_avg4", "ev_surprise_trend"]] = np.nan   # 오래된 정보는 비움
    if grades is not None and len(grades):
        g = grades.groupby(level=0).sum().sort_index()
        net = (g["up"] - g["down"]).astype(float)
        tot = (g["up"] + g["down"]).astype(float)
        cal = pd.date_range(min(g.index.min(), d.min()) - pd.Timedelta(days=100), d.max())
        net_c = net.reindex(cal, fill_value=0.0)
        tot_c = tot.reindex(cal, fill_value=0.0)
        # t 이전(엄격) 30/90일 합: 하루 밀어서 당일 변경은 제외
        n30 = net_c.rolling(30, min_periods=1).sum().shift(1)
        a30 = tot_c.rolling(30, min_periods=1).sum().shift(1)
        n90 = net_c.rolling(90, min_periods=1).sum().shift(1)
        first = g.index.min()
        ok = d > first
        out["ev_net_upgrades_30d"] = np.where(ok, n30.reindex(d).values, np.nan)
        out["ev_actions_30d"] = np.where(ok, a30.reindex(d).values, np.nan)
        out["ev_net_upgrades_90d"] = np.where(ok, n90.reindex(d).values, np.nan)
    return out


def add_event_features(df, fetch=None):
    """features 패널(df: date, ticker, ...)에 이벤트 피처와 날짜별 순위 버전을 붙인다. 반환: (df, 피처 이름 목록, 커버리지)."""
    fetch = fetch or fetch_events
    parts, cover = [], {}
    for t, g in df.groupby("ticker", sort=False):
        try:
            earn, grades = fetch(t)
        except Exception as e:
            print(f"⚠️ [{t}] 이벤트 실패({e}) — 생략")
            earn, grades = None, None
        f = event_features(g["date"].values, earn, grades)
        f["ticker"], f["date"] = t, pd.to_datetime(g["date"].values)
        cover[t] = {"earn": 0 if earn is None else int(len(earn)), "grades": 0 if grades is None else int(len(grades))}
        parts.append(f.reset_index(drop=True))
    ev = pd.concat(parts, ignore_index=True)
    out = df.merge(ev, on=["date", "ticker"], how="left")
    rk = out.groupby("date")[EVENT_FEATURES].rank(pct=True).add_prefix("cs_")
    out = pd.concat([out, rk], axis=1)
    return out, EVENT_FEATURES + [f"cs_{c}" for c in EVENT_FEATURES], cover
