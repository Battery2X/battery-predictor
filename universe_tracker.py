"""
universe_tracker.py — 확장 종목(단일종목 2배 ETF·관찰 종목·비교군)을 '과거 백필 + 오늘부터 누적'으로 비교 (v39, 읽기 전용 분석).

왜
- 실전 로그만 기다리면 판정에 수년이 걸린다. 그래서 (1) 과거 16년을 규칙 그대로 백필해 큰 표본을 먼저 보고,
  (2) 등록일(universe.json registered_on) 이후는 매일 상태를 기록해 '사후 조정 없는' 실전 표본을 따로 쌓는다.
- 방향 모델·피처·임계값·예측 로그는 건드리지 않는다 → 검증 시계 리셋 없음. 새 로그 파일(logs/universe_daily.csv)만 추가.

규칙(사전등록, 숫자 고정): HOLD · TREND200 · MOM126 (universe.json) + VOL_GATE · CRASH_GATE (strategy_rules.json)
- 신호는 모두 '기초자산'의 전일 종가까지 정보로 계산해 다음 날 상품 수익률에 적용(ex-ante).
- 레버리지 상품 이력이 짧으면 기초자산 × 배수 − 연 비용/252 로 합성(근사). 실물 ETF 가 있으면 실제 배수(회귀 기울기)를 따로 확인.

한계: 거래비용·세금 미반영. 종목 선택 자체가 '이미 오른 종목' 편향(그래서 INTC·QCOM 비교군 포함). 종목끼리 상관이 커서
'15종목 = 독립 표본 15개'가 아니다. 백테스트는 실전 성적이 아니다.

실행
  python universe_tracker.py backfill [출력=reports] [시작일=2010-01-01]
  python universe_tracker.py daily    [logs=logs] [출력=reports]
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

import strategy_sim as ss

HERE = os.path.dirname(os.path.abspath(__file__))
UNIVERSE_PATH = os.path.join(HERE, "universe.json")
DAILY_LOG = "universe_daily.csv"
RULES = ["HOLD", "TREND200", "MOM126", "VOL_GATE", "CRASH_GATE", "SWITCH_TREND", "SWITCH_MOM"]
SWITCH = ("SWITCH_TREND", "SWITCH_MOM")
LOG_COLS = ["price_date", "ticker", "under", "lev", "group", "close", "close_under", "synthetic"] + [f"pos_{r}" for r in RULES] + ["run_at"]


def load_universe(path=UNIVERSE_PATH):
    try:
        with open(path, encoding="utf-8") as f:
            u = json.load(f)
        assert u.get("items") and u.get("registered_on")
        return u
    except Exception as e:
        print(f"⚠️ universe.json 읽기 실패({e}) — 생략")
        return None


# ---------------------------------------------------------------- 신호 (기초자산 종가 → 다음 날 보유비중, ex-ante)
def trend_pos(close, window=200):
    c = pd.Series(np.asarray(close, float))
    ma = c.rolling(window, min_periods=window).mean()
    state = (c > ma).astype(float).where(ma.notna(), 1.0)   # 워밍업 구간은 보유(다른 규칙과 같은 기간 비교)
    return state.shift(1).fillna(1.0).values                # i 번째 수익률엔 i-1 종가의 상태


def mom_pos(close, window=126):
    c = pd.Series(np.asarray(close, float))
    m = c / c.shift(window) - 1
    state = (m > 0).astype(float).where(m.notna(), 1.0)
    return state.shift(1).fillna(1.0).values


def positions(close_under, cfg_sim, cfg_uni):
    """기초자산 종가 배열 → {규칙: 보유비중 배열} (수익률 배열과 같은 길이 = len(close)-1)."""
    c = np.asarray(close_under, float)
    r = c[1:] / c[:-1] - 1
    R, U = cfg_sim["rules"], cfg_uni["rules"]
    return {"HOLD": np.ones(len(r)),
            "TREND200": trend_pos(c, U["TREND200"]["window"])[1:],
            "MOM126": mom_pos(c, U["MOM126"]["window"])[1:],
            "VOL_GATE": ss.vol_gate(r, R["VOL_GATE"]["lam"], R["VOL_GATE"]["mult"], R["VOL_GATE"]["min_obs"]),
            "CRASH_GATE": ss.crash_gate(r, R["CRASH_GATE"]["dd_window"], R["CRASH_GATE"]["dd_trigger"], R["CRASH_GATE"]["reentry_ma"]),
            "SWITCH_TREND": 2 * trend_pos(c, U["SWITCH_TREND"]["window"])[1:] - 1,     # +1 롱 / -1 인버스
            "SWITCH_MOM": 2 * mom_pos(c, U["SWITCH_MOM"]["window"])[1:] - 1}


def strategy_returns(ru, pos, lev, inv_lev, cost_long, cost_inv):
    """기초자산 수익률 ru, 보유 pos(+1 롱/-1 인버스/0 현금) → 상품 수익률. 인버스는 -inv_lev×ru − 비용."""
    ru, pos = np.asarray(ru, float), np.asarray(pos, float)
    lr = lev * ru - cost_long / 252.0
    ir = -inv_lev * ru - cost_inv / 252.0
    return np.where(pos > 0, pos * lr, np.where(pos < 0, -pos * ir, 0.0))


def compare_switch(ru, rv_hold, pos, lev, inv_lev, cost_long, cost_inv):
    s = strategy_returns(ru, pos, lev, inv_lev, cost_long, cost_inv)
    m = ss.metrics(s)
    h = ss.metrics(rv_hold)
    if m is None:
        return None
    m["exposure_inverse"] = float((np.asarray(pos) < 0).mean())
    m["vs_hold"] = {"hold_total": h["total"], "hold_mdd": h["mdd"],
                    "daily_logdiff_ci": ss.block_bootstrap_mean(np.log1p(s) - np.log1p(rv_hold))}
    return m


def _costs(cfg_uni, lev, inv_lev):
    cm = cfg_uni.get("lev_cost_annual", {})
    return float(cm.get(str(lev), 0.0)), float(cm.get(str(inv_lev), 0.0)) + (0.0095 if inv_lev == 1 else 0.0)


def next_day_state(close_under, cfg_sim, cfg_uni):
    """오늘 종가까지 보고 정한 '내일' 보유 상태(로그 기록용). 수익률 하나를 가짜로 덧붙여 마지막 신호를 읽는다."""
    c = np.asarray(close_under, float)
    ext = np.append(c, c[-1])
    p = positions(ext, cfg_sim, cfg_uni)
    return {k: float(v[-1]) for k, v in p.items()}


def realized_leverage(etf_ret, under_ret):
    a, b = np.asarray(etf_ret, float), np.asarray(under_ret, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 60 or np.var(b[ok]) == 0:
        return None
    return float(np.cov(a[ok], b[ok])[0, 1] / np.var(b[ok], ddof=1))


# ---------------------------------------------------------------- 백필
def _fetch(sym, start, fetcher):
    try:
        df = fetcher(sym, start)
    except Exception as e:
        print(f"⚠️ [{sym}] 시세 실패({e})")
        return None
    if df is None or "Close" not in df or len(df) < 260:
        return None
    return df["Close"].dropna()


def backfill(start="2010-01-01", fetcher=None, cfg_sim=None, cfg_uni=None):
    cfg_sim, cfg_uni = cfg_sim or ss.load_rules(), cfg_uni or load_universe()
    if cfg_sim is None or cfg_uni is None:
        return None
    if fetcher is None:
        import common as cm
        fetcher = cm.fetch_ticker_data
    out = {}
    for it in cfg_uni["items"]:
        t, u, lev = it["ticker"], it["under"], int(it["lev"])
        cu = _fetch(u, start, fetcher)
        if cu is None:
            print(f"⚠️ [{t}] 기초자산 {u} 데이터 부족 — 생략")
            continue
        ru = (cu / cu.shift(1) - 1).dropna()
        cost = float(cfg_uni.get("lev_cost_annual", {}).get(str(lev), 0.0))
        rv = ss.synth_leveraged(ru.values, lev, cost) if lev != 1 else ru.values
        pos = positions(cu.values, cfg_sim, cfg_uni)
        sw = {k: pos.pop(k) for k in SWITCH}
        res = {"group": it.get("group"), "lev": lev, "under": u, "period": [str(ru.index[0].date()), str(ru.index[-1].date())],
               "synthetic": lev != 1, "rules": ss.compare(rv, pos, cfg_uni["registered_on"], ru.index)}
        inv = int(cfg_uni.get("inverse_lev", {}).get(it.get("group"), 0))
        if inv:
            cl, ci_ = _costs(cfg_uni, lev, inv)
            for k, p in sw.items():
                res["rules"][k] = compare_switch(ru.values, rv, p, lev, inv, cl, ci_)
            res["inverse_hold"] = ss.metrics(strategy_returns(ru.values, -np.ones(len(ru)), lev, inv, cl, ci_))
        if lev != 1:   # 실물 ETF 가 있으면 실제 배수와 실물 기준 성적(짧은 기간)도
            ce = _fetch(t, start, fetcher)
            if ce is not None:
                j = pd.concat([ce.rename("e"), cu.rename("u")], axis=1).dropna()
                re_, ru2 = j["e"].pct_change().dropna(), j["u"].pct_change().dropna()
                res["real_etf"] = {"period": [str(re_.index[0].date()), str(re_.index[-1].date())],
                                   "realized_lev": realized_leverage(re_.values, ru2.values),
                                   "hold": ss.metrics(re_.values)}
        out[t] = res
    return out


def pooled(res):
    """규칙별로 종목 전체를 모아 본 요약: HOLD 대비 CAGR·MDD 변화의 중앙값과 'MDD 개선 종목 수'."""
    agg = {}
    for rule in RULES[1:]:
        dc, dm, better, n = [], [], 0, 0
        for v in res.values():
            m, h = v["rules"].get(rule), v["rules"].get("HOLD")
            if not m or not h or m.get("cagr") is None or h.get("cagr") is None:
                continue
            n += 1
            dc.append(m["cagr"] - h["cagr"])
            dm.append(m["mdd"] - h["mdd"])
            better += m["mdd"] > h["mdd"]
        if n:
            agg[rule] = {"n": n, "median_dcagr": float(np.median(dc)), "median_dmdd": float(np.median(dm)), "mdd_improved": int(better)}
    return agg


def render_backfill(res, cfg_uni):
    L = [f"=== 확장 종목 백필 (규칙 등록일 {cfg_uni['registered_on']}, 그 이전은 표본 내 참고 · 거래비용 미반영 · 실전 성적 아님) ===",
         "종목 선택 자체가 '이미 오른 종목' 편향 · 종목끼리 상관 큼 → 종목 수 = 독립 표본 수가 아님"]
    for t, v in res.items():
        h = v["rules"]["HOLD"]
        tag = f"{v['under']}×{v['lev']} 합성" if v["synthetic"] else t
        L.append(f"\n[{t}] ({v['group']}, {tag}) {v['period'][0]}~{v['period'][1]}")
        for r in RULES:
            m = v["rules"].get(r)
            if not m:
                continue
            ci = m["vs_hold"]["daily_logdiff_ci"]
            sig = "" if r == "HOLD" or not ci else (" ↑유의" if ci[1] > 0 else (" ↓유의" if ci[2] < 0 else ""))
            cg = "n/a" if m["cagr"] is None else f"{m['cagr'] * 100:5.1f}%"
            L.append(f"  {r:<10} 연 {cg} · MDD {m['mdd'] * 100:6.1f}% · 보유 {m['exposure'] * 100:3.0f}%{sig}")
        if v.get("inverse_hold"):
            ih = v["inverse_hold"]
            L.append(f"  (참고) 인버스만 계속 보유: 총 {ih['total'] * 100:+.1f}% · MDD {ih['mdd'] * 100:.1f}%")
        if v.get("real_etf"):
            e = v["real_etf"]
            lv = "n/a" if e["realized_lev"] is None else f"{e['realized_lev']:.2f}배"
            L.append(f"  실물 {t} {e['period'][0]}~: 실제 배수 {lv} · 보유 수익 {e['hold']['total'] * 100:+.0f}% · MDD {e['hold']['mdd'] * 100:.0f}%")
    L.append("\n[종목 전체 요약: HOLD 대비]")
    for r, a in pooled(res).items():
        L.append(f"  {r:<10} 연수익 중앙값 {a['median_dcagr'] * 100:+.1f}%p · MDD 중앙값 {a['median_dmdd'] * 100:+.1f}%p · MDD 개선 {a['mdd_improved']}/{a['n']}종목")
    return "\n".join(L)


# ---------------------------------------------------------------- 매일 누적 (등록일 이후 실전 표본)
def load_daily(log_dir):
    p = os.path.join(log_dir, DAILY_LOG)
    if not os.path.exists(p):
        return pd.DataFrame(columns=LOG_COLS)
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"⚠️ {DAILY_LOG} 읽기 실패({e}) — 새로 시작하지 않고 이번 기록만 생략")
        return None
    if not {"price_date", "ticker"}.issubset(df.columns):
        print(f"⚠️ {DAILY_LOG} 형식이 깨져 보임(열 {list(df.columns)[:4]}) — 덮어쓰지 않고 이번 기록만 생략")
        return None
    for c in LOG_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[LOG_COLS]


def daily_rows(fetcher=None, cfg_sim=None, cfg_uni=None, now=None, start=None):
    cfg_sim, cfg_uni = cfg_sim or ss.load_rules(), cfg_uni or load_universe()
    if cfg_sim is None or cfg_uni is None:
        return []
    if fetcher is None:
        import common as cm
        fetcher = cm.fetch_ticker_data
    start = start or (pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
    run_at = str(now or pd.Timestamp.utcnow().tz_localize(None))
    rows = []
    for it in cfg_uni["items"]:
        t, u, lev = it["ticker"], it["under"], int(it["lev"])
        cu = _fetch(u, start, fetcher)
        if cu is None:
            continue
        ce = None if lev == 1 else _fetch(t, start, fetcher)
        st = next_day_state(cu.values, cfg_sim, cfg_uni)
        d = cu.index[-1]
        close = float(cu.iloc[-1]) if lev == 1 else (float(ce.loc[d]) if ce is not None and d in ce.index else np.nan)
        rows.append({"price_date": str(pd.Timestamp(d).date()), "ticker": t, "under": u, "lev": lev, "group": it.get("group"), "close": close,
                     "close_under": float(cu.iloc[-1]), "synthetic": bool(lev != 1 and np.isnan(close)),
                     **{f"pos_{k}": v for k, v in st.items()}, "run_at": run_at})
    return rows


def append_daily(rows, log_dir):
    if not rows:
        return None
    old = load_daily(log_dir)
    if old is None:
        return None
    new = pd.DataFrame(rows)[LOG_COLS]
    df = new if old.empty else pd.concat([old, new], ignore_index=True)
    df = df.sort_values("run_at").drop_duplicates(["price_date", "ticker"], keep="last").sort_values(["ticker", "price_date"])
    os.makedirs(log_dir, exist_ok=True)
    tmp = os.path.join(log_dir, DAILY_LOG + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, os.path.join(log_dir, DAILY_LOG))
    return df


def forward_eval(df, cfg_uni, cost_map=None):
    """로그의 연속 행으로 등록일 이후 규칙 성적을 계산. t 일 수익률엔 t-1 행의 상태를 쓴다."""
    out = {}
    if df is None or df.empty:
        return out
    reg = pd.Timestamp(cfg_uni["registered_on"])
    cost_map = cost_map or cfg_uni.get("lev_cost_annual", {})
    for t, g in df.groupby("ticker"):
        g = g.sort_values("price_date")
        g = g[pd.to_datetime(g["price_date"]) >= reg]
        if len(g) < 2:
            continue
        lev = int(g["lev"].iloc[-1])
        real = g["close"].astype(float)
        if real.notna().all() and lev != 1 or lev == 1:
            r = (real / real.shift(1) - 1).values[1:]
        else:   # 실물 종가가 빠진 날이 있으면 기초자산 합성으로
            ru = (g["close_under"].astype(float) / g["close_under"].astype(float).shift(1) - 1).values[1:]
            r = ss.synth_leveraged(ru, lev, float(cost_map.get(str(lev), 0.0)))
        pos = {k: g[f"pos_{k}"].astype(float).values[:-1] for k in RULES if f"pos_{k}" in g}
        ru = (g["close_under"].astype(float) / g["close_under"].astype(float).shift(1) - 1).values[1:]
        grp = g["group"].iloc[-1] if "group" in g else None
        inv = int(cfg_uni.get("inverse_lev", {}).get(grp, 0)) if grp else 0
        rules = {}
        for k, p in pos.items():
            if k in SWITCH:
                if inv and not np.isnan(p).all():
                    cl, ci_ = _costs(cfg_uni, lev, inv)
                    rules[k] = ss.metrics(strategy_returns(ru, np.nan_to_num(p), lev, inv, cl, ci_))
            else:
                rules[k] = ss.metrics(r, p)
        out[t] = {"days": len(r), "rules": rules}
    return out


def status_section(log_dir="logs", cfg_uni=None):
    """텔레그램용 한 덩어리(표시 전용). 실패하면 빈 문자열."""
    try:
        cfg_uni = cfg_uni or load_universe()
        df = load_daily(log_dir)
        if cfg_uni is None or df is None or df.empty:
            return ""
        last = df.sort_values("price_date").groupby("ticker").tail(1)
        mark = lambda x: "○" if x == 1 else ("✕" if x == 0 else "-")
        L = ["🧪 규칙 상태 (미검증 · 표시 전용 · 매매 신호 아님) ○보유 ✕현금", "종목  추세 모멘텀 변동성 낙폭 | 전환(추세)"]
        for _, r in last.iterrows():
            if r["ticker"] in ("SOXL", "TQQQ", "SPXL", "NVDL", "AMDL", "AVL", "MUU", "TSMX"):
                side = {1.0: "롱", -1.0: "인버스"}.get(r.get("pos_SWITCH_TREND"), "-")
                L.append(f"{r['ticker']:<5} {mark(r['pos_TREND200'])}    {mark(r['pos_MOM126'])}     {mark(r['pos_VOL_GATE'])}     {mark(r['pos_CRASH_GATE'])}  | {side}")
        fe = forward_eval(df, cfg_uni)
        n = max([v["days"] for v in fe.values()], default=0)
        L.append(f"등록일({cfg_uni['registered_on']}) 이후 누적 {n}거래일 — 60일 전에는 판정하지 않음")
        return "\n".join(L)
    except Exception as e:
        print(f"⚠️ 규칙 상태 섹션 생략({e})")
        return ""


def main(argv):
    mode = argv[1] if len(argv) > 1 else "daily"
    if mode == "backfill":
        out_dir = argv[2] if len(argv) > 2 else "reports"
        start = argv[3] if len(argv) > 3 else "2010-01-01"
        res = backfill(start)
        if not res:
            print("결과 없음 — 생략")
            return 0
        txt = render_backfill(res, load_universe())
        print(txt)
        os.makedirs(out_dir, exist_ok=True)
        open(os.path.join(out_dir, "universe_backfill.txt"), "w", encoding="utf-8").write(txt + "\n")
        json.dump({"items": res, "pooled": pooled(res)}, open(os.path.join(out_dir, "universe_backfill.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1, default=float)
        return 0
    log_dir = argv[2] if len(argv) > 2 else "logs"
    out_dir = argv[3] if len(argv) > 3 else "reports"
    df = append_daily(daily_rows(), log_dir)
    if df is None:
        print("오늘 기록 없음 — 생략")
        return 0
    txt = status_section(log_dir)
    print(txt)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "universe_status.txt"), "w", encoding="utf-8").write(txt + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
