"""
paper_trade.py — Claude 모의투자 장부 (v45). 실제 돈 아님 · 사용자에게 하는 매매 권유 아님.

흐름
- Claude(매일 예약 작업)가 paper/orders_pending.json 에 '목표 비중'과 이유를 쓴다(decided_at = 결정 시각, UTC).
- 매일 GitHub Actions(predict.yml)가 이 스크립트를 실행:
  1) fill: 결정 시각 '이후' 처음 확정된 미국 종가(장 마감 16:00 ET 기준)로 목표 비중까지 매매(매도 먼저, 편도 5bp 비용).
     가격을 못 구한 종목이 있으면 그 주문은 다음 날 다시 시도(조용히 보류).
  2) mark: 최신 종가로 평가액 기록(paper/equity.csv), 비교 지수(SPY·QQQ·SMH)는 시작일 기준 같은 금액으로 환산.
- 결정 시점보다 미래 가격으로 체결하지 않도록 '결정 이후 첫 종가'만 쓴다(장중 결정 → 그날 종가, 장 마감 후 결정 → 다음 거래일 종가).

파일: paper/state.json · paper/orders_pending.json · paper/trades.csv · paper/equity.csv · paper/journal.md(Claude 가 작성)
실행: python paper_trade.py run [paper=paper] [reports=reports]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

BENCH = ["SPY", "QQQ", "SMH"]
COST = 0.0005
CLOSE_ET = pd.Timedelta(hours=16)


def _read_json(p, default):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except Exception as e:
        print(f"⚠️ {p} 읽기 실패({e})")
        return None


def _write_json(p, obj):
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def default_fetch(sym, start):
    import common as cm
    df = cm.fetch_ticker_data(sym, start)
    return None if df is None else df["Close"]


def close_time_utc(d):
    """미국 장 마감 시각(UTC). 서머타임은 미 동부 시간대로 계산."""
    ts = pd.Timestamp(d).tz_localize("America/New_York") + CLOSE_ET
    return ts.tz_convert("UTC").tz_localize(None)


def get_prices(tickers, start, fetch):
    out = {}
    for t in sorted(set(tickers)):
        try:
            s = fetch(t, start)
        except Exception as e:
            print(f"⚠️ [{t}] 가격 실패({e})")
            s = None
        if s is not None and len(s):
            s = s.copy()
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            out[t] = s.dropna()
    return out


def fill(state, order, prices):
    """order: {decided_at, targets:{t:w}, note}. 반환: (state, trades, fill_date) 또는 (state, [], None) — 아직 체결 불가."""
    dec = pd.Timestamp(order["decided_at"])
    dec = dec.tz_convert("UTC").tz_localize(None) if dec.tzinfo is not None else dec      # 시간대 없으면 UTC 로 간주
    need = set(order["targets"]) | set(state["positions"])
    if not need.issubset(prices):
        return state, [], None
    # 결정 이후 첫 종가 날짜(모든 종목 공통)
    cands = None
    for t in need:
        ds = [d for d in prices[t].index if close_time_utc(d) > dec]
        cands = set(ds) if cands is None else cands & set(ds)
    if not cands:
        return state, [], None
    d = min(cands)
    px = {t: float(prices[t].loc[d]) for t in need}
    value = state["cash"] + sum(p["shares"] * px[t] for t, p in state["positions"].items())
    trades = []
    targets = {t: float(w) for t, w in order["targets"].items() if float(w) > 0}
    if sum(targets.values()) > 1.0000001:
        raise ValueError("목표 비중 합이 100% 초과")
    # 매도 먼저
    for t, p in list(state["positions"].items()):
        want = targets.get(t, 0.0) * value / px[t]
        if want < p["shares"] - 1e-9:
            q = p["shares"] - want
            state["cash"] += q * px[t] * (1 - COST)
            p["shares"] = want
            trades.append({"date": str(d.date()), "ticker": t, "side": "SELL", "shares": q, "price": px[t]})
            if p["shares"] <= 1e-9:
                del state["positions"][t]
    for t, w in targets.items():
        cur = state["positions"].get(t, {"shares": 0.0, "cost": 0.0})
        want = w * value / px[t]
        if want > cur["shares"] + 1e-9:
            q = min(want - cur["shares"], state["cash"] / (px[t] * (1 + COST)))
            if q <= 0:
                continue
            state["cash"] -= q * px[t] * (1 + COST)
            cur["cost"] = (cur["cost"] * cur["shares"] + q * px[t]) / (cur["shares"] + q)
            cur["shares"] += q
            state["positions"][t] = cur
            trades.append({"date": str(d.date()), "ticker": t, "side": "BUY", "shares": q, "price": px[t]})
    if state.get("start_date") is None:
        state["start_date"] = str(d.date())
        state["bench_start"] = {b: float(prices[b].loc[d]) for b in BENCH if b in prices and d in prices[b].index}
    return state, trades, d


def mark(state, prices):
    if state.get("start_date") is None:          # 첫 체결 전에는 평가 기록 안 함
        return None
    held = list(state["positions"])
    dates = None
    for t in held + [b for b in BENCH if b in prices]:
        if t not in prices:
            return None
        dates = set(prices[t].index) if dates is None else dates & set(prices[t].index)
    if not dates:
        return None
    d = max(dates)
    val = state["cash"] + sum(p["shares"] * float(prices[t].loc[d]) for t, p in state["positions"].items())
    row = {"date": str(d.date()), "value": val, "cash": state["cash"]}
    for b, b0 in (state.get("bench_start") or {}).items():
        if b in prices and d in prices[b].index:
            row[f"bench_{b}"] = state["initial"] * float(prices[b].loc[d]) / b0
    return row


def run(paper_dir="paper", report_dir="reports", fetch=None, start="2026-10-01"):
    fetch = fetch or default_fetch
    os.makedirs(paper_dir, exist_ok=True)
    sp = os.path.join(paper_dir, "state.json")
    state = _read_json(sp, {"initial": 100000.0, "cash": 100000.0, "positions": {}, "start_date": None})
    orders = _read_json(os.path.join(paper_dir, "orders_pending.json"), [])
    if state is None or orders is None:
        return None
    if isinstance(orders, dict):
        orders = [orders]
    tick = set(state["positions"]) | set(BENCH) | {t for o in orders for t in o.get("targets", {})}
    prices = get_prices(tick, start, fetch)
    done, keep = [], []
    for o in sorted(orders, key=lambda o: o["decided_at"]):
        state, tr, d = fill(state, o, prices)
        if d is None:
            keep.append(o)
            print(f"⏳ 주문({o['decided_at']}) 아직 체결 불가 — 보류")
            continue
        done.append({**o, "filled_on": str(d.date())})
        if tr:
            tp = os.path.join(paper_dir, "trades.csv")
            pd.DataFrame(tr).assign(decided_at=o["decided_at"]).to_csv(tp, mode="a", header=not os.path.exists(tp), index=False)
    _write_json(sp, state)
    _write_json(os.path.join(paper_dir, "orders_pending.json"), keep)
    if done:
        hp = os.path.join(paper_dir, "orders_filled.json")
        _write_json(hp, (_read_json(hp, []) or []) + done)
    row = mark(state, prices)
    txt = ""
    if row:
        ep = os.path.join(paper_dir, "equity.csv")
        eq = pd.read_csv(ep) if os.path.exists(ep) else pd.DataFrame()
        eq = pd.concat([eq, pd.DataFrame([row])], ignore_index=True).drop_duplicates("date", keep="last").sort_values("date")
        eq.to_csv(ep, index=False)
        txt = status_text(state, eq)
        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, "paper_status.txt"), "w", encoding="utf-8") as f:
            f.write(txt + "\n")
        print(txt)
    return state, row, txt


def status_text(state, eq):
    last = eq.iloc[-1]
    ini = state["initial"]
    L = [f"🧾 Claude 모의투자 (가상 $100,000 · 실제 돈·매매 권유 아님) 기준 {last['date']}",
         f"평가액 ${last['value']:,.0f} ({(last['value'] / ini - 1) * 100:+.2f}%) · 현금 {last['cash'] / last['value'] * 100:.0f}%"]
    bs = [f"{c[6:]} {(last[c] / ini - 1) * 100:+.2f}%" for c in eq.columns if c.startswith("bench_") and pd.notna(last[c])]
    if bs:
        L.append("같은 기간 지수: " + " · ".join(bs))
    if len(eq) >= 2:
        L.append(f"전일 대비 {(last['value'] / eq.iloc[-2]['value'] - 1) * 100:+.2f}%")
    return "\n".join(L)


if __name__ == "__main__":
    a = sys.argv
    run(a[2] if len(a) > 2 else "paper", a[3] if len(a) > 3 else "reports")
