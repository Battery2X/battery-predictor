"""
xs_top20.py — '20일 뒤 상위 10%' 전략: 거래비용 포함 시뮬레이션 + 매일 실전 기록 (v43, 연구용 · 매매 신호 아님).

규칙(xs_config.json 'top20_v43' 사전등록, 숫자 고정)
- 29종목 각각의 '20거래일 뒤 수익률'을 v42 V0 방식(LightGBM, 원래 수익률 학습)으로 예측.
- 예측 상위 10%(최소 3종목)를 같은 비중으로 담고 20거래일 동안 보유, 20거래일마다 교체. 교체 시 편도 10bp 비용.
- 비교 대상: 같은 날 29종목 동일가중 보유.
- 시작일 운을 없애려고 교체 시작일을 0~19일로 20가지 바꿔 모두 계산(중앙값·범위 보고).

매일 기록(logs/top20_predictions.csv): 날짜별 전 종목 예측·순위·상위 여부·종가. 20거래일 뒤 자동 정산.
'공식 바구니': 등록일부터 20거래일마다 교체되는 상위 10% 묶음 — 텔레그램에 현재 바구니와 교체 후 수익(동일가중 대비) 표시.
기존 방향 모델·로그 불변 → 검증 시계 리셋 없음. 백테스트는 실전 성적 아님.

실행
  python xs_top20.py backtest [출력=reports]
  python xs_top20.py daily    [logs=logs] [출력=reports]
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

import xs_predictor as xp
import xs_magnitude as xm
import xs_variants as xv

LOG_NAME = "top20_predictions.csv"
LOG_COLS = ["price_date", "ticker", "pred20", "rank_pct", "is_top", "close", "ret20", "settled", "run_at"]


def _cfg(cfg):
    return cfg["top20_v43"]


def _feats():
    return list(xp.FEATURES) + ["absavg20"]


def n_picks(n, t):
    return max(int(t["min_picks"]), int(math.ceil(n * float(t["top_frac"]))))


# ---------------------------------------------------------------- 시뮬레이션
def simulate(pred, cfg, offset=0):
    """pred: 날짜·종목별 pred_ret 과 실제 fret20. 교체일마다 상위 묶음 → 20일 수익(비용 후) vs 동일가중."""
    t = _cfg(cfg)
    h, step, cost = int(t["horizon"]), int(t["rebalance_every"]), t["cost_bp_per_side"] / 1e4
    y = f"fret{h}"
    p = pred.dropna(subset=[y, "pred_ret"])
    dates = np.sort(p["date"].unique())[offset::step]
    rows, prev = [], frozenset()
    for d in dates:
        g = p[p["date"] == d]
        if len(g) < 10:
            continue
        k = n_picks(len(g), t)
        top = g.nlargest(k, "pred_ret")
        picks = frozenset(top["ticker"])
        turn = 1.0 if not prev else 1 - len(picks & prev) / k
        gross = float(top[y].mean())
        rows.append({"date": d, "top": gross - 2 * cost * turn, "top_gross": gross, "ew": float(g[y].mean()),
                     "turnover": turn, "picks": ",".join(sorted(picks))})
        prev = picks
    R = pd.DataFrame(rows)
    if R.empty:
        return None
    R["excess"] = R["top"] - R["ew"]
    return R


def summarize(R, h=20):
    per_year = 252 / h
    eq = lambda r: np.cumprod(1 + r.values)
    yrs = len(R) / per_year
    cagr = lambda r: float(eq(r)[-1] ** (1 / yrs) - 1) if yrs >= 1 else None
    def mdd(r):
        e = eq(r)
        return float((e / np.maximum.accumulate(e) - 1).min())
    ci = xp.block_ci(R["excess"].values, block=3)
    yr = R.groupby(pd.to_datetime(R["date"]).dt.year)["excess"].sum()
    return {"periods": int(len(R)), "cagr_top": cagr(R["top"]), "cagr_ew": cagr(R["ew"]), "mdd_top": mdd(R["top"]), "mdd_ew": mdd(R["ew"]),
            "excess_mean": float(R["excess"].mean()), "excess_ci": ci, "beat_share": float((R["excess"] > 0).mean()),
            "turnover": float(R["turnover"].mean()), "years_pos": int((yr > 0).sum()), "years": int(len(yr)),
            "by_year": {int(k): float(v) for k, v in yr.items()}}


def backtest(df, cfg):
    t = _cfg(cfg)
    pred = xv.walk_variant(df, cfg, int(t["horizon"]), _feats(), cfg["variants_v42"][t["model"]])
    if pred is None:
        return None
    runs = []
    for off in range(int(t["rebalance_every"])):
        R = simulate(pred, cfg, off)
        if R is not None and len(R) >= 10:
            s = summarize(R, int(t["horizon"]))
            s["offset"] = off
            runs.append((s, R))
    if not runs:
        return None
    S = pd.DataFrame([s for s, _ in runs])
    main = runs[0]
    return {"main": main[0], "main_periods": main[1].assign(date=main[1]["date"].astype(str)).to_dict("records"),
            "offsets": {"n": int(len(S)),
                        "cagr_top_med": float(S["cagr_top"].median()), "cagr_top_min": float(S["cagr_top"].min()), "cagr_top_max": float(S["cagr_top"].max()),
                        "cagr_ew_med": float(S["cagr_ew"].median()),
                        "excess_mean_med": float(S["excess_mean"].median()), "excess_pos_offsets": int((S["excess_mean"] > 0).sum()),
                        "mdd_top_med": float(S["mdd_top"].median()), "mdd_ew_med": float(S["mdd_ew"].median()),
                        "beat_share_med": float(S["beat_share"].median())}}


def render_bt(res, cfg):
    t = _cfg(cfg)
    m, o = res["main"], res["offsets"]
    pc = lambda x, d=1: "n/a" if x is None else f"{x * 100:+.{d}f}%"
    ci = m["excess_ci"]
    L = [f"=== '20일 뒤 상위 10%' 전략 시뮬레이션 (등록 {t['registered_on']}, 백테스트 · 실전 성적 아님 · 종목 선택 편향 있음) ===",
         f"규칙: 29종목 중 예측 상위 10%(최소 {t['min_picks']}개) 동일비중, {t['rebalance_every']}거래일마다 교체, 편도 {t['cost_bp_per_side']}bp 비용 반영",
         "", f"[시작일 20가지 모두] 연평균 상위10% {pc(o['cagr_top_med'])} (범위 {pc(o['cagr_top_min'])} ~ {pc(o['cagr_top_max'])}) vs 동일가중 {pc(o['cagr_ew_med'])}",
         f"  20일마다 동일가중보다 더 번 평균 {pc(o['excess_mean_med'], 2)} · 플러스였던 시작일 {o['excess_pos_offsets']}/{o['n']} · 이긴 구간 비율 {o['beat_share_med'] * 100:.0f}%",
         f"  최대낙폭 상위10% {pc(o['mdd_top_med'])} vs 동일가중 {pc(o['mdd_ew_med'])}",
         "", f"[대표 시작일] {m['periods']}구간 · 연평균 {pc(m['cagr_top'])} vs {pc(m['cagr_ew'])} · 20일 초과 평균 {pc(m['excess_mean'], 2)}"
         f" {'' if not ci else f'[{ci[1] * 100:+.2f}, {ci[2] * 100:+.2f}]'} · 교체율 {m['turnover'] * 100:.0f}% · 플러스였던 해 {m['years_pos']}/{m['years']}",
         "  연도별 초과(합): " + " ".join(f"{y}:{v * 100:+.0f}%" for y, v in m["by_year"].items())]
    return "\n".join(L)


# ---------------------------------------------------------------- 매일 기록
def load_log(log_dir):
    p = os.path.join(log_dir, LOG_NAME)
    if not os.path.exists(p):
        return pd.DataFrame(columns=LOG_COLS)
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"⚠️ {LOG_NAME} 읽기 실패({e}) — 덮어쓰지 않음")
        return None
    if not {"price_date", "ticker"}.issubset(df.columns):
        print(f"⚠️ {LOG_NAME} 형식이 깨져 보임 — 덮어쓰지 않음")
        return None
    for c in LOG_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[LOG_COLS]


def settle(log, feats, h=20):
    if log is None or log.empty:
        return log
    px = feats[["date", "ticker", "close"]].copy().sort_values(["ticker", "date"])
    px["fut"] = px.groupby("ticker")["close"].shift(-h)
    px["price_date"] = px["date"].dt.strftime("%Y-%m-%d")
    m = log.drop(columns=[c for c in ("fut",) if c in log]).merge(px[["price_date", "ticker", "fut"]], on=["price_date", "ticker"], how="left")
    need = (m["settled"] != True) & m["fut"].notna() & m["close"].notna()   # noqa: E712
    m.loc[need, "ret20"] = m.loc[need, "fut"] / m.loc[need, "close"] - 1
    m.loc[need, "settled"] = True
    return m[LOG_COLS]


def daily(log_dir="logs", cfg=None, fetcher=None, now=None):
    cfg = cfg or xp.load_cfg()
    if cfg is None or "top20_v43" not in cfg:
        return None
    t = _cfg(cfg)
    h = int(t["horizon"])
    panel = xp.load_panel(cfg, fetcher)
    if panel is None:
        return None
    df = xm.add_labels(xp.build_features(panel), [h])
    last = df["date"].max()
    dates = np.sort(df["date"].unique())
    cutoff = dates[-(h + 2)] if len(dates) > h + 2 else None
    if cutoff is None:
        return None
    tr = df[df["date"] <= cutoff].dropna(subset=_feats() + [f"fret{h}"])
    te = df[df["date"] == last].dropna(subset=_feats())
    if len(tr) < 2000 or len(te) < 10:
        print("⚠️ 학습/예측 표본 부족 — 생략")
        return None
    pred, _ = xv._fit_predict(tr, te, _feats(), f"fret{h}", cfg["variants_v42"][t["model"]], h)
    te = te.copy()
    te["pred20"] = pred
    te["rank_pct"] = te["pred20"].rank(pct=True)
    k = n_picks(len(te), t)
    te["is_top"] = te["pred20"].rank(ascending=False, method="first") <= k
    te["price_date"] = pd.Timestamp(last).strftime("%Y-%m-%d")
    te["ret20"], te["settled"] = np.nan, False
    te["run_at"] = str(now or pd.Timestamp.utcnow().tz_localize(None))
    old = load_log(log_dir)
    if old is None:
        return None
    new = te[LOG_COLS]
    log = new if old.empty else pd.concat([old, new], ignore_index=True)
    log = log.sort_values("run_at").drop_duplicates(["price_date", "ticker"], keep="last")
    log = settle(log, df, h).sort_values(["price_date", "ticker"])
    os.makedirs(log_dir, exist_ok=True)
    tmp = os.path.join(log_dir, LOG_NAME + ".tmp")
    log.to_csv(tmp, index=False)
    os.replace(tmp, os.path.join(log_dir, LOG_NAME))
    return log, df


def basket(log, cfg, closes):
    """등록일부터 20거래일마다 교체되는 '공식 바구니': (교체일, 종목들, 교체 후 수익, 동일가중 수익, 경과 거래일)."""
    t = _cfg(cfg)
    reg = pd.Timestamp(t["registered_on"])
    ds = sorted(d for d in log["price_date"].unique() if pd.Timestamp(d) >= reg)
    if not ds:
        return None
    reb = ds[::int(t["rebalance_every"])][-1]
    g = log[log["price_date"] == reb]
    picks = sorted(g[g["is_top"] == True]["ticker"])   # noqa: E712
    last = closes["date"].max()
    c0 = closes[closes["date"] == pd.Timestamp(reb)].set_index("ticker")["close"]
    c1 = closes[closes["date"] == last].set_index("ticker")["close"]
    r = (c1 / c0 - 1).dropna()
    if r.empty:
        return None
    return {"rebalance_date": reb, "picks": picks, "ret_top": float(r.reindex(picks).mean()), "ret_ew": float(r.mean()),
            "days": int(len([d for d in ds if d > reb]))}


def status_text(log, cfg, closes):
    if log is None or log.empty:
        return ""
    t = _cfg(cfg)
    last = log[log["price_date"] == log["price_date"].max()].sort_values("pred20", ascending=False)
    L = [f"📌 20일 상위10% 전략 (미검증 · 연구용 · 매매 신호 아님) 기준 {last['price_date'].iloc[0]}",
         "오늘 예측 상위: " + ", ".join(last[last["is_top"] == True]["ticker"])]   # noqa: E712
    b = basket(log, cfg, closes)
    if b:
        L.append(f"공식 바구니(교체일 {b['rebalance_date']}): {', '.join(b['picks'])} · 이후 {b['ret_top'] * 100:+.1f}% vs 29종목 평균 {b['ret_ew'] * 100:+.1f}% ({b['days']}거래일 경과)")
    s = log[(log["settled"] == True) & (pd.to_datetime(log["price_date"]) >= pd.Timestamp(t["registered_on"]))]   # noqa: E712
    if len(s):
        ex = s.groupby("price_date").apply(lambda g: g[g["is_top"] == True]["ret20"].mean() - g["ret20"].mean(), include_groups=False)   # noqa: E712
        L.append(f"정산된 날 {len(ex)}일: 상위10% 20일 초과 평균 {ex.mean() * 100:+.2f}% (독립 구간 3개 전에는 판정 안 함)")
    return "\n".join(L)


def main(argv):
    mode = argv[1] if len(argv) > 1 else "daily"
    cfg = xp.load_cfg()
    if cfg is None or "top20_v43" not in cfg:
        print("설정 없음 — 생략")
        return 0
    if mode == "backtest":
        out_dir = argv[2] if len(argv) > 2 else "reports"
        panel = xp.load_panel(cfg)
        if panel is None:
            return 0
        df = xm.add_labels(xp.build_features(panel), [int(_cfg(cfg)["horizon"])])
        res = backtest(df, cfg)
        if not res:
            print("평가 불가 — 생략")
            return 0
        txt = render_bt(res, cfg)
        print(txt)
        os.makedirs(out_dir, exist_ok=True)
        open(os.path.join(out_dir, "top20_backtest.txt"), "w", encoding="utf-8").write(txt + "\n")
        json.dump(res, open(os.path.join(out_dir, "top20_backtest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
        return 0
    log_dir = argv[2] if len(argv) > 2 else "logs"
    out_dir = argv[3] if len(argv) > 3 else "reports"
    out = daily(log_dir, cfg)
    if not out:
        return 0
    log, df = out
    txt = status_text(log, cfg, df[["date", "ticker", "close"]])
    if txt:
        print(txt)
        os.makedirs(out_dir, exist_ok=True)
        open(os.path.join(out_dir, "top20_status.txt"), "w", encoding="utf-8").write(txt + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
