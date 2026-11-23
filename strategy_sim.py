"""
strategy_sim.py — 신호를 '실제로 따라 했다면' 수익률·최대낙폭이 어땠는지 계산하는 읽기 전용 시뮬레이터 (v38).

왜 필요한가
- 지금까지는 "방향을 맞혔나(적중률)"만 채점했다. 레버리지 보유자에게 더 중요한 건
  "최악일 때 얼마나 잃었나(최대낙폭)"와 "그 대가로 수익을 얼마나 포기했나"다.
- 방향 모델·피처·임계값·예측 로그는 전혀 건드리지 않는다 → 검증 시계 리셋 없음.
- 오픈소스 vectorbt 의 Portfolio 지표(총수익·CAGR·MDD·노출도) 개념을 빌렸지만, 의존성을 늘리지 않으려고
  (requirements 버전 고정 원칙, 라이선스 조건) numpy/pandas 로 직접 구현했다.

재발 방지 장치
- 규칙 숫자는 strategy_rules.json 에 '사전등록'. 결과를 보고 숫자를 바꾸면 등록일이 바뀌고,
  판정은 등록일 이후 구간으로만 한다(과거 구간은 '표본 내 참고').
- '방향 신호가 사실상 상수' 경보: 롱 상품에서 상승 예측 비율이 95% 이상이면 맨 위에 표시.
- 모든 신호는 '그날 종가까지 알 수 있던 정보'만 쓴다(다음 날 수익률에 적용). 테스트로 확인.

한계
- 거래비용·세금·슬리피지 미반영(현금 전환이 잦은 규칙일수록 실제보다 좋게 나온다).
- bench 모드의 합성 3배는 근사(3×일수익률 − 연 비용/252). 실제 SOXL 과 다를 수 있다.
- live 모드는 표본이 수십~백여 거래일이라 우연과 구분이 거의 안 된다. 백테스트는 실전 성적이 아니다.

실행
  python strategy_sim.py live  [logs폴더=logs] [출력=reports]
  python strategy_sim.py bench [출력=reports] [시작일=2010-01-01]   # 인터넷(야후) 필요, Actions 에서 실행
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategy_rules.json")
LONGS = {"TQQQ": "QQQ", "SOXL": "SMH", "SPXL": "SPY"}
DEGENERATE_SHARE = 0.95


# ---------------------------------------------------------------- 설정
def load_rules(path=RULES_PATH):
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        if "rules" not in cfg or "registered_on" not in cfg:
            raise ValueError("rules/registered_on 누락")
        return cfg
    except Exception as e:  # 설정이 깨져도 조용히 죽지 않고 이유를 남긴다
        print(f"⚠️ 규칙 파일을 읽지 못함({e}) — 시뮬레이션 생략")
        return None


# ---------------------------------------------------------------- 신호 (모두 ex-ante: i 번째 수익률엔 i-1 까지 정보만)
def ewma_vol_exante(ret, lam=0.94, min_obs=60):
    """i 번째 값 = ret[0..i-1] 로 만든 EWMA 일변동성. 관측이 부족하면 NaN."""
    r = pd.Series(np.asarray(ret, float))
    var = (r ** 2).ewm(alpha=1 - lam, adjust=False).mean()
    vol = np.sqrt(var).shift(1)                    # 당일 수익률은 쓰지 않는다
    cnt = r.notna().cumsum().shift(1).fillna(0)
    vol[cnt < min_obs] = np.nan
    return vol.values


def vol_gate(ret, lam=0.94, mult=2.0, min_obs=60):
    vol = pd.Series(ewma_vol_exante(ret, lam, min_obs))
    med = vol.expanding().median().shift(1)        # 기준(중앙값)도 과거만
    pos = pd.Series(1.0, index=vol.index)
    pos[(vol >= mult * med) & vol.notna() & med.notna()] = 0.0
    return pos.values


def crash_gate(ret, dd_window=60, dd_trigger=-0.30, reentry_ma=20):
    """가격 지수를 수익률로 복원 → 60일 고점 대비 낙폭이 기준 이하면 현금, 20일선 회복 시 복귀. 상태는 전일 종가 기준."""
    r = np.nan_to_num(np.asarray(ret, float))
    px = pd.Series(np.cumprod(1 + r))              # px[i] = i 번째 수익률 '이후' 종가
    peak = px.rolling(dd_window, min_periods=1).max()
    dd = px / peak - 1
    ma = px.rolling(reentry_ma, min_periods=reentry_ma).mean()
    pos = np.ones(len(r))
    out = False
    for i in range(1, len(r)):                     # i 번째 수익률은 i-1 종가에서 결정
        j = i - 1
        if not out and dd.iloc[j] <= dd_trigger:
            out = True
        elif out and not math.isnan(ma.iloc[j]) and px.iloc[j] > ma.iloc[j]:
            out = False
        pos[i] = 0.0 if out else 1.0
    return pos


def prob_gate(prob, min_prob=50.0):
    p = np.asarray(prob, float)
    return np.where(np.isnan(p), np.nan, (p >= min_prob).astype(float))


def edge_gate(edge, min_edge=3.0):
    e = np.asarray(edge, float)
    return np.where(np.isnan(e), np.nan, (e >= min_edge).astype(float))


# ---------------------------------------------------------------- 지표
def synth_leveraged(ret, lev=3.0, cost_annual=0.0889):
    return lev * np.asarray(ret, float) - cost_annual / 252.0


def metrics(ret, pos=None):
    """ret: 상품 일수익률, pos: 0~1 보유비중(없으면 1). 현금 수익은 0으로 가정."""
    r = np.asarray(ret, float)
    p = np.ones_like(r) if pos is None else np.asarray(pos, float)
    ok = ~(np.isnan(r) | np.isnan(p))
    r, p = r[ok], p[ok]
    n = len(r)
    if n == 0:
        return None
    s = p * r
    eq = np.cumprod(1 + s)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    dd = eq / peak - 1
    under, longest = 0, 0
    for x in dd:
        under = under + 1 if x < 0 else 0
        longest = max(longest, under)
    years = n / 252.0
    total = eq[-1] - 1
    cagr = (eq[-1] ** (1 / years) - 1) if years >= 1 and eq[-1] > 0 else np.nan
    return {"days": n, "total": float(total), "cagr": float(cagr) if not np.isnan(cagr) else None,
            "ann_vol": float(np.std(s) * math.sqrt(252)), "mdd": float(dd.min()),
            "longest_underwater_days": int(longest), "worst_day": float(s.min()),
            "exposure": float(p.mean()), "recovery_needed": float(1 / (1 + dd.min()) - 1) if dd.min() > -1 else None}


def block_bootstrap_mean(d, block=5, n_boot=2000, seed=0):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2 * block:
        return None
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        st = rng.integers(0, n - block + 1, nb)
        means[b] = np.concatenate([d[s:s + block] for s in st])[:n].mean()
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(ret, positions, registered_on=None, dates=None):
    """positions: {규칙명: 보유비중 배열}. 각 규칙을 HOLD 와 같은 날짜끼리만 비교."""
    ret = np.asarray(ret, float)
    out = {}
    for name, pos in positions.items():
        pos = np.asarray(pos, float)
        ok = ~(np.isnan(ret) | np.isnan(pos))
        if ok.sum() == 0:
            out[name] = None
            continue
        m = metrics(ret[ok], pos[ok])
        h = metrics(ret[ok])
        d = np.log1p(pos[ok] * ret[ok]) - np.log1p(ret[ok])
        m["vs_hold"] = {"hold_total": h["total"], "hold_mdd": h["mdd"],
                        "daily_logdiff_ci": block_bootstrap_mean(d)}
        if registered_on is not None and dates is not None:
            dts = pd.to_datetime(pd.Series(dates))[ok].values
            post = dts >= np.datetime64(pd.Timestamp(registered_on))
            m["post_registration_days"] = int(post.sum())
        out[name] = m
    return out


def degenerate_share(prob):
    p = np.asarray(prob, float)
    p = p[~np.isnan(p)]
    return float((p >= 50).mean()) if len(p) else None


# ---------------------------------------------------------------- 데이터 적재
def load_live(log_dir="logs"):
    """predictions_scalp.csv 의 롱 레버리지 행만. 같은 price_date 는 마지막 실행만 남긴다. 정산 안 된 행 제외."""
    path = os.path.join(log_dir, "predictions_scalp.csv")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"⚠️ 실전 로그를 읽지 못함({e}) — live 생략")
        return {}
    need = {"ticker", "price_date", "predicted_up_prob", "actual_return", "run_date"}
    if not need.issubset(df.columns):
        print(f"⚠️ 실전 로그 열 부족: {sorted(need - set(df.columns))} — live 생략")
        return {}
    edge = None
    try:
        eh = pd.read_csv(os.path.join(log_dir, "edge_history.csv"))
        eh = eh[eh["log"] == "scalp"][["ticker", "price_date", "edge"]].drop_duplicates(["ticker", "price_date"], keep="last")
        edge = eh
    except Exception:
        pass
    out = {}
    for t in LONGS:
        g = df[df["ticker"] == t].copy()
        if g.empty:
            continue
        g = g.sort_values("run_date").drop_duplicates("price_date", keep="last")
        g = g.dropna(subset=["actual_return"]).sort_values("price_date").reset_index(drop=True)
        if edge is not None:
            g = g.merge(edge[edge["ticker"] == t][["price_date", "edge"]], on="price_date", how="left")
        else:
            g["edge"] = np.nan
        out[t] = g
    return out


def run_live(log_dir="logs", cfg=None):
    cfg = cfg or load_rules()
    if cfg is None:
        return None
    R = cfg["rules"]
    res = {}
    for t, g in load_live(log_dir).items():
        if len(g) < 10:
            print(f"⚠️ [{t}] 정산된 행 {len(g)}개 — 너무 적어 생략")
            continue
        r = g["actual_return"].values
        pos = {"HOLD": np.ones(len(r)),
               "PROB_GATE": prob_gate(g["predicted_up_prob"], R["PROB_GATE"]["min_prob"]),
               "EDGE_GATE": edge_gate(g["edge"], R["EDGE_GATE"]["min_edge"]),
               "VOL_GATE": vol_gate(r, R["VOL_GATE"]["lam"], R["VOL_GATE"]["mult"], R["VOL_GATE"]["min_obs"]),
               "CRASH_GATE": crash_gate(r, R["CRASH_GATE"]["dd_window"], R["CRASH_GATE"]["dd_trigger"], R["CRASH_GATE"]["reentry_ma"])}
        res[t] = {"period": [str(g["price_date"].iloc[0]), str(g["price_date"].iloc[-1])],
                  "up_pred_share": degenerate_share(g["predicted_up_prob"]),
                  "edge_coverage": float(g["edge"].notna().mean()),
                  "rules": compare(r, pos, cfg["registered_on"], g["price_date"])}
    return res


def run_bench(start="2010-01-01", cfg=None):
    """야후에서 실제 레버리지 ETF + 벤치마크 합성 3배. 방향 모델 신호는 쓰지 않는다(장기 이력이 없으므로)."""
    cfg = cfg or load_rules()
    if cfg is None:
        return None
    R = cfg["rules"]
    try:
        import common as cm
    except Exception as e:
        print(f"⚠️ common 불러오기 실패({e}) — bench 생략")
        return None
    res = {}
    for t, b in LONGS.items():
        for name, sym, synth in ((t, t, False), (f"{b}x3(합성)", b, True)):
            try:
                df = cm.fetch_ticker_data(sym, start)
            except Exception as e:
                print(f"⚠️ [{sym}] 시세 실패({e}) — 생략")
                df = None
            if df is None or len(df) < 300:
                continue
            r = df["Close"].pct_change().dropna()
            rv = synth_leveraged(r.values, 3.0, cfg.get("leverage_cost_annual", 0.0889)) if synth else r.values
            pos = {"HOLD": np.ones(len(rv)),
                   "VOL_GATE": vol_gate(rv, R["VOL_GATE"]["lam"], R["VOL_GATE"]["mult"], R["VOL_GATE"]["min_obs"]),
                   "CRASH_GATE": crash_gate(rv, R["CRASH_GATE"]["dd_window"], R["CRASH_GATE"]["dd_trigger"], R["CRASH_GATE"]["reentry_ma"])}
            res[name] = {"period": [str(r.index[0].date()), str(r.index[-1].date())],
                         "rules": compare(rv, pos, cfg["registered_on"], r.index)}
    return res


# ---------------------------------------------------------------- 보고서
def _pct(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:+.{d}f}%"


def render(res, title, cfg):
    L = [f"=== {title} ===",
         f"규칙 등록일 {cfg['registered_on']} (이전 구간은 표본 내 참고) · 거래비용 미반영 · 실전 성적 아님"]
    for t, v in (res or {}).items():
        L.append(f"\n[{t}] {v['period'][0]} ~ {v['period'][1]}")
        s = v.get("up_pred_share")
        if s is not None and s >= DEGENERATE_SHARE:
            L.append(f"  ⚠️ 방향 신호가 사실상 상수: 상승 예측 {s * 100:.0f}% → PROB_GATE ≈ HOLD")
        for name, m in v["rules"].items():
            if m is None:
                L.append(f"  {name:<11} 데이터 없음")
                continue
            ci = m["vs_hold"]["daily_logdiff_ci"]
            sig = ""
            if ci:
                sig = " (HOLD 대비 유의하게 좋음)" if ci[1] > 0 else (" (유의하게 나쁨)" if ci[2] < 0 else " (HOLD와 구분 불가)")
                if m.get("post_registration_days", 1) == 0 and ci[1] > 0:   # 등록 전 구간의 '유의'는 판정이 아님
                    sig = " (HOLD보다 좋았음 — 등록 전 구간이라 표본 내 참고, 판정 아님)"
                elif m.get("post_registration_days", 1) == 0 and ci[2] < 0:
                    sig = " (HOLD보다 나빴음 — 등록 전 구간이라 표본 내 참고)"
            L.append(f"  {name:<11} 수익 {_pct(m['total'])} · 최대낙폭 {_pct(m['mdd'])} · 보유비중 {m['exposure'] * 100:.0f}%"
                     f" · {m['days']}일{sig if name != 'HOLD' else ''}")
    return "\n".join(L)


def main(argv):
    mode = argv[1] if len(argv) > 1 else "live"
    cfg = load_rules()
    if cfg is None:
        return 0
    if mode == "bench":
        out_dir = argv[2] if len(argv) > 2 else "reports"
        start = argv[3] if len(argv) > 3 else "2010-01-01"
        res, title = run_bench(start, cfg), "전략 시뮬레이션 (장기, 방향 모델 미사용)"
        stem = "strategy_sim_bench"
    else:
        log_dir = argv[2] if len(argv) > 2 else "logs"
        out_dir = argv[3] if len(argv) > 3 else "reports"
        res, title = run_live(log_dir, cfg), "전략 시뮬레이션 (실전 로그, 1일)"
        stem = "strategy_sim_live"
    if not res:
        print("결과 없음 — 생략")
        return 0
    txt = render(res, title, cfg)
    print(txt)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, stem + ".txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    with open(os.path.join(out_dir, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
