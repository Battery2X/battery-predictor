"""
xs_predictor.py — 고성장 개별주 '다음 날' 예측기 (v40, 기존 방향 모델과 별개인 연구 모듈).

무엇을 예측하나 (두 가지를 같은 피처로)
1) 상대 순위: 내일 이 종목이 '유니버스 평균보다' 더 오를까 → LightGBM 회귀(내일 수익률 − 그날 종목 평균). 평가: 날짜별 RankIC.
2) 절대 방향: 내일 오를까 → LightGBM 분류. 평가: 적중률·Brier 를 '그날 알 수 있던 상승 비율'(ex-ante 기준선)과 비교.
   하루에 종목 수만큼 표본이 생겨 검증이 지수 1개보다 훨씬 빠르다(단, 같은 날 종목끼리는 상관 → 날짜 블록 부트스트랩).

설계 출처(오픈소스 참고, 코드 복사 없음)
- Microsoft Qlib: Alpha158 계열 피처(수익률·이동평균 비율·변동성·고저 범위 위치), 날짜별 순위 정규화(CSRankNorm),
  롤링 재학습, IC/RankIC 평가.
- alphalens(-reloaded): RankIC 시계열, 분위 수익률(상위-하위) 관점.

규칙
- 미래정보 차단: t 일 피처는 t 종가까지. 라벨은 t+1 수익률. 재학습 시점 T 에는 라벨이 확정된 행(날짜 ≤ T−1)만 학습.
- 설정은 xs_config.json 사전등록(해시 테스트). 기존 방향 모델·예측 로그와 무관 → 검증 시계 리셋 없음.
- 한계: 종목 선택 편향(이미 성장한 종목), 상장 기간이 짧은 종목(PLTR·CRWD 등)은 뒤에만 참여, 거래비용은 근사, 백테스트는 실전 성적 아님.

실행
  python xs_predictor.py backtest [출력=reports]
  python xs_predictor.py daily    [logs=logs] [출력=reports]
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(HERE, "xs_config.json")
LOG_NAME = "xs_predictions.csv"
LOG_COLS = ["price_date", "ticker", "pred_rank_score", "pred_rank_pct", "pred_up_prob", "base_up", "close", "next_ret", "settled", "run_at"]
FEATURES = None  # build_features 가 채움


def load_cfg(path=CFG_PATH):
    try:
        with open(path, encoding="utf-8") as f:
            c = json.load(f)
        assert c.get("universe") and c.get("registered_on")
        return c
    except Exception as e:
        print(f"⚠️ xs_config 읽기 실패({e}) — 생략")
        return None


# ---------------------------------------------------------------- 데이터
def fetch_ohlcv(sym, start):
    import common as cm
    df = cm._yf_download_with_retry(sym, start)
    if df is None or df.empty:
        return None
    cols = {}
    for c in ("Open", "High", "Low", "Close", "Volume"):
        s = cm._flat_col(df, c)
        if s is not None:
            cols[c] = s
    if "Close" not in cols:
        return None
    out = pd.DataFrame(cols)
    out = cm.drop_partial_us_bar(out)
    return out.dropna(subset=["Close"])


def load_panel(cfg, fetcher=None):
    fetcher = fetcher or fetch_ohlcv
    frames = []
    for t in cfg["universe"]:
        try:
            df = fetcher(t, cfg["start"])
        except Exception as e:
            print(f"⚠️ [{t}] 시세 실패({e}) — 생략")
            df = None
        if df is None or len(df) < 300:
            print(f"⚠️ [{t}] 데이터 부족 — 생략")
            continue
        d = df.copy()
        for c in ("High", "Low"):
            if c not in d:
                d[c] = d["Close"]
        if "Volume" not in d:
            d["Volume"] = np.nan
        d["ticker"] = t
        d.index = pd.to_datetime(d.index)
        d.index.name = "date"
        frames.append(d[["High", "Low", "Close", "Volume", "ticker"]])
    if not frames:
        return None
    return pd.concat(frames).reset_index()


# ---------------------------------------------------------------- 피처 (Alpha158 축약) · 라벨
def _per_ticker(g):
    c, h, l, v = g["Close"], g["High"], g["Low"], g["Volume"]
    r = c.pct_change()
    f = pd.DataFrame(index=g.index)
    for n in (1, 2, 3, 5, 10, 20, 60):
        f[f"ret{n}"] = c / c.shift(n) - 1
    for n in (5, 10, 20, 60):
        f[f"ma{n}"] = c / c.rolling(n).mean() - 1
    for n in (5, 20, 60):
        f[f"std{n}"] = r.rolling(n).std()
    for n in (10, 20, 60):
        hi, lo = h.rolling(n).max(), l.rolling(n).min()
        f[f"rsv{n}"] = (c - lo) / (hi - lo).replace(0, np.nan)
    f["hl1"] = (h - l) / c
    f["hl5"] = ((h - l) / c).rolling(5).mean()
    up = r.clip(lower=0).rolling(14).mean()
    dn = (-r.clip(upper=0)).rolling(14).mean()
    f["rsi14"] = up / (up + dn).replace(0, np.nan)
    lv = np.log1p(v)
    f["vol5"] = lv - lv.rolling(20).mean() if v.notna().any() else np.nan
    f["skew20"] = r.rolling(20).skew()
    f["next_ret"] = c.shift(-1) / c - 1           # 라벨: 내일 수익률 (피처 아님)
    return f


def build_features(panel):
    global FEATURES
    p = panel.sort_values(["ticker", "date"]).copy()
    parts = []
    for t, g in p.groupby("ticker", sort=False):
        g = g.set_index("date")
        f = _per_ticker(g)
        f["ticker"], f["close"] = t, g["Close"]
        parts.append(f.reset_index())
    df = pd.concat(parts, ignore_index=True)
    base = [c for c in df.columns if c not in ("date", "ticker", "close", "next_ret")]
    # 시장(유니버스 동일가중) 맥락: 절대 방향 예측용
    mk = df.groupby("date")[["ret1", "ret5", "ret20", "std20"]].mean().add_prefix("mkt_")
    df = df.merge(mk, left_on="date", right_index=True, how="left")
    # 날짜별 순위 정규화(0~1): 상대 순위 예측의 표준 전처리
    rk = df.groupby("date")[base].rank(pct=True).add_prefix("cs_")
    df = pd.concat([df, rk], axis=1)
    FEATURES = [f"cs_{c}" for c in base] + list(mk.columns) + ["ret1", "ret5", "ret20", "std20"]
    df["n_on_date"] = df.groupby("date")["ticker"].transform("count")
    df["excess_next"] = df["next_ret"] - df.groupby("date")["next_ret"].transform("mean")
    df["up_next"] = (df["next_ret"] > 0).astype(float).where(df["next_ret"].notna())
    return df


# ---------------------------------------------------------------- 학습·워크포워드
def _models(cfg):
    import lightgbm as lgb
    p = dict(cfg["lgbm"])
    return lgb.LGBMRegressor(**p), lgb.LGBMClassifier(**p)


def walk_forward(df, cfg, min_stocks=5):
    """재학습 시점마다 '라벨 확정 행'만 학습하고 다음 구간을 예측. 반환: 예측이 붙은 행들."""
    dates = np.sort(df["date"].unique())
    ok = df.dropna(subset=FEATURES + ["next_ret"]).copy()
    ok = ok[ok["n_on_date"] >= min_stocks]
    if ok.empty:
        return None
    start = pd.Timestamp(dates[0]) + pd.DateOffset(years=int(cfg["min_train_years"]))
    test_dates = dates[dates >= np.datetime64(start)]
    step = int(cfg["retrain_every"])
    out = []
    for i in range(0, len(test_dates), step):
        block = test_dates[i:i + step]
        T = block[0]
        tr = ok[ok["date"] < T]                       # 날짜 < T: 라벨(다음 날)이 T 이하 → 확정
        tr = tr[tr["date"] < tr["date"].max()] if len(tr) else tr   # 1일 퍼지(경계 겹침 방지)
        te = df[df["date"].isin(block)].dropna(subset=FEATURES)
        if len(tr) < 2000 or te.empty:
            continue
        reg, clf = _models(cfg)
        reg.fit(tr[FEATURES], tr["excess_next"])
        clf.fit(tr[FEATURES], tr["up_next"].astype(int))
        te = te.copy()
        te["pred_rank_score"] = reg.predict(te[FEATURES])
        te["pred_up_prob"] = clf.predict_proba(te[FEATURES])[:, 1]
        te["base_up"] = float(tr["up_next"].mean())            # ex-ante 기준선: 학습 구간 상승 비율
        te["train_end"] = tr["date"].max()
        out.append(te)
    return pd.concat(out, ignore_index=True) if out else None


# ---------------------------------------------------------------- 평가
def _spearman(a, b):
    a, b = pd.Series(a).rank(), pd.Series(b).rank()
    if a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def block_ci(x, block=20, n_boot=2000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2 * block:
        return None
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))
    m = np.empty(n_boot)
    for b in range(n_boot):
        st = rng.integers(0, n - block + 1, nb)
        m[b] = np.concatenate([x[s:s + block] for s in st])[:n].mean()
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def evaluate(pred, cfg, min_stocks=5):
    p = pred.dropna(subset=["next_ret", "pred_rank_score", "pred_up_prob"])
    if p.empty:
        return None
    k, cost = int(cfg["top_k"]), cfg["cost_bp_per_side"] / 1e4
    daily = []
    for d, g in p.groupby("date"):
        if len(g) < max(min_stocks, 2 * k):
            continue
        g = g.sort_values("pred_rank_score")
        top, bot = g.tail(k), g.head(k)
        daily.append({"date": d, "ic": _spearman(g["pred_rank_score"], g["next_ret"]),
                      "top": top["next_ret"].mean(), "bot": bot["next_ret"].mean(), "ew": g["next_ret"].mean(),
                      "top_set": frozenset(top["ticker"]), "n": len(g)})
    D = pd.DataFrame(daily).sort_values("date")
    if D.empty:
        return None
    # 회전율(상위 k 교체 비율) → 비용
    turn = [np.nan] + [1 - len(a & b) / k for a, b in zip(D["top_set"].iloc[:-1], D["top_set"].iloc[1:])]
    D["turnover"] = turn
    D["top_net"] = D["top"] - 2 * cost * D["turnover"].fillna(1)
    D["excess_net"] = D["top_net"] - D["ew"]
    D["ls"] = D["top"] - D["bot"]
    eq = lambda r: float(np.prod(1 + np.nan_to_num(r)) - 1)
    years = len(D) / 252
    cagr = lambda r: float((1 + eq(r)) ** (1 / years) - 1) if years >= 1 else None
    # 방향
    acc = float(((p["pred_up_prob"] >= 0.5) == (p["next_ret"] > 0)).mean())
    base_acc = float(((p["base_up"] >= 0.5) == (p["next_ret"] > 0)).mean())
    y = (p["next_ret"] > 0).astype(float)
    brier_m = (p["pred_up_prob"] - y) ** 2
    brier_b = (p["base_up"] - y) ** 2
    bd = (brier_m - brier_b).groupby(p["date"]).mean()
    hc = p[(p["pred_up_prob"] - p["base_up"]).abs() >= 0.03]         # 엣지 3%p 이상만
    hc_acc = float(((hc["pred_up_prob"] >= 0.5) == (hc["next_ret"] > 0)).mean()) if len(hc) else None
    D["year"] = pd.to_datetime(D["date"]).dt.year
    by_year = D.groupby("year").agg(ic=("ic", "mean"), excess=("excess_net", "sum"), days=("ic", "size")).reset_index()
    per_ticker = p.groupby("ticker").apply(lambda g: pd.Series({
        "acc": float(((g["pred_up_prob"] >= 0.5) == (g["next_ret"] > 0)).mean()),
        "base_acc": float(((g["base_up"] >= 0.5) == (g["next_ret"] > 0)).mean()),
        "up_pred_share": float((g["pred_up_prob"] >= 0.5).mean()), "n": len(g)}), include_groups=False).reset_index()
    return {
        "period": [str(pd.Timestamp(D["date"].iloc[0]).date()), str(pd.Timestamp(D["date"].iloc[-1]).date())], "days": len(D),
        "rank_ic": block_ci(D["ic"]), "icir": float(D["ic"].mean() / D["ic"].std()) if D["ic"].std() > 0 else None,
        "ic_pos_share": float((D["ic"] > 0).mean()),
        "long_short_daily": block_ci(D["ls"]), "top_excess_net_daily": block_ci(D["excess_net"]),
        "cagr_top_net": cagr(D["top_net"]), "cagr_ew": cagr(D["ew"]), "mean_turnover": float(D["turnover"].mean()),
        "dir_acc": acc, "dir_base_acc": base_acc, "dir_n": int(len(p)), "dir_brier_diff": block_ci(bd.values),
        "dir_up_pred_share": float((p["pred_up_prob"] >= 0.5).mean()), "dir_edge3_acc": hc_acc, "dir_edge3_n": int(len(hc)),
        "by_year": by_year.to_dict("records"), "per_ticker": per_ticker.to_dict("records"),
    }


def render(ev, cfg):
    f = lambda x, d=2: "n/a" if x is None else f"{x * 100:+.{d}f}%"
    ci = lambda c, s=100, d=2: "n/a" if not c else f"{c[0] * s:+.{d}f} [{c[1] * s:+.{d}f}, {c[2] * s:+.{d}f}]"
    e3 = "n/a" if ev["dir_edge3_acc"] is None else f"{ev['dir_edge3_acc'] * 100:.1f}%"
    L = [f"=== 교차종목 다음날 예측 워크포워드 ({ev['period'][0]}~{ev['period'][1]}, {ev['days']}거래일, 종목 {len(cfg['universe'])}개 등록) ===",
         f"설정 등록일 {cfg['registered_on']} · 백테스트는 실전 성적 아님 · 종목 선택 편향 있음 · 거래비용 편도 {cfg['cost_bp_per_side']}bp 반영(상위 {cfg['top_k']}개)",
         "", "[1] 상대 순위 (어느 종목이 평균보다 더 오를까)",
         f"  RankIC 평균(×100) {ci(ev['rank_ic'])} · ICIR {ev['icir']:.3f} · IC>0 인 날 {ev['ic_pos_share'] * 100:.0f}%" if ev["icir"] is not None else "  RankIC n/a",
         f"  상위{cfg['top_k']}−하위{cfg['top_k']} 일평균(%) {ci(ev['long_short_daily'])}",
         f"  상위{cfg['top_k']}(비용 후) − 동일가중 일평균(%) {ci(ev['top_excess_net_daily'])} · 회전율 {ev['mean_turnover'] * 100:.0f}%/일",
         f"  연환산: 상위{cfg['top_k']} 비용후 {f(ev['cagr_top_net'], 1)} vs 동일가중 {f(ev['cagr_ew'], 1)}",
         "", "[2] 절대 방향 (내일 오를까)",
         f"  적중률 {ev['dir_acc'] * 100:.1f}% vs 기준선(늘 다수 방향) {ev['dir_base_acc'] * 100:.1f}% · n={ev['dir_n']:,} · 상승 예측 비율 {ev['dir_up_pred_share'] * 100:.0f}%",
         f"  Brier(모델−기준선, 낮을수록 모델 우위, ×100) {ci(ev['dir_brier_diff'], 100, 3)}",
         f"  엣지 ≥3%p 인 예측만: 적중률 {e3} (n={ev['dir_edge3_n']:,})",
         "", "[3] 연도별 RankIC(×100) · 상위 초과수익 합(비용 후)"]
    for r in ev["by_year"]:
        L.append(f"  {r['year']}: IC {r['ic'] * 100:+.2f} · 초과 {r['excess'] * 100:+.1f}% · {r['days']}일")
    L += ["", "[4] 종목별 방향 적중률 (모델 vs 기준선, 상승 예측 비율)"]
    for r in sorted(ev["per_ticker"], key=lambda r: -r["acc"] + r["base_acc"]):
        L.append(f"  {r['ticker']:<5} {r['acc'] * 100:5.1f}% vs {r['base_acc'] * 100:5.1f}% ({(r['acc'] - r['base_acc']) * 100:+.1f}%p) · 상승예측 {r['up_pred_share'] * 100:3.0f}% · n={int(r['n'])}")
    return "\n".join(L)


# ---------------------------------------------------------------- 매일 예측 기록 (등록일 이후 실전 표본)
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


def settle(log, feats):
    """미정산 행의 next_ret 을 채운다(같은 종목의 다음 거래일 종가 기준)."""
    if log is None or log.empty:
        return log
    px = feats[["date", "ticker", "close"]].copy()
    px["price_date"] = px["date"].dt.strftime("%Y-%m-%d")
    px = px.sort_values(["ticker", "date"])
    px["nxt"] = px.groupby("ticker")["close"].shift(-1)
    m = log.merge(px[["price_date", "ticker", "nxt"]], on=["price_date", "ticker"], how="left")
    need = (m["settled"] != True) & m["nxt"].notna() & m["close"].notna()   # noqa: E712
    m.loc[need, "next_ret"] = m.loc[need, "nxt"] / m.loc[need, "close"] - 1
    m.loc[need, "settled"] = True
    return m[LOG_COLS]


def daily(log_dir="logs", cfg=None, fetcher=None, now=None):
    cfg = cfg or load_cfg()
    if cfg is None:
        return None
    panel = load_panel(cfg, fetcher)
    if panel is None:
        return None
    feats = build_features(panel)
    last = feats["date"].max()
    tr = feats.dropna(subset=FEATURES + ["next_ret"])
    te = feats[(feats["date"] == last)].dropna(subset=FEATURES)
    if len(tr) < 2000 or len(te) < 5:
        print("⚠️ 학습/예측 표본 부족 — 생략")
        return None
    reg, clf = _models(cfg)
    reg.fit(tr[FEATURES], tr["excess_next"])
    clf.fit(tr[FEATURES], tr["up_next"].astype(int))
    te = te.copy()
    te["pred_rank_score"] = reg.predict(te[FEATURES])
    te["pred_rank_pct"] = te["pred_rank_score"].rank(pct=True)
    te["pred_up_prob"] = clf.predict_proba(te[FEATURES])[:, 1]
    te["base_up"] = float(tr["up_next"].mean())
    te["price_date"] = last.strftime("%Y-%m-%d")
    te["next_ret"], te["settled"] = np.nan, False
    te["run_at"] = str(now or pd.Timestamp.utcnow().tz_localize(None))
    old = load_log(log_dir)
    if old is None:
        return None
    new = te[LOG_COLS]
    log = new if old.empty else pd.concat([old, new], ignore_index=True)
    log = log.sort_values("run_at").drop_duplicates(["price_date", "ticker"], keep="last")
    log = settle(log, feats).sort_values(["price_date", "ticker"])
    os.makedirs(log_dir, exist_ok=True)
    tmp = os.path.join(log_dir, LOG_NAME + ".tmp")
    log.to_csv(tmp, index=False)
    os.replace(tmp, os.path.join(log_dir, LOG_NAME))
    return log


def status_text(log, cfg, k=None):
    """텔레그램용: 오늘 상위/하위 종목 + 등록 후 실전 성적(표본이 쌓이기 전에는 숫자만)."""
    if log is None or log.empty:
        return ""
    k = k or int(cfg["top_k"])
    last = log[log["price_date"] == log["price_date"].max()].sort_values("pred_rank_score", ascending=False)
    top = ", ".join(last["ticker"].head(k))
    bot = ", ".join(last["ticker"].tail(k))
    s = log[(log["settled"] == True) & (pd.to_datetime(log["price_date"]) >= pd.Timestamp(cfg["registered_on"]))]   # noqa: E712
    L = [f"🔬 교차종목 예측 (미검증 · 연구용 · 매매 신호 아님) 기준 {last['price_date'].iloc[0]}",
         f"상대 강세 예상 상위{k}: {top}", f"상대 약세 예상 하위{k}: {bot}"]
    nd = s["price_date"].nunique()
    if nd:
        ics = [_spearman(g["pred_rank_score"], g["next_ret"]) for _, g in s.groupby("price_date") if len(g) >= 5]
        acc = float(((s["pred_up_prob"] >= 0.5) == (s["next_ret"] > 0)).mean())
        L.append(f"등록 후 {nd}거래일: RankIC 평균 {np.nanmean(ics) * 100:+.1f} · 방향 적중 {acc * 100:.1f}% (60거래일 전 판정 안 함)")
    return "\n".join(L)


def main(argv):
    mode = argv[1] if len(argv) > 1 else "daily"
    cfg = load_cfg()
    if cfg is None:
        return 0
    if mode == "backtest":
        out_dir = argv[2] if len(argv) > 2 else "reports"
        panel = load_panel(cfg)
        if panel is None:
            print("데이터 없음 — 생략")
            return 0
        feats = build_features(panel)
        pred = walk_forward(feats, cfg)
        ev = evaluate(pred, cfg) if pred is not None else None
        if not ev:
            print("평가 불가 — 생략")
            return 0
        txt = render(ev, cfg)
        print(txt)
        os.makedirs(out_dir, exist_ok=True)
        open(os.path.join(out_dir, "xs_backtest.txt"), "w", encoding="utf-8").write(txt + "\n")
        json.dump(ev, open(os.path.join(out_dir, "xs_backtest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
        return 0
    log_dir = argv[2] if len(argv) > 2 else "logs"
    out_dir = argv[3] if len(argv) > 3 else "reports"
    log = daily(log_dir, cfg)
    txt = status_text(log, cfg)
    if txt:
        print(txt)
        os.makedirs(out_dir, exist_ok=True)
        open(os.path.join(out_dir, "xs_status.txt"), "w", encoding="utf-8").write(txt + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
