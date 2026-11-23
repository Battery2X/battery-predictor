"""
xs_magnitude.py — '몇 % 오르나'(수익률 크기) 예측 검증 (v41, 연구용 · 읽기 전용).

질문 두 개를 나눠서 본다
1) 수익률 자체(부호 포함, 예: +2.3%)를 맞히나?
   비교 기준: '학습 구간 평균 수익률'을 그대로 예측값으로 쓰는 것(가장 단순한 예측).
   지표: 예측 오차가 기준보다 얼마나 줄었나(설명력 R², 0보다 크면 기준보다 나음) + 예측을 10단계로 나눴을 때 실제 수익률이 순서대로 나오나.
2) 움직임 크기(방향 무시, 예: ±3%)를 맞히나?
   비교 기준: '최근 20일 평균 변동폭'(이미 아는 정보로 만든 단순 예측).
   지표: 예측 크기와 실제 크기의 상관, 오차가 기준보다 줄었나.
호라이즌 1·5·20거래일. 5·20일은 겹치는 표본이라 신뢰구간은 h일 간격 날짜만으로 계산.

규칙: 피처는 xs_predictor 와 동일(t 종가까지). h일 라벨은 t+h 종가까지 필요 → 재학습 시점 T 에는 날짜 < T−h 인 행만 학습(퍼지).
설정은 xs_config.json 'magnitude'(사전등록). 기존 방향 모델·로그 불변 → 검증 시계 리셋 없음. 백테스트는 실전 성적 아님.

실행: python xs_magnitude.py [출력=reports]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import xs_predictor as xp


def add_labels(df, horizons):
    df = df.sort_values(["ticker", "date"]).copy()
    for h in horizons:
        df[f"fret{h}"] = df.groupby("ticker")["close"].shift(-h) / df["close"] - 1
    r1 = df.groupby("ticker")["close"].pct_change()
    df["absavg20"] = r1.abs().groupby(df["ticker"]).transform(lambda s: s.rolling(20).mean())   # t 까지
    return df


def walk(df, cfg, h):
    import lightgbm as lgb
    feats = xp.FEATURES
    y = f"fret{h}"
    ok = df.dropna(subset=feats + [y, "absavg20"])
    dates = np.sort(df["date"].unique())
    start = pd.Timestamp(dates[0]) + pd.DateOffset(years=int(cfg["min_train_years"]))
    test = dates[dates >= np.datetime64(start)]
    step = int(cfg["magnitude"]["retrain_every"])
    out = []
    for i in range(0, len(test), step):
        block = test[i:i + step]
        T = block[0]
        prior = dates[dates < T]
        if len(prior) <= h + 1:
            continue
        cutoff = prior[-(h + 1)]                     # 이 날짜까지의 행은 라벨(t+h)이 T 이전에 확정
        tr = ok[ok["date"] <= cutoff]
        te = df[df["date"].isin(block)].dropna(subset=feats + ["absavg20"])
        if len(tr) < 2000 or te.empty:
            continue
        p = dict(cfg["lgbm"])
        m_ret = lgb.LGBMRegressor(**p).fit(tr[feats + ["absavg20"]], tr[y])
        m_abs = lgb.LGBMRegressor(**p).fit(tr[feats + ["absavg20"]], tr[y].abs())
        te = te.copy()
        te["pred_ret"] = m_ret.predict(te[feats + ["absavg20"]])
        te["pred_abs"] = m_abs.predict(te[feats + ["absavg20"]])
        te["base_ret"] = float(tr[y].mean())
        te["base_abs"] = te["absavg20"] * np.sqrt(h) * 1.0     # 최근 변동폭을 h일로 늘린 단순 예측
        te["train_cut"] = cutoff
        out.append(te)
    return pd.concat(out, ignore_index=True) if out else None


def _r2_by_date(p, y, pred, base, h):
    d = p.dropna(subset=[y]).copy()
    d["e_m"] = (d[y] - d[pred]) ** 2
    d["e_b"] = (d[y] - d[base]) ** 2
    g = d.groupby("date")[["e_m", "e_b"]].sum()
    g = g.iloc[::h]                                   # 겹침 제거: h일 간격 날짜만
    r2 = 1 - g["e_m"].sum() / g["e_b"].sum()
    rel = (g["e_b"] - g["e_m"]) / g["e_b"].mean()     # 날짜별 개선량(정규화) → 부트스트랩
    ci = xp.block_ci(rel.values, block=10)
    return float(r2), ci, int(len(g))


def evaluate(pred, h):
    y = f"fret{h}"
    p = pred.dropna(subset=[y])
    if p.empty:
        return None
    r2, r2ci, nd = _r2_by_date(p, y, "pred_ret", "base_ret", h)
    p = p.copy()
    p["actual_abs"] = p[y].abs()
    ar2, ar2ci, _ = _r2_by_date(p, "actual_abs", "pred_abs", "base_abs", h)
    # 10단계 표(예측 수익률 순서 → 실제 평균)
    p["dec"] = p.groupby("date")["pred_ret"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) if len(s) >= 10 else np.nan)
    dec = p.dropna(subset=["dec"]).groupby("dec").agg(pred=("pred_ret", "mean"), actual=(y, "mean"), up=(y, lambda s: (s > 0).mean()), n=(y, "size")).reset_index()
    big = p[p["pred_ret"] >= p["pred_ret"].quantile(0.9)]
    return {"h": h, "n": int(len(p)), "dates_indep": nd,
            "r2_ret": r2, "r2_ret_ci": r2ci,
            "corr_ret": float(p["pred_ret"].corr(p[y], method="spearman")),
            "r2_abs": ar2, "r2_abs_ci": ar2ci,
            "corr_abs_model": float(p["pred_abs"].corr(p["actual_abs"], method="spearman")),
            "corr_abs_base": float(p["base_abs"].corr(p["actual_abs"], method="spearman")),
            "mae_ret_model": float((p[y] - p["pred_ret"]).abs().mean()), "mae_ret_base": float((p[y] - p["base_ret"]).abs().mean()),
            "top10_pred_mean": float(big["pred_ret"].mean()), "top10_actual_mean": float(big[y].mean()),
            "avg_actual": float(p[y].mean()), "deciles": dec.to_dict("records")}


def render(res, cfg):
    pc = lambda x, d=2: f"{x * 100:+.{d}f}%"
    L = ["=== '몇 % 오르나' 예측 검증 (29종목, 2013~, 백테스트 · 실전 성적 아님 · 종목 선택 편향 있음) ===",
         "R²: 0보다 크면 단순 기준보다 정확, 0 이하면 기준보다 못함. [ ] 는 95% 구간(0을 넘어야 우연 아님)."]
    for r in res:
        h = r["h"]
        ci = r["r2_ret_ci"]
        ai = r["r2_abs_ci"]
        L += ["", f"[{h}거래일 뒤] 예측 {r['n']:,}건 · 독립 날짜 {r['dates_indep']}개",
              f"  수익률(부호 포함) R² {r['r2_ret'] * 100:+.2f}% · 개선 구간 {'n/a' if not ci else f'[{ci[1]:+.3f}, {ci[2]:+.3f}]'}"
              f" · 순위 상관 {r['corr_ret']:+.3f}",
              f"  평균 오차: 모델 {r['mae_ret_model'] * 100:.2f}%p vs 기준 {r['mae_ret_base'] * 100:.2f}%p",
              f"  '많이 오른다' 예측 상위 10%: 예측 평균 {pc(r['top10_pred_mean'])} → 실제 평균 {pc(r['top10_actual_mean'])} (전체 평균 {pc(r['avg_actual'])})",
              f"  움직임 크기 R²(기준=최근 20일 변동폭) {r['r2_abs'] * 100:+.1f}% · 개선 구간 {'n/a' if not ai else f'[{ai[1]:+.3f}, {ai[2]:+.3f}]'}"
              f" · 상관 모델 {r['corr_abs_model']:+.3f} vs 기준 {r['corr_abs_base']:+.3f}",
              "  10단계(예측 낮음→높음) 실제 평균: " + " ".join(f"{d['actual'] * 100:+.2f}" for d in r["deciles"])]
    return "\n".join(L)


def main(argv):
    out_dir = argv[1] if len(argv) > 1 else "reports"
    cfg = xp.load_cfg()
    if cfg is None or "magnitude" not in cfg:
        print("설정 없음 — 생략")
        return 0
    panel = xp.load_panel(cfg)
    if panel is None:
        print("데이터 없음 — 생략")
        return 0
    df = add_labels(xp.build_features(panel), cfg["magnitude"]["horizons"])
    res = []
    for h in cfg["magnitude"]["horizons"]:
        pr = walk(df, cfg, h)
        ev = evaluate(pr, h) if pr is not None else None
        if ev:
            res.append(ev)
    if not res:
        print("평가 불가 — 생략")
        return 0
    txt = render(res, cfg)
    print(txt)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "xs_magnitude.txt"), "w", encoding="utf-8").write(txt + "\n")
    json.dump(res, open(os.path.join(out_dir, "xs_magnitude.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
