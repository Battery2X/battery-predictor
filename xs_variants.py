"""
xs_variants.py — '몇 % 오르나' 개선안 비교 (v42, 연구용 · 읽기 전용).

V0 기존(v41): 원래 수익률을 그대로 학습.
V1 보정: (a) 극단값 1%/99% 잘라 학습, (b) 이상치에 덜 휘둘리는 Huber 손실 + 더 강한 규제,
         (c) 예측 크기 보정 — 학습 구간 뒤쪽 20%로 '예측이 실제보다 몇 배 과장되는지'를 재서(기울기, 0~1로 제한) 줄인다.
         (v41 에서 예측이 실제의 2~4배였고, 합성 점검에서 3배 과장만으로 R² 가 +0.9% → -3.1% 로 뒤집힘을 확인.)
V2 보정 + 정성/이벤트 데이터: 실적 서프라이즈(최근·4분기 평균·추세·발표 후 경과일), 애널리스트 의견 변경(30·90일 순상향).

추가 점검(#1): 상위 10% 효과가 해마다 꾸준한지, 업종 그룹·비교군(INTC·QCOM)에서도 나오는지.
설정은 xs_config.json 'variants_v42' 사전등록. 기존 방향 모델·로그 불변 → 검증 시계 리셋 없음. 백테스트는 실전 성적 아님.

실행: python xs_variants.py [출력=reports]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import xs_predictor as xp
import xs_magnitude as xm
import xs_events as xe


def _fit_predict(tr, te, feats, y, cfg_v, h):
    import lightgbm as lgb
    params = dict(cfg_v["lgbm"])
    if cfg_v.get("huber"):
        params.update(objective="huber", alpha=float(cfg_v["huber_delta"]) * np.sqrt(h))
    yt = tr[y].copy()
    if cfg_v.get("winsor"):
        lo, hi = yt.quantile(cfg_v["winsor"]), yt.quantile(1 - cfg_v["winsor"])
        yt = yt.clip(lo, hi)
    base = float(tr[y].mean())
    beta = 1.0
    if cfg_v.get("calibrate"):
        ds = np.sort(tr["date"].unique())
        cut = ds[int(len(ds) * 0.8)]
        a, b = tr["date"] < cut - np.timedelta64(int(h * 1.5) + 1, "D"), tr["date"] >= cut   # 안쪽 검증도 퍼지
        if a.sum() > 1000 and b.sum() > 200:
            m0 = lgb.LGBMRegressor(**params).fit(tr.loc[a, feats], yt[a])
            pv = m0.predict(tr.loc[b, feats])
            pv = pv - pv.mean()
            yv = tr.loc[b, y].values - tr.loc[b, y].mean()
            var = float((pv ** 2).sum())
            beta = float(np.clip((pv * yv).sum() / var, 0.0, 1.0)) if var > 0 else 0.0
    m = lgb.LGBMRegressor(**params).fit(tr[feats], yt)
    raw = m.predict(te[feats])
    pt = m.predict(tr[feats]).mean()
    return base + beta * (raw - pt) if cfg_v.get("calibrate") else raw, beta


def walk_variant(df, cfg, h, feats, cfg_v):
    y = f"fret{h}"
    ok = df.dropna(subset=[y, "absavg20"] + [f for f in xp.FEATURES])
    dates = np.sort(df["date"].unique())
    start = pd.Timestamp(dates[0]) + pd.DateOffset(years=int(cfg["min_train_years"]))
    test = dates[dates >= np.datetime64(start)]
    step = int(cfg["magnitude"]["retrain_every"])
    out = []
    for i in range(0, len(test), step):
        block = test[i:i + step]
        prior = dates[dates < block[0]]
        if len(prior) <= h + 1:
            continue
        cutoff = prior[-(h + 1)]
        tr = ok[ok["date"] <= cutoff]
        te = df[df["date"].isin(block)].dropna(subset=list(xp.FEATURES) + ["absavg20"])
        if len(tr) < 2000 or te.empty:
            continue
        te = te.copy()
        te["pred_ret"], te["beta"] = _fit_predict(tr, te, feats, y, cfg_v, h)
        te["base_ret"] = float(tr[y].mean())
        te["base_abs"] = te["absavg20"] * np.sqrt(h)
        te["pred_abs"] = te["base_abs"]          # 변형 비교는 부호 있는 수익률에 집중(크기는 v41 결과 유지)
        te["train_cut"] = cutoff
        out.append(te)
    return pd.concat(out, ignore_index=True) if out else None


def top_decile_checks(p, h, groups):
    """#1: 상위 10%의 '그날 평균 대비 초과수익'을 연도·그룹별로."""
    y = f"fret{h}"
    p = p.dropna(subset=[y]).copy()
    p["ex"] = p[y] - p.groupby("date")[y].transform("mean")
    p["is_top"] = p.groupby("date")["pred_ret"].transform(lambda s: s >= s.quantile(0.9))
    t = p[p["is_top"]]
    t = t[t["date"].isin(np.sort(p["date"].unique())[::h])]          # 겹침 제거
    yr = t.groupby(pd.to_datetime(t["date"]).dt.year)["ex"].mean()
    inv = {tk: g for g, tks in groups.items() for tk in tks}
    t = t.assign(group=t["ticker"].map(inv).fillna("other"))
    gr = t.groupby("group")["ex"].agg(["mean", "size"])
    ci = xp.block_ci(t.groupby("date")["ex"].mean().values, block=10)
    return {"years_pos": int((yr > 0).sum()), "years": int(len(yr)), "by_year": {int(k): float(v) for k, v in yr.items()},
            "by_group": {k: {"mean": float(r["mean"]), "n": int(r["size"])} for k, r in gr.iterrows()}, "excess_ci": ci}


def run(df, cfg, ev_feats=None):
    V = cfg["variants_v42"]
    res = {}
    for name in ("V0", "V1", "V2"):
        cv = V[name]
        feats = list(xp.FEATURES) + ["absavg20"] + (ev_feats if cv.get("events") else [])
        if cv.get("events") and not ev_feats:
            print("⚠️ 이벤트 피처 없음 — V2 생략")
            continue
        res[name] = {}
        for h in cfg["magnitude"]["horizons"]:
            pr = walk_variant(df, cfg, h, feats, cv)
            if pr is None:
                continue
            ev = xm.evaluate(pr, h)
            ev["beta_mean"] = float(pr["beta"].mean())
            ev["top10"] = top_decile_checks(pr, h, V["groups"])
            res[name][h] = ev
    return res


def render(res, cover):
    pc = lambda x: f"{x * 100:+.2f}%"
    L = ["=== '몇 % 오르나' 개선안 비교 (29종목, 2013~, 백테스트 · 실전 성적 아님 · 종목 선택 편향 있음) ===",
         "V0 기존 / V1 크기 보정(극단값 자르기·Huber·과장 줄이기) / V2 = V1 + 실적 서프라이즈·애널리스트 의견 변경",
         "R²: 0보다 크면 '그냥 평균'보다 정확. [ ] 95% 구간 하한이 0을 넘어야 우연 아님."]
    if cover:
        ne = sum(1 for v in cover.values() if v["earn"] > 0)
        ng = sum(1 for v in cover.values() if v["grades"] > 0)
        avg_e = np.mean([v["earn"] for v in cover.values()])
        L.append(f"이벤트 데이터 확보: 실적 {ne}/{len(cover)}종목(평균 {avg_e:.0f}분기) · 의견 변경 {ng}/{len(cover)}종목")
    hs = sorted({h for v in res.values() for h in v})
    for h in hs:
        L += ["", f"[{h}거래일 뒤]"]
        for name, v in res.items():
            if h not in v:
                continue
            r = v[h]
            ci = r["r2_ret_ci"]
            t = r["top10"]
            eci = t["excess_ci"]
            L.append(f"  {name}: R² {r['r2_ret'] * 100:+.2f}% {'' if not ci else f'[{ci[1]:+.3f}, {ci[2]:+.3f}]'} · 평균오차 {r['mae_ret_model'] * 100:.2f} vs 기준 {r['mae_ret_base'] * 100:.2f}%p"
                     f" · 과장 보정계수 {r['beta_mean']:.2f}")
            L.append(f"      상위10% 예측 {pc(r['top10_pred_mean'])} → 실제 {pc(r['top10_actual_mean'])} (평균 {pc(r['avg_actual'])}) · "
                     f"그날 평균 대비 초과 {'n/a' if not eci else f'{eci[0] * 100:+.2f}% [{eci[1] * 100:+.2f}, {eci[2] * 100:+.2f}]'} · 플러스였던 해 {t['years_pos']}/{t['years']}")
            L.append("      그룹별 초과: " + " · ".join(f"{g} {d['mean'] * 100:+.2f}%" for g, d in sorted(t["by_group"].items())))
    return "\n".join(L)


def main(argv):
    out_dir = argv[1] if len(argv) > 1 else "reports"
    cfg = xp.load_cfg()
    if cfg is None or "variants_v42" not in cfg:
        print("설정 없음 — 생략")
        return 0
    panel = xp.load_panel(cfg)
    if panel is None:
        print("데이터 없음 — 생략")
        return 0
    df = xm.add_labels(xp.build_features(panel), cfg["magnitude"]["horizons"])
    try:
        df, evf, cover = xe.add_event_features(df)
    except Exception as e:
        print(f"⚠️ 이벤트 피처 전체 실패({e}) — V2 생략")
        evf, cover = None, {}
    res = run(df, cfg, evf)
    if not res:
        print("결과 없음")
        return 0
    txt = render(res, cover)
    print(txt)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "xs_variants.txt"), "w", encoding="utf-8").write(txt + "\n")
    json.dump({"cover": cover, "res": {k: {str(h): r for h, r in v.items()} for k, v in res.items()}},
              open(os.path.join(out_dir, "xs_variants.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
