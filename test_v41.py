"""test_v41.py — '몇 % 오르나' 검증(xs_magnitude.py): 합성 데이터, 네트워크 없음.
라벨 정합 · 퍼지(미래정보 차단) · 심어둔 크기 신호 회수 · 무신호면 기준선을 못 넘음 · 보고서 · 모델 불변."""
import copy
import numpy as np
import pandas as pd

import xs_predictor as xp
import xs_magnitude as xm

fails = []
def ck(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)

cfg = copy.deepcopy(xp.load_cfg())
cfg["universe"] = [f"S{i:02d}" for i in range(12)]
cfg["lgbm"]["n_estimators"] = 60
cfg["min_train_years"] = 2
cfg["magnitude"]["retrain_every"] = 63

def make(signal, seed, n=1100, k=12):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    vol = np.full(k, 0.015)
    R = np.zeros((n, k))
    R[:5] = rng.normal(0, 0.015, (5, k))
    for t in range(5, n):
        past = R[t - 5:t].sum(0)
        z = (past - past.mean()) / (past.std() + 1e-12)
        vol = np.sqrt(0.0000045 + 0.1 * R[t - 1] ** 2 + 0.88 * vol ** 2)      # 변동성 뭉침(크기는 예측 가능하게)
        R[t] = signal * 0.006 * z + rng.normal(0, 1, k) * vol
    px = 100 * np.cumprod(1 + R, 0)
    return {f"S{j:02d}": pd.DataFrame({"High": px[:, j] * 1.01, "Low": px[:, j] * 0.99, "Close": px[:, j], "Volume": 1e6}, index=idx) for j in range(k)}

def run(data):
    df = xm.add_labels(xp.build_features(xp.load_panel(cfg, lambda s, st: data.get(s))), [1, 5])
    return df, {h: xm.evaluate(xm.walk(df, cfg, h), h) for h in (1, 5)}

sig_df, sig = run(make(1.0, 1))
g = sig_df[sig_df["ticker"] == "S03"].reset_index(drop=True)
ck("라벨: 5일 뒤 수익률 = close[t+5]/close[t]-1", abs(g.loc[100, "fret5"] - (g.loc[105, "close"] / g.loc[100, "close"] - 1)) < 1e-12)
ck("라벨: 마지막 5일은 5일 라벨 없음", g["fret5"].tail(5).isna().all() and g["fret5"].iloc[-6] == g["fret5"].iloc[-6])
pr = xm.walk(sig_df, cfg, 5)
dates = np.sort(sig_df["date"].unique())
pos = {d: i for i, d in enumerate(dates)}
gap = pr.apply(lambda r: pos[np.datetime64(r["date"])] - pos[np.datetime64(r["train_cut"])], axis=1)
ck("퍼지: 학습 마지막 행과 예측 날짜 사이가 h(5)거래일 초과 → 라벨이 예측 전에 확정", (gap > 5).all())
ck(f"신호 있음(1일): 수익률 R² > 0 이고 구간 하한 > 0 (R² {sig[1]['r2_ret'] * 100:.2f}%)", sig[1]["r2_ret"] > 0 and sig[1]["r2_ret_ci"][1] > 0)
ck("신호 있음: 예측 상위 10% 의 실제 평균 > 전체 평균", sig[1]["top10_actual_mean"] > sig[1]["avg_actual"])
ck("변동성 뭉침: 움직임 크기 예측 상관 > 0.1", sig[1]["corr_abs_model"] > 0.1 and sig[1]["corr_abs_base"] > 0.1)
_, noi = run(make(0.0, 7))
ck(f"무신호(1일): 수익률 R² 구간이 0 이하 포함 (R² {noi[1]['r2_ret'] * 100:.2f}%)", noi[1]["r2_ret_ci"][1] <= 0)
ck(f"무신호(5일): 수익률 R² 구간이 0 이하 포함 (R² {noi[5]['r2_ret'] * 100:.2f}%)", noi[5]["r2_ret_ci"][1] <= 0)
txt = xm.render([sig[1], sig[5]], cfg)
ck("보고서: 쉬운 설명·실전 아님 문구", "기준보다" in txt and "실전 성적 아님" in txt)
ck("빈 예측이면 None", xm.evaluate(pr.iloc[:0], 5) is None)
import common
ck("모델 불변: 피처 20개·라벨 방식·로그 스키마 그대로", len(common.FEATURE_COLUMNS) == 20 and common.LABEL_MODE == "all" and len(common.LOG_COLUMNS) == 15)

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + str(fails)} ({len(fails)} fail)")
raise SystemExit(1 if fails else 0)
