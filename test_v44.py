"""test_v44.py — 모델 vs 단순 규칙 비교(xs_top20.compare_baselines) 검증: 합성 데이터, 네트워크 없음.
변동성만 보상받는 세계 → 변동성 규칙이 강하고 모델과 차이 없음 / 무작위 ≈ 고르게 보유 / 점수는 t 종가까지 / 모델 불변."""
import copy
import numpy as np
import pandas as pd

import xs_predictor as xp
import xs_magnitude as xm
import xs_top20 as xt

fails = []
def ck(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)

cfg = copy.deepcopy(xp.load_cfg())
K = 20
cfg["universe"] = [f"S{i:02d}" for i in range(K)]
cfg["min_train_years"] = 2
cfg["magnitude"]["retrain_every"] = 63
cfg["variants_v42"]["V0"]["lgbm"] = dict(cfg["variants_v42"]["V0"]["lgbm"], n_estimators=60)
cfg["baselines_v44"]["random_seeds"] = 5

def make(seed, n=1200, vol_premium=True):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-01", periods=n)
    vols = np.linspace(0.008, 0.035, K)
    drift = 0.15 * vols ** 2 * 50 if vol_premium else np.full(K, 0.0004)       # 변동성 클수록 평균 수익 큼
    R = rng.normal(0, 1, (n, K)) * vols + drift
    px = 100 * np.cumprod(1 + R, 0)
    return {f"S{j:02d}": pd.DataFrame({"High": px[:, j] * 1.01, "Low": px[:, j] * 0.99, "Close": px[:, j], "Volume": 1e6}, index=idx) for j in range(K)}

df = xm.add_labels(xp.build_features(xp.load_panel(cfg, lambda s, st, d=make(1): d.get(s))), [20])
res = xt.compare_baselines(df, cfg)
ck(f"변동성 보상 세계: 변동성 규칙 연평균 > 고르게 ({res['VOL60']['cagr_med'] * 100:.0f}% vs {res['VOL60']['cagr_ew'] * 100:.0f}%)", res["VOL60"]["cagr_med"] > res["VOL60"]["cagr_ew"])
ck(f"변동성 보상 세계: 모델 선택이 변동성 상위와 많이 겹침 ({res['overlap_with_vol'] * 100:.0f}%)", res["overlap_with_vol"] >= 0.5)
ci = res["model_minus_vol"]["ci"]
ck(f"변동성 보상 세계: 모델−변동성 규칙 차이 구간이 0 포함 ({ci[1] * 100:+.2f}~{ci[2] * 100:+.2f}%)", ci[1] <= 0 <= ci[2])
ck(f"무작위: 20일 초과수익(산술) ≈ 0, 비용만큼 약간 음수 ({res['RANDOM']['excess_med'] * 100:+.2f}%)", -0.01 < res["RANDOM"]["excess_med"] < 0.002)
ck("보고서 렌더링", "변동성 상위" in xt.render_cmp(res, cfg) and "실전 성적 아님" in xt.render_cmp(res, cfg))
# 점수(std60·ret60)는 t 종가까지: t 이후 가격을 바꿔도 t 점수 불변
p = xp.load_panel(cfg, lambda s, st, d=make(2): d.get(s))
cut = p["date"].sort_values().unique()[700]
p2 = p.copy(); p2.loc[p2["date"] > cut, ["Close", "High", "Low"]] *= 2
a = xp.build_features(p).set_index(["date", "ticker"]).loc[lambda x: x.index.get_level_values(0) <= cut, ["std60", "ret60"]].sort_index()
b = xp.build_features(p2).set_index(["date", "ticker"]).loc[lambda x: x.index.get_level_values(0) <= cut, ["std60", "ret60"]].sort_index()
ck("단순 규칙 점수는 미래 가격과 무관(t 종가까지)", np.allclose(a.values, b.values, equal_nan=True))
import ast, inspect
tree = ast.parse(open(xt.__file__, encoding="utf-8").read())
last = tree.body[-1]
ck("스크립트 실행 시 모든 함수가 먼저 정의됨(__main__ 블록이 파일 맨 끝)", isinstance(last, ast.If) and "__main__" in ast.dump(last.test))
import common
ck("모델 불변: 피처 20개·라벨 방식·로그 스키마 그대로", len(common.FEATURE_COLUMNS) == 20 and common.LABEL_MODE == "all" and len(common.LOG_COLUMNS) == 15)
print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + str(fails)} ({len(fails)} fail)")
raise SystemExit(1 if fails else 0)
