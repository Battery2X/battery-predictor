"""test_v42.py — 개선안(xs_variants.py)·이벤트 피처(xs_events.py) 검증: 합성 데이터, 네트워크 없음.
이벤트 미래정보 차단 · 실패 방어 · 과장 보정이 R² 를 개선 · 심어둔 실적 신호는 V2 만 잡음 · 상위10% 점검 구조 · 모델 불변."""
import copy
import numpy as np
import pandas as pd

import xs_predictor as xp
import xs_magnitude as xm
import xs_events as xe
import xs_variants as xv

fails = []
def ck(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)

# ============ 1) 이벤트 피처: 날짜 < t 만
days = pd.bdate_range("2020-01-01", periods=120)
earn = pd.DataFrame({"surprise": [5.0, -3.0, 10.0]}, index=pd.to_datetime(["2020-01-15", "2020-03-02", "2020-05-01"]))
grades = pd.DataFrame({"up": [1, 0, 1], "down": [0, 1, 0]}, index=pd.to_datetime(["2020-02-10", "2020-02-11", "2020-03-02"]))
f = xe.event_features(days, earn, grades)
ck("실적: 발표 당일(3/2)은 이전 값(+5) 사용", f.loc["2020-03-02", "ev_last_surprise"] == 5.0)
ck("실적: 다음 날(3/3)부터 새 값(-3) 반영", f.loc["2020-03-03", "ev_last_surprise"] == -3.0)
ck("실적: 첫 발표 이전은 비어 있음", np.isnan(f.loc["2020-01-10", "ev_last_surprise"]))
ck("실적: 경과일 계산(3/3 → 1일)", f.loc["2020-03-03", "ev_days_since_earn"] == 1.0)
ck("의견: 변경 당일(2/10)은 미포함, 다음 날 반영", np.nan_to_num(f.loc["2020-02-10", "ev_net_upgrades_30d"]) == 0 and f.loc["2020-02-11", "ev_net_upgrades_30d"] == 1)
ck("의견: 2/12 에는 +1 -1 = 0, 활동 2건", f.loc["2020-02-12", "ev_net_upgrades_30d"] == 0 and f.loc["2020-02-12", "ev_actions_30d"] == 2)
g0 = xe.event_features(days, None, None)
ck("실패/없음: 전부 NaN(예외 없음)", g0.isna().all().all())

# ============ 2) 합성 패널: 실적 서프라이즈가 이후 5일 수익률을 끌어올리는 세계
cfg = copy.deepcopy(xp.load_cfg())
cfg["universe"] = [f"S{i:02d}" for i in range(12)]
cfg["min_train_years"] = 2
cfg["magnitude"]["horizons"] = [1, 5]
for v in ("V0", "V1", "V2"):
    cfg["variants_v42"][v]["lgbm"] = dict(cfg["variants_v42"][v]["lgbm"], n_estimators=50)

def make(seed, n=1000, k=12, drift=0.0, mom=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-01", periods=n)
    E = {}
    R = rng.normal(0.0003, 0.018, (n, k))
    for j in range(k):
        ed = idx[rng.integers(5, 63)::63]
        sur = rng.normal(0, 10, len(ed))
        E[f"S{j:02d}"] = pd.DataFrame({"surprise": sur}, index=ed)
        for d, s in zip(ed, sur):
            t0 = idx.get_loc(d) + 1                     # 발표 다음 날부터 표류
            R[t0:t0 + 10, j] += drift * s / 10 * 0.01
    if mom:
        for t in range(5, n):
            past = R[t - 5:t].sum(0)
            R[t] += mom * 0.004 * (past - past.mean()) / (past.std() + 1e-12)
    px = 100 * np.cumprod(1 + R, 0)
    data = {f"S{j:02d}": pd.DataFrame({"High": px[:, j] * 1.01, "Low": px[:, j] * 0.99, "Close": px[:, j], "Volume": 1e6}, index=idx) for j in range(k)}
    return data, E

def prep(data, E, fail=False):
    df = xm.add_labels(xp.build_features(xp.load_panel(cfg, lambda s, st: data.get(s))), [1, 5])
    def fe(t):
        if fail:
            raise RuntimeError("down")
        return E[t], None
    return xe.add_event_features(df, fetch=fe)

data, E = make(1, drift=1.0)
df, evf, cover = prep(data, E)
ck("커버리지 기록: 12종목 실적 있음", sum(v["earn"] > 0 for v in cover.values()) == 12)
res = xv.run(df, cfg, evf)
r1, r2 = res["V1"][5], res["V2"][5]
ck(f"심어둔 실적 신호: V2(이벤트 포함) 5일 R² > 0, 하한 > 0 (R² {r2['r2_ret'] * 100:.2f}%)", r2["r2_ret"] > 0 and r2["r2_ret_ci"][1] > 0)
ck(f"같은 세계에서 V2 R² > V1 R² (V1 {r1['r2_ret'] * 100:.2f}%)", r2["r2_ret"] > r1["r2_ret"])
ck("보정계수는 0~1", 0 <= res["V1"][1]["beta_mean"] <= 1)
t = r2["top10"]
ck("상위10% 점검: 연도별·그룹별 결과 구조", t["years"] >= 1 and isinstance(t["by_group"], dict) and t["excess_ci"] is not None)

data2, E2 = make(5, mom=0.0)
df2, evf2, _ = prep(data2, E2)
res2 = xv.run(df2, cfg, evf2)
ck(f"무신호: V1(보정) R² 가 V0 보다 0에 가깝거나 높음 (V0 {res2['V0'][1]['r2_ret'] * 100:.2f}% / V1 {res2['V1'][1]['r2_ret'] * 100:.2f}%)",
   res2["V1"][1]["r2_ret"] >= res2["V0"][1]["r2_ret"])

df3, evf3, cover3 = prep(data, E, fail=True)
ck("이벤트 전부 실패: 예외 없이 커버리지 0, 피처 NaN", all(v["earn"] == 0 for v in cover3.values()) and df3["ev_last_surprise"].isna().all())
ck("보고서 렌더링(커버리지·그룹 포함)", "그룹별" in xv.render(res, cover) and "이벤트 데이터 확보" in xv.render(res, cover))

import common
ck("모델 불변: 피처 20개·라벨 방식·로그 스키마 그대로", len(common.FEATURE_COLUMNS) == 20 and common.LABEL_MODE == "all" and len(common.LOG_COLUMNS) == 15)
print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + str(fails)} ({len(fails)} fail)")
raise SystemExit(1 if fails else 0)
