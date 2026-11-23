"""test_v40.py — 교차종목 다음날 예측기(xs_predictor.py) 검증: 합성 데이터, 네트워크 없음.
미래정보 차단 · 심어둔 신호 회수 · 무신호면 우위 없음 · 실패 방어 · 매일 기록/정산 · 사전등록 · 모델 불변."""
import copy, hashlib, json, os, tempfile
import numpy as np
import pandas as pd

import xs_predictor as xp

fails = []
def ck(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)

cfg = xp.load_cfg()
tcfg = copy.deepcopy(cfg)
tcfg["universe"] = [f"S{i:02d}" for i in range(12)] + ["BAD", "SHORT"]
tcfg["lgbm"]["n_estimators"] = 60
tcfg["min_train_years"] = 2
tcfg["start"] = "2015-01-01"

def make_panel(signal, n=1100, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    k = 12
    R = np.zeros((n, k))
    R[:5] = rng.normal(0, 0.015, (5, k))
    for t in range(5, n):
        past = R[t - 5:t].sum(0)
        z = (past - past.mean()) / (past.std() + 1e-12)
        R[t] = rng.normal(0.0003, 0.004) + signal * 0.004 * z + rng.normal(0, 0.015, k)
    px = 100 * np.cumprod(1 + R, 0)
    data = {}
    for j in range(k):
        c = px[:, j]
        data[f"S{j:02d}"] = pd.DataFrame({"High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1e6 + rng.random(n) * 1e5}, index=idx)
    return data

def fetcher_from(data):
    def f(sym, start):
        if sym == "BAD":
            raise RuntimeError("API down")
        if sym == "SHORT":
            return data["S00"].iloc[:100]
        return data.get(sym)
    return f

# ============ 1) 데이터·피처
sig = make_panel(1.0)
panel = xp.load_panel(tcfg, fetcher_from(sig))
ck("적재: 실패·짧은 종목은 조용히 생략 → 12종목", panel is not None and panel["ticker"].nunique() == 12)
F = xp.build_features(panel)
ck("피처: 날짜별 순위 정규화 값은 0~1", F[[c for c in xp.FEATURES if c.startswith("cs_")]].stack().between(0, 1).all())
ck("라벨: 마지막 날 next_ret 은 비어 있음(미래 없음)", F[F["date"] == F["date"].max()]["next_ret"].isna().all())
# 미래정보 차단: t 이후 가격을 바꿔도 t 피처는 동일
cut = panel["date"].sort_values().unique()[600]
p2 = panel.copy()
p2.loc[p2["date"] > cut, ["Close", "High", "Low"]] *= 1.5
F2 = xp.build_features(p2)
a = F[F["date"] <= cut].set_index(["date", "ticker"])[xp.FEATURES].sort_index()
b = F2[F2["date"] <= cut].set_index(["date", "ticker"])[xp.FEATURES].sort_index()
ck("미래정보 차단: 이후 가격을 바꿔도 그 이전 피처는 그대로", np.allclose(a.values, b.values, equal_nan=True))

# ============ 2) 워크포워드: 학습 끝 < 예측일, 심어둔 신호 회수
pred = xp.walk_forward(F, tcfg)
ck("워크포워드: 모든 예측의 학습 마지막 날 < 예측 날짜", (pred["train_end"] < pred["date"]).all())
ev = xp.evaluate(pred, tcfg)
ck(f"신호 있음: RankIC 하한 > 0 (평균 {ev['rank_ic'][0] * 100:.1f})", ev["rank_ic"][1] > 0)
ck("신호 있음: 상위-하위 스프레드 하한 > 0", ev["long_short_daily"][1] > 0)
noise = make_panel(0.0, seed=3)
Fn = xp.build_features(xp.load_panel(tcfg, fetcher_from(noise)))
evn = xp.evaluate(xp.walk_forward(Fn, tcfg), tcfg)
ck(f"무신호: RankIC 구간이 0 포함 (평균 {evn['rank_ic'][0] * 100:.2f})", evn["rank_ic"][1] <= 0 <= evn["rank_ic"][2])
ck("무신호: 방향 Brier 가 기준선보다 유의하게 좋지 않음", not (evn["dir_brier_diff"][2] < 0))
txt = xp.render(ev, tcfg)
ck("보고서: 편향·실전 아님 문구 포함", "선택 편향" in txt and "실전 성적 아님" in txt)
ck("평가: 빈 예측이면 None", xp.evaluate(pred.iloc[:0], tcfg) is None)

# ============ 3) 매일 기록·정산
with tempfile.TemporaryDirectory() as td:
    short = {k: v.iloc[:-1] for k, v in sig.items()}            # 어제까지 데이터로 예측
    log1 = xp.daily(td, tcfg, fetcher_from(short), now="2026-11-24 08:00")
    d1 = log1["price_date"].max()
    ck("daily: 종목 수만큼 예측 기록, 아직 미정산", len(log1) == 12 and (log1["settled"] == False).all())
    log1b = xp.daily(td, tcfg, fetcher_from(short), now="2026-11-24 09:00")
    ck("daily: 같은 날 재실행해도 중복 없음", len(log1b) == 12)
    log2 = xp.daily(td, tcfg, fetcher_from(sig), now="2026-11-25 08:00")
    s = log2[log2["price_date"] == d1]
    exp = (sig["S03"]["Close"].iloc[-1] / sig["S03"]["Close"].iloc[-2] - 1)
    got = float(s[s["ticker"] == "S03"]["next_ret"].iloc[0])
    ck("정산: 다음 날 종가로 next_ret 채움(값 일치)", s["settled"].all() and abs(got - exp) < 1e-9)
    ck("상태 문구: 상위/하위 종목 표시", "상위5" in xp.status_text(log2, tcfg))
    open(os.path.join(td, xp.LOG_NAME), "w").write("\x00junk")
    before = open(os.path.join(td, xp.LOG_NAME), "rb").read()
    r = xp.daily(td, tcfg, fetcher_from(sig), now="2026-11-26 08:00")
    ck("깨진 로그는 덮어쓰지 않음", r is None and open(os.path.join(td, xp.LOG_NAME), "rb").read() == before)
    ck("데이터 전부 실패하면 None", xp.daily(td, dict(tcfg, universe=["BAD"]), fetcher_from(sig)) is None)

# ============ 4) 사전등록 · 모델 불변
h = hashlib.sha256(json.dumps({k: cfg[k] for k in ("universe", "start", "min_train_years", "retrain_every", "top_k", "cost_bp_per_side", "lgbm")}, sort_keys=True).encode()).hexdigest()[:6]
FROZEN = {"2026-11-24": "59d149"}
ck(f"사전등록: 설정 해시({h}) 일치 — 바꾸면 registered_on 을 올리고 FROZEN 추가", FROZEN.get(cfg["registered_on"]) == h)
import common
ck("모델 불변: 피처 20개·라벨 방식·로그 스키마 그대로", len(common.FEATURE_COLUMNS) == 20 and common.LABEL_MODE == "all" and len(common.LOG_COLUMNS) == 15)

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + str(fails)} ({len(fails)} fail)")
raise SystemExit(1 if fails else 0)
