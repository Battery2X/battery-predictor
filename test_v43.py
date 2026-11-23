"""test_v43.py — '20일 상위 10%' 전략(xs_top20.py) 검증: 합성 데이터, 네트워크 없음.
비용·교체율 계산 · 심어둔 20일 신호 회수 · 무신호면 우위 없음 · 시작일 20가지 · 매일 기록/20일 뒤 정산 · 공식 바구니 · 모델 불변."""
import copy, os, tempfile
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

# ============ 1) 시뮬레이션 계산
d = pd.bdate_range("2020-01-01", periods=60)
rows = []
for i, day in enumerate(d):
    for j in range(K):
        rows.append({"date": day, "ticker": f"S{j:02d}", "pred_ret": j * 0.001, "fret20": 0.01 if j >= 17 else 0.0})
P = pd.DataFrame(rows)
R = xt.simulate(P, cfg, 0)
ck("교체일 = 20거래일 간격 → 3구간", len(R) == 3)
ck("상위 10%(20종목 → 3개 = 최소 3) 고름", R["picks"].iloc[0] == "S17,S18,S19")
ck("첫 구간 교체율 100% → 비용 2×10bp 차감", abs(R["top"].iloc[0] - (0.01 - 0.002)) < 1e-12)
ck("같은 종목 유지 시 교체율 0 → 비용 없음", R["turnover"].iloc[1] == 0 and abs(R["top"].iloc[1] - 0.01) < 1e-12)
ck("동일가중 비교 = 그날 전 종목 평균", abs(R["ew"].iloc[0] - 0.03 / K * 1) < 1e-12)
s = xt.summarize(R)
ck("요약: 이긴 구간 비율 100%", s["beat_share"] == 1.0)

# ============ 2) 합성 세계: 60일 모멘텀이 다음 20일 수익을 정함
def make(sig, seed, n=1200):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-01", periods=n)
    Rr = rng.normal(0.0004, 0.016, (n, K))
    for t in range(60, n):
        past = Rr[t - 60:t].sum(0)
        Rr[t] += sig * 0.0015 * (past - past.mean()) / (past.std() + 1e-12)
    px = 100 * np.cumprod(1 + Rr, 0)
    return {f"S{j:02d}": pd.DataFrame({"High": px[:, j] * 1.01, "Low": px[:, j] * 0.99, "Close": px[:, j], "Volume": 1e6}, index=idx) for j in range(K)}

def prep(data):
    return xm.add_labels(xp.build_features(xp.load_panel(cfg, lambda s, st: data.get(s))), [20])

sig = xt.backtest(prep(make(1.0, 1)), cfg)
ck(f"신호 있음: 20일 초과 평균 > 0 (대표 {sig['main']['excess_mean'] * 100:+.2f}%)", sig["main"]["excess_mean"] > 0)
ck(f"신호 있음: 시작일 20가지 중 대부분 플러스 ({sig['offsets']['excess_pos_offsets']}/{sig['offsets']['n']})", sig["offsets"]["excess_pos_offsets"] >= 15)
noi = xt.backtest(prep(make(0.0, 9)), cfg)
ci = noi["main"]["excess_ci"]
ck(f"무신호: 초과 평균 구간이 0 포함 ({ci[1] * 100:+.2f}~{ci[2] * 100:+.2f}%)", ci[1] <= 0 <= ci[2])
ck("보고서: 실전 아님·편향 문구", "실전 성적 아님" in xt.render_bt(sig, cfg) and "편향" in xt.render_bt(sig, cfg))

# ============ 3) 매일 기록 · 20일 뒤 정산 · 공식 바구니
data = make(1.0, 2)
cfg2 = copy.deepcopy(cfg)
idx = data["S00"].index
cfg2["top20_v43"]["registered_on"] = str(idx[-45].date())
with tempfile.TemporaryDirectory() as td:
    def at(n_cut):
        return {k: v.iloc[:len(v) - n_cut] for k, v in data.items()}
    out = xt.daily(td, cfg2, lambda s, st: at(25).get(s), now="t1")
    log, df = out
    d0 = log["price_date"].max()
    ck("daily: 전 종목 기록, 상위 3개 표시, 미정산", len(log) == K and int(log["is_top"].sum()) == 3 and (log["settled"] == False).all())
    log_b, _ = xt.daily(td, cfg2, lambda s, st: at(25).get(s), now="t2")
    ck("daily: 같은 날 재실행해도 중복 없음", len(log_b) == K)
    for n_cut in range(24, -1, -1):
        log, df = xt.daily(td, cfg2, lambda s, st, n=n_cut: at(n).get(s), now=f"t{100 - n_cut}")
    r = log[(log["price_date"] == d0) & (log["ticker"] == "S05")]
    c = data["S05"]["Close"]
    i0 = c.index.get_loc(pd.Timestamp(d0))
    ck("정산: 20거래일 뒤 종가로 ret20 계산(값 일치)", bool(r["settled"].iloc[0]) and abs(float(r["ret20"].iloc[0]) - (c.iloc[i0 + 20] / c.iloc[i0] - 1)) < 1e-9)
    ck("정산 전 날짜(최근 20일)는 아직 미정산", (log[log["price_date"] == log["price_date"].max()]["settled"] == False).all())
    b = xt.basket(log, cfg2, df[["date", "ticker", "close"]])
    ck("공식 바구니: 등록일 이후 20거래일 단위 교체일과 3종목", b is not None and len(b["picks"]) == 3)
    st = xt.status_text(log, cfg2, df[["date", "ticker", "close"]])
    ck("상태 문구: 매매 신호 아님·바구니 표시", "매매 신호 아님" in st and "공식 바구니" in st)
    open(os.path.join(td, xt.LOG_NAME), "w").write("\x00junk")
    before = open(os.path.join(td, xt.LOG_NAME), "rb").read()
    ck("깨진 로그는 덮어쓰지 않음", xt.daily(td, cfg2, lambda s, st: data.get(s)) is None and open(os.path.join(td, xt.LOG_NAME), "rb").read() == before)
    ck("데이터 실패면 None", xt.daily(td, dict(cfg2, universe=["X"]), lambda s, st: None) is None)

import common
ck("모델 불변: 피처 20개·라벨 방식·로그 스키마 그대로", len(common.FEATURE_COLUMNS) == 20 and common.LABEL_MODE == "all" and len(common.LOG_COLUMNS) == 15)
print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + str(fails)} ({len(fails)} fail)")
raise SystemExit(1 if fails else 0)
