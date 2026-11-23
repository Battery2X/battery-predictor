"""test_v39.py — 확장 종목 추적(universe_tracker.py) 검증: 합성 데이터, 네트워크 없음.
ex-ante 신호 · 실제 배수 추정 · 백필 방어 코드 · 매일 누적(중복·깨진 로그) · 등록일 이후만 판정 · 모델 불변."""
import hashlib, json, os, tempfile
import numpy as np
import pandas as pd

import strategy_sim as ss
import universe_tracker as ut

fails = []
def ck(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)

cfg_sim, cfg_uni = ss.load_rules(), ut.load_universe()
rng = np.random.default_rng(7)

# ============ 1) 신호: ex-ante
c = np.concatenate([np.linspace(100, 200, 300), [100.0], np.full(10, 100.0)])
tp = ut.trend_pos(c, 200)
ck("추세: 폭락 당일(300번) 신호는 전일 상태(보유)", tp[300] == 1.0)
ck("추세: 폭락 다음 날부터 현금(종가 < 200일선)", tp[301] == 0.0)
ck("추세: 워밍업(200일 전)은 보유", (tp[:200] == 1).all())
mc = np.concatenate([np.linspace(50, 100, 130), np.full(5, 40.0)])
mp = ut.mom_pos(mc, 126)
ck("모멘텀: 오르던 동안 보유, 126일 전보다 낮아진 다음 날부터 현금", mp[129] == 1.0 and mp[130] == 1.0 and mp[131] == 0.0)
ck("모멘텀: 횡보(수익률 0)는 보유 아님", ut.mom_pos(np.full(140, 100.0), 126)[-1] == 0.0)
pos = ut.positions(c, cfg_sim, cfg_uni)
ck("규칙 5개 · 길이 = 수익률 길이", set(pos) == set(ut.RULES) and all(len(v) == len(c) - 1 for v in pos.values()))
st = ut.next_day_state(c, cfg_sim, cfg_uni)
ck("내일 상태: 오늘 종가(폭락 후) 기준 추세 현금", st["TREND200"] == 0.0)

# ============ 2) 실제 배수 추정
u = rng.normal(0, 0.02, 500)
ck("실제 배수: 2배 상품 ≈ 2.0", abs(ut.realized_leverage(2 * u + rng.normal(0, 0.001, 500), u) - 2) < 0.05)
ck("실제 배수: 표본 60 미만이면 None", ut.realized_leverage(u[:30], u[:30]) is None)

# ============ 3) 백필 (가짜 시세)
idx = pd.bdate_range("2012-01-02", periods=1500)
UND = {s: 100 * np.cumprod(1 + rng.normal(0.0004, 0.02, 1500)) for s in ["SMH", "QQQ", "SPY", "NVDA", "AMD", "AVGO", "MU", "TSM", "TSLA", "AAPL", "TER", "MP", "ASML", "INTC", "QCOM"]}
def fake(sym, start):
    if sym == "MP":
        raise RuntimeError("API down")                     # 실패 경로
    if sym == "NVDL":                                      # 실물 ETF: 최근 400일만, 2배
        ru = UND["NVDA"][1:] / UND["NVDA"][:-1] - 1
        e = 50 * np.cumprod(1 + 2 * ru[-400:])
        return pd.DataFrame({"Close": e}, index=idx[-400:])
    if sym in UND:
        return pd.DataFrame({"Close": UND[sym]}, index=idx)
    return None                                            # 없는 ETF
res = ut.backfill("2012-01-01", fetcher=fake, cfg_sim=cfg_sim, cfg_uni=cfg_uni)
ck("백필: 기초자산 실패 종목(MP)은 조용히 생략", "MP" not in res and "TER" in res)
ck("백필: 실물 ETF 없으면 합성만(AMDL)", "real_etf" not in res["AMDL"] and res["AMDL"]["synthetic"])
ck("백필: 실물 ETF 있으면 실제 배수 ≈ 2 (NVDL)", abs(res["NVDL"]["real_etf"]["realized_lev"] - 2) < 0.05)
ck("백필: 1배 종목은 합성 아님", res["TER"]["synthetic"] is False)
pl = ut.pooled(res)
ck("백필: 종목 전체 요약에 규칙 4개", set(pl) >= {"TREND200", "MOM126", "VOL_GATE", "CRASH_GATE", "SWITCH_TREND"} and pl["TREND200"]["n"] == len(res))
ck("백필: 보고서 렌더링(편향 경고 포함)", "편향" in ut.render_backfill(res, cfg_uni))

# ============ 4) 매일 누적
with tempfile.TemporaryDirectory() as td:
    rows = ut.daily_rows(fetcher=fake, cfg_sim=cfg_sim, cfg_uni=cfg_uni, now="2026-11-24 08:00", start="2012-01-01")
    ck("daily: 성공한 종목만 행 생성(MP 제외)", len(rows) == len(cfg_uni["items"]) - 1)
    ck("daily: 실물 없는 2배 상품은 synthetic 표시", [r for r in rows if r["ticker"] == "AMDL"][0]["synthetic"] is True)
    ut.append_daily(rows, td)
    df = ut.append_daily(rows, td)
    ck("daily: 같은 날 재실행해도 중복 없음", len(df) == len(rows))
    open(os.path.join(td, ut.DAILY_LOG), "w").write("\x00garbage,\n\"")
    before = open(os.path.join(td, ut.DAILY_LOG), "rb").read()
    r2 = ut.append_daily(rows, td)
    after = open(os.path.join(td, ut.DAILY_LOG), "rb").read()
    ck("daily: 깨진 로그는 덮어쓰지 않음(그대로 보존)", r2 is None and before == after)
    ck("상태 섹션: 로그 없으면 빈 문자열", ut.status_section(os.path.join(td, "none")) == "")

# ============ 5) 등록일 이후 성적: t 일 수익률엔 t-1 상태
base = {"ticker": "TER", "under": "TER", "lev": 1, "synthetic": False, "run_at": "x"}
log = pd.DataFrame([
    dict(base, price_date="2026-11-21", close=100, close_under=100, **{f"pos_{k}": 1.0 for k in ut.RULES}),   # 등록 전
    dict(base, price_date="2026-11-24", close=100, close_under=100, **{f"pos_{k}": (0.0 if k == "TREND200" else 1.0) for k in ut.RULES}),
    dict(base, price_date="2026-11-25", close=50, close_under=50, **{f"pos_{k}": 1.0 for k in ut.RULES}),
    dict(base, price_date="2026-11-26", close=60, close_under=60, **{f"pos_{k}": 1.0 for k in ut.RULES}),
])
fe = ut.forward_eval(log, cfg_uni)
ck("등록일 이후만: 등록 전 행 제외 → 2거래일", fe["TER"]["days"] == 2)
ck("t-1 상태 적용: TREND200 은 11/24 -50% 를 피함(현금)", abs(fe["TER"]["rules"]["TREND200"]["total"] - 0.2) < 1e-12
   and abs(fe["TER"]["rules"]["HOLD"]["total"] - (0.5 * 1.2 - 1)) < 1e-12)
with tempfile.TemporaryDirectory() as td:
    log.to_csv(os.path.join(td, ut.DAILY_LOG), index=False)
    s = ut.status_section(td, cfg_uni)
    ck("상태 섹션: 표시 전용 문구와 누적 일수", "매매 신호 아님" in s and "누적 2거래일" in s)


# ============ 5b) 롱↔인버스 전환
ru = np.array([0.01, -0.02, 0.03])
sr = ut.strategy_returns(ru, np.array([1, -1, 0]), 3, 3, 0.0, 0.0)
ck("전환: 롱 +3배, 인버스 -3배, 현금 0", np.allclose(sr, [0.03, 0.06, 0.0]))
sr1 = ut.strategy_returns(np.array([0.02]), np.array([-1]), 2, 1, 0.0, 0.0)
ck("전환: 단일종목 인버스는 -1배", np.allclose(sr1, [-0.02]))
chop = np.tile([0.05, -0.0476190476], 100)        # 기초자산 제자리 왕복
inv_only = ut.strategy_returns(chop, -np.ones(200), 3, 3, 0.0, 0.0)
ck("전환: 횡보장에서 -3배 인버스는 녹는다(제자리 기초자산인데 손실)", np.prod(1 + inv_only) < 0.9)
sw = ut.positions(c, cfg_sim, cfg_uni)["SWITCH_TREND"]
ck("전환: 200일선 아래로 내려간 다음 날 인버스(-1)", sw[299] == 1 and sw[300] == -1)

# ============ 6) 사전등록 · 모델 불변
h = hashlib.sha256(json.dumps([cfg_uni["rules"], cfg_uni["items"], cfg_uni["lev_cost_annual"], cfg_uni.get("inverse_lev")], sort_keys=True).encode()).hexdigest()[:6]
FROZEN = {"2026-11-23": "5fc581", "2026-11-24": "746065"}
ck(f"사전등록: universe 규칙·종목 해시({h}) 일치 — 바꾸면 registered_on 을 올리고 FROZEN 추가", FROZEN.get(cfg_uni["registered_on"]) == h)
import common
ck("모델 불변: 피처 20개·라벨 방식·로그 스키마 그대로", len(common.FEATURE_COLUMNS) == 20 and common.LABEL_MODE == "all" and len(common.LOG_COLUMNS) == 15)

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + str(fails)} ({len(fails)} fail)")
raise SystemExit(1 if fails else 0)
