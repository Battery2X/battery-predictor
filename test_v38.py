"""test_v38.py — 전략 시뮬레이터(strategy_sim.py) 검증: 합성 데이터, 네트워크 없음.
지표 계산 · 미래정보 차단(ex-ante) · 방어 코드(빈 데이터·열 누락·NaN) · 사전등록 규칙 · 모델 불변."""
import hashlib, json, os, tempfile
import numpy as np
import pandas as pd

import strategy_sim as ss

fails = []
def ck(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)

# ============ 1) 지표
r = np.array([0.10, -0.50, 0.20])
m = ss.metrics(r)
ck("지표: 총수익 = 누적곱 (1.1*0.5*1.2-1 = -34%)", abs(m["total"] - (1.1 * 0.5 * 1.2 - 1)) < 1e-12)
ck("지표: 최대낙폭 -50% (고점 1.1 → 0.55)", abs(m["mdd"] + 0.5) < 1e-12)
ck("지표: 회복 필요 상승률 = +100%", abs(m["recovery_needed"] - 1.0) < 1e-12)
ck("지표: 1년 미만은 CAGR 계산 안 함", m["cagr"] is None)
m0 = ss.metrics(r, np.zeros(3))
ck("지표: 전부 현금이면 수익 0·낙폭 0·보유비중 0", m0["total"] == 0 and m0["mdd"] == 0 and m0["exposure"] == 0)
ck("지표: 빈 입력은 None", ss.metrics(np.array([])) is None)
ck("지표: NaN 행은 제외", ss.metrics(np.array([0.1, np.nan, -0.1]))["days"] == 2)
lr = ss.synth_leveraged(np.array([0.01]), 3.0, 0.0889)
ck("합성 3배: 3×1% − 연 8.89%/252", abs(lr[0] - (0.03 - 0.0889 / 252)) < 1e-12)

# ============ 2) 미래정보 차단: 신호 i 는 수익률 0..i-1 만 사용
rng = np.random.default_rng(1)
base = rng.normal(0, 0.01, 300)
a, b = base.copy(), base.copy()
b[200] = -0.40                                    # 200번째 날 폭락
va, vb = ss.vol_gate(a), ss.vol_gate(b)
ck("ex-ante 변동성: 폭락 당일(200) 신호는 폭락 전과 동일", va[200] == vb[200])
ck("ex-ante 변동성: 폭락 다음 날(201)부터 현금 전환", vb[201] == 0.0)
ca, cb = ss.crash_gate(a), ss.crash_gate(b)
ck("ex-ante 낙폭: 폭락 당일 신호는 폭락 전과 동일", ca[200] == cb[200])
ck("ex-ante 낙폭: -40% 다음 날 현금 전환", cb[201] == 0.0)
up = np.concatenate([np.full(60, -0.01), [-0.40], np.full(80, 0.02)])
cg = ss.crash_gate(up)
ck("낙폭 규칙: 20일선 회복 후 복귀", cg[61 + 1] == 0.0 and cg[-1] == 1.0)
ve = ss.ewma_vol_exante(np.full(50, 0.01), min_obs=60)
ck("변동성: 관측 60개 미만이면 NaN → 보유 유지(1)", np.isnan(ve).all() and (ss.vol_gate(np.full(50, 0.01)) == 1).all())

# ============ 3) 확률·엣지 게이트
ck("확률 게이트: 50 이상 보유, NaN 은 NaN", list(np.nan_to_num(ss.prob_gate([49.9, 50, np.nan]), nan=-1)) == [0, 1, -1])
ck("엣지 게이트: +3%p 이상만 보유", list(ss.edge_gate([2.9, 3.0, -5])) == [0, 1, 0])
ck("상수 경보: 상승 예측 비율 계산", ss.degenerate_share([60, 55, 40, np.nan]) == 2 / 3)

# ============ 4) 오라클은 HOLD 보다 좋고, 반대 오라클은 나쁨 (부트스트랩 방향)
ret = rng.normal(0.0005, 0.03, 400)
res = ss.compare(ret, {"HOLD": np.ones(400), "ORACLE": (ret > 0).astype(float), "ANTI": (ret <= 0).astype(float)})
ck("비교: 오라클 HOLD 대비 유의하게 좋음", res["ORACLE"]["vs_hold"]["daily_logdiff_ci"][1] > 0)
ck("비교: 반대 오라클 최대낙폭 ≥ 오라클 낙폭(덜 나쁨 아님)", res["ANTI"]["mdd"] <= 0 and res["ORACLE"]["mdd"] == 0)
ck("비교: 신호가 NaN 인 날은 HOLD 도 같은 날만 비교", ss.compare(ret[:4], {"X": [1, np.nan, 1, 1]})["X"]["days"] == 3)
ck("부트스트랩: 표본이 너무 적으면 None", ss.block_bootstrap_mean([0.1, 0.2]) is None)

# ============ 5) 로그 적재 방어 코드
with tempfile.TemporaryDirectory() as td:
    ck("live: 로그 파일 없으면 빈 결과(예외 없음)", ss.load_live(td) == {})
    pd.DataFrame({"ticker": ["SOXL"], "x": [1]}).to_csv(os.path.join(td, "predictions_scalp.csv"), index=False)
    ck("live: 필수 열 누락이면 빈 결과", ss.load_live(td) == {})
    days = pd.bdate_range("2026-01-01", periods=30).strftime("%Y-%m-%d")
    rows = [{"run_date": f"{d} 08:00", "ticker": "SOXL", "price_date": d, "predicted_up_prob": 60.0, "actual_return": 0.01} for d in days]
    rows.append({"run_date": f"{days[5]} 09:00", "ticker": "SOXL", "price_date": days[5], "predicted_up_prob": 40.0, "actual_return": 0.01})  # 재실행 중복
    rows.append({"run_date": "2026-03-01 08:00", "ticker": "SOXL", "price_date": "2026-02-27", "predicted_up_prob": 60.0, "actual_return": np.nan})  # 미정산
    rows.append({"run_date": f"{days[0]} 08:00", "ticker": "SOXS", "price_date": days[0], "predicted_up_prob": 40.0, "actual_return": -0.01})  # 인버스는 제외
    pd.DataFrame(rows).to_csv(os.path.join(td, "predictions_scalp.csv"), index=False)
    L = ss.load_live(td)
    ck("live: 롱 상품만, 같은 가격일은 마지막 실행만, 미정산 제외", list(L) == ["SOXL"] and len(L["SOXL"]) == 30
       and L["SOXL"].loc[L["SOXL"].price_date == days[5], "predicted_up_prob"].iloc[0] == 40.0)
    out = ss.run_live(td)
    ck("live: edge_history 없으면 EDGE_GATE 는 '데이터 없음'으로 처리", out["SOXL"]["rules"]["EDGE_GATE"] is None and out["SOXL"]["edge_coverage"] == 0)
    ck("live: 상승 예측 비율 96.7% → 상수 경보 대상", out["SOXL"]["up_pred_share"] >= ss.DEGENERATE_SHARE)
    ck("live: 보고서 렌더링에 경보 문구 포함", "사실상 상수" in ss.render(out, "t", ss.load_rules()))
    fake = {"T": {"period": ["a", "b"], "rules": {"X": {"total": 0.1, "mdd": -0.1, "exposure": 0.5, "days": 30, "post_registration_days": 0,
                                                       "vs_hold": {"daily_logdiff_ci": (0.01, 0.001, 0.02)}}}}}
    ck("보고서: 등록 전 구간의 '유의'는 판정이 아닌 표본 내 참고로 표시", "판정 아님" in ss.render(fake, "t", ss.load_rules()))
    rc = ss.main(["x", "live", td, td])
    ck("live: main 실행 후 txt/json 생성", rc == 0 and os.path.exists(os.path.join(td, "strategy_sim_live.json")))
    bad = os.path.join(td, "bad.json"); open(bad, "w").write("{")
    ck("규칙 파일 깨짐 → None(조용히 생략)", ss.load_rules(bad) is None)

# ============ 6) 사전등록 규칙: 숫자가 바뀌면 등록일도 바뀌어야 함
cfg = ss.load_rules()
FROZEN = {"2026-11-23": "c30e57"}  # 등록일 → 규칙 해시 앞 6자리
h = hashlib.sha256(json.dumps(cfg["rules"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:6]
ck(f"사전등록: 규칙 해시({h})가 등록일({cfg['registered_on']})의 기록과 일치 — 다르면 등록일을 올리고 FROZEN 에 추가할 것",
   FROZEN.get(cfg["registered_on"]) == h)

# ============ 7) 모델 불변
try:
    import common
    ck("모델 불변: 피처 20개·라벨 방식·로그 스키마 그대로", len(common.FEATURE_COLUMNS) == 20 and common.LABEL_MODE == "all" and len(common.LOG_COLUMNS) == 15)
except Exception as e:
    ck(f"모델 불변 확인용 common import ({e})", False)

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + str(fails)} ({len(fails)} fail)")
raise SystemExit(1 if fails else 0)
