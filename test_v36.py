"""test_v36.py — ETF 속 들여다보기(etf_drivers.py) 검증: 합성 데이터, 네트워크 없음.
구성종목 정리 · 기여도 · 실적 노출 · 재무 지표 · 가중 요약 · 캐시/로그 · 리포트 조립 · 실패 시 조용한 건너뛰기."""
import hashlib, json, os, tempfile
import numpy as np
import pandas as pd

import common
import easy_report as er
import etf_drivers as ed

fails = []
def ck(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)

nosleep = lambda s: None

# ============ 1) 구성종목 정리
raw = [{"symbol": "AMD", "weight": "0.09"}, {"symbol": "nvda", "weight": 0.08}, {"symbol": "n/a", "weight": "0.01"},
       {"symbol": "BRK.B", "weight": "0.02"}, {"symbol": "AMD", "weight": "0.01"}, {"symbol": "X", "weight": "-0.1"},
       {"symbol": "Y", "weight": "oops"}, {"symbol": "", "weight": 0.5}, None and {}]
h = ed.clean_holdings([r for r in raw if r])
ck("구성종목: 기호 없는 항목·음수·깨진 값 제외, 중복 합산, 대문자·점→하이픈, 비중순", [x["symbol"] for x in h] == ["AMD", "NVDA", "BRK-B"] and abs(h[0]["weight"] - 0.10) < 1e-12)
ck("구성종목: 퍼센트 단위(8.5)면 비율로 환산", abs(ed.clean_holdings([{"symbol": "A", "weight": 8.5}, {"symbol": "B", "weight": 4.0}])[0]["weight"] - 0.085) < 1e-12)
ck("구성종목: 빈 입력 → []", ed.clean_holdings(None) == [] and ed.clean_holdings([]) == [])

seed = ed.load_seed()
ck("시드: SOXX 29종목·QQQ 100종목, 비중 합 0.95~1.0", len(seed["SOXX"]["holdings"]) == 29 and len(seed["QQQ"]["holdings"]) == 100
   and all(0.95 < sum(x["weight"] for x in seed[k]["holdings"]) <= 1.0 for k in ("SOXX", "QQQ")))
ov = ed.overlap(seed["QQQ"]["holdings"], seed["SOXX"]["holdings"])
ck("겹침: QQQ 안의 반도체지수 종목 비중 ≈ 29.5% (19종목)", abs(ov[0] - 0.295) < 0.002 and ov[1] == 19)

T = "2026-11-17"
st = {}
got = ed.get_holdings("SOXX", st, seed, T)
ck("구성종목 조회: 조회 수단이 없으면 시드 사용", got["source"] == "seed" and len(got["holdings"]) == 29)
good = lambda i: [{"symbol": f"S{k}", "weight": 0.05} for k in range(12)]
got = ed.get_holdings("SPY", st, seed, T, av_fetch=good)
ck("구성종목 조회: AV 성공 → 상태에 저장", got["source"] == "alphavantage" and st["holdings"]["SPY"]["asof"] == T)
calls = []
got2 = ed.get_holdings("SPY", st, seed, "2026-11-20", av_fetch=lambda i: calls.append(i) or good(i))
ck("구성종목 조회: 7일 이내면 다시 받지 않음", calls == [] and got2["asof"] == T)
bad = lambda i: [{"symbol": "ONLY", "weight": 0.9}]
got3 = ed.get_holdings("SPY", st, seed, "2026-12-01", av_fetch=bad, yf_fetch=lambda i: 1 / 0)
ck("구성종목 조회: 오래됐는데 새 응답이 비정상·실패면 이전 값 유지", got3["asof"] == T and len(got3["holdings"]) == 12)
got4 = ed.get_holdings("SPY", {}, seed, T, av_fetch=bad, yf_fetch=good)
ck("구성종목 조회: AV 비정상 → yfinance 로 대체", got4["source"] == "yfinance")
ck("구성종목 조회: 아무것도 없으면 None", ed.get_holdings("SPY", {}, {}, T) is None)

# ============ 2) 기여도
days = pd.bdate_range("2026-10-01", periods=30)
syms = [f"S{k}" for k in range(10)]
rng = np.random.default_rng(1)
px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (30, 10)), axis=0)), index=days, columns=syms)
hold = [{"symbol": s, "weight": w} for s, w in zip(syms, [0.2, 0.15, 0.12, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02])]
a = ed.attribution(hold, px)
r1 = px.iloc[-1] / px.iloc[-2] - 1
w = pd.Series({x["symbol"]: x["weight"] for x in hold})
ck("기여도: 종목 기여 = 비중 × 수익률", all(abs(a[1]["contrib"][s] - w[s] * r1[s]) < 1e-12 for s in syms))
ck("기여도: 지수 추정 = 기여 합 / 쓴 비중 합", abs(a[1]["est"] - (w * r1).sum() / w.sum()) < 1e-12 and abs(a[1]["covered"] - 0.85) < 1e-12)
ck("기여도: 오른 종목 비중·개수", abs(a[1]["up_weight"] - w[r1 > 0].sum() / w.sum()) < 1e-12 and a[1]["up_count"] == int((r1 > 0).sum()))
ck("기여도: 5일 수익률은 5거래일 전 대비", abs(a[5]["est"] - (w * (px.iloc[-1] / px.iloc[-6] - 1)).sum() / w.sum()) < 1e-12)
ck("기여도: 쏠림(상위5 비중·유효 종목 수)", abs(a["top5_w"] - 0.65) < 1e-12 and abs(a["eff_n"] - 1 / ((w / w.sum()) ** 2).sum()) < 1e-9)
full = hold + [{"symbol": f"Z{k}", "weight": 0.0015} for k in range(100)]
af = ed.attribution(hold, px, full_holdings=full)
ck("기여도(v37): 쏠림은 전체 구성 기준(가격은 상위만 써도 유효 종목 수는 전체로)", af["n_total"] == 110 and af["eff_n"] > a["eff_n"] and af[1]["est"] == a[1]["est"])
# ---- 동조도
mk = np.random.default_rng(2).normal(0, 0.02, 60)
cd = pd.bdate_range("2026-08-20", periods=61)
cp = pd.DataFrame({s: 100 * np.exp(np.cumsum(np.r_[0, mk * (2.0 if s == "S0" else 1.0)])) for s in syms}, index=cd)
cm = ed.comovement(hold, cp, "S0", window=60)
idx_ret = sum(w[s] * (2.0 if s == "S0" else 1.0) for s in syms) / w.sum()     # 지수 = 시장 × 이 값, S0 = 시장 × 2
ck("동조도: 종목이 시장의 2배로 움직이면 기울기 = 지수 민감도/2 (로그·단순 수익률 차이 허용)", abs(cm["beta"] - idx_ret / 2.0) < 0.02 and abs(cm["direct"] - 0.2 / 0.85) < 1e-9)
ck("동조도: 직접 효과(비중)보다 큼 = 같이 움직임이 반영됨", cm["beta"] > cm["direct"])
ck("동조도: 없는 종목·짧은 이력·None → None", ed.comovement(hold, cp, "NOPE") is None and ed.comovement(hold, cp.tail(10), "S0") is None and ed.comovement(hold, None, "S0") is None)
px2 = px.copy(); px2.iloc[-1, 0] = px2.iloc[-2, 0] * 20      # 분할·오류로 보이는 값
a2 = ed.attribution(hold, px2)
ck("기여도: 하루 +1900% 같은 값은 제외", "S0" not in a2[1]["contrib"] and a2[1]["n"] == 9)
px3 = px.copy(); px3["S1"] = np.nan; px3.iloc[-1, 2] = 0.0
a3 = ed.attribution(hold, px3)
ck("기여도: 결측·0원 종목은 빼고 계산", a3 is not None and "S1" not in a3[1]["contrib"] and "S2" not in a3[1]["contrib"])
ck("기여도: 종목 5개 미만·가격 1행·None → None", ed.attribution(hold[:3], px) is None and ed.attribution(hold, px.iloc[:1]) is None and ed.attribution(hold, None) is None and ed.attribution([], px) is None)
ck("기여도: 구성종목에 없는 열은 무시", ed.attribution(hold, px.assign(ZZZ=1.0))[1]["n"] == 10)

# ============ 3) 실적 노출
csv = "symbol,name,reportDate,fiscalDateEnding,estimate,currency,timeOfTheDay\nS0,a,2026-11-19,2026-10-31,1,USD,post\nS0,a,2027-02-20,2027-01-31,1,USD,post\nS1,b,2026-12-20,2026-10-31,,USD,\nS2,c,2026-11-10,2026-09-30,,USD,\nS3,d,bad,2026-09-30,,USD,\n"
ed_ = ed.parse_earnings_csv(csv, T)
ck("실적 CSV: 오늘 이후 가장 가까운 날짜만, 지난 날짜·깨진 날짜 제외", ed_ == {"S0": "2026-11-19", "S1": "2026-12-20"})
ck("실적 CSV: 빈·엉뚱한 응답 → {}", ed.parse_earnings_csv("", T) == {} and ed.parse_earnings_csv("{\"Information\": \"limit\"}", T) == {} and ed.parse_earnings_csv(None, T) == {})
ex = ed.earnings_exposure(hold, ed_, T)
ck("실적 노출: 10거래일 안의 종목만 비중 합산", abs(ex["weight"] - 0.2) < 1e-12 and ex["names"] == [("S0", "2026-11-19", 0.2)] and abs(ex["known"] - 0.35) < 1e-12)
ck("실적 노출: 경계(오늘·10거래일째 포함, 11거래일째 제외)", ed.earnings_exposure(hold, {"S0": T, "S1": "2026-12-01", "S2": "2026-12-02"}, T)["weight"] == 0.35)
ck("실적 노출: 날짜 없음 → 0", ed.earnings_exposure(hold, {}, T)["weight"] == 0.0)

# ============ 4) 재무 지표
q = pd.to_datetime(["2026-09-30", "2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"])
inc = pd.DataFrame({q[0]: [1200, 720, 480, 360], q[1]: [1100, 640, 460, 300], q[2]: [1000, 560, 440, 250], q[3]: [950, 520, 430, 230], q[4]: [800, 400, 400, 160]},
                   index=["Total Revenue", "Gross Profit", "Cost Of Revenue", "Operating Income"])
bal = pd.DataFrame({q[0]: [600.0], q[1]: [500.0]}, index=["Inventory"])
cf = pd.DataFrame({q[0]: [-300.0, 200.0], q[1]: [-250.0, 150.0], q[2]: [-240.0, 100.0], q[3]: [-230.0, 90.0], q[4]: [-200.0, 80.0]}, index=["Capital Expenditure", "Free Cash Flow"])
et = pd.DataFrame({"current": [2.0, 10.5, 12.6], "30daysAgo": [1.9, 10.0, 12.0]}, index=["0q", "0y", "+1y"])
f = ed.fundamentals_from_statements(inc, bal, cf, {"forwardPE": 25.0}, et)
ck("재무: 매출 성장(전년비 +50%, 전분기비 +9.1%)", abs(f["rev_yoy"] - 0.5) < 1e-12 and abs(f["rev_qoq"] - (1200 / 1100 - 1)) < 1e-12 and f["quarter"] == "2026-09-30")
ck("재무: 매출총이익률 60%, 전년비 +10%p", abs(f["gross_margin"] - 0.6) < 1e-12 and abs(f["gm_chg_yoy"] - 0.1) < 1e-12)
ck("재무: 영업이익률 30%, FCF 마진", abs(f["op_margin"] - 0.3) < 1e-12 and abs(f["fcf_margin"] - 200 / 1200) < 1e-12)
ck("재무: 재고일수 = 재고/매출원가×91, 전분기 대비 변화", abs(f["dio"] - 600 / 480 * 91) < 1e-9 and abs(f["dio_chg"] - (600 / 480 - 500 / 460) * 91) < 1e-9)
ck("재무: 설비투자는 양수로, 전년비 +50%", f["capex"] == 300.0 and abs(f["capex_yoy"] - 0.5) < 1e-12)
ck("재무: 선행 PER, EPS 추정 30일 변화(+1y 우선)", f["fwd_pe"] == 25.0 and abs(f["eps_rev30"] - 0.05) < 1e-12)
f4 = ed.fundamentals_from_statements(inc.iloc[:, :4], bal, cf.iloc[:, :4])
ck("재무: 분기 4개뿐이면 전년비는 None, 전분기비는 계산", f4["rev_yoy"] is None and f4["rev_qoq"] is not None and f4["gm_chg_yoy"] is None and f4["gm_chg_qoq"] is not None and f4["capex_yoy"] is None)
inc_nogp = inc.drop(index="Gross Profit")
ck("재무: 매출총이익 행이 없으면 매출-원가로 계산", abs(ed.fundamentals_from_statements(inc_nogp, None, None)["gross_margin"] - 0.6) < 1e-12)
e0 = ed.fundamentals_from_statements(None, None, None)
ck("재무: 재무제표가 전부 없으면 모든 항목 None(예외 없음)", all(v is None for v in e0.values()))
e1 = ed.fundamentals_from_statements(pd.DataFrame(), pd.DataFrame({"x": ["a"]}, index=["Inventory"]), "oops", {"forwardPE": "nan"}, pd.DataFrame())
ck("재무: 빈 표·문자 값·엉뚱한 형식에도 예외 없음", all(v is None for v in e1.values()))
ck("재무: 음수·터무니없는 PER 은 버림", ed.fundamentals_from_statements(None, None, None, {"forwardPE": -5})["fwd_pe"] is None and ed.fundamentals_from_statements(None, None, None, {"forwardPE": 9000})["fwd_pe"] is None)
bank = pd.DataFrame({q[0]: [1000.0], q[1]: [900.0]}, index=["Total Revenue"])
fb = ed.fundamentals_from_statements(bank, pd.DataFrame({q[0]: [5.0]}, index=["Cash"]), None)
ck("재무: 재고·원가가 없는 업종(은행 등)은 재고일수 None", fb["dio"] is None and fb["gross_margin"] is None and fb["rev_qoq"] is not None)

fund = {"S0": {"rev_yoy": 0.5, "gross_margin": 0.6, "fwd_pe": 20.0, "quarter": "x"}, "S1": {"rev_yoy": 0.1, "gross_margin": None, "fwd_pe": 40.0, "quarter": "x"},
        "S2": {"rev_yoy": float("nan"), "fwd_pe": None}}
ag = ed.aggregate_fundamentals(hold[:3], fund)       # 비중 0.2/0.15/0.12, 값 있는 비중 0.35/0.47 = 74%
ck("가중 요약(v37): 가중 중앙값 — 비중 큰 쪽 값", ag["rev_yoy"] == 0.5 and abs(ag["cov_rev_yoy"] - 0.35) < 1e-12)
ck("가중 요약: PER 은 이익수익률 가중(조화평균)", abs(ag["fwd_pe"] - 0.35 / (0.2 / 20 + 0.15 / 40)) < 1e-12)
ck("가중 요약(v37): 값 있는 종목 비중이 60% 미만인 항목은 요약하지 않음", "gross_margin" not in ag and "dio" not in ag)
ck("가중 요약(v37): 기준을 낮추면 한 종목만으로도 계산", abs(ed.aggregate_fundamentals(hold[:3], fund, min_cov=0.0)["gross_margin"] - 0.6) < 1e-12)
skew = {"S0": {"rev_yoy": 0.3}, "S1": {"rev_yoy": 0.4}, "S2": {"rev_yoy": 3.8}}        # 한 종목이 +380%
ck("가중 요약(v37): 극단값 한 종목이 요약을 끌고 가지 않음(중앙값 0.4, 평균이면 1.2)", ed.aggregate_fundamentals(hold[:3], skew)["rev_yoy"] == 0.4)
ck("가중 요약: 빈 입력 → {}", ed.aggregate_fundamentals([], fund) == {} and ed.aggregate_fundamentals(hold, {}) == {})
cx = ed.hyperscaler_capex({s: {"capex": 100.0, "capex_yoy": 0.25} for s in ed.HYPERSCALERS})
ck("빅테크 설비투자: 4사 합과 전년비", cx["sum"] == 400.0 and abs(cx["yoy"] - 0.25) < 1e-12)
cx2 = ed.hyperscaler_capex({"MSFT": {"capex": 100.0, "capex_yoy": 0.25}, "META": {"capex": 50.0}})
ck("빅테크 설비투자: 일부만 있으면 합만(전년비 None), 전혀 없으면 None", cx2["sum"] == 150.0 and cx2["yoy"] is None and ed.hyperscaler_capex({}) is None)

# ---- 재무 캐시: 돌아가며 갱신
stt = {}
fetched = []
def ff(s):
    fetched.append(s)
    if s == "BAD":
        raise RuntimeError("x")
    if s == "EMPTY":
        return {k: None for k in ("quarter", "rev_yoy")}
    return {"quarter": "2026-09-30", "rev_yoy": 0.3}
up = ed.refresh_fundamentals(stt, [f"T{k}" for k in range(8)] + ["BAD", "EMPTY", "T0"], T, ff, per_run=4, sleeper=nosleep)
ck("재무 캐시: 한 번에 per_run 개만 조회", len(fetched) == 4 and len(up) == 4)
up2 = ed.refresh_fundamentals(stt, [f"T{k}" for k in range(8)] + ["BAD", "EMPTY"], T, ff, per_run=20, sleeper=nosleep)
ck("재무 캐시: 이미 받은 종목은 건너뛰고, 실패·빈 응답 종목은 캐시에 값으로 넣지 않음", set(up2) == {f"T{k}" for k in range(4, 8)} and "BAD" not in stt["fund"] and "quarter" not in stt["fund"].get("EMPTY", {}))
fetched.clear()
ed.refresh_fundamentals(stt, ["T0", "T1"], "2026-11-20", ff, sleeper=nosleep)
ck("재무 캐시: 7일 이내면 다시 받지 않음", fetched == [])
ed.refresh_fundamentals(stt, ["T0", "T1"], "2026-11-26", ff, sleeper=nosleep)
ck("재무 캐시: 7일 지나면 다시 받음", fetched == ["T0", "T1"])

# ============ 5) collect 전체 흐름 (전부 가짜 조회)
big_days = pd.bdate_range("2026-10-01", "2026-11-16")
def fake_closes(symbols):
    r = np.random.default_rng(9)
    return pd.DataFrame(100 * np.exp(np.cumsum(r.normal(0, 0.02, (len(big_days), len(symbols))), axis=0)), index=big_days, columns=list(symbols))
spy_h = lambda i: [{"symbol": s, "weight": w} for s, w in (("NVDA", .075), ("AAPL", .07), ("MSFT", .06), ("AMZN", .04), ("GOOGL", .03), ("META", .028), ("AVGO", .025), ("TSLA", .02), ("BRK.B", .017), ("JPM", .015))]
def fake_fund(s):
    return ed.fundamentals_from_statements(inc, bal, cf, {"forwardPE": 30.0}, et)
earn_txt = lambda: "symbol,name,reportDate,fiscalDateEnding,estimate,currency,timeOfTheDay\n" + "NVDA,n,2026-11-17,2026-10-31,2.49,USD,post-market\n" + "".join(f"X{k},n,2026-12-0{1 + k % 9},2026-10-31,,USD,\n" for k in range(30))
with tempfile.TemporaryDirectory() as td:
    lg, rp = os.path.join(td, "logs"), os.path.join(td, "reports")
    res = ed.collect(lg, rp, today=T, closes_fetch=fake_closes, av_holdings_fetch=lambda i: spy_h(i) if i == "SPY" else None, yf_holdings_fetch=None,
                     earnings_text_fetch=earn_txt, yf_earnings_fetch=None, fund_fetch=fake_fund, sleeper=nosleep, use_network=False)
    ck("collect: 세 지수 모두 기여도 계산", set(res["indexes"]) == {"SOXX", "QQQ", "SPY"} and all(r["attr"] and 1 in r["attr"] for r in res["indexes"].values()))
    ck("collect: SOXX·QQQ 는 시드, SPY 는 받은 값", res["indexes"]["SOXX"]["source"] == "seed" and res["indexes"]["SPY"]["source"] == "alphavantage")
    ck("collect: NVDA 실적(11/17)이 세 지수의 실적 노출에 잡힘", all(r["earn"]["names"][0][0] == "NVDA" for r in res["indexes"].values())
       and abs(res["indexes"]["SOXX"]["earn"]["weight"] - 0.0818) < 1e-9)
    ck("collect: 겹침 29.5%", abs(res["overlap"][0] - 0.295) < 0.002)
    ck("collect: 재무는 이번 실행에 10종목만 갱신", len(res["fund_updated"]) == ed.FUND_PER_RUN)
    d1 = pd.read_csv(os.path.join(lg, ed.DAILY_LOG))
    ck("collect: 일일 로그 지수당 1행, 합치기용 키 열", len(d1) == 3 and {"run_date", "ticker", "horizon_days", "price_date"} <= set(d1.columns) and set(d1.ticker) == {"SOXX", "QQQ", "SPY"})
    f1 = pd.read_csv(os.path.join(lg, ed.FUND_LOG))
    ck("collect: 재무 로그 (종목, 분기) 1행씩", len(f1) == ed.FUND_PER_RUN and f1.duplicated(["ticker", "price_date"]).sum() == 0)
    state = ed.load_state(rp)
    ck("collect: 상태 파일에 구성종목·실적일·재무 캐시 저장", "SPY" in state["holdings"] and state["earnings"]["dates"].get("NVDA") == "2026-11-17" and len(state["fund"]) == ed.FUND_PER_RUN)
    res2 = ed.collect(lg, rp, today=T, closes_fetch=fake_closes, earnings_text_fetch=lambda: 1 / 0, fund_fetch=fake_fund, sleeper=nosleep, use_network=False)
    d2 = pd.read_csv(os.path.join(lg, ed.DAILY_LOG))
    ck("collect: 같은 날 재실행 → 일일 로그 행이 늘지 않고, 실적일은 캐시 사용, 재무는 다음 10종목", len(d2) == 3 and res2["indexes"]["SOXX"]["earn"]["weight"] > 0 and len(ed.load_state(rp)["fund"]) == 2 * ed.FUND_PER_RUN)
    import push_logs
    ck("로그: push_logs.merge_log_frames 로 합쳐도 행 수 불변", len(push_logs.merge_log_frames(d2.iloc[:2].copy(), d2.copy())) == 3)
    lines = ed.driver_lines(res2)
    ck("리포트 줄: 지수별 기여·쏠림·실적 노출·겹침", sum("지수 추정" in l for l in lines) == 3 and any("10거래일 내 실적" in l and "NVDA(11/17)" in l for l in lines) and any("겹침" in l for l in lines))
    # 재무가 다 찬 뒤 체력 줄
    for _ in range(6):
        res3 = ed.collect(lg, rp, today=T, closes_fetch=fake_closes, fund_fetch=fake_fund, sleeper=nosleep, use_network=False)
    lines3 = ed.driver_lines(res3)
    ck("리포트 줄: 재무가 5종목 이상 차면 체력 줄(매출·이익률·재고·PER·EPS추정)", any("체력" in l and "매출 +50%" in l and "재고" in l for l in lines3 if True) and any("빅테크 4사 설비투자" in l and "+50%" in l for l in lines3))
    ck("collect(v37): 실적 임박 종목의 동조도를 계산해 둠", "NVDA" in (res3["indexes"]["SOXX"].get("comove") or {}))
    ck("리포트 줄(v37): 0.01%p 미만 기여는 표시하지 않음", not any("+0.00%p" in l or "-0.00%p" in l for l in lines3))
    ck("리포트 줄: 재고일수는 반도체지수에만 표시", sum("재고" in l for l in lines3) == 1)

    # ---- 요약 리포트 조립
    now = pd.Timestamp("2026-11-17 09:20")
    sums = {"scalp": {"horizon_days": 1, "generated_at": str(now), "items": [
        {"ticker": "SOXL", "benchmark": "SMH", "desc": "x", "direction": 1, "up_prob": 56.0, "raw_up_prob": 60.0, "base_rate": 55.0, "edge": 1.0,
         "price": 58.32, "vol": 1.5, "drawdown": -0.6, "ret1": -0.1, "ret5": 0.2, "price_date": "2026-11-13", "signal": None}]}}
    base = er.build_sections(sums, [], [], None, [], None, [], now, risk={})
    withd = er.build_sections(sums, [], [], None, [], None, [], now, risk={"_drivers": res3})
    import re
    nums = [int(x) for s in withd for x in re.findall(r"<b>(\d+)\. ", s)]
    ck("요약: 드라이버 섹션 1개 추가, 번호 연속", len(withd) == len(base) + 1 and any("ETF 속 들여다보기" in s for s in withd) and nums == list(range(1, len(nums) + 1)))
    ck("요약: '&' 등 HTML 이스케이프(S&P500)", any("S&amp;P500" in s for s in withd) and not any("S&P500" in s for s in withd))
    ck("요약: 깨진 드라이버 값에도 예외 없음", len(er.build_sections(sums, [], [], None, [], None, [], now, risk={"_drivers": "x"})) == len(base)
       and len(er.build_sections(sums, [], [], None, [], None, [], now, risk={"_drivers": {"indexes": {"SOXX": None}}})) >= len(base))

# ---- 전부 실패
with tempfile.TemporaryDirectory() as td:
    boom = lambda *a, **k: 1 / 0
    res = ed.collect(os.path.join(td, "l"), os.path.join(td, "r"), today=T, closes_fetch=boom, av_holdings_fetch=boom, yf_holdings_fetch=boom,
                     earnings_text_fetch=boom, yf_earnings_fetch=boom, fund_fetch=boom, sleeper=nosleep, use_network=False)
    ck("전부 실패: 예외 없이 시드 구성종목만으로 진행(기여도 없음)", set(res["indexes"]) == {"SOXX", "QQQ"} and all(r["attr"] is None for r in res["indexes"].values()))
    ck("전부 실패: 리포트 줄은 '시세를 받지 못해 생략' + 겹침만", all(("생략" in l) or ("겹침" in l) or ("구성종목 비중" in l) for l in ed.driver_lines(res)))
    ck("전부 실패: 일일 로그는 만들지 않음", not os.path.exists(os.path.join(td, "l", ed.DAILY_LOG)))
    res = ed.collect(os.path.join(td, "l"), os.path.join(td, "r"), today=T, seed_path=os.path.join(td, "none.json"), closes_fetch=boom, use_network=False)
    ck("시드도 없고 조회도 안 되면 빈 결과 + 안내", res["indexes"] == {} and len(res["notes"]) == 3 and ed.driver_lines(res) != [])
    open(os.path.join(td, "r", ed.STATE_FILE), "w").write("{broken")
    ck("깨진 상태 파일 → 빈 상태로 시작", ed.load_state(os.path.join(td, "r")) == {})
ck("driver_lines: None·빈 dict → []", ed.driver_lines(None) == [] and ed.driver_lines({}) == [])

# ============ 5-b) 롱·인버스 결과 대칭 점검
import direction_scorecard as ds
pl = pd.DataFrame({'ticker': ['TQQQ', 'SQQQ', 'TQQQ', 'SQQQ', 'SOXL', 'SOXS', 'TQQQ', 'SPXL'],
                   'price_date': ['2026-11-02', '2026-11-02', '2026-11-03', '2026-11-03', '2026-11-02', '2026-11-02', '2026-11-04', '2026-11-02'],
                   'actual_return': [0.03, -0.03, -0.02, -0.01, 0.01, 0.004, 0.02, np.nan]})
po = ds.pair_outcomes(pl)
ck("쌍 결과: 짝이 맞는 정산분만 세고 둘 다 하락·둘 다 상승을 구분", po == {'pairs': 3, 'both_down': 1, 'both_up': 1})
ck("쌍 결과: 열이 없거나 빈 로그 → None", ds.pair_outcomes(pd.DataFrame({'ticker': []})) is None and ds.pair_outcomes(None) is None and ds.pair_outcomes(pl.iloc[:0]) is None)

# ============ 6) 모델 불변
ck("모델 불변: 피처 20개·벤치마크·로그 스키마 그대로", len(common.FEATURE_COLUMNS) == 20 and common.LOG_COLUMNS[-1] == "price_date" and len(common.LOG_COLUMNS) == 15
   and common.SOXX_TOP_HOLDINGS_WEIGHTS["AMD"] == 8.54)

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + str(fails)} ({len(fails)} fail)")
raise SystemExit(1 if fails else 0)
