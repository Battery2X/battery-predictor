"""test_v45.py — 모의투자 장부(paper_trade.py) 검증: 합성 가격, 네트워크 없음.
결정 이후 첫 종가 체결(미래가격 금지) · 비중·비용 계산 · 가격 없으면 보류 · 재실행 중복 없음 · 비교 지수 · 모델 불변."""
import json, os, tempfile
import numpy as np
import pandas as pd
import paper_trade as pt

fails = []
def ck(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)

idx = pd.bdate_range("2026-11-16", "2026-11-27")
P = {"SPY": 100.0, "QQQ": 200.0, "SMH": 300.0, "AAA": 10.0, "BBB": 50.0}
def fetch_upto(last):
    def f(sym, start):
        if sym not in P:
            return None
        n = len(idx)
        s = pd.Series(P[sym] * np.cumprod(np.r_[1, np.full(n - 1, 1.01 if sym == "AAA" else 1.0)]), index=idx)
        return s[s.index <= pd.Timestamp(last)]
    return f

ck("장 마감 시각(UTC): 11/23(겨울) 16:00 ET = 21:00 UTC", pt.close_time_utc("2026-11-23") == pd.Timestamp("2026-11-23 21:00"))
with tempfile.TemporaryDirectory() as td:
    pdir, rdir = os.path.join(td, "paper"), os.path.join(td, "rep")
    os.makedirs(pdir)
    # 장중(14:55 UTC) 결정 → 그날(11/23) 종가 체결
    json.dump([{"decided_at": "2026-11-23T14:55:00Z", "targets": {"AAA": 0.5, "BBB": 0.3}, "note": "t"}], open(os.path.join(pdir, "orders_pending.json"), "w"))
    r = pt.run(pdir, rdir, fetch_upto("2026-11-20"))          # 11/23 종가가 아직 없음
    ck("결정일 종가가 아직 없으면 보류(주문 유지)", len(json.load(open(os.path.join(pdir, "orders_pending.json")))) == 1)
    st, row, txt = pt.run(pdir, rdir, fetch_upto("2026-11-24"))
    tr = pd.read_csv(os.path.join(pdir, "trades.csv"))
    a = tr[tr.ticker == "AAA"].iloc[0]
    exp_px = 10.0 * 1.01 ** list(idx).index(pd.Timestamp("2026-11-23"))
    ck("체결: 결정 이후 첫 종가(11/23) 가격 사용", a["date"] == "2026-11-23" and abs(a["price"] - exp_px) < 1e-9)
    ck("비중: AAA 50% 금액 = 50,000 / 가격", abs(a["shares"] - 50000 / exp_px) < 1e-6)
    ck("비용: 현금 = 100,000 − 80,000×(1+5bp)", abs(st["cash"] - (100000 - 80000 * 1.0005)) < 1e-6)
    ck("주문 처리 후 대기열 비움·체결 기록", json.load(open(os.path.join(pdir, "orders_pending.json"))) == [] and len(json.load(open(os.path.join(pdir, "orders_filled.json")))) == 1)
    ck("평가: 11/24 기준 AAA 1% 상승 반영", row["date"] == "2026-11-24" and abs(row["value"] - (st["cash"] + st["positions"]["AAA"]["shares"] * exp_px * 1.01 + st["positions"]["BBB"]["shares"] * 50)) < 1e-6)
    ck("비교 지수: 시작일 기준 $100,000 환산", abs(row["bench_SPY"] - 100000) < 1e-6)
    pt.run(pdir, rdir, fetch_upto("2026-11-24"))
    ck("같은 날 재실행해도 평가 기록 중복 없음", len(pd.read_csv(os.path.join(pdir, "equity.csv"))) == 1)
    # 장 마감 후(22:00 UTC) 결정 → 다음 거래일 종가, AAA 전량 매도
    json.dump([{"decided_at": "2026-11-24T22:00:00Z", "targets": {"BBB": 0.3}, "note": "sell AAA"}], open(os.path.join(pdir, "orders_pending.json"), "w"))
    st2, row2, _ = pt.run(pdir, rdir, fetch_upto("2026-11-26"))
    tr2 = pd.read_csv(os.path.join(pdir, "trades.csv"))
    s = tr2[(tr2.side == "SELL") & (tr2.ticker == "AAA")].iloc[0]
    ck("장 마감 후 결정은 다음 거래일(11/25) 종가 체결 — 미래가격 금지", s["date"] == "2026-11-25")
    ck("목표에 없는 종목은 전량 매도", "AAA" not in st2["positions"])
    ck("상태 문구: 실제 돈·권유 아님 표시", "매매 권유 아님" in open(os.path.join(rdir, "paper_status.txt"), encoding="utf-8").read())
    # 가격 못 구한 종목 → 보류
    json.dump([{"decided_at": "2026-11-26T15:00:00Z", "targets": {"ZZZ": 0.1}, "note": "x"}], open(os.path.join(pdir, "orders_pending.json"), "w"))
    pt.run(pdir, rdir, fetch_upto("2026-11-27"))
    ck("가격 없는 종목 주문은 보류(조용히)", len(json.load(open(os.path.join(pdir, "orders_pending.json")))) == 1)
    try:
        pt.fill({"cash": 1, "positions": {}, "start_date": None}, {"decided_at": "2026-11-23T14:00:00Z", "targets": {"AAA": 0.8, "BBB": 0.3}}, {k: fetch_upto("2026-11-27")(k, None) for k in P})
        ck("비중 합 100% 초과는 거부", False)
    except ValueError:
        ck("비중 합 100% 초과는 거부", True)

import ast
t = ast.parse(open(pt.__file__, encoding="utf-8").read())
ck("__main__ 블록이 파일 끝", isinstance(t.body[-1], ast.If) and "__main__" in ast.dump(t.body[-1].test))
import common
ck("모델 불변: 피처 20개·라벨 방식·로그 스키마 그대로", len(common.FEATURE_COLUMNS) == 20 and common.LABEL_MODE == "all" and len(common.LOG_COLUMNS) == 15)
print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + str(fails)} ({len(fails)} fail)")
raise SystemExit(1 if fails else 0)
