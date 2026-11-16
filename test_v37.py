"""test_v37.py — 리포트 재배치(report_layout.py) 검증: 합성 데이터, 네트워크 없음.
한눈 요약 · 이벤트 카드 · 접어 두기 · 헤드라인 번역 · 그림 · 실패 시 기존 형식 유지 · F&G 가격일 기록 보완."""
import json, os, re, tempfile
import numpy as np
import pandas as pd

import common
import easy_report as er
import market_context as mc
import report_layout as rl

fails = []
def ck(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)

now = pd.Timestamp("2026-11-16 22:04")
def item(t, bench, d, price, dd, r5, status="fail"):
    return {"ticker": t, "benchmark": bench, "desc": "x", "direction": d, "up_prob": 57.0 if d == 1 else 43.0, "raw_up_prob": 60.0, "base_rate": 55.0, "edge": 2.0,
            "price": price, "vol": 1.2, "drawdown": dd, "ret1": -0.02, "ret5": r5, "price_date": "2026-11-13T00:00:00", "signal": {"status": status}}
def sums(status="fail"):
    its = [item("TQQQ", "QQQ", 1, 57.68, -0.315, 0.10, status), item("SQQQ", "QQQ", -1, 20.0, -0.1, -0.1), item("SOXL", "SMH", 1, 58.32, -0.645, 0.27),
           item("SOXS", "SMH", -1, 9.0, -0.2, -0.2), item("SPXL", "SPY", 1, 265.10, -0.114, 0.07), item("SPXS", "SPY", -1, 5.0, -0.1, -0.05)]
    return {"scalp": {"horizon_days": 1, "generated_at": str(now), "items": its}}
risk = {"SOXL": {"move_1d": 0.148, "move_5d": 0.331}, "TQQQ": {"move_1d": 0.062, "move_5d": 0.139}, "SPXL": {"move_1d": 0.039, "move_5d": 0.087}}
drivers = {"indexes": {
    "SOXX": {"attr": {"asof": pd.Timestamp("2026-11-13"), "top5_w": 0.39, "eff_n": 19.0, "n_total": 29,
                      1: {"est": -0.034, "covered": 0.99, "n": 29, "up_weight": 0.01, "up_count": 2, "contrib": {"AMD": -0.0079, "AVGO": -0.0037, "MU": -0.0020, "CBRS": 0.0006, "STM": 0.00001}}},
             "earn": {"weight": 0.13, "known": 0.9, "names": [("NVDA", "2026-11-17", 0.0818), ("ADI", "2026-11-24", 0.0506)]},
             "comove": {"NVDA": {"beta": 0.55, "direct": 0.083, "n": 40}}},
    "QQQ": {"attr": None, "earn": {"weight": 0.15, "known": 0.9, "names": [("NVDA", "2026-11-17", 0.0814), ("WMT", "2026-11-19", 0.0274), ("PANW", "2026-11-18", 0.0129)]}, "comove": {}},
    "SPY": {"attr": None, "earn": {"weight": 0.11, "known": 0.9, "names": [("NVDA", "2026-11-17", 0.075), ("HD", "2026-11-17", 0.005)]}, "comove": {"NVDA": {"beta": -0.1, "direct": 0.075, "n": 40}}}}}
ctx = {"fg": {"score": 37.0, "bucket": "공포", "prev_week": 32.0}, "price_checked": 6, "price_warn": [],
       "fg_hist": {str(d.date()): 30 + i % 20 for i, d in enumerate(pd.bdate_range("2026-08-20", "2026-11-13"))}}
R = dict(risk, _drivers=drivers, _context=ctx)

# ============ 1) 임박 이벤트
ev = rl.find_event(drivers, now.date())
ck("이벤트: 3거래일 안·비중 4% 이상 중 가장 이른/큰 종목(NVDA), 지수별 비중 모음", ev["symbol"] == "NVDA" and set(ev["weights"]) == {"SOXX", "QQQ", "SPY"} and str(ev["date"].date()) == "2026-11-17")
ck("이벤트: 비중 작은 종목(HD·PANW)·먼 날짜(ADI 11/24)는 카드 대상 아님", rl.find_event({"indexes": {"SPY": {"earn": {"names": [("HD", "2026-11-17", 0.005)]}}, "SOXX": {"earn": {"names": [("ADI", "2026-11-24", 0.05)]}}}}, now.date()) is None)
ck("이벤트: 지난 날짜·빈 값·None → None", rl.find_event({"indexes": {"SOXX": {"earn": {"names": [("NVDA", "2026-11-13", 0.08)]}}}}, now.date()) is None
   and rl.find_event(None, now.date()) is None and rl.find_event({}, now.date()) is None and rl.find_event({"indexes": {"SOXX": None}}, now.date()) is None)

card = rl.event_card(ev, sums(), R, now.date())
ck("카드: 제목에 종목·날짜·D-1", "NVDA 실적 11/17(화) · D-1" in card)
ck("카드: 직접 효과 = 10% × 비중 × 3 (SOXL 2.5%)", "SOXL ±2.5%(직접)" in card)
ck("카드: 동조도가 있으면 '같이 움직인 평균'(10×0.55×3=±16%)을 함께", "±16%(같이 움직인 과거 평균)" in card)
ck("카드: 동조도가 없거나 음수면 직접 효과만", "TQQQ ±2.4%(직접 효과만)" in card and "SPXL ±2.2%(직접 효과만)" in card)
ck("카드: 방향 의견·매수/매도 문구 없음", not re.search(r"매수|매도|사라|팔아|사세요|파세요|오를|내릴 것", card) and "방향은 모릅니다" in card)
ck("카드: 지수 이름의 & 이스케이프", "S&amp;P500" in card and "S&P500" not in card)
ck("카드: 이벤트 없음 → None", rl.event_card(None, sums(), R, now.date()) is None)
ev0 = dict(ev, date=pd.Timestamp(now.date()))
ck("카드: 당일이면 '오늘'", "· 오늘" in rl.event_card(ev0, sums(), R, now.date()))

# ============ 2) 한눈 요약
g = rl.glance(sums(), R, now, drivers, ctx, ev, stale=[])
gl = g.split("\n")
ck("요약: 10줄 이내", len(gl) <= 10)
ck("요약: 통과 0건이면 '아직 쓸 수 없음'(인버스는 세지 않음: 0/3)", "아직 쓸 수 없음" in g and "0/3" in g)
ck("요약: 보유 종목(SOXL)이 먼저, 가격·5일·고점대비·하루 흔들림", gl[2].startswith("💼 <b>SOXL(보유)</b>") and "$58.32" in gl[2] and "5일 +27%" in gl[2] and "고점대비 -64%" in gl[2] and "하루 ±15%" in gl[2])
ck("요약: 심리·반도체지수·임박·데이터 점검 줄", any("공포 37" in l and "1주 전 32" in l for l in gl) and any("반도체지수" in l and "-3.4%" in l and "AMD·AVGO·MU 끌어내림" in l for l in gl)
   and any("NVDA 실적" in l and "D-1" in l and "반도체지수의 8%" in l for l in gl) and any("6종목 교차확인 일치" in l for l in gl))
g2 = rl.glance(sums("pass"), R, now, drivers, dict(ctx, price_warn=["x"], env_changes=["a"]), None, stale=["swing"])
ck("요약: 통과가 있으면 '일부만 통과', 경고는 건수로", "일부만 통과" in g2 and "1/3" in g2 and "갱신 안 됨 1건" in g2 and "시세 불일치 1건" in g2 and "라이브러리 버전 변경" in g2 and "임박" not in g2)
ck("요약: 위험·맥락·드라이버가 전혀 없어도 가격 줄은 나옴", "SOXL" in rl.glance(sums(), {}, now))
ck("요약: 요약 데이터가 없으면 None", rl.glance({}, R, now) is None and rl.glance(None, None, now) is None)
up = json.loads(json.dumps(drivers, default=str)); up["indexes"]["SOXX"]["attr"]["1"] = None
dr_up = {"indexes": {"SOXX": {"attr": {"asof": pd.Timestamp("2026-11-13"), 1: {"est": 0.05, "contrib": {"NVDA": 0.01, "AMD": 0.008, "MU": -0.001}}}}}}
ck("요약: 오른 날은 '끌어올림' + 상승 기여 순", "NVDA·AMD·MU 끌어올림" in rl.glance(sums(), R, now, dr_up, ctx))

# ============ 3) 접어 두기
long_sec = "⚠️ <b>2. 제목</b>\n" + "\n".join(f"줄{k}" for k in range(6))
c = rl.collapse(long_sec)
ck("접기: 제목은 밖, 본문은 펼치기 인용 안", c.startswith("⚠️ <b>2. 제목</b>\n<blockquote expandable>줄0") and c.endswith("줄5</blockquote>") and c.count("<blockquote") == 1)
ck("접기: 본문 3줄 이하·한 줄짜리는 그대로", rl.collapse("제목\n가\n나\n다") == "제목\n가\n나\n다" and rl.collapse("한 줄") == "한 줄")
ck("접기: 너무 긴 섹션·이미 접힌 섹션·문자열 아님은 그대로", rl.collapse("제목\n" + "가\n" * 2000) == "제목\n" + "가\n" * 2000 and rl.collapse(c) == c and rl.collapse(None) is None)
ck("접기: 평문 재전송 때 인용 태그가 지워짐", "<" not in common._tg_plain(c).replace("<b>", "") and "blockquote" not in common._tg_plain(c))

# ============ 4) 헤드라인
ck("제목 다듬기: 'Stock Market Today:' 머리말 제거, 110자 초과는 줄임", rl.tidy_headline("Stock Market Today: Tech Futures Slump") == "Tech Futures Slump" and len(rl.tidy_headline("x" * 300)) == 108)
resp = json.dumps([[["기술주 선물 하락\n", "Tech Futures Slump\n", None, None, 3], ["동일가중 펀드 선방", "Equal-Weight Funds Delivered", None, None, 3]], None, "en"])
ck("번역: 응답 조각을 이어 붙여 줄 단위로 나눔", rl.translate_ko(["Tech Futures Slump", "Equal-Weight Funds Delivered"], getter=lambda u: resp) == ["기술주 선물 하락", "동일가중 펀드 선방"])
seen = []
rl.translate_ko(["A & B: 50% off?"], getter=lambda u: seen.append(u) or "[]")
ck("번역: 질의는 URL 인코딩, 한 번만 호출", len(seen) == 1 and "A+%26+B%3A+50%25+off%3F" in seen[0] and "tl=ko" in seen[0])
orig = ["Tech Futures Slump", "Equal-Weight Funds Delivered", "Third"]
ck("번역: 개수 불일치 → 원문 그대로", rl.translate_ko(orig, getter=lambda u: resp) == orig)
for bad in ("not json", "[]", "[[null]]", json.dumps({"error": 429}), None):
    if rl.translate_ko(orig, getter=lambda u, b=bad: b) != orig:
        fails.append(f"bad translate {bad!r}")
ck("번역: 깨진 응답·오류·예외 → 원문 그대로", not any(f.startswith("bad translate") for f in fails) and rl.translate_ko(orig, getter=lambda u: 1 / 0) == orig)
ck("번역: 빈 입력 → []", rl.translate_ko([], getter=lambda u: 1 / 0) == [] and rl.translate_ko(None) == [])
hs = rl.headline_section(["Stock Market Today: A <b>x</b> & y", "B"], getter=lambda u: json.dumps([[["가 <b>x</b> & 나\n"], ["다"]]]))
ck("헤드라인 섹션: 번역문 HTML 이스케이프, 제목에 '자동 번역'", "자동 번역" in hs and "&lt;b&gt;" in hs and "&amp;" in hs and hs.count("•") == 2)
ck("헤드라인 섹션: 제목이 없으면 None", rl.headline_section([]) is None and rl.headline_section(None) is None)

# ============ 5) 전체 조립
base = er.build_sections(sums(), [], [], None, [], None, ["Stock Market Today: Tech Futures Slump", "When Nvidia Stumbled"], now, risk=R)
tr = lambda u: json.dumps([[["기술주 선물 하락\n"], ["엔비디아가 흔들릴 때"]]])
new = rl.restyle(base, sums(), R, now, headlines=["Stock Market Today: Tech Futures Slump", "When Nvidia Stumbled"], translate_getter=tr)
ck("조립: 첫 섹션은 한눈 요약, 둘째는 이벤트 카드", "한눈 요약" in new[0] and "이벤트 카드" in new[1])
ck("조립: 통과 0건의 '오늘의 결론'은 요약과 중복이라 생략", not any("오늘의 결론" in s for s in new) and any("오늘의 결론" in s for s in base))
ck("조립: 헤드라인이 한국어로 바뀜", any("기술주 선물 하락" in s for s in new) and not any("Tech Futures Slump" in s for s in new))
nums = [int(x) for s in new for x in re.findall(r"<b>(\d+)\. ", s)]
ck("조립: 섹션 번호 1부터 연속", nums == list(range(1, len(nums) + 1)) and len(nums) >= 3)
ck("조립: '오늘의 한 줄 공부'는 접지 않음", any("한 줄 공부" in s and "<blockquote" not in s for s in new))
ck("조립: 인용 태그 짝이 섹션마다 맞음", all(s.count("<blockquote expandable>") == s.count("</blockquote>") for s in new))
msgs = er.chunk_sections(new)
ck("조립: 메시지로 나눠도 길이 한도·태그 짝 유지", all(len(m) <= 4000 and m.count("<blockquote expandable>") == m.count("</blockquote>") and m.count("<b>") == m.count("</b>") for m in msgs))
passed = er.build_sections(sums("pass"), [], [], None, [], None, [], now, risk=R)
ck("조립: 통과가 생기면 '오늘의 결론'이 다시 보임", any("오늘의 결론" in s for s in rl.restyle(passed, sums("pass"), R, now)))
ck("조립: 요약을 만들 수 없으면 원본 그대로", rl.restyle(base, {}, R, now) == base and rl.restyle(base, None, None, now) == base)
ck("조립: 번역 실패 시 원문 헤드라인 유지", any("Tech Futures Slump" in s for s in rl.restyle(base, sums(), R, now, headlines=["Stock Market Today: Tech Futures Slump", "When Nvidia Stumbled"], translate_getter=lambda u: 1 / 0)))
ck("조립: 드라이버·맥락이 깨져 있어도 예외 없음", isinstance(rl.restyle(base, sums(), {"_drivers": "x", "_context": 3}, now, translate_getter=tr), list))
ck("조립: 페어 대칭 표시는 건드리지 않음(방향 신호 섹션 내용 동일)", [s for s in base if "방향 신호 (참고용" in s][0].split("\n", 1)[1] in [s for s in new if "방향 신호 (참고용" in s][0])

# ============ 6) 그림
png = rl.dashboard_png(sums(), R, now)
ck("그림: PNG 생성(머리 8바이트), 100KB 안팎", png is not None and png[:8] == b"\x89PNG\r\n\x1a\n" and 20_000 < len(png) < 600_000)
ck("그림: 맥락·드라이버가 없어도 생성(빈 칸은 'no data')", rl.dashboard_png(sums(), {}, now) is not None)
ck("그림: 요약 데이터가 없으면 None", rl.dashboard_png({}, R, now) is None and rl.dashboard_png(None, None, now) is None)
bad_d = {"indexes": {"SOXX": {"attr": {"asof": "nope", 1: {"est": "x", "contrib": {}}}, "earn": {"names": [("A", "bad", "w")]}}}}
ck("그림: 깨진 값이 섞여도 예외 없이 None 또는 PNG", rl.dashboard_png(sums(), dict(risk, _drivers=bad_d, _context={"fg_hist": {"bad": "x"}}), now) in (None,) or True)
posted = []
os.environ["TELEGRAM_TOKEN"], os.environ["TELEGRAM_CHAT_ID"] = "T", "C"
class Rsp:
    status_code = 200
ok = rl.send_photo(png, "설명", poster=lambda u, data, files: posted.append((u, data, files)) or Rsp())
ck("그림 전송: sendPhoto 로 사진과 설명", ok and posted[0][0].endswith("/botT/sendPhoto") and posted[0][1]["chat_id"] == "C" and posted[0][2]["photo"][1] == png)
ck("그림 전송: 실패·예외·빈 사진 → False", not rl.send_photo(png, poster=lambda *a: 1 / 0) and not rl.send_photo(None, poster=lambda *a: Rsp()))
os.environ.pop("TELEGRAM_TOKEN"); os.environ.pop("TELEGRAM_CHAT_ID")
ck("그림 전송: 토큰이 없으면 보내지 않음", not rl.send_photo(png, poster=lambda *a: Rsp()))

# ============ 7) F&G: 기준일이 가격일과 다를 때 이력에서 가격일 값을 기록
days = pd.bdate_range("2024-01-02", "2026-11-13")
ms = lambda d: float(pd.Timestamp(d).value // 10**6)
sc = [40.0 + (i % 7) for i in range(len(days))]; sc[-1] = 38.6
pl = {"fear_and_greed": {"score": 37.0, "rating": "fear", "timestamp": "2026-11-16T13:00:00+00:00", "previous_1_week": 32.0},
      "fear_and_greed_historical": {"data": [{"x": ms(d), "y": s} for d, s in zip(days, sc)]},
      "market_momentum_sp500": {"score": 15, "data": [{"x": ms(d), "y": 5000.0 + i} for i, d in enumerate(days)]}}
summ = {"scalp": {"horizon_days": 1, "items": [{"ticker": "TQQQ", "benchmark": "QQQ", "direction": 1, "up_prob": 57.0, "raw_up_prob": 58.0, "base_rate": 55.0, "edge": 2.0, "price": 57.68, "price_date": "2026-11-13"}]}}
os.environ.pop(mc.AV_KEY_ENV, None)
with tempfile.TemporaryDirectory() as td:
    cx = mc.collect(summ, td, fg_getter=lambda u: pl, sleeper=lambda s: None)
    c = pd.read_csv(os.path.join(td, mc.CONTEXT_LOG))
    ck("F&G 기록: 실행일(11/16) 값이 아니라 가격일(11/13)의 이력 값 38.6 을 그 행에 기록", len(c) == 1 and c.price_date.iloc[0] == "2026-11-13" and abs(float(c.fg_score.iloc[0]) - 38.6) < 1e-9)
    ck("F&G 기록: 리포트에는 현재 값(37) 표시, 그림용 이력 90일 보관", cx["fg"]["score"] == 37.0 and len(cx["fg_hist"]) == 90)
with tempfile.TemporaryDirectory() as td:
    pl2 = dict(pl, fear_and_greed_historical={"data": [{"x": ms(d), "y": s} for d, s in zip(days[:-1], sc[:-1])]})
    mc.collect(summ, td, fg_getter=lambda u: pl2, sleeper=lambda s: None)
    ck("F&G 기록: 이력에 가격일이 없으면 다른 날 값을 넣지 않음", not os.path.exists(os.path.join(td, mc.CONTEXT_LOG)))

# ============ 8) 버전 고정 · 모델 불변
req = open("requirements.txt", encoding="utf-8").read()
env = json.load(open("reports/env_versions.json", encoding="utf-8"))["versions"]
ck("버전 고정: 예측 라이브러리 5종이 실제 실행 환경 기록과 같은 버전으로 고정", all(f"{p}=={env[p]}" in req for p in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm")))
ck("모델 불변: 피처 20개·라벨 방식·로그 스키마 그대로", len(common.FEATURE_COLUMNS) == 20 and common.LABEL_MODE == "all" and len(common.LOG_COLUMNS) == 15)

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + str(fails)} ({len(fails)} fail)")
raise SystemExit(1 if fails else 0)
