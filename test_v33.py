"""test_v33.py — v33 표시/기록 보조 수정 검증(합성 데이터). 방향 모델·피처·임계값은 건드리지 않는다."""
import json, os, tempfile
from datetime import datetime
import pandas as pd
import common, easy_report as er

fails = []
def ck(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)

# 1) 실적 임박: 기준일 = 가격일
common.STOCK_EARNINGS_DATES['ZZZ'] = '2026-11-17'
w = common.earnings_proximity_warning
ck("실적 당일 종가 기준(다음날 반응 예측) → 경고", w('ZZZ', 5, ref_date=pd.Timestamp('2026-11-17')) is not None)
ck("실적 5일 전 종가 → 경고", w('ZZZ', 5, ref_date='2026-11-12') is not None)
ck("실적 6일 전 종가 → 없음", w('ZZZ', 5, ref_date='2026-11-11') is None)
ck("실적 다음날 종가(반응 끝남) → 없음", w('ZZZ', 5, ref_date='2026-11-18') is None)
ck("tz 있는 기준일도 처리", w('ZZZ', 5, ref_date=pd.Timestamp('2026-11-17', tz='UTC')) is not None)
ck("ref_date=NaT → 예외 없이 기존 동작", w('ZZZ', 5, ref_date=pd.NaT) in (None,) or isinstance(w('ZZZ', 5, ref_date=pd.NaT), str))
ck("ref_date 깨진 값 → 예외 없음", (w('ZZZ', 5, ref_date='not-a-date') is None) or True)
ck("등록 안 된 종목 → None", w('NOPE', 5, ref_date='2026-11-17') is None)
ck("NVDA 실적일 11/17", common.STOCK_EARNINGS_DATES['NVDA'] == '2026-11-17')
del common.STOCK_EARNINGS_DATES['ZZZ']

# 2) 달력 만료 점검
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, 'h.json')
    json.dump({'updated': '2026-10-16', 'issues': [{'title': 'a', 'until': '2026-10-30'}]}, open(p, 'w'))
    c = er.calendar_staleness({'2026-11-10': {}}, {'AAPL': '2026-10-29', 'NVDA': '2026-11-17'}, hot_path=p, today='2026-11-11')
    ck("지난 실적일·빈 일정표·만료 메모 3건 모두 감지", len(c) == 3 and 'AAPL' in c[0] and 'NVDA' not in c[0])
    c2 = er.calendar_staleness({'2026-12-09': {}}, {'NVDA': '2026-11-17'}, hot_path=p, today='2026-10-20')
    ck("모두 유효하면 빈 리스트", c2 == [])
    ck("파일 없음/깨진 입력에도 예외 없음", er.calendar_staleness(None, None, hot_path=os.path.join(d, 'x.json')) == ['경제 일정표(MARKET_EVENTS)에 앞으로의 일정이 없음'])
    ck("깨진 날짜", isinstance(er.calendar_staleness({'bad': {}}, {'A': 'bad'}, hot_path=p, today='2026-11-11'), list))

# 3) 리포트 문구
item = dict(ticker='SOXL', benchmark='SMH', direction=1, up_prob=52.4, edge=-2.0, price=56.06, vol=2.29,
            drawdown=-0.659, ret5=-0.23, price_date='2026-11-10', signal={'status': 'insufficient'})
summ = {'scalp': {'horizon_days': 1, 'items': [item]}}
now = datetime(2026, 11, 11, 9, 0, tzinfo=er.KST)
base = {'SOXL': {'move_1d': 0.154, 'move_5d': 0.345, 'regime': '고변동'}}
txt = "\n".join(er.build_sections(summ, [], [], None, [], None, [], now, risk=dict(base)))
ck("'본전' 표현 제거, 60일 고점 기준 명시", '본전까지' not in txt and '60일 고점 회복까지 +193% 필요' in txt and '60일 고점대비 -65.9%' in txt)
ck("편향 정보 없으면 주석 없음", '실제 흔들림은 예상의' not in txt)
r2 = dict(base); r2['_vol_bias'] = {1: {'ratio': 1.62, 'n_eff': 18.0, 'n': 54}, 5: {'ratio': 1.94, 'n_eff': 2.8, 'n': 42}}
r2['_calendar_warn'] = ['핫이슈 <만료>']
txt2 = "\n".join(er.build_sections(summ, [], [], None, [], None, [], now, risk=r2))
ck("과소예측 주석 표시", '약 1.6배(하루)·1.9배(5일)' in txt2 and '아직 못 믿는 수치' in txt2)
ck("달력 경고 표시 + HTML 이스케이프", '갱신 필요' in txt2 and '&lt;만료&gt;' in txt2)
r3 = dict(base); r3['_vol_bias'] = {1: {'ratio': 1.05, 'n_eff': 100.0, 'n': 300}}
ck("보정 양호(1.25배 미만)면 주석 없음", '실제 흔들림은 예상의' not in "\n".join(er.build_sections(summ, [], [], None, [], None, [], now, risk=r3)))
r4 = dict(base); r4['_vol_bias'] = {'x': None}; r4['_calendar_warn'] = None
ck("깨진 보조 정보에도 리포트 생성", len(er.build_sections(summ, [], [], None, [], None, [], now, risk=r4)) > 0)
ck("risk=None 도 동작", len(er.build_sections(summ, [], [], None, [], None, [], now, risk=None)) > 0)

# 4) 일정표
ev = er.upcoming_events(common.MARKET_EVENTS, common.STOCK_EARNINGS_DATES, today='2026-11-30', window=14)
ck("12월 FOMC·CPI·고용 일정 노출", len(ev) == 3)

print("\nALL PASS" if not fails else f"\nFAILED: {fails}")
raise SystemExit(1 if fails else 0)
