"""daily_summary.collect_risk + easy_report 위험 상황판 연동 검증 (합성 데이터, 네트워크 불필요)."""
import os, shutil, tempfile
from datetime import datetime
import numpy as np
import pandas as pd
import daily_summary as ds
import easy_report as er
import risk_model as rm

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS" if cond else "FAIL"), name, extra)
    ok &= bool(cond)

rng = np.random.default_rng(3)
idx = pd.bdate_range('2025-01-02', periods=330)
full = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(idx)))), index=idx)

def fetcher_upto(n):
    return lambda t: pd.DataFrame({'Close': full.iloc[:n]})

KST = er.KST
tmp = tempfile.mkdtemp(); logd = os.path.join(tmp, 'logs'); repd = os.path.join(tmp, 'reports')
os.makedirs(logd)
# 방향 로그가 있어도 건드리지 않는지 확인용
pred_path = os.path.join(logd, 'predictions_scalp.csv'); open(pred_path, 'w').write('run_date,ticker\n2026-10-01,TQQQ\n')
before = open(pred_path, 'rb').read()

n0 = 300
now0 = datetime(2026, 10, 16, 9, 0, tzinfo=KST)
r0 = ds.collect_risk(['TQQQ', 'SOXL'], now0, logd, repd, fetcher=fetcher_upto(n0))
check("C1 스냅샷 2종목 생성", set(r0) == {'TQQQ', 'SOXL'} and r0['SOXL']['move_5d'] > 0)
vl = rm.load_vol_log(logd)
check("C2 변동성 로그: 2종목×3호라이즌=6행, 모두 미정산", len(vl) == 6 and (vl['settled'] == 0).all())
check("C3 방향 로그는 바이트 단위로 불변", open(pred_path, 'rb').read() == before)
check("C4 성적표 파일 생성(정산 전 안내 문구)", '아직 정산된 행이 없습니다' in open(os.path.join(repd, 'vol_scorecard.txt'), encoding='utf-8').read())

# 같은 날 재실행 → 중복 없음
ds.collect_risk(['TQQQ', 'SOXL'], now0, logd, repd, fetcher=fetcher_upto(n0))
check("C5 같은 가격일 재실행해도 6행 유지", len(rm.load_vol_log(logd)) == 6)

# 6거래일 뒤: h=1,5 정산, h=20 보류, 새 가격일 행 추가
n1 = n0 + 6
now1 = datetime(2026, 10, 23, 9, 0, tzinfo=KST)
ds.collect_risk(['TQQQ', 'SOXL'], now1, logd, repd, fetcher=fetcher_upto(n1))
vl = rm.load_vol_log(logd)
old = vl[vl['price_date'] == full.index[n0 - 1]]
st = old.groupby('horizon_days')['settled'].min().to_dict()
check("C6 6거래일 뒤: 과거 행 h=1,5 정산·h=20 보류", st == {1: 1, 5: 1, 20: 0}, str(st))
exp = rm.realized_vol_after(full.iloc[:n1], full.index[n0 - 1], 5)
got = old[(old['ticker'] == 'SOXL') & (old['horizon_days'] == 5)]['vol_realized'].iloc[0]
check("C7 정산값이 독립 계산값과 일치", abs(got - exp) < 1e-12)
check("C8 새 가격일 행 추가됨(총 12행)", len(vl) == 12)
txt = open(os.path.join(repd, 'vol_scorecard.txt'), encoding='utf-8').read()
check("C9 성적표에 후보·QLIKE 표시", 'vol_ewma' in txt and 'qlike' in txt)

# 실패 방어
class Boom(Exception): pass
def boom(t): raise Boom("net")
check("C10 시세 전부 실패해도 예외 없이 {} 반환", ds.collect_risk(['TQQQ'], now1, logd, repd, fetcher=boom) == {})
check("C11 빈 데이터도 {} 반환", ds.collect_risk(['TQQQ'], now1, logd, repd, fetcher=lambda t: pd.DataFrame({'Close': []})) == {})
check("C12 데이터 짧으면 스냅샷 없이 {}", ds.collect_risk(['TQQQ'], now1, logd, repd, fetcher=fetcher_upto(40)) == {})
# 로그 파일이 깨졌을 때
open(os.path.join(logd, 'vol_forecasts.csv'), 'w').write('garbage,,,\n1,2\n')
res = ds.collect_risk(['SOXL'], now1, logd, repd, fetcher=fetcher_upto(n1))
check("C13 깨진 로그 파일에서도 죽지 않음", isinstance(res, dict))
check("C13b 깨진 원본은 덮어쓰기 전에 .corrupt_* 로 보존됨",
      any('.corrupt_' in f for f in os.listdir(logd)))
shutil.rmtree(tmp)

# 리포트 연동: 선택 인자, 텍스트, HTML 안전
import selftest_v25 as st25
S = st25.make_summaries()
now = datetime(2026, 10, 16, 12, 0, tzinfo=KST)
secs_no = er.build_sections(S, [], [], None, [], None, [], now)
secs_rk = er.build_sections(S, [], [], None, [], None, [], now, risk={'SOXL': r0['SOXL']})
t_no, t_rk = '\n'.join(secs_no), '\n'.join(secs_rk)
check("C14 risk 없으면 기존 출력과 동일(하위호환)", '예상 흔들림' not in t_no)
check("C15 risk 있으면 해당 종목에만 예상 흔들림 표시", t_rk.count('예상 흔들림') == 1 and '방향 예측 아님' in t_rk)
check("C16 상태 문구 HTML 안전·길이 한도", all(len(m) <= er.TELEGRAM_LIMIT for m in er.chunk_sections(secs_rk)) and '<<' not in t_rk)
check("C17 섹션 번호 연속", [int(x) for x in __import__('re').findall(r'<b>(\d+)\. ', t_rk)] == list(range(1, len(__import__('re').findall(r'<b>(\d+)\. ', t_rk)) + 1)))
check("C18 '매수/매도/사라' 문구 없음", all(w not in t_rk for w in ('매수', '매도', '사라', '목표가')))
print("\n" + ("ALL PASS" if ok else "SOME FAIL"))
raise SystemExit(0 if ok else 1)
