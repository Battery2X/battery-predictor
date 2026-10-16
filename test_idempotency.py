"""append_predictions 멱등성·동시성 극한 검증 (v28)."""
import sys, tempfile, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np, common
def fresh():
    d=tempfile.mkdtemp(); common.LOG_DIR=d; return d
def rec(rd,t='SOXL',h=1,pd_=None,up=60.0):
    return common.make_prediction_record(pd.Timestamp(rd),'SMH',t,h,up,up,100.0,price_date=pd_)
def n(): return len(common.load_log('x'))
res=[]
def case(name,ok): res.append((name,ok)); print(('OK  ' if ok else 'HOLE'),name)
# 1 same batch twice
fresh(); b=[rec('2026-10-14 09:17'),rec('2026-10-14 09:17','SOXS')]
common.append_predictions('x',b); common.append_predictions('x',b); case('같은 배치 2회',n()==2)
# 2 duplicate inside one batch
fresh(); common.append_predictions('x',[rec('2026-10-14 09:17'),rec('2026-10-14 09:17')]); case('한 배치 내부 중복',n()==1)
# 3 same price_date, different run_date inside ONE batch (Sat+Mon style)
fresh(); common.append_predictions('x',[rec('2026-10-17 09:17',pd_='2026-10-16'),rec('2026-10-19 09:17',pd_='2026-10-16')]); case('한 배치 내 같은 종가기준 2건',n()==1)
# 4 sat run then mon run (separate)
fresh(); common.append_predictions('x',[rec('2026-10-17 09:17',pd_='2026-10-16')]); common.append_predictions('x',[rec('2026-10-19 09:17',pd_='2026-10-16')]); case('토→월 재실행(같은 종가)',n()==1)
# 5 horizon float vs int
fresh(); common.append_predictions('x',[rec('2026-10-14 09:17',h=5)]); common.append_predictions('x',[rec('2026-10-14 09:17',h=5.0)]); case('호라이즌 5 vs 5.0',n()==1)
# 6 ticker case/space
fresh(); common.append_predictions('x',[rec('2026-10-14 09:17','SOXL')]); common.append_predictions('x',[rec('2026-10-14 09:17','soxl ')]); case('티커 대소문자/공백',n()==1)
# 7 tz-aware run_date
fresh(); common.append_predictions('x',[rec('2026-10-14 09:17')]);
try:
    common.append_predictions('x',[common.make_prediction_record(pd.Timestamp('2026-10-14 09:17',tz='UTC'),'SMH','SOXL',1,60,60,100.0)]); case('tz-aware run_date',n()==1)
except Exception as e: case('tz-aware run_date 예외 '+type(e).__name__,False)
# 8 string run_date
fresh(); common.append_predictions('x',[rec('2026-10-14 09:17')]); r=rec('2026-10-14 09:17'); r['run_date']='2026-10-14 23:59'; common.append_predictions('x',[r]); case('문자열 run_date 같은 날',n()==1)
# 9 real concurrent processes
import subprocess, textwrap
d=fresh()
code=textwrap.dedent("""
import sys; sys.path.insert(0,'.'); import common, pandas as pd
common.LOG_DIR=sys.argv[1]; k=int(sys.argv[2])
rows=[common.make_prediction_record(pd.Timestamp('2026-10-14 09:17')+pd.Timedelta(seconds=k),'SMH',f'T{k}_{i}',1,60,60,100.0) for i in range(3)]
common.append_predictions('x',rows)
""")
ps=[subprocess.Popen([sys.executable,'-c',code,d,str(k)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,cwd=os.path.dirname(os.path.abspath(__file__))) for k in range(8)]
[p.wait() for p in ps]
case('동시 8프로세스 append(24행 모두 보존)',n()==24)
# 10 settled row preserved on re-append
d=fresh(); common.append_predictions('x',[rec('2026-10-14 09:17')]); df=common.load_log('x'); df.loc[0,['actual_label','correct']]=[1,1]; common.save_log('x',df)
common.append_predictions('x',[rec('2026-10-14 09:17')]); df=common.load_log('x'); case('정산 완료 행 보존',len(df)==1 and df.loc[0,'correct']==1)
# 11 NaT price_date new row vs existing
fresh(); common.append_predictions('x',[rec('2026-10-14 09:17')]); common.append_predictions('x',[rec('2026-10-15 09:17',pd_='2026-10-14')]); case('다른 날·다른 종가(정상 2건)',n()==2)
bad=sum(not o for _,o in res); print(bad,'holes'); sys.exit(1 if bad else 0)
