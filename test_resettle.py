"""resettle_logs 합성 데이터 검증 (네트워크 불필요)."""
import os, shutil, tempfile
import numpy as np
import pandas as pd
import common
import resettle_logs as R

ok = True
def check(name, cond):
    global ok
    print(("PASS" if cond else "FAIL"), name)
    ok &= bool(cond)

# 합성 시세: 영업일 60일, 알려진 값
idx = pd.bdate_range('2026-08-03', periods=60)
close = pd.Series(100.0 + np.arange(60) * 1.0, index=idx)
# 일부 구간을 하락으로 바꿔 라벨이 갈리게 함
close.iloc[11] = 90.0   # 8/18
px = pd.DataFrame({'Close': close})

# 로그: 수요일(8/19) 실행, 1일 호라이즌 예측. price_date 없음(구 로그) → 직전 영업일 8/18 종가가 entry
# 구(舊) 기준: 실행일 + 1영업일(8/20) 종가로 채점. 신 기준: 8/18 → 8/19(1거래일 뒤)
run = pd.Timestamp('2026-08-19')
entry_old = float(close[pd.Timestamp('2026-08-18')])      # 90
rows = [{
    'run_date': run, 'benchmark': 'QQQ', 'ticker': 'TQQQ', 'horizon_days': 1,
    'predicted_label': 1, 'predicted_up_prob': 0.55, 'raw_up_prob': 0.55,
    'price_at_prediction': entry_old, 'target_date': pd.Timestamp('2026-08-20'),
    'actual_price': float(close[pd.Timestamp('2026-08-20')]),
    'actual_return': float(close[pd.Timestamp('2026-08-20')]) / entry_old - 1,
    'actual_label': 1, 'correct': 1, 'near_earnings': 0, 'price_date': pd.NaT},
    # 미정산 행
    {'run_date': pd.Timestamp('2026-10-14'), 'benchmark': 'QQQ', 'ticker': 'TQQQ', 'horizon_days': 5,
     'predicted_label': 1, 'predicted_up_prob': 0.5, 'raw_up_prob': 0.5,
     'price_at_prediction': 150.0, 'target_date': pd.NaT, 'actual_price': np.nan, 'actual_return': np.nan,
     'actual_label': np.nan, 'correct': np.nan, 'near_earnings': 0, 'price_date': pd.NaT}]
df = pd.DataFrame(rows, columns=common.LOG_COLUMNS)
df['price_date'] = pd.to_datetime(df['price_date'])

new, changed = R.resettle_df(df, lambda t: px)
r0 = new.iloc[0]
exp_actual = float(close[pd.Timestamp('2026-08-19')])      # 8/19 종가 (8/18의 +1거래일)
check("price_date 보강(8/18)", pd.Timestamp(r0['price_date']) == pd.Timestamp('2026-08-18'))
check("새 기준 actual_price = 8/19 종가", abs(r0['actual_price'] - exp_actual) < 1e-9)
check("target_date = 8/19", pd.Timestamp(r0['target_date']) == pd.Timestamp('2026-08-19'))
check("수익률 정확", abs(r0['actual_return'] - (exp_actual / entry_old - 1)) < 1e-9)
check("원본 df 불변(복사본 사용)", df.iloc[0]['actual_price'] == float(close[pd.Timestamp('2026-08-20')]))
check("미정산 행은 정산 안 됨", pd.isna(new.iloc[1]['actual_label']))
check("미정산 행 target_date 교정(10/13+5BD=10/20)", pd.Timestamp(new.iloc[1]['target_date']) == pd.Timestamp('2026-10-20'))

# 라벨이 실제로 뒤집히는 케이스: 8/18=90 → 8/19=가격 > 90 이면 up(1), 8/20 도 up. 뒤집힘 확인용으로 하락 구간 추가
close2 = close.copy(); close2.iloc[12] = 85.0   # 8/19 = 85 (하락), 8/20 = 113 (상승)
px2 = pd.DataFrame({'Close': close2})
df2 = df.copy(); df2.loc[0, 'actual_price'] = float(close2.iloc[13]); df2.loc[0, 'actual_label'] = 1; df2.loc[0, 'correct'] = 1
new2, ch2 = R.resettle_df(df2, lambda t: px2)
check("라벨 뒤집힘 감지(changed=1)", ch2 == 1)
check("뒤집힌 후 correct=0 (예측 상승, 실제 하락)", int(new2.iloc[0]['correct']) == 0)

# 시세 실패 → 행 유지
new3, ch3 = R.resettle_df(df, lambda t: None)
check("시세 실패 시 변경 0", ch3 == 0 and new3.iloc[0]['actual_price'] == df.iloc[0]['actual_price'])

# 파일 왕복 + 백업 (main --apply)
tmp = tempfile.mkdtemp()
common.LOG_DIR = tmp
common.save_log('scalp', df)
before_bytes = open(os.path.join(tmp, 'predictions_scalp.csv'), 'rb').read()
common.fetch_ticker_data = lambda t, **k: px2
import sys
sys.argv = ['resettle_logs.py', tmp, '--apply']
R.main()
baks = [f for f in os.listdir(tmp) if '.bak_v25_' in f]
check("백업 파일 생성", len(baks) == 1)
check("백업 = 원본 바이트 동일", open(os.path.join(tmp, baks[0]), 'rb').read() == before_bytes)
re = common.load_log('scalp')
check("저장 후 재로딩 가능 & 새 기준 반영", int(re.iloc[0]['correct']) == 0)
# dry-run은 파일 불변
common.save_log('scalp', df)
b2 = open(os.path.join(tmp, 'predictions_scalp.csv'), 'rb').read()
sys.argv = ['resettle_logs.py', tmp]
R.main()
check("dry-run 파일 불변", open(os.path.join(tmp, 'predictions_scalp.csv'), 'rb').read() == b2)
shutil.rmtree(tmp)
print("\nALL PASS" if ok else "\nSOME FAIL")
raise SystemExit(0 if ok else 1)
