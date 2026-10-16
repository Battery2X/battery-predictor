"""
resettle_logs.py — 과거 정산 행을 '모델 목표와 같은 기준'으로 다시 채점한다 (v25 정산 수정용, 1회성).

배경: v24 이전 정산은 '실행일 + h영업일 이후 첫 종가'를 써서, 화~금 실행분이 h+1거래일 수익률로
      채점됐다(모델은 종가 t → t+h 를 학습). 이 스크립트는 이미 정산된 행의 actual_*, correct 를
      price_date(없으면 직전 영업일 폴백) 기준으로 다시 계산해 바로잡는다.

실행: python resettle_logs.py [logs 폴더, 기본 'logs'] [--apply]
  --apply 없이 실행하면 변경 전/후 적중률만 보여주고 파일은 건드리지 않는다(dry-run).
  --apply 를 붙이면 <파일>.bak_v25_YYYYMMDD 로 백업 후 덮어쓴다.

주의
- 예측 모델·피처·임계값은 바뀌지 않는다. 바뀌는 건 '채점 기준'뿐이다.
- 시장 데이터(yfinance) 접근이 필요하다 → GitHub Actions(workflow_dispatch)나 로컬 PC에서 실행.
- 시세 조회가 실패한 종목은 해당 행을 건드리지 않고 그대로 둔다.
"""
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import common

LOG_NAMES = ['scalp', 'swing', 'position', 'stock_scalp', 'stock_swing', 'stock_position']


def resettle_df(df, price_lookup):
    """
    df: load_log 결과. price_lookup(ticker) -> Close 컬럼 DataFrame 또는 None.
    반환: (새 df, 변경된 행 수). 정산 완료 행만 대상, 미정산 행은 그대로.
    """
    df = df.copy()
    changed = 0
    done = df['actual_label'].notna()
    # price_date 보강 (모든 행: 정산 완료/미정산 무관)
    miss = df['price_date'].isna()
    if miss.any():
        df.loc[miss, 'price_date'] = df.loc[miss, 'run_date'].apply(common._fallback_price_date)
    for ticker in df.loc[done, 'ticker'].unique():
        px = price_lookup(ticker)
        if px is None or len(px) == 0:
            print(f"  ⚠️ {ticker}: 시세 조회 실패 — 해당 종목 정산 행은 그대로 둠")
            continue
        px = px.ffill().dropna()
        for idx, row in df[done & (df['ticker'] == ticker)].iterrows():
            res = common.compute_aligned_outcome(px, row['price_date'], row['horizon_days'],
                                                 row['price_at_prediction'])
            if res is None:
                continue          # 새 기준으로는 아직 h거래일이 안 지남 → 다음에 다시
            new_correct = int(res['actual_label'] == int(row['predicted_label']))
            if (res['actual_label'] != row['actual_label']) or (new_correct != row['correct']):
                changed += 1
            df.loc[idx, 'actual_price'] = res['actual_price']
            df.loc[idx, 'actual_return'] = res['actual_return']
            df.loc[idx, 'actual_label'] = res['actual_label']
            df.loc[idx, 'correct'] = new_correct
            df.loc[idx, 'target_date'] = res['target_used']
    # 미정산 행의 target_date 도 새 기준으로 교정
    pend = df['actual_label'].isna()
    if pend.any():
        df.loc[pend, 'target_date'] = [pd.Timestamp(d) + pd.offsets.BDay(int(h)) for d, h in
                                       zip(df.loc[pend, 'price_date'], df.loc[pend, 'horizon_days'])]
    return df, changed


def accuracy_table(df):
    s = df[df['correct'].notna()]
    if s.empty:
        return {}
    return {t: (len(g), float(g['correct'].mean())) for t, g in s.groupby('ticker')}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    apply_changes = '--apply' in sys.argv
    logs_dir = args[0] if args else common.LOG_DIR
    common.LOG_DIR = logs_dir
    stamp = datetime.now().strftime('%Y%m%d')
    cache = {}

    def lookup(t):
        if t not in cache:
            cache[t] = common.fetch_ticker_data(t)
        return cache[t]

    for name in LOG_NAMES:
        path = os.path.join(logs_dir, f'predictions_{name}.csv')
        if not os.path.exists(path):
            continue
        df = common.load_log(name)
        new_df, changed = resettle_df(df, lookup)
        before, after = accuracy_table(df), accuracy_table(new_df)
        print(f"\n[{name}] 정산 행 {int(df['correct'].notna().sum())}건 중 결과가 바뀐 행 {changed}건")
        for t in sorted(set(before) | set(after)):
            b = before.get(t, (0, np.nan))
            a = after.get(t, (0, np.nan))
            print(f"   {t:5s} 변경 전 {b[1]*100:5.1f}% (n={b[0]}) → 변경 후 {a[1]*100:5.1f}% (n={a[0]})")
        if apply_changes and changed >= 0:
            shutil.copy2(path, f"{path}.bak_v25_{stamp}")
            common.save_log(name, new_df)
            print(f"   ✅ 적용 완료 (백업: {os.path.basename(path)}.bak_v25_{stamp})")
    if not apply_changes:
        print("\n(dry-run) 파일은 변경하지 않았습니다. 적용하려면 --apply 를 붙이세요.")


if __name__ == '__main__':
    main()
