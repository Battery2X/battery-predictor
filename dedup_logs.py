"""
logs/predictions_*.csv 에서 "같은 날 같은 ticker/horizon" 중복 실행 기록을 정리한다.
규칙: (날짜, ticker, horizon_days)가 같은 행이 여러 개면, run_date가 가장 이른(그날 첫 실행)
      행 하나만 남기고 나머지는 삭제한다. (cron 스케줄과 가장 가까운 실행을 대표값으로 채택)

실행: python dedup_logs.py [logs 폴더 경로, 기본값 'logs'] [--apply]
  --apply 없이 실행하면 무엇이 삭제될지만 보여주고 파일은 건드리지 않음 (dry-run).
  --apply 를 붙이면 실제로 정리하고, 원본은 <파일명>.bak_YYYYMMDD_HHMMSS 로 백업 후 덮어씀.

주의:
- 이 스크립트는 모델/피처/임계값을 바꾸지 않는다. 로그 "표본 수"만 줄어든다.
- 실행 전 반드시 git 커밋이 된 상태인지 확인할 것 (백업 파일과 별개로 git 히스토리로도 복구 가능해야 함).
"""
import sys
import glob
import os
import shutil
from datetime import datetime
import pandas as pd


def dedup_file(path, apply_changes):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  ⚠️ 읽기 실패, 스킵: {e}")
        return

    if df.empty or 'run_date' not in df.columns:
        print(f"  ⚠️ 빈 파일이거나 예상 스키마 아님, 스킵")
        return

    original_len = len(df)

    # run_date 파싱용 임시 컬럼 (원본 문자열 컬럼은 그대로 보존)
    parsed = pd.to_datetime(df['run_date'], errors='coerce')
    if parsed.isna().any():
        n_bad = parsed.isna().sum()
        print(f"  ⚠️ run_date 파싱 실패한 행 {n_bad}개 발견 — 해당 행은 정리 대상에서 제외하고 그대로 둠")

    df['_run_date_parsed'] = parsed
    df['_run_date_only'] = parsed.dt.date

    valid_mask = parsed.notna()
    df_valid = df[valid_mask].copy()
    df_invalid = df[~valid_mask].copy()  # 파싱 안 된 행은 그대로 보존

    if 'ticker' not in df_valid.columns or 'horizon_days' not in df_valid.columns:
        print(f"  ⚠️ ticker/horizon_days 컬럼 없음, 스킵")
        return

    # 그룹별로 가장 이른 run_date 행 하나만 유지
    df_valid_sorted = df_valid.sort_values('_run_date_parsed')
    keep_idx = df_valid_sorted.groupby(
        ['_run_date_only', 'ticker', 'horizon_days'], as_index=False
    ).head(1).index

    dropped = df_valid.drop(index=keep_idx)
    kept_valid = df_valid.loc[keep_idx]

    result = pd.concat([kept_valid, df_invalid], ignore_index=True)
    result = result.sort_values('_run_date_parsed', na_position='last')
    result = result.drop(columns=['_run_date_parsed', '_run_date_only'])

    n_dropped = len(dropped)
    new_len = len(result)

    print(f"  원본 {original_len}행 → 정리 후 {new_len}행 (제거 {n_dropped}행)")

    if n_dropped == 0:
        print(f"  중복 없음, 변경 없이 넘어감")
        return

    if not apply_changes:
        print(f"  [DRY-RUN] 실제로는 아무 것도 바뀌지 않았습니다. --apply 를 붙여서 재실행하세요.")
        return

    # 백업 후 덮어쓰기
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{path}.bak_{ts}"
    shutil.copy2(path, backup_path)
    print(f"  백업 생성: {backup_path}")

    # 원본 컬럼 순서 유지
    original_cols = pd.read_csv(path, nrows=0).columns.tolist()
    result = result[original_cols]
    result.to_csv(path, index=False)
    print(f"  ✅ 정리 완료: {path}")


def main():
    args = sys.argv[1:]
    apply_changes = '--apply' in args
    positional = [a for a in args if a != '--apply']
    logs_dir = positional[0] if positional else 'logs'

    pattern = os.path.join(logs_dir, 'predictions_*.csv')
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"'{pattern}' 패턴에 맞는 파일이 없습니다.")
        return

    mode = "실제 적용" if apply_changes else "DRY-RUN (미리보기만, 파일 미변경)"
    print(f"모드: {mode}\n")

    for f in files:
        print(f"📄 {f}")
        dedup_file(f, apply_changes)
        print()


if __name__ == '__main__':
    main()
