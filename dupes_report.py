"""
logs/predictions_*.csv 에서 "같은 날 같은 ticker/horizon"으로 여러 번 실행된
중복 기록이 얼마나 있는지 집계하는 진단 스크립트.

실행: python dupes_report.py [logs 폴더 경로, 기본값 'logs']

- 로직/모델 변경 없음. 읽기 전용 (파일 수정 안 함).
- 파일이 없거나 비어있으면 조용히 스킵.
"""
import sys
import glob
import os
import pandas as pd


def analyze_file(path):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  ⚠️ 읽기 실패, 스킵: {e}")
        return None
    if df.empty or 'run_date' not in df.columns:
        print(f"  ⚠️ 빈 파일이거나 예상 스키마 아님, 스킵")
        return None

    df['run_date'] = pd.to_datetime(df['run_date'], errors='coerce')
    df = df.dropna(subset=['run_date'])
    df['run_date_only'] = df['run_date'].dt.date

    group_cols = ['run_date_only', 'ticker', 'horizon_days']
    counts = df.groupby(group_cols).size().reset_index(name='n_runs')
    dupes = counts[counts['n_runs'] > 1].sort_values(['run_date_only', 'ticker'])

    total_rows = len(df)
    total_groups = len(counts)
    dup_groups = len(dupes)
    extra_rows = int((dupes['n_runs'] - 1).sum())  # 중복 제거시 사라질 행 수

    print(f"  전체 행 수: {total_rows}")
    print(f"  고유 (날짜,종목,호라이즌) 조합 수: {total_groups}")
    if total_rows:
        print(f"  중복된 조합 수: {dup_groups}  |  제거될 행 수: {extra_rows} "
              f"({extra_rows/total_rows*100:.1f}% of total)")

    if dup_groups > 0:
        print(f"  --- 중복 상세 (상위 15개) ---")
        for _, row in dupes.head(15).iterrows():
            print(f"    {row['run_date_only']} | {row['ticker']} | {int(row['horizon_days'])}일 "
                  f"→ {int(row['n_runs'])}회 실행")
        if dup_groups > 15:
            print(f"    ... 외 {dup_groups - 15}건 더")

    return {
        'file': os.path.basename(path),
        'total_rows': total_rows,
        'total_groups': total_groups,
        'dup_groups': dup_groups,
        'extra_rows': extra_rows,
    }


def main():
    logs_dir = sys.argv[1] if len(sys.argv) > 1 else 'logs'
    pattern = os.path.join(logs_dir, 'predictions_*.csv')
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"'{pattern}' 패턴에 맞는 파일이 없습니다.")
        return

    summary = []
    for f in files:
        print(f"\n📄 {f}")
        result = analyze_file(f)
        if result:
            summary.append(result)

    if not summary:
        print("\n분석 가능한 파일이 없었습니다.")
        return

    print("\n" + "=" * 50)
    print("전체 요약")
    print("=" * 50)
    total_rows = sum(s['total_rows'] for s in summary)
    total_extra = sum(s['extra_rows'] for s in summary)
    for s in summary:
        pct = (s['extra_rows'] / s['total_rows'] * 100) if s['total_rows'] else 0
        print(f"  {s['file']}: {s['total_rows']}행 중 {s['extra_rows']}행 중복 ({pct:.1f}%)")
    overall_pct = (total_extra / total_rows * 100) if total_rows else 0
    print(f"\n  합계: {total_rows}행 중 {total_extra}행이 같은 날 재실행분 ({overall_pct:.1f}%)")


if __name__ == '__main__':
    main()
