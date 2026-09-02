"""
check_logs.py — logs/predictions_*.csv 6개 파일이 각자 있어야 할 종목/호라이즌만
담고 있는지 확인하는 검증 스크립트. 파일을 손으로 옮기다 실수로 다른 로그 내용을
엉뚱한 파일에 덮어썼을 때(예: swing.csv 내용이 stock_scalp.csv에 들어간 경우)
바로 잡아낸다.

실행: python check_logs.py [logs 폴더 경로, 기본값 'logs']
- 읽기 전용. 파일을 고치지 않는다.
- git commit 하기 전에 습관적으로 한 번 돌려보는 걸 권장.
"""
import sys
import os
import pandas as pd

EXPECTED = {
    'predictions_scalp.csv':          ({1},  {'TQQQ', 'SQQQ', 'SOXL', 'SOXS', 'SPXL', 'SPXS'}),
    'predictions_swing.csv':          ({5},  {'TQQQ', 'SQQQ', 'SOXL', 'SOXS', 'SPXL', 'SPXS'}),
    'predictions_position.csv':       ({20}, {'TQQQ', 'SQQQ', 'SOXL', 'SOXS', 'SPXL', 'SPXS'}),
    'predictions_stock_scalp.csv':    ({1},  {'AAPL', 'NVDA', 'TSLA'}),
    'predictions_stock_swing.csv':    ({5},  {'AAPL', 'NVDA', 'TSLA'}),
    'predictions_stock_position.csv': ({20}, {'AAPL', 'NVDA', 'TSLA'}),
}


def main():
    logs_dir = sys.argv[1] if len(sys.argv) > 1 else 'logs'
    all_ok = True

    for fname, (expected_horizons, expected_tickers) in EXPECTED.items():
        path = os.path.join(logs_dir, fname)
        if not os.path.exists(path):
            print(f"⚠️ {fname}: 파일 없음 (스킵)")
            continue
        df = pd.read_csv(path)
        if df.empty:
            print(f"⚠️ {fname}: 빈 파일 (스킵)")
            continue

        actual_horizons = set(df['horizon_days'].unique())
        actual_tickers = set(df['ticker'].unique())

        bad_horizons = actual_horizons - expected_horizons
        bad_tickers = actual_tickers - expected_tickers

        if bad_horizons or bad_tickers:
            all_ok = False
            print(f"❌ {fname}: {len(df)}행 — 이 파일에 있으면 안 되는 데이터 발견!")
            if bad_horizons:
                print(f"     예상 못한 horizon_days: {bad_horizons} (원래는 {expected_horizons}만 있어야 함)")
            if bad_tickers:
                print(f"     예상 못한 ticker: {bad_tickers} (원래는 {expected_tickers}만 있어야 함)")
            print(f"     -> 다른 로그 파일 내용이 잘못 섞였을 가능성이 높습니다. 덮어쓰기 전 파일을 다시 확인하세요.")
        else:
            print(f"✅ {fname}: {len(df)}행 — 정상 (ticker={sorted(actual_tickers)}, horizon={sorted(actual_horizons)})")

    print()
    if all_ok:
        print("모든 로그 파일 정상 — commit 해도 됩니다.")
    else:
        print("문제 있는 파일이 있습니다 — 위 내용을 고친 뒤 다시 확인하세요. 이 상태로 commit하지 마세요.")
        sys.exit(1)


if __name__ == '__main__':
    main()
