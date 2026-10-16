"""
health_check.py — 하루 실행이 '조용히 실패'했는지 마지막에 점검한다 (v27).

왜 필요한가
- predict.yml 의 분석 단계는 모두 continue-on-error 라서, 스크립트가 전부 죽어도 워크플로우는 초록색으로 끝난다.
  GitHub 실패 메일도, 텔레그램 요약(그날 갱신 안 된 항목만 표시)도 '아무 기록이 안 쌓인 날'을 크게 알려주지 못한다.
- 그래서 마지막 단계에서 6개 로그 각각의 '가장 최근 가격일'과 그날 기록된 행 수를 확인하고,
  문제가 있으면 텔레그램으로 알리고 종료코드 1(=GitHub 실패 메일)로 끝낸다. 정상이면 조용히 0.

판정(보수적: 미국 공휴일에도 오경보가 나지 않게 4일 여유)
- 로그 파일이 없거나 비어 있음 → 문제
- 가장 최근 price_date 가 오늘로부터 4일 넘게 오래됨 → 문제(실행이 며칠째 안 쌓임)
- 가장 최근 price_date 의 행 수가 기대치보다 적음(레버리지 6, 개별주 3) → 문제(일부 종목만 기록)
예측 로그는 읽기만 한다. 모델·로그 스키마에 영향 없음.
"""
import os
import sys
from datetime import datetime

import pandas as pd

LOG_EXPECTED = {'scalp': 6, 'swing': 6, 'position': 6, 'stock_scalp': 3, 'stock_swing': 3, 'stock_position': 3}
MAX_LAG_DAYS = 4
OVERDUE_DAYS = 45


def check_health(log_dir="logs", today=None, expected=None, max_lag_days=MAX_LAG_DAYS):
    """문제 목록(문자열 리스트)을 돌려준다. 빈 리스트 = 정상. 어떤 입력에서도 예외를 던지지 않는다."""
    expected = expected or LOG_EXPECTED
    today = pd.Timestamp(today or datetime.now()).normalize()
    issues = []
    for name, n_exp in expected.items():
        path = os.path.join(log_dir, f"predictions_{name}.csv")
        try:
            if not os.path.exists(path):
                issues.append(f"{name}: 로그 파일 없음")
                continue
            df = pd.read_csv(path)
            if df.empty or 'ticker' not in df.columns:
                issues.append(f"{name}: 로그가 비어 있거나 형식이 깨짐")
                continue
            pdv = pd.to_datetime(df['price_date'], errors='coerce') if 'price_date' in df.columns else pd.Series(pd.NaT, index=df.index)
            if pdv.isna().all():       # 구 로그(price_date 없음): 실행일 사용
                pdv = pd.to_datetime(df['run_date'], errors='coerce')
            last = pdv.max()
            if pd.isna(last):
                issues.append(f"{name}: 날짜를 읽을 수 없음")
                continue
            lag = (today - last.normalize()).days
            if lag > max_lag_days:
                issues.append(f"{name}: 최근 기록이 {lag}일 전({last.date()}) — 실행이 며칠째 쌓이지 않음")
                continue
            _rd = pd.to_datetime(df['run_date'], errors='coerce').dt.date if 'run_date' in df.columns else None
            if _rd is not None and 'horizon_days' in df.columns:       # v28: 같은 날·종목·호라이즌 중복 행(멱등성 위반) 감시
                _dup = int(pd.DataFrame({'d': _rd, 't': df['ticker'].astype(str).str.strip().str.upper(), 'h': df['horizon_days']}).duplicated().sum())
                if _dup:
                    issues.append(f"{name}: 같은 날·종목·호라이즌 중복 행 {_dup}건 — 표본이 부풀려졌을 수 있음")
            if 'actual_label' in df.columns and 'target_date' in df.columns:      # v29: 상장폐지·시세 중단 등으로 정산이 무기한 밀리는 행
                _td = pd.to_datetime(df['target_date'], errors='coerce')
                _late = int((df['actual_label'].isna() & (_td < today - pd.Timedelta(days=OVERDUE_DAYS))).sum())
                if _late:
                    issues.append(f"{name}: 목표일이 {OVERDUE_DAYS}일 넘게 지났는데 정산 안 된 행 {_late}건 — 시세 중단/상장폐지/조회 실패 의심")
            n_last = int(df.loc[pdv.dt.normalize() == last.normalize(), ['ticker', 'horizon_days']].drop_duplicates().shape[0])
            if n_last < n_exp:
                issues.append(f"{name}: {last.date()} 기록 {n_last}건(기대 {n_exp}건) — 일부 종목/벤치마크 누락")
        except Exception as e:
            issues.append(f"{name}: 점검 중 오류({type(e).__name__})")
    return issues


def main():
    from common import send_telegram
    issues = check_health()
    if not issues:
        print("✅ 건강 점검 통과: 6개 로그 모두 최근 가격일 기록이 기대 건수 이상")
        return 0
    msg = "🩺 <b>실행 건강 점검: 문제 발견</b>\n" + "\n".join(f"• {i}" for i in issues) + \
          "\n(분석 단계는 오류가 나도 워크플로우가 초록색으로 끝나므로 이 점검이 대신 알립니다. Actions 로그를 확인하세요.)"
    print(msg)
    send_telegram(msg)
    return 1


if __name__ == "__main__":
    sys.exit(main())
