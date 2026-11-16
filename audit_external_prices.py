"""
audit_external_prices.py — 예측 로그의 가격·정산을 '다른 출처' 종가와 대조한다 (v34, 읽기 전용: 로그를 건드리지 않음)

왜 만들었나
- 로그의 기준가·정산가·적중 여부는 전부 yfinance 한 곳에서 나온다. 출처가 하나면 그 출처의 오류(수정주가 변경,
  날짜 밀림 등)를 로그 스스로는 잡아낼 수 없다. 다른 출처(예: Alpha Vantage 일별 종가)를 CSV로 넣어 대조한다.
- 방향 모델·기존 로그·검증 시계에는 영향이 없다.

점검 항목 (행 단위)
  1) 기준가: price_at_prediction 이 외부 종가(price_date)와 tol 이내로 같은가
  2) 정산가: actual_price 가 외부 종가(target_date)와 tol 이내로 같은가
  3) 적중 판정: 외부 종가로 다시 계산한 방향(target/base - 1 > 0)과 로그의 actual_label·correct 가 같은가
     (인버스 종목도 그 종목 자신의 가격으로 채점하므로 같은 식을 쓴다)

입력 CSV: ticker,date,close  (헤더 필수, 수정 전 종가도 가능 — 배당락은 tol 로 흡수, 분할은 '가격 불일치'로 드러남)
실행: python audit_external_prices.py 외부종가.csv [로그폴더=logs] [허용오차=0.01] [출력폴더=reports]
"""
import glob
import os
import sys

import pandas as pd

LOG_GLOB = "predictions_*.csv"


def load_external(path):
    """외부 종가 CSV → {(ticker, 'YYYY-MM-DD'): close}. 깨진 행은 건너뛴다."""
    out = {}
    try:
        ext = pd.read_csv(path)
    except Exception:
        return out
    need = {'ticker', 'date', 'close'}
    if not need.issubset(set(ext.columns)):
        return out
    for _, r in ext.iterrows():
        try:
            c = float(r['close'])
            if c > 0:
                out[(str(r['ticker']).strip().upper(), str(pd.Timestamp(r['date']).date()))] = c
        except Exception:
            continue
    return out


def _day(x):
    try:
        return str(pd.Timestamp(x).date())
    except Exception:
        return None


def audit_log(df, ext, tol=0.01):
    """한 로그(DataFrame)를 대조해 (요약 dict, 불일치 행 리스트)를 돌려준다. 외부 종가가 없는 행은 '대조 불가'로만 센다."""
    s = dict(rows=len(df), base_checked=0, base_bad=0, settle_checked=0, settle_bad=0,
             label_checked=0, label_bad=0, correct_bad=0, uncovered=0)
    bad = []
    for col in ('ticker', 'price_date', 'price_at_prediction', 'target_date'):
        if col not in df.columns:
            return s, bad                       # 옛 스키마(price_date 없음 등)는 대조 불가
    for i, r in df.iterrows():
        t = str(r['ticker']).strip().upper()
        pd_, td_ = _day(r['price_date']), _day(r['target_date'])
        eb = ext.get((t, pd_)) if pd_ else None
        et = ext.get((t, td_)) if td_ else None
        if eb is None:
            s['uncovered'] += 1
            continue
        try:
            lb = float(r['price_at_prediction'])
        except Exception:
            lb = float('nan')
        if lb == lb:
            s['base_checked'] += 1
            if abs(lb / eb - 1) > tol:
                s['base_bad'] += 1
                bad.append((i, t, pd_, 'base', lb, eb))
        settled = pd.notna(r.get('actual_price')) and pd.notna(r.get('correct'))
        if not settled or et is None:
            continue
        la = float(r['actual_price'])
        s['settle_checked'] += 1
        if abs(la / et - 1) > tol:
            s['settle_bad'] += 1
            bad.append((i, t, td_, 'settle', la, et))
        ext_ret = et / eb - 1
        if abs(ext_ret) <= tol / 2:             # 0 근처는 출처 간 반올림 차이로 부호가 뒤집힐 수 있어 판정 제외
            continue
        ext_label = 1.0 if ext_ret > 0 else 0.0
        s['label_checked'] += 1
        try:
            al, pl, cr = float(r['actual_label']), float(r['predicted_label']), float(r['correct'])
        except Exception:
            continue
        if al != ext_label:
            s['label_bad'] += 1
            bad.append((i, t, td_, 'label', al, ext_label))
        if cr != float(pl == ext_label):
            s['correct_bad'] += 1
            bad.append((i, t, td_, 'correct', cr, float(pl == ext_label)))
    return s, bad


def run(ext_path, log_dir="logs", tol=0.01, out_dir=None):
    ext = load_external(ext_path)
    lines = [f"외부 종가 대조 (허용오차 ±{tol * 100:.1f}%, 외부 종가 {len(ext)}건) — 읽기 전용, 로그 불변"]
    if not ext:
        lines.append("외부 종가 파일을 읽지 못했거나 비어 있습니다(형식: ticker,date,close). 대조를 건너뜁니다.")
    total_bad = 0
    for f in sorted(glob.glob(os.path.join(log_dir, LOG_GLOB))) if ext else []:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            lines.append(f"[{os.path.basename(f)}] 읽기 실패(건너뜀): {e}")
            continue
        s, bad = audit_log(df, ext, tol)
        total_bad += len(bad)
        lines.append(f"[{os.path.basename(f)}] 행 {s['rows']} · 기준가 {s['base_checked']}건 중 불일치 {s['base_bad']} · "
                     f"정산가 {s['settle_checked']}건 중 불일치 {s['settle_bad']} · "
                     f"방향 {s['label_checked']}건 중 불일치 {s['label_bad']} · 적중판정 불일치 {s['correct_bad']} · 대조불가 {s['uncovered']}")
        for b in bad[:12]:
            lines.append(f"    행{b[0]} {b[1]} {b[2]} [{b[3]}] 로그 {b[4]:.4f} vs 외부 {b[5]:.4f}")
        if len(bad) > 12:
            lines.append(f"    … 외 {len(bad) - 12}건")
    if ext:
        lines.append("결론: " + ("불일치 없음 — 로그의 가격·정산이 외부 출처와 일치합니다." if total_bad == 0
                               else f"불일치 {total_bad}건 — 위 목록을 확인하세요(분할·배당락·날짜 밀림 가능성)."))
    text = "\n".join(lines) + "\n"
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "audit_external.txt"), "w", encoding="utf-8") as fh:
            fh.write(text)
    return text


if __name__ == "__main__":
    a = sys.argv
    if len(a) < 2:
        print(__doc__)
        sys.exit(0)
    print(run(a[1], a[2] if len(a) > 2 else "logs", float(a[3]) if len(a) > 3 else 0.01, a[4] if len(a) > 4 else None))
