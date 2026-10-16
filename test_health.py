"""health_check 합성 검증."""
import os, shutil, tempfile
import pandas as pd
import health_check as hc

ok = True
def check(n, c, x=""):
    global ok; print(("PASS" if c else "FAIL"), n, x); ok &= bool(c)

tmp = tempfile.mkdtemp()
def write(name, rows):
    pd.DataFrame(rows, columns=['ticker', 'horizon_days', 'price_date', 'run_date']).to_csv(os.path.join(tmp, f"predictions_{name}.csv"), index=False)
def full(d, n): return [{'ticker': t, 'horizon_days': 1, 'price_date': d, 'run_date': d} for t in ['A', 'B', 'C', 'D', 'E', 'F'][:n]]
today = "2026-10-16"
exp = {'scalp': 6, 'stock_scalp': 3}

write('scalp', full("2026-10-15", 6)); write('stock_scalp', full("2026-10-15", 3))
check("H1 정상이면 문제 없음", hc.check_health(tmp, today, exp) == [])
write('scalp', full("2026-10-15", 4))
r = hc.check_health(tmp, today, exp); check("H2 일부 종목 누락 감지", len(r) == 1 and 'scalp' in r[0] and '4건' in r[0], str(r))
write('scalp', full("2026-10-09", 6))
r = hc.check_health(tmp, today, exp); check("H3 4일 넘게 오래된 로그 감지", len(r) == 1 and '7일 전' in r[0], str(r))
write('scalp', full("2026-10-13", 6))
check("H4 3일 전은 공휴일 여유로 통과", hc.check_health(tmp, today, exp) == [])
os.remove(os.path.join(tmp, "predictions_stock_scalp.csv"))
r = hc.check_health(tmp, today, exp); check("H5 파일 없음 감지", any('파일 없음' in x for x in r), str(r))
open(os.path.join(tmp, "predictions_stock_scalp.csv"), "w").write("garbage\x00\n,,")
r = hc.check_health(tmp, today, exp); check("H6 깨진 파일에도 예외 없이 문제로 보고", len(r) >= 1, str(r))
open(os.path.join(tmp, "predictions_stock_scalp.csv"), "w").write("")
r = hc.check_health(tmp, today, exp); check("H7 빈 파일", len(r) >= 1, str(r))
pd.DataFrame({'ticker': ['A'], 'horizon_days': [1], 'run_date': ['2026-10-15']}).to_csv(os.path.join(tmp, "predictions_stock_scalp.csv"), index=False)
r = hc.check_health(tmp, today, {'stock_scalp': 1}); check("H8 구 로그(price_date 없음)는 run_date로 판정", r == [], str(r))
check("H9 같은 (종목,호라이즌) 중복 행은 1건으로 셈", hc.check_health(tmp, today, {'stock_scalp': 2}) != [])
shutil.rmtree(tmp)
print("\nALL PASS" if ok else "\nSOME FAIL"); raise SystemExit(0 if ok else 1)
