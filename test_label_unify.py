"""v31 라벨 통일 검증: (1) 'deadzone' 은 v30 동작과 비트 단위로 동일 (2) 'all' 은 V1 변형과 동일 (3) 라벨 누수/대칭/방어."""
import sys, numpy as np, pandas as pd
import common as cm, backtest_variants as V, audit_math as A
res = []
def check(n, c): res.append(bool(c)); print(("OK   " if c else "FAIL ") + n)
df, idx = A.make(drift=0.0005, seed=4, n=1500); m = A.macro(idx)
real = cm.fetch_ticker_data; cm.fetch_ticker_data = lambda t, **k: df.copy()
kw = dict(min_train_rows=300, model_params={'n_estimators': 60, 'max_depth': 3, 'learning_rate': .1})
check("기본 LABEL_MODE 는 'all'", cm.LABEL_MODE == 'all')
for h, cap in ((1, .04), (5, .08), (20, .15)):
    cm.LABEL_MODE = 'deadzone'; old = cm.train_benchmark_model('QQQ', m, h, cap, **kw); ref = V.train_variant('deadzone', 'iso', 'QQQ', m, h, cap, **kw)
    check(f"h{h}: deadzone 모드 = v30 동작(확률·기준선 동일)", abs(old['up_prob'] - ref['up_prob']) < 1e-9 and abs(old['base_rate'] - ref['base_rate']) < 1e-9)
    cm.LABEL_MODE = 'all'; new = cm.train_benchmark_model('QQQ', m, h, cap, **kw); v1 = V.train_variant('all', 'iso', 'QQQ', m, h, cap, **kw)
    check(f"h{h}: all 모드 = V1 변형(백테스트에서 검증된 것과 동일)", abs(new['up_prob'] - v1['up_prob']) < 1e-9 and abs(new['base_rate'] - v1['base_rate']) < 1e-9)
    check(f"h{h}: 상승+하락=100%, 확률 1~99%, 값 유한", abs(new['up_prob'] + new['down_prob'] - 100) < 1e-9 and 1 <= new['up_prob'] <= 99 and np.isfinite(new['edge']))
# 라벨 정의 자체
d = A.prepared(df, idx, 5, .08); cm.LABEL_MODE = 'all'; d = cm.apply_label_mode(d)
check("마지막 h행(미래 수익률 미확정)은 라벨 NaN — 미래 누수 없음", d['Target'].iloc[-5:].isna().all() and d['Target'].iloc[:-5].notna().all())
check("라벨 = (h일 수익률>0)", (d['Target'].dropna() == (d['Target_Ret'].dropna() > 0).astype(float)).all())
tr = d['Target'].dropna(); check("기준선(라벨 평균) = ret>0 비율", abs(tr.mean() - (d['Target_Ret'].dropna() > 0).mean()) < 1e-12)
# 방어: Target_Ret 없음/상수 가격 → 죽지 않음
check("Target_Ret 없는 프레임은 그대로 통과", cm.apply_label_mode(pd.DataFrame({'a': [1, 2]})).shape == (2, 1))
flat = pd.DataFrame({'High': 101., 'Low': 99., 'Close': 100.}, index=idx); cm.fetch_ticker_data = lambda t, **k: flat.copy()
check("상수 가격(라벨 한 클래스)은 None 으로 스킵", cm.train_benchmark_model('QQQ', m, 5, .08, **kw) is None)
cm.fetch_ticker_data = real
print(f"{sum(res)}/{len(res)} 통과"); sys.exit(0 if all(res) else 1)
