"""push_logs.py 검증 — 진짜 git(bare 원격 + 클론 2개)으로 push 경쟁·충돌·실패를 재현한다. 네트워크 불필요."""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

import push_logs as P

COLS = ['run_date', 'benchmark', 'ticker', 'horizon_days', 'predicted_label', 'predicted_up_prob', 'raw_up_prob',
        'price_at_prediction', 'target_date', 'actual_price', 'actual_return', 'actual_label', 'correct',
        'near_earnings', 'price_date']
results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(("OK   " if cond else "FAIL ") + name)


def sh(cwd, *a):
    return subprocess.run(['git'] + list(a), cwd=cwd, capture_output=True, text=True, check=True).stdout


def row(day, tk, h=1, settled=False, pdate=None):
    d = pd.Timestamp(day)
    return {'run_date': d + pd.Timedelta(hours=9, minutes=17), 'benchmark': 'SMH', 'ticker': tk, 'horizon_days': h,
            'predicted_label': 1, 'predicted_up_prob': 60.0, 'raw_up_prob': 60.0, 'price_at_prediction': 10.0,
            'target_date': d + pd.offsets.BDay(h), 'actual_price': 11.0 if settled else np.nan,
            'actual_return': 0.1 if settled else np.nan, 'actual_label': 1.0 if settled else np.nan,
            'correct': 1.0 if settled else np.nan, 'near_earnings': False,
            'price_date': pd.Timestamp(pdate) if pdate else d.normalize() - pd.offsets.BDay(1)}


def write_log(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows, columns=COLS).to_csv(path, index=False)


def setup():
    root = tempfile.mkdtemp()
    bare = os.path.join(root, 'remote.git')
    subprocess.run(['git', 'init', '-q', '--bare', '-b', 'DiE', bare], check=True)
    a, b = os.path.join(root, 'a'), os.path.join(root, 'b')
    subprocess.run(['git', 'clone', '-q', bare, a], check=True, capture_output=True)
    for c in (a,):
        sh(c, 'config', 'user.name', 't'); sh(c, 'config', 'user.email', 't@t')
        sh(c, 'checkout', '-q', '-b', 'DiE')
    write_log(os.path.join(a, 'logs/predictions_x.csv'), [row('2026-10-12', 'SOXL'), row('2026-10-13', 'SOXL')])
    open(os.path.join(a, 'code.py'), 'w').write('v1\n')
    sh(a, 'add', '.'); sh(a, 'commit', '-qm', 'init'); sh(a, 'push', '-q', 'origin', 'DiE')
    subprocess.run(['git', 'clone', '-q', '-b', 'DiE', bare, b], check=True, capture_output=True)
    sh(b, 'config', 'user.name', 'u'); sh(b, 'config', 'user.email', 'u@t')
    return root, bare, a, b


def remote_log(bare, root):
    chk = os.path.join(root, 'chk')
    shutil.rmtree(chk, ignore_errors=True)
    subprocess.run(['git', 'clone', '-q', '-b', 'DiE', bare, chk], check=True, capture_output=True)
    return pd.read_csv(os.path.join(chk, 'logs/predictions_x.csv')), chk


quiet = lambda *_: None
nosleep = lambda s: None

# A. 충돌 없음
root, bare, a, b = setup()
write_log(os.path.join(a, 'logs/predictions_x.csv'), [row('2026-10-12', 'SOXL'), row('2026-10-13', 'SOXL'), row('2026-10-14', 'SOXL')])
rc = P.sync_and_push('DiE', 'm', cwd=a, log=quiet, sleep=nosleep)
r, _ = remote_log(bare, root)
check('A 평상시 push', rc == 0 and len(r) == 3)

# B. 변경 없음
rc = P.sync_and_push('DiE', 'm', cwd=a, log=quiet, sleep=nosleep)
check('B 변경 없으면 아무 일 없음(rc 0)', rc == 0)

# C. 원격이 로그를 건드리지 않고 앞서감
root, bare, a, b = setup()
open(os.path.join(b, 'code.py'), 'w').write('v2\n'); sh(b, 'commit', '-qam', 'code'); sh(b, 'push', '-q', 'origin', 'DiE')
write_log(os.path.join(a, 'logs/predictions_x.csv'), [row('2026-10-12', 'SOXL'), row('2026-10-13', 'SOXL'), row('2026-10-14', 'SOXL')])
rc = P.sync_and_push('DiE', 'm', cwd=a, log=quiet, sleep=nosleep)
r, chk = remote_log(bare, root)
check('C 원격 코드 커밋만 앞섬 → 우리 로그 + 원격 코드 모두 보존', rc == 0 and len(r) == 3 and open(os.path.join(chk, 'code.py')).read() == 'v2\n')

# D. 원격이 같은 로그를 수정(다른 행 추가) → 예전 방식이면 충돌로 유실
root, bare, a, b = setup()
write_log(os.path.join(b, 'logs/predictions_x.csv'), [row('2026-10-12', 'SOXL'), row('2026-10-13', 'SOXL'), row('2026-10-14', 'SOXS')])
sh(b, 'commit', '-qam', 'other writer'); sh(b, 'push', '-q', 'origin', 'DiE')
write_log(os.path.join(a, 'logs/predictions_x.csv'), [row('2026-10-12', 'SOXL'), row('2026-10-13', 'SOXL'), row('2026-10-14', 'SOXL')])
rc = P.sync_and_push('DiE', 'm', cwd=a, log=quiet, sleep=nosleep)
r, _ = remote_log(bare, root)
check('D 같은 로그 동시 수정 → 두 쪽 행 모두 보존(4행)', rc == 0 and len(r) == 4 and set(r.ticker) == {'SOXL', 'SOXS'})
# (구 방식이 실제로 유실하는지 대조)
root, bare, a, b = setup()
write_log(os.path.join(b, 'logs/predictions_x.csv'), [row('2026-10-12', 'SOXL'), row('2026-10-13', 'SOXL'), row('2026-10-14', 'SOXS')])
sh(b, 'commit', '-qam', 'other'); sh(b, 'push', '-q', 'origin', 'DiE')
write_log(os.path.join(a, 'logs/predictions_x.csv'), [row('2026-10-12', 'SOXL'), row('2026-10-13', 'SOXL'), row('2026-10-14', 'SOXL')])
sh(a, 'add', 'logs'); sh(a, 'commit', '-qm', 'old way')
pr = subprocess.run('git pull --rebase origin DiE || git rebase --abort; git push', shell=True, cwd=a, capture_output=True, text=True)
check('D0 (대조) 구 방식은 충돌 시 push 실패 = 유실 재현', pr.returncode != 0)

# E. 원격엔 정산 완료, 우리는 미정산 같은 행 → 정산본 유지
root, bare, a, b = setup()
write_log(os.path.join(b, 'logs/predictions_x.csv'), [row('2026-10-12', 'SOXL', settled=True), row('2026-10-13', 'SOXL')])
sh(b, 'commit', '-qam', 'resettle'); sh(b, 'push', '-q', 'origin', 'DiE')
write_log(os.path.join(a, 'logs/predictions_x.csv'), [row('2026-10-12', 'SOXL'), row('2026-10-13', 'SOXL'), row('2026-10-14', 'SOXL')])
rc = P.sync_and_push('DiE', 'm', cwd=a, log=quiet, sleep=nosleep)
r, _ = remote_log(bare, root)
first = r[r.run_date.str.startswith('2026-10-12')]
check('E 정산 완료 행이 미정산본에 덮이지 않음', rc == 0 and len(r) == 3 and first.actual_label.notna().all())

# F. 반대: 우리는 정산 완료, 원격은 미정산 → 정산본 채택
root, bare, a, b = setup()
write_log(os.path.join(b, 'logs/predictions_x.csv'), [row('2026-10-12', 'SOXL'), row('2026-10-13', 'SOXL'), row('2026-10-14', 'SOXS')])
sh(b, 'commit', '-qam', 'o'); sh(b, 'push', '-q', 'origin', 'DiE')
write_log(os.path.join(a, 'logs/predictions_x.csv'), [row('2026-10-12', 'SOXL', settled=True), row('2026-10-13', 'SOXL'), row('2026-10-14', 'SOXL')])
rc = P.sync_and_push('DiE', 'm', cwd=a, log=quiet, sleep=nosleep)
r, _ = remote_log(bare, root)
check('F 우리 정산본 채택 + 원격 신규행 보존', rc == 0 and len(r) == 4 and r[r.run_date.str.startswith('2026-10-12')].actual_label.notna().all())

# G. 재시도 중에도 원격이 계속 앞서감(경쟁) — 3번 연속 새치기
root, bare, a, b = setup()
write_log(os.path.join(a, 'logs/predictions_x.csv'), [row('2026-10-12', 'SOXL'), row('2026-10-13', 'SOXL'), row('2026-10-14', 'SOXL')])
cnt = {'n': 0}
write_log(os.path.join(b, 'logs/predictions_x.csv'), [row('2026-10-12', 'SOXL'), row('2026-10-13', 'SOXL'), row('2026-10-14', 'T0')])
sh(b, 'commit', '-qam', 'race0'); sh(b, 'push', '-q', 'origin', 'DiE')


def racing_sleep(s):
    cnt['n'] += 1
    if cnt['n'] <= 3:
        sh(b, 'pull', '-q', '--rebase', 'origin', 'DiE')
        df = pd.read_csv(os.path.join(b, 'logs/predictions_x.csv'))
        df = pd.concat([df, pd.DataFrame([row('2026-10-14', f'T{cnt["n"]}')]).assign(run_date=lambda d: d.run_date.astype(str), target_date=lambda d: d.target_date.astype(str), price_date=lambda d: d.price_date.astype(str))], ignore_index=True)
        df.to_csv(os.path.join(b, 'logs/predictions_x.csv'), index=False)
        sh(b, 'commit', '-qam', f'race{cnt["n"]}'); sh(b, 'push', '-q', 'origin', 'DiE')


rc = P.sync_and_push('DiE', 'm', cwd=a, log=quiet, sleep=racing_sleep)
r, _ = remote_log(bare, root)
check('G 원격이 연달아 앞서가도 결국 push, 모든 행 보존', rc == 0 and {'SOXL', 'T0', 'T1', 'T2', 'T3'} <= set(r.ticker))

# H. 원격 CSV 가 깨져 있음 → 로컬본 사용, 죽지 않음
root, bare, a, b = setup()
open(os.path.join(b, 'logs/predictions_x.csv'), 'w').write('\x00\x01garbage,,\n"unterminated\n')
sh(b, 'commit', '-qam', 'corrupt'); sh(b, 'push', '-q', 'origin', 'DiE')
write_log(os.path.join(a, 'logs/predictions_x.csv'), [row('2026-10-12', 'SOXL'), row('2026-10-13', 'SOXL'), row('2026-10-14', 'SOXL')])
rc = P.sync_and_push('DiE', 'm', cwd=a, log=quiet, sleep=nosleep)
r, _ = remote_log(bare, root)
check('H 원격 CSV 가 깨져도 우리 로그를 살려 push', rc == 0 and len(r) == 3)

# I. push 가 끝내 불가능(원격 사라짐) → rc 1, 로그는 rescue_logs 에 보존
root, bare, a, b = setup()
write_log(os.path.join(a, 'logs/predictions_x.csv'), [row('2026-10-12', 'SOXL'), row('2026-10-14', 'SOXL')])
shutil.rmtree(bare)
rc = P.sync_and_push('DiE', 'm', cwd=a, log=quiet, sleep=nosleep, max_tries=3)
saved = os.path.join(a, 'rescue_logs', 'logs', 'predictions_x.csv')
check('I 원격 불가 → rc=1 이고 rescue_logs 에 그날 로그 보존', rc == 1 and os.path.exists(saved) and len(pd.read_csv(saved)) == 2)

# J. 변동성 로그 형태(actual_label 없고 settled 있음) 합치기
rem = pd.DataFrame({'run_date': ['2026-10-14'], 'ticker': ['SOXL'], 'price_date': ['2026-10-13'], 'horizon_days': [5], 'settled': [1]})
loc = pd.DataFrame({'run_date': ['2026-10-14', '2026-10-15'], 'ticker': ['SOXL', 'SOXL'], 'price_date': ['2026-10-13', '2026-10-14'], 'horizon_days': [5, 5], 'settled': [0, 0]})
m = P.merge_log_frames(rem, loc)
check('J 변동성 로그: 같은 키 1건(정산본) + 신규 1건', len(m) == 2 and int(m.iloc[0]['settled']) == 1)

# K. 합치기 멱등성·행 순서·열 보존
a_ = pd.DataFrame([row('2026-10-12', 'SOXL'), row('2026-10-13', 'SOXL')])
m1 = P.merge_log_frames(a_, a_)
m2 = P.merge_log_frames(m1, a_)
check('K 같은 로그끼리 합쳐도 행 수 불변·재합성도 동일', len(m1) == 2 and len(m2) == 2 and list(m1.columns) == list(a_.columns))

# L. 토요일/월요일 같은 종가 기준 중복은 이른 쪽만
sat = row('2026-10-17', 'SOXL', pdate='2026-10-16'); mon = row('2026-10-19', 'SOXL', pdate='2026-10-16')
m = P.merge_log_frames(pd.DataFrame([sat]), pd.DataFrame([mon]))
check('L 같은 종가기준 중복 → 이른 실행 1건', len(m) == 1 and str(m.iloc[0]['run_date']).startswith('2026-10-17'))

# M. 빈/None/열 누락 입력
check('M1 원격 비어있음', len(P.merge_log_frames(pd.DataFrame(), a_)) == 2)
check('M2 원격 None', len(P.merge_log_frames(None, a_)) == 2)
check('M3 필수 열 없는 원격(깨진 형식)', len(P.merge_log_frames(pd.DataFrame({'x': [1]}), a_)) >= 2)

# N. 양쪽 모두 정산 완료인데 값이 다름 → 이 실행(로컬)의 최신 채점이 이김
old = pd.DataFrame([row('2026-10-12', 'SOXL', settled=True)]); new = old.copy(); new['actual_return'] = 0.5; new['correct'] = 0.0
m = P.merge_log_frames(old, new)
check('N 정산본끼리 충돌 → 로컬 최신 채점 채택', len(m) == 1 and float(m.iloc[0]['actual_return']) == 0.5)

# O. 실제 DiE 로그로 멱등성: 자기 자신과 합쳐도 값 불변
real = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'predictions_scalp.csv')
if os.path.exists(real):
    r0 = pd.read_csv(real)
    mm = P.merge_log_frames(r0, r0)
    exp = r0.iloc[pd.to_datetime(r0['run_date']).argsort(kind='stable')].reset_index(drop=True)
    same = len(mm) == len(r0) and np.allclose(pd.to_numeric(mm['predicted_up_prob']).values, pd.to_numeric(exp['predicted_up_prob']).values) \
        and (mm['ticker'].values == exp['ticker'].values).all()
    check('O 실제 scalp 로그 자기합성 시 행 수·확률 불변', same)

bad = [n for n, ok in results if not ok]
print(f"\n{len(results) - len(bad)}/{len(results)} 통과")
sys.exit(1 if bad else 0)
