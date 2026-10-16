"""
push_logs.py — 로그를 커밋하고 push 한다. push 가 거절돼도 '그날 기록'을 잃지 않는다 (v28).

왜 필요한가
- 기존 방식: `git pull --rebase || git rebase --abort` 뒤 `git push`.
  리베이스가 충돌하면 abort 후 push 가 거절되고, 러너(임시 컨테이너)가 사라지면서 그날 로그가 통째로 없어졌다.
  push 직전에 원격이 앞서가는 경쟁 상태에서도 같은 일이 일어난다.
- 방식: push 실패 → 원격 최신본을 받아 'CSV 를 행 단위로 합쳐서'(원격 행 + 우리 행) 다시 커밋·push. 최대 N회 재시도.
  끝내 실패하면 로그를 rescue_logs/ 로 복사하고 종료코드 1 (워크플로우가 artifact 로 보존).

합치기 규칙(merge_log_frames) — 예측/변동성 로그 공통
- 키 (실행일, 티커, 호라이즌) 또는 (종가기준일, 티커, 호라이즌)이 같으면 같은 예측으로 본다.
- 같은 예측이 양쪽에 있으면: 정산 끝난 행 > 안 끝난 행, 그다음엔 가장 이른 run_date(=그날 첫 실행).
  동률이면 정산본끼리는 이 실행(로컬)의 최신 채점, 미정산끼리는 원격 우선.
- 예측 모델·로직은 건드리지 않는다. 로그 파일의 '합치기'만 한다.

사용: python push_logs.py <브랜치> "<커밋 메시지>" [경로 ...]   (경로 기본: logs reports)
"""
import io
import os
import shutil
import subprocess
import sys
import time

import pandas as pd

MAX_TRIES = 5
LAST_MERGE_CONFLICTS = []   # 정산본끼리 값이 달랐던 행(로컬 채택) 기록 — 사람이 확인할 수 있게 로그에 남긴다


# ---------------------------------------------------------------- 합치기 (순수 함수, 네트워크·git 불필요)
def _eff_price_date(df):
    """종가 기준일: price_date 가 없으면 run_date 직전 영업일(common._fallback_price_date 와 동일 규칙)."""
    pdv = pd.to_datetime(df['price_date'], errors='coerce') if 'price_date' in df.columns else pd.Series(pd.NaT, index=df.index)
    rd = pd.to_datetime(df['run_date'], errors='coerce')
    miss = pdv.isna()
    if miss.any():
        pdv = pdv.copy()
        pdv[miss] = rd[miss].dt.normalize() - pd.offsets.BDay(1)
    return pdv.dt.normalize()


def _settled(df):
    for c in ('actual_label', 'settled'):
        if c in df.columns:
            v = df[c]
            if c == 'settled':
                return v.fillna(0).astype(str).str.lower().isin(['1', '1.0', 'true'])
            return v.notna()
    return pd.Series(False, index=df.index)


def merge_log_frames(remote, local):
    """
    원격본과 로컬본을 합친다. 원칙: '각 쪽에 원래 있던 행은 함부로 지우지 않는다'.
    - 로컬 행이 원격의 어떤 행과 같은 예측(키 일치)이면 1:1 로 짝지어, 더 나은 쪽 하나로 대체한다.
        정산본 > 미정산본 / 둘 다 정산본이면 로컬(이 실행의 최신 채점) / 둘 다 미정산이면 더 이른 run_date(동률은 원격).
    - 짝이 없는 로컬 행(원격에 같은 예측이 전혀 없음)은 새 행으로 추가한다.
    - 원격 안에 이미 있던 중복·이상 행은 건드리지 않는다(과거 기록의 표본 수를 몰래 바꾸지 않기 위해).
    - 어느 한쪽이 비어 있거나 형식이 달라도 죽지 않는다.
    """
    need = {'run_date', 'ticker'}
    if remote is None or len(remote) == 0:
        return local
    if local is None or len(local) == 0:
        return remote
    if not (need <= set(remote.columns) and need <= set(local.columns)):
        both = pd.concat([remote, local], ignore_index=True)
        return both.drop_duplicates().reset_index(drop=True)
    cols = list(remote.columns) + [c for c in local.columns if c not in remote.columns]
    R = remote.reindex(columns=cols).reset_index(drop=True)
    L = local.reindex(columns=cols).reset_index(drop=True)

    def keys(df):
        rd = pd.to_datetime(df['run_date'], errors='coerce')
        tk = df['ticker'].astype(str).str.strip().str.upper()
        hz = pd.to_numeric(df['horizon_days'], errors='coerce') if 'horizon_days' in df.columns else pd.Series(0.0, index=df.index)
        k1 = list(zip(rd.dt.normalize(), tk, hz))
        k2 = list(zip(_eff_price_date(df), tk, hz))
        return rd, k1, k2, _settled(df).values

    r_rd, r_k1, r_k2, r_set = keys(R)
    l_rd, l_k1, l_k2, l_set = keys(L)
    by1, by2 = {}, {}
    for i, (a, b) in enumerate(zip(r_k1, r_k2)):
        if pd.notna(r_rd.iloc[i]):
            by1.setdefault(a, []).append(i)
            by2.setdefault(b, []).append(i)
    used = set()
    out = {i: R.iloc[i] for i in range(len(R))}
    new_rows = []
    for j in range(len(L)):
        if pd.isna(l_rd.iloc[j]) or str(L.at[j, 'ticker']) in ('nan', 'None', ''):
            continue                                          # 키 없는 로컬 행은 로그에 둘 가치가 없다(load_log 도 제거함)
        cand = [i for i in by1.get(l_k1[j], []) if i not in used] or [i for i in by2.get(l_k2[j], []) if i not in used]
        if cand:
            i = cand[0]
            used.add(i)
            if l_set[j] and not r_set[i]:
                take = True
            elif l_set[j] and r_set[i]:
                take = True
                try:
                    diff = any(str(L.at[j, c]) != str(R.at[i, c]) for c in ('actual_label', 'correct') if c in cols)
                except Exception:
                    diff = False
                if diff:
                    LAST_MERGE_CONFLICTS.append(f"{L.at[j, 'ticker']} {str(L.at[j, 'run_date'])[:10]}")
            elif not l_set[j] and not r_set[i]:
                take = bool(l_rd.iloc[j] < r_rd.iloc[i])
            else:
                take = False
            if take:
                out[i] = L.iloc[j]
        elif l_k1[j] in by1 or l_k2[j] in by2:
            continue                                          # 원격에 같은 예측이 이미 있음(짝은 다른 로컬 행이 차지) → 중복이므로 버림
        else:
            new_rows.append(L.iloc[j])
    rows = [out[i] for i in range(len(R))] + new_rows
    merged = pd.DataFrame(rows).reset_index(drop=True)
    order = pd.to_datetime(merged['run_date'], errors='coerce')
    return merged.iloc[order.argsort(kind='stable')].reset_index(drop=True)[cols]


def merge_csv_text(remote_text, local_path):
    """원격 CSV 텍스트와 로컬 CSV 파일을 합쳐 DataFrame 을 돌려준다. 원격 텍스트가 깨졌으면 로컬만 쓴다."""
    local = pd.read_csv(local_path)
    try:
        remote = pd.read_csv(io.StringIO(remote_text)) if remote_text and remote_text.strip() else None
    except Exception:
        remote = None
    return merge_log_frames(remote, local)


# ---------------------------------------------------------------- git 보조
def _git(args, cwd, check=False):
    p = subprocess.run(['git'] + args, cwd=cwd, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 실패: {p.stderr.strip()}")
    return p


def _staged_any(cwd):
    return _git(['diff', '--staged', '--quiet'], cwd).returncode != 0


def _add(paths, cwd):
    for p in paths:
        if os.path.exists(os.path.join(cwd, p)):
            _git(['add', p], cwd)


def _rescue(paths, cwd):
    dst = os.path.join(cwd, 'rescue_logs')
    os.makedirs(dst, exist_ok=True)
    for p in paths:
        src = os.path.join(cwd, p)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(dst, p), dirs_exist_ok=True)
        elif os.path.exists(src):
            shutil.copy2(src, dst)
    return dst


def _csvs(paths, cwd):
    out = []
    for p in paths:
        full = os.path.join(cwd, p)
        if os.path.isdir(full):
            for f in sorted(os.listdir(full)):
                if f.endswith('.csv'):
                    out.append(os.path.join(p, f).replace(os.sep, '/'))
        elif p.endswith('.csv') and os.path.exists(full):
            out.append(p)
    return out


def sync_and_push(branch, message, paths=('logs', 'reports'), cwd='.', max_tries=MAX_TRIES,
                  remote='origin', sleep=lambda s: time.sleep(s), log=print):
    """0 = 성공(또는 올릴 변경 없음), 1 = 끝내 실패(로그는 rescue_logs/ 에 보존)."""
    paths = list(paths)
    _add(paths, cwd)
    if not _staged_any(cwd):
        log("변경된 로그 없음 — 커밋 스킵")
        return 0
    _git(['commit', '-m', message], cwd, check=True)

    for attempt in range(1, max_tries + 1):
        p = _git(['push', remote, f'HEAD:{branch}'], cwd)
        if p.returncode == 0:
            log(f"push 성공(시도 {attempt}회)")
            return 0
        log(f"⚠️ push 거절(시도 {attempt}/{max_tries}): {p.stderr.strip().splitlines()[-1] if p.stderr.strip() else '?'}")
        f = _git(['fetch', remote, branch], cwd)
        if f.returncode != 0:
            sleep(min(2 ** attempt, 20))
            continue
        # 우리 쪽 파일(작업 폴더)을 보존한 채, HEAD 만 원격 최신으로 옮긴다(=리베이스 충돌이 아예 생기지 않음).
        snap = {c: open(os.path.join(cwd, c), 'rb').read() for c in _csvs(paths, cwd) if os.path.exists(os.path.join(cwd, c))}
        if _git(['reset', '--mixed', f'{remote}/{branch}'], cwd).returncode != 0:
            sleep(min(2 ** attempt, 20))
            continue
        for rel, data in snap.items():
            tmp_local = os.path.join(cwd, rel)
            show = _git(['show', f'{remote}/{branch}:{rel}'], cwd)
            remote_text = show.stdout if show.returncode == 0 else None
            try:
                merged = merge_csv_text(remote_text, tmp_local)
                tmp = tmp_local + f".tmp_{os.getpid()}"
                merged.to_csv(tmp, index=False)
                os.replace(tmp, tmp_local)
            except Exception as e:          # 합치기 실패 → 우리 파일 그대로(원격 행은 git 히스토리에 남아 있음)
                log(f"⚠️ {rel} 합치기 실패({type(e).__name__}) — 로컬본 사용")
                with open(tmp_local, 'wb') as fh:
                    fh.write(data)
        if LAST_MERGE_CONFLICTS:
            log(f"ℹ️ 정산 결과가 서로 다른 행 {len(LAST_MERGE_CONFLICTS)}건은 로컬(최신 채점)을 채택: {', '.join(LAST_MERGE_CONFLICTS[:10])}")
            del LAST_MERGE_CONFLICTS[:]
        _add(paths, cwd)
        if not _staged_any(cwd):
            log("원격에 이미 같은 내용이 있음 — 올릴 변경 없음")
            return 0
        _git(['commit', '-m', message], cwd, check=True)
        sleep(min(attempt, 5))

    dst = _rescue(paths, cwd)
    log(f"❌ push 를 {max_tries}회 시도했지만 실패 — 로그를 {dst} 에 보존했습니다(워크플로우가 artifact 로 저장).")
    return 1


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    branch, message = sys.argv[1], sys.argv[2]
    paths = sys.argv[3:] or ['logs', 'reports']
    return sync_and_push(branch, message, paths)


if __name__ == '__main__':
    sys.exit(main())
