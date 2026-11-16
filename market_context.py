"""
market_context.py — 시장 맥락 수집 (v35): CNN 공포탐욕지수 · Alpha Vantage 교차확인 · 엣지/맥락 로그

원칙 (검증 시계 영향 없음)
- 여기서 모으는 값은 방향 모델의 입력이 아니다. 피처·임계값·학습·예측 로그(predictions_*.csv)는 건드리지 않는다.
- 용도는 두 가지뿐이다: (1) 텔레그램 요약에 '지금 상태'로 표시, (2) logs/ 에 날짜별로 쌓아 두었다가
  나중에 "이 정보가 예측에 도움이 되는가"를 워크포워드로 검증(쌓이지 않은 데이터는 검증도 못 한다).
- 모든 외부 호출은 실패·빈 응답·형식 변경·키 없음에서 조용히 None/{} 을 돌려준다. 요약 리포트는 계속 나간다.

데이터
- CNN 공포탐욕지수: https://production.dataviz.cnn.io/index/fearandgreed/graphdata[/YYYY-MM-DD]
  (CNN 이 페이지 그래프에 쓰는 공개 JSON. 공식 API 가 아니라 예고 없이 막히거나 바뀔 수 있음)
- Alpha Vantage: 환경변수 ALPHAVANTAGE_API_KEY 가 있을 때만 호출. 무료 키는 하루 25회·초당 1회라 호출 수를 세어 제한한다.
    GLOBAL_QUOTE      : 종가·거래일 교차확인(야후 시세가 갱신 안 된 날을 잡는다)
    EARNINGS_CALENDAR : 다음 실적일(수동 달력이 지났을 때 알려 준다)
    풋/콜 비율         : 맥락 로그 기록용(함수명은 AV_PCR_FUNCTION, 응답 형식이 다르면 조용히 생략)

로그 (둘 다 push_logs 의 행 단위 합치기와 호환되도록 run_date·ticker·horizon_days 열을 둔다)
- logs/edge_history.csv  : 그날 모델이 낸 기준선(base_rate)·엣지. 예측 로그엔 이 값이 없어 '엣지가 맞는가'를 검증할 수 없었다.
- logs/context_daily.csv : 가격일별 공포탐욕지수(구성요소 포함)·풋콜비율·교차확인 결과.
"""
import io
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

FG_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
FG_HISTORY_START = "2021-01-04"          # 이보다 이른 날짜는 CNN 이 500 을 돌려준다(2026-11 확인)
FG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://edition.cnn.com/markets/fear-and-greed",
    "Origin": "https://edition.cnn.com",
}
FG_BUCKETS = [(0, 25, "극단적 공포"), (25, 45, "공포"), (45, 55, "중립"), (55, 75, "탐욕"), (75, 100.0001, "극단적 탐욕")]
FG_COMPONENTS = {
    "market_momentum_sp500": "모멘텀", "stock_price_strength": "신고가·신저가", "stock_price_breadth": "거래량 폭",
    "put_call_options": "풋콜", "market_volatility_vix": "VIX", "junk_bond_demand": "정크본드", "safe_haven_demand": "안전자산",
}
EDGE_LOG = "edge_history.csv"
CONTEXT_LOG = "context_daily.csv"
EDGE_COLUMNS = ['run_date', 'ticker', 'horizon_days', 'price_date', 'log', 'benchmark', 'direction',
                'up_prob', 'raw_up_prob', 'base_rate', 'edge', 'logged_at']
# run_date 열에는 '가격일'을 넣는다: push_logs 가 (run_date 날짜, 종목, 호라이즌)을 같은 예측의 키로 쓰기 때문에,
# 실제 실행 시각을 넣으면 같은 날 두 번 돌린 서로 다른 가격일 행이 하나로 합쳐진다. 실제 기록 시각은 logged_at.

AV_URL = "https://www.alphavantage.co/query"
AV_KEY_ENV = "ALPHAVANTAGE_API_KEY"
AV_DAILY_BUDGET = 20                     # 무료 한도 25회보다 여유
AV_MIN_INTERVAL = 1.3                    # 초당 1회 제한
AV_PCR_FUNCTION = "HISTORICAL_PUT_CALL_RATIO"
_AV_STATE = {"calls": 0, "last": 0.0, "blocked": False}


# ------------------------------------------------------------------ 공통
def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def fg_bucket(score):
    s = _num(score)
    if s is None:
        return None
    for lo, hi, name in FG_BUCKETS:
        if lo <= s < hi:
            return name
    return None


def _atomic_csv(df, path):
    tmp = f"{path}.tmp_{os.getpid()}"
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _read_csv(path, columns=None):
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception as e:
        print(f"⚠️ {os.path.basename(path)} 읽기 실패(새로 시작, 원본은 .corrupt 로 보존): {e}")
        try:
            os.replace(path, f"{path}.corrupt_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        except Exception:
            pass
    return pd.DataFrame(columns=columns or [])


# ------------------------------------------------------------------ 1) CNN 공포탐욕지수
def parse_fear_greed(payload):
    """CNN JSON(dict) → {'score','rating','asof','prev_close','prev_week','prev_month','prev_year','components','history'} 또는 None.
    history: DataFrame(index=날짜, 열 score[, sp500]). 형식이 조금 달라도 읽을 수 있는 만큼만 읽는다."""
    try:
        if not isinstance(payload, dict):
            return None
        fg = payload.get("fear_and_greed") or {}
        score = _num(fg.get("score"))
        if score is None or not (0 <= score <= 100):
            return None
        out = {"score": score, "rating": str(fg.get("rating") or ""), "bucket": fg_bucket(score),
               "prev_close": _num(fg.get("previous_close")), "prev_week": _num(fg.get("previous_1_week")),
               "prev_month": _num(fg.get("previous_1_month")), "prev_year": _num(fg.get("previous_1_year")),
               "components": {}, "history": None, "asof": None}
        try:
            ts = pd.Timestamp(fg.get("timestamp"))
            out["asof"] = (ts.tz_convert("America/New_York") if ts.tzinfo is not None else ts).tz_localize(None).normalize()
        except Exception:
            out["asof"] = None
        for key in FG_COMPONENTS:
            c = payload.get(key)
            if isinstance(c, dict) and _num(c.get("score")) is not None:
                out["components"][key] = {"score": _num(c.get("score")), "rating": str(c.get("rating") or "")}

        def series(key):
            rows = ((payload.get(key) or {}).get("data")) or []
            idx, val = [], []
            for p in rows:
                x, y = _num(p.get("x")) if isinstance(p, dict) else None, _num(p.get("y")) if isinstance(p, dict) else None
                if x is None or y is None:
                    continue
                idx.append(pd.Timestamp(int(x), unit="ms").normalize())
                val.append(y)
            s = pd.Series(val, index=pd.DatetimeIndex(idx), dtype=float)
            return s[~s.index.duplicated(keep="last")].sort_index()

        hs = series("fear_and_greed_historical")
        if len(hs):
            hist = pd.DataFrame({"score": hs})
            sp = series("market_momentum_sp500")
            if len(sp):
                hist["sp500"] = sp.reindex(hist.index)
            out["history"] = hist
            if out["asof"] is None:
                out["asof"] = hist.index[-1]
        return out
    except Exception as e:
        print(f"⚠️ 공포탐욕지수 해석 실패(무시): {type(e).__name__}: {e}")
        return None


def fetch_fear_greed(getter=None, start=FG_HISTORY_START, timeout=12):
    """CNN 에서 받아 parse_fear_greed 결과를 돌려준다. 긴 이력 요청이 실패하면 기본(약 1년) 주소로 한 번 더. 전부 실패하면 None.
    getter(url) -> dict 를 넘기면 네트워크 대신 그것을 쓴다(테스트용)."""
    urls = ([f"{FG_URL}/{start}"] if start else []) + [FG_URL]
    for url in urls:
        try:
            if getter is not None:
                payload = getter(url)
            else:
                import requests
                r = requests.get(url, headers=FG_HEADERS, timeout=timeout)
                if getattr(r, "status_code", 200) != 200:
                    print(f"⚠️ 공포탐욕지수 응답 {r.status_code} ({url[-22:]})")
                    continue
                payload = r.json()
            out = parse_fear_greed(payload)
            if out is not None:
                return out
        except Exception as e:
            print(f"⚠️ 공포탐욕지수 조회 실패({type(e).__name__}) — 다음 주소 시도/생략")
    return None


def fg_forward_stats(history, score, horizons=(5, 20), min_days=250, min_episodes=5):
    """
    '지금과 같은 구간이었던 과거의 날들 뒤에 S&P500 이 어땠나'를 센다(설명용 통계, 예측이 아님).
    반환 {h: {'n','episodes','mean','up','vol','all_mean','all_up','all_vol'}} — 이력이 짧거나 독립 구간이 적으면 그 h 는 뺀다.
    episodes: 같은 구간에 연속으로 머문 날을 한 덩어리로 센 수(겹치는 표본을 독립 표본으로 착각하지 않기 위해).
    """
    out = {}
    try:
        if history is None or "sp500" not in history.columns:
            return out
        h_df = history.dropna(subset=["score", "sp500"])
        h_df = h_df[h_df["sp500"] > 0]
        if len(h_df) < min_days:
            return out
        b = fg_bucket(score)
        if b is None:
            return out
        lo, hi = next((lo, hi) for lo, hi, name in FG_BUCKETS if name == b)
        sp = h_df["sp500"].values
        sc = h_df["score"].values
        r = np.diff(np.log(sp), prepend=np.nan)
        n = len(sp)
        for h in horizons:
            fwd = np.full(n, np.nan)
            fvol = np.full(n, np.nan)
            for i in range(n - h):
                fwd[i] = sp[i + h] / sp[i] - 1
                fvol[i] = np.sqrt(np.mean(r[i + 1:i + h + 1] ** 2) * 252)
            ok = np.isfinite(fwd)
            sel = ok & (sc >= lo) & (sc < hi)
            idx = np.where(sel)[0]
            if len(idx) == 0:
                continue
            episodes = 1 + int(np.sum(np.diff(idx) > h))
            if episodes < min_episodes:
                continue
            out[h] = {"n": int(len(idx)), "episodes": episodes,
                      "mean": float(fwd[idx].mean()), "up": float((fwd[idx] > 0).mean()), "vol": float(fvol[idx].mean()),
                      "all_mean": float(fwd[ok].mean()), "all_up": float((fwd[ok] > 0).mean()), "all_vol": float(fvol[ok].mean()),
                      "since": str(h_df.index[0].date())}
    except Exception as e:
        print(f"⚠️ 공포탐욕 과거 통계 계산 실패(무시): {e}")
        return {}
    return out


# ------------------------------------------------------------------ 2) Alpha Vantage
def av_available():
    return bool(os.environ.get(AV_KEY_ENV)) and not _AV_STATE["blocked"]


def av_get(function, getter=None, sleeper=time.sleep, as_text=False, **params):
    """Alpha Vantage 한 번 호출. 키 없음·예산 초과·한도 안내문('Information'/'Note')·오류면 None.
    한도 안내문을 받으면 이번 실행에서는 더 부르지 않는다(남은 호출을 낭비하지 않기 위해)."""
    key = os.environ.get(AV_KEY_ENV)
    if not key or _AV_STATE["blocked"] or _AV_STATE["calls"] >= AV_DAILY_BUDGET:
        return None
    try:
        wait = AV_MIN_INTERVAL - (time.time() - _AV_STATE["last"])
        if wait > 0:
            sleeper(wait)
        _AV_STATE["calls"] += 1
        _AV_STATE["last"] = time.time()
        q = dict(function=function, apikey=key, **params)
        if getter is not None:
            text = getter(q)
        else:
            import requests
            r = requests.get(AV_URL, params=q, timeout=15)
            if getattr(r, "status_code", 200) != 200:
                return None
            text = r.text
        if text is None:
            return None
        head = str(text).lstrip()[:1]
        if head == "{":
            data = json.loads(text)
            if isinstance(data, dict) and any(k in data for k in ("Information", "Note", "Error Message", "error")):
                msg = str(data.get("Information") or data.get("Note") or data.get("Error Message") or data.get("error"))
                if any(w in msg.lower() for w in ("rate limit", "premium", "per day", "requests per")):
                    _AV_STATE["blocked"] = True
                print(f"ℹ️ Alpha Vantage [{function}] 안내/오류 응답 — 생략: {msg[:90]}")
                return None
            return text if as_text else data
        return text if as_text else None
    except Exception as e:
        print(f"⚠️ Alpha Vantage [{function}] 호출 실패(무시): {type(e).__name__}")
        return None


def av_quotes(tickers, getter=None, sleeper=time.sleep):
    """{티커: {'price','day','prev_close'}} — 실패한 티커는 빠진다."""
    out = {}
    for t in tickers:
        d = av_get("GLOBAL_QUOTE", getter=getter, sleeper=sleeper, symbol=t)
        try:
            q = (d or {}).get("Global Quote") or {}
            price, day = _num(q.get("05. price")), q.get("07. latest trading day")
            if price and price > 0 and day:
                out[t] = {"price": price, "day": pd.Timestamp(day).normalize(), "prev_close": _num(q.get("08. previous close"))}
        except Exception:
            continue
    return out


def crosscheck_prices(items, quotes, tol=0.01):
    """
    요약 항목(items: [{'ticker','price','price_date'}])을 Alpha Vantage 종가와 대조해 경고 문자열 리스트를 돌려준다.
    - 우리 가격일이 AV 최근 거래일보다 이르면: 야후 시세가 갱신되지 않은 채 예측한 것(2026-09-04·09-23 실제 발생).
    - 같은 날인데 가격이 tol 넘게 다르면: 장중가·분할·오류 의심(배당락 수준 차이는 tol 로 흡수).
    - 우리 가격일이 AV 보다 늦으면 AV 쪽이 늦은 것이므로 경고하지 않는다.
    """
    warns, checked = [], 0
    seen = set()
    for it in items or []:
        try:
            t = str(it.get("ticker")).strip().upper()
            if t in seen or t not in quotes:
                continue
            seen.add(t)
            ours_p, ours_d = _num(it.get("price")), pd.Timestamp(it.get("price_date")).normalize()
            q = quotes[t]
            if ours_p is None or pd.isna(ours_d):
                continue
            checked += 1
            if ours_d < q["day"]:
                warns.append(f"{t}: 우리 가격일 {ours_d.month}/{ours_d.day} < 다른 출처 최근 거래일 {q['day'].month}/{q['day'].day} — 시세 미갱신 상태의 예측일 수 있음")
            elif ours_d == q["day"] and abs(ours_p / q["price"] - 1) > tol:
                warns.append(f"{t}: 같은 날 종가가 출처마다 다름(우리 ${ours_p:.2f} vs ${q['price']:.2f}, {abs(ours_p / q['price'] - 1) * 100:.1f}%)")
        except Exception:
            continue
    return warns, checked


def av_next_earnings(tickers, today=None, getter=None, sleeper=time.sleep):
    """{티커: 'YYYY-MM-DD'} — 오늘 이후 가장 가까운 실적일. 캘린더에 없으면 빠진다. (응답은 CSV)"""
    today = pd.Timestamp(today or datetime.now().date()).normalize()
    out = {}
    for t in tickers:
        txt = av_get("EARNINGS_CALENDAR", getter=getter, sleeper=sleeper, as_text=True, symbol=t, horizon="6month")
        try:
            if not txt or "reportDate" not in txt:
                continue
            df = pd.read_csv(io.StringIO(txt))
            d = pd.to_datetime(df.loc[df["symbol"].astype(str).str.upper() == t, "reportDate"], errors="coerce").dropna()
            d = d[d >= today]
            if len(d):
                out[t] = str(d.min().date())
        except Exception:
            continue
    return out


def earnings_calendar_notes(manual, av_dates, today=None):
    """수동 달력(common.STOCK_EARNINGS_DATES)과 AV 캘린더를 비교한 안내 문구. 수동 값을 자동으로 바꾸지는 않는다(잘못된 자동 갱신 방지)."""
    today = pd.Timestamp(today or datetime.now().date()).normalize()
    notes = []
    for t, av in sorted((av_dates or {}).items()):
        try:
            m = pd.Timestamp(manual.get(t)).normalize() if manual.get(t) else None
        except Exception:
            m = None
        if m is None or m < today:
            notes.append(f"{t} 다음 실적일 후보 {av} (Alpha Vantage 캘린더, 추정일 수 있음) — common.STOCK_EARNINGS_DATES 에 반영 필요")
        elif str(m.date()) != av:
            notes.append(f"{t} 실적일 불일치: 입력값 {m.date()} vs Alpha Vantage {av} — 회사 IR 로 확인 필요")
    return notes


def av_put_call(symbols, date=None, getter=None, sleeper=time.sleep):
    """{심볼: 전체 옵션체인 풋/콜 비율}. 응답 형식이 예상과 다르면 그 심볼은 빠진다(기록만 안 될 뿐 다른 동작엔 영향 없음)."""
    out = {}
    for s in symbols:
        params = {"symbol": s}
        if date:
            params["date"] = str(pd.Timestamp(date).date())
        d = av_get(AV_PCR_FUNCTION, getter=getter, sleeper=sleeper, **params)
        try:
            body = d.get("data", d) if isinstance(d, dict) else None
            if isinstance(body, list) and body:
                body = body[0]
            v = _num((body or {}).get("put_call_ratio_full_chain"))
            if v is not None and 0 < v < 50:
                out[s] = v
        except Exception:
            continue
    return out


# ------------------------------------------------------------------ 3) 로그
def append_edge_history(summaries, log_dir="logs", run_date=None):
    """reports/summary_*.json 내용(load_summaries 결과)에서 기준선·엣지를 edge_history.csv 에 쌓는다. 같은 (가격일, 종목, 호라이즌)은 처음 값 유지."""
    try:
        rows = []
        run_date = pd.Timestamp(run_date or datetime.now())
        for name, data in (summaries or {}).items():
            h = data.get("horizon_days")
            for it in data.get("items", []):
                pdv = pd.to_datetime(it.get("price_date"), errors="coerce")
                if pd.isna(pdv) or _num(it.get("base_rate")) is None or _num(it.get("up_prob")) is None:
                    continue
                rows.append({"run_date": str(pdv.date()), "logged_at": str(run_date), "ticker": str(it.get("ticker")).strip().upper(), "horizon_days": h,
                             "price_date": str(pdv.date()), "log": name, "benchmark": it.get("benchmark"),
                             "direction": it.get("direction"), "up_prob": _num(it.get("up_prob")),
                             "raw_up_prob": _num(it.get("raw_up_prob")), "base_rate": _num(it.get("base_rate")),
                             "edge": _num(it.get("edge"))})
        if not rows:
            return 0
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, EDGE_LOG)
        old = _read_csv(path, EDGE_COLUMNS)
        new = pd.DataFrame(rows, columns=EDGE_COLUMNS)
        both = pd.concat([old, new], ignore_index=True) if len(old) else new
        both["price_date"] = both["price_date"].astype(str).str[:10]
        before = len(old)
        both = both.drop_duplicates(["price_date", "ticker", "horizon_days", "log"], keep="first")
        both = both.sort_values(["price_date", "log", "ticker"], kind="stable").reset_index(drop=True)
        if len(both) != before:
            _atomic_csv(both[[c for c in EDGE_COLUMNS if c in both.columns]], path)
        return len(both) - before
    except Exception as e:
        print(f"⚠️ 엣지 이력 기록 실패(무시): {e}")
        return 0


def append_context(price_date, fg=None, pcr=None, extra=None, log_dir="logs", run_date=None):
    """가격일 1행. 이미 그 날짜 행이 있으면 비어 있는 칸만 채운다(이미 기록된 값은 덮어쓰지 않음)."""
    try:
        pdv = pd.to_datetime(price_date, errors="coerce")
        if pd.isna(pdv):
            return False
        row = {"run_date": str(pdv.date()), "ticker": "_MARKET", "horizon_days": 0, "price_date": str(pdv.date()),
               "logged_at": str(pd.Timestamp(run_date or datetime.now()))}
        if fg:
            row.update({"fg_score": fg.get("score"), "fg_rating": fg.get("rating"),
                        "fg_asof": str(pd.Timestamp(fg["asof"]).date()) if fg.get("asof") is not None else None,
                        "fg_prev_close": fg.get("prev_close"), "fg_prev_week": fg.get("prev_week")})
            for k, c in (fg.get("components") or {}).items():
                row[f"fg_{k}"] = c.get("score")
        for s, v in (pcr or {}).items():
            row[f"pcr_{s}"] = v
        row.update(extra or {})
        if len(row) <= 5:
            return False
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, CONTEXT_LOG)
        old = _read_csv(path)
        if len(old) and "price_date" in old.columns:
            old["price_date"] = old["price_date"].astype(str).str[:10]
            old = old.drop_duplicates(["price_date"], keep="first")
            m = old["price_date"] == row["price_date"]
            if m.any():
                i = old.index[m][0]
                changed = False
                for k, v in row.items():
                    if k in ("run_date", "ticker", "horizon_days", "price_date", "logged_at") or v is None:
                        continue
                    if k not in old.columns:
                        old[k] = np.nan
                        if isinstance(v, str):
                            old[k] = old[k].astype(object)
                    if pd.isna(old.at[i, k]):
                        if isinstance(v, str) and old[k].dtype != object:
                            old[k] = old[k].astype(object)
                        old.at[i, k] = v
                        changed = True
                if changed:
                    _atomic_csv(old, path)
                return changed
            out = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
        else:
            out = pd.DataFrame([row])
        out = out.sort_values("price_date", kind="stable").reset_index(drop=True)
        _atomic_csv(out, path)
        return True
    except Exception as e:
        print(f"⚠️ 맥락 로그 기록 실패(무시): {e}")
        return False


# ------------------------------------------------------------------ 3-b) 실행 환경 기록
ENV_PACKAGES = ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "yfinance")
ENV_FILE = "env_versions.json"


def record_env(report_dir="reports", versions=None, today=None):
    """
    예측에 영향을 주는 라이브러리 버전을 reports/env_versions.json 에 남기고, 직전 기록과 달라진 항목을 돌려준다.
    왜: requirements.txt 에 버전이 고정돼 있지 않아 GitHub Actions 가 어느 날 새 xgboost/lightgbm/scikit-learn 을 설치하면
    코드를 한 줄도 안 바꿨는데 예측값이 달라질 수 있다(= 모르는 사이의 검증 시계 오염). 바뀐 날을 알 수 있게 기록한다.
    반환: ["xgboost 2.1.4 → 3.0.0", ...] (처음 기록이거나 변화 없으면 [])
    """
    try:
        if versions is None:
            import platform
            from importlib import metadata
            versions = {"python": platform.python_version()}
            for p in ENV_PACKAGES:
                try:
                    versions[p] = metadata.version(p)
                except Exception:
                    versions[p] = None
        os.makedirs(report_dir, exist_ok=True)
        path = os.path.join(report_dir, ENV_FILE)
        prev = {}
        try:
            with open(path, encoding="utf-8") as f:
                prev = json.load(f)
        except Exception:
            prev = {}
        old = prev.get("versions") or {}
        changes = [f"{k} {old.get(k)} → {v}" for k, v in versions.items() if old and old.get(k) != v]
        day = str(pd.Timestamp(today or datetime.now().date()).date())
        history = list(prev.get("history") or [])
        if changes:
            history.append({"date": day, "changes": changes})
        if changes or not old:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"versions": versions, "since": day if (changes or not old) else prev.get("since"),
                           "history": history[-50:]}, f, ensure_ascii=False, indent=1)
        return changes
    except Exception as e:
        print(f"⚠️ 실행 환경 기록 실패(무시): {e}")
        return []


# ------------------------------------------------------------------ 4) 한 번에 모으기 (daily_summary 가 호출)
def collect(summaries, log_dir="logs", earnings_manual=None, today=None, fg_getter=None, av_getter=None, sleeper=time.sleep, report_dir=None,
            quote_tickers=("TQQQ", "SOXL", "SPXL", "SQQQ", "SOXS", "SPXS"),   # v36: 추적 대상 6개 ETF 만(호출 수를 구성종목·실적 캘린더에 양보)
            pcr_symbols=("SPY", "QQQ", "SMH", "SOXL", "NVDA"), earnings_tickers=("AAPL", "NVDA", "TSLA")):
    """요약 리포트용 dict 를 돌려준다. 어떤 부분이 실패해도 나머지는 채운다. 예외를 밖으로 내지 않는다."""
    ctx = {"fg": None, "fg_stats": {}, "price_warn": [], "price_checked": 0, "earnings_notes": [], "pcr": {}, "av_used": False}
    items, pdates = [], []
    try:
        for data in (summaries or {}).values():
            for it in data.get("items", []):
                items.append(it)
                d = pd.to_datetime(it.get("price_date"), errors="coerce")
                if not pd.isna(d):
                    pdates.append(d.normalize())
    except Exception:
        pass
    price_date = max(pdates) if pdates else None

    try:
        ctx["edge_rows_added"] = append_edge_history(summaries, log_dir)
    except Exception:
        pass
    if report_dir:
        ctx["env_changes"] = record_env(report_dir, today=today)
    try:
        fg = fetch_fear_greed(getter=fg_getter)
        if fg:
            ctx["fg"] = {k: v for k, v in fg.items() if k != "history"}
            ctx["fg_stats"] = fg_forward_stats(fg.get("history"), fg["score"])
    except Exception as e:
        print(f"⚠️ 공포탐욕지수 단계 실패(무시): {e}")
    try:
        if os.environ.get(AV_KEY_ENV):
            ctx["av_used"] = True
            quotes = av_quotes(quote_tickers, getter=av_getter, sleeper=sleeper)
            ctx["price_warn"], ctx["price_checked"] = crosscheck_prices(items, quotes)
            ctx["pcr"] = av_put_call(pcr_symbols, date=price_date, getter=av_getter, sleeper=sleeper)
            manual = earnings_manual or {}
            td = pd.Timestamp(today or datetime.now().date()).normalize()
            stale = [t for t in earnings_tickers if not manual.get(t) or pd.to_datetime(manual.get(t), errors="coerce") < td]
            # 호출 수 절약: 수동 달력이 지난 종목만 매일, 나머지는 월요일에만 대조
            ask = stale if td.weekday() != 0 else list(earnings_tickers)
            if ask:
                ctx["earnings_notes"] = earnings_calendar_notes(manual, av_next_earnings(ask, today=td, getter=av_getter, sleeper=sleeper), today=td)
    except Exception as e:
        print(f"⚠️ Alpha Vantage 단계 실패(무시): {e}")
    try:
        if price_date is not None:
            fg_row = ctx["fg"]
            # 공포탐욕 값은 그 값의 기준일이 우리 가격일과 같을 때만 그 날짜 행에 적는다(주말 실행·지연 시 다른 날 값이 섞이지 않게)
            if fg_row and fg_row.get("asof") is not None and pd.Timestamp(fg_row["asof"]).normalize() != price_date:
                fg_row = None
            append_context(price_date, fg=fg_row, pcr=ctx["pcr"],
                           extra={"av_checked": ctx["price_checked"], "av_warn": len(ctx["price_warn"])} if ctx["av_used"] else None,
                           log_dir=log_dir)
    except Exception as e:
        print(f"⚠️ 맥락 로그 단계 실패(무시): {e}")
    return ctx


def context_lines(ctx):
    """텔레그램 섹션용 줄 리스트(HTML 이스케이프는 호출부에서). 보여줄 게 없으면 []."""
    lines = []
    try:
        fg = (ctx or {}).get("fg")
        if fg:
            def f(v):
                return "-" if v is None else f"{v:.0f}"
            lines.append(f"공포탐욕지수 {fg['score']:.0f} ({fg.get('bucket') or fg.get('rating')}) · 전일 {f(fg.get('prev_close'))} · 1주 전 {f(fg.get('prev_week'))} · 1달 전 {f(fg.get('prev_month'))}")
            comps = fg.get("components") or {}
            if comps:
                lo = sorted(comps.items(), key=lambda kv: kv[1]["score"])
                parts = [f"{FG_COMPONENTS.get(k, k)} {v['score']:.0f}" for k, v in lo]
                lines.append("  └ 구성(0 공포~100 탐욕): " + " · ".join(parts))
            st = (ctx.get("fg_stats") or {}).get(20) or (ctx.get("fg_stats") or {}).get(5)
            if st:
                h = 20 if (ctx.get("fg_stats") or {}).get(20) else 5
                lines.append(f"  └ {st['since'][:4]}년 이후 같은 구간 뒤 S&P500 {h}거래일: 평균 {st['mean'] * 100:+.1f}% (전체 평균 {st['all_mean'] * 100:+.1f}%) · "
                             f"흔들림 연 {st['vol'] * 100:.0f}% (전체 {st['all_vol'] * 100:.0f}%)")
                lines.append(f"     독립 구간 {st['episodes']}개뿐인 과거 집계 — 예측이 아니며 3배 ETF에는 그대로 적용되지 않음")
        ec = (ctx or {}).get("env_changes") or []
        if ec:
            lines.append("⚠️ 라이브러리 버전이 바뀜: " + ", ".join(ec[:6]) + " — 코드가 같아도 예측값이 달라질 수 있음(검증 시계 주의)")
        pw = (ctx or {}).get("price_warn") or []
        if pw:
            lines.append("⚠️ 시세 교차확인(Alpha Vantage):")
            lines += [f"  • {w}" for w in pw[:6]]
        elif (ctx or {}).get("price_checked"):
            lines.append(f"시세 교차확인: {ctx['price_checked']}종목 종가·거래일이 다른 출처(Alpha Vantage)와 일치")
        pcr = (ctx or {}).get("pcr") or {}
        if pcr:
            lines.append("풋/콜 비율(전일, 기록용): " + " · ".join(f"{k} {v:.2f}" for k, v in pcr.items()))
    except Exception:
        return lines
    return lines
