"""
etf_drivers.py — ETF 속 들여다보기 (v36): 구성종목 기여도 · 실적 노출 · 재무(펀더멘털) 추적

무엇을 추적하나 (대상: SOXL/SOXS, TQQQ/SQQQ, SPXL/SPXS)
  3배 ETF 는 지수의 '하루 수익률 ×3'이라 결국 지수를 움직이는 종목이 ETF 를 움직인다. 그래서 종목이 아니라 지수 단위로 본다.
    SOXL·SOXS → NYSE 반도체지수(구성은 SOXX 로 봄, 30종목)   ※ 방향 모델의 벤치마크 SMH 와는 다른 지수
    TQQQ·SQQQ → 나스닥100(QQQ)
    SPXL·SPXS → S&P500(SPY)
  1) 기여도(매일)   : 비중 × 수익률 = 그 종목이 지수를 몇 %p 움직였나. 상위 기여/하락 종목, 오른 종목의 비중, 쏠림(상위5 비중·유효 종목 수)
  2) 실적 노출(매일): 앞으로 10거래일 안에 실적을 내는 종목들의 비중 합 → 지수의 몇 %가 실적 이벤트에 걸려 있나
  3) 재무(돌아가며 갱신): 매출 성장, 매출총이익률, 영업이익률, 재고일수(반도체 사이클), 설비투자(빅테크 = 반도체 수요),
                       선행 PER, EPS 추정치 30일 변화 → 비중 가중 평균으로 지수 단위 요약
  4) 겹침          : QQQ 안의 반도체지수 종목 비중 → TQQQ 와 SOXL 을 같이 들면 얼마나 같은 베팅인가

원칙
- 방향 모델의 입력이 아니다. 피처·임계값·학습·예측 로그는 건드리지 않는다(검증 시계 영향 없음). 표시 + logs/ 기록 전용.
- 외부 조회(yfinance · Alpha Vantage)는 실패·빈 응답·형식 변경에서 조용히 건너뛴다. 계산 함수는 전부 순수 함수라 합성 데이터로 검증한다.
- 재무 수치는 '매수/매도 신호'가 아니라 지수의 체력을 읽는 재료다. 분기 1회 갱신되는 느린 정보라 하루·5일 방향과는 직접 관계가 약하다.

상태 파일: reports/drivers_state.json (구성종목·실적일·재무 캐시)  /  로그: logs/drivers_daily.csv, logs/fundamentals.csv
"""
import io
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

INDEXES = {            # 지수 키 → (구성 ETF, 표시 이름, 추종 레버리지 ETF 쌍)
    "SOXX": {"name": "반도체지수", "pair": "SOXL·SOXS", "long": "SOXL"},
    "QQQ": {"name": "나스닥100", "pair": "TQQQ·SQQQ", "long": "TQQQ"},
    "SPY": {"name": "S&P500", "pair": "SPXL·SPXS", "long": "SPXL"},
}
HYPERSCALERS = ("MSFT", "GOOGL", "AMZN", "META")       # 설비투자 = AI 반도체 수요의 원천
SEED_FILE = "holdings_seed.json"
STATE_FILE = "drivers_state.json"
DAILY_LOG = "drivers_daily.csv"
FUND_LOG = "fundamentals.csv"
HOLDINGS_MAX_AGE = 7          # 일. 구성종목 비중 갱신 주기
EARNINGS_MAX_AGE = 3
FUND_MAX_AGE = 7
FUND_PER_RUN = 10             # 한 번 실행에 재무를 갱신할 종목 수(오래된 순). 조회 부담·차단 위험을 나눠서 진다
TOP_FOR_FUND = 15             # 지수별 재무 추적 종목 수(비중 상위)
MAX_PRICE_NAMES = 40          # 지수별 기여도 계산에 쓸 종목 수(비중 상위)
EARN_WINDOW_BDAYS = 10


# ------------------------------------------------------------------ 공통
def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _today(today=None):
    return pd.Timestamp(today or datetime.now().date()).normalize()


def _age_days(asof, today=None):
    try:
        return int((_today(today) - pd.Timestamp(asof).normalize()).days)
    except Exception:
        return 10 ** 6


def clean_holdings(rows, max_names=None):
    """[{symbol, weight}] → 중복·현금·선물·기호 없는 항목을 뺀 리스트(비중 내림차순). weight 는 0~1 비율. 1보다 크면 %로 보고 100으로 나눈다."""
    out = {}
    for r in rows or []:
        try:
            s = str(r.get("symbol") or "").strip().upper().replace(".", "-")
            w = _num(r.get("weight"))
            if not s or s in ("N/A", "NAN", "NONE", "CASH", "USD") or w is None or w <= 0:
                continue
            out[s] = out.get(s, 0.0) + w
        except Exception:
            continue
    if not out:
        return []
    if max(out.values()) > 1.0 or sum(out.values()) > 1.5:
        out = {k: v / 100.0 for k, v in out.items()}
    lst = sorted(({"symbol": k, "weight": v} for k, v in out.items()), key=lambda d: -d["weight"])
    return lst[:max_names] if max_names else lst


def load_state(report_dir="reports"):
    try:
        with open(os.path.join(report_dir, STATE_FILE), encoding="utf-8") as f:
            s = json.load(f)
        return s if isinstance(s, dict) else {}
    except Exception:
        return {}


def save_state(state, report_dir="reports"):
    try:
        os.makedirs(report_dir, exist_ok=True)
        path = os.path.join(report_dir, STATE_FILE)
        tmp = f"{path}.tmp_{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=0, default=str)
        os.replace(tmp, path)
        return True
    except Exception as e:
        print(f"⚠️ 드라이버 상태 저장 실패(무시): {e}")
        return False


def load_seed(path=SEED_FILE):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ------------------------------------------------------------------ 1) 구성종목
def get_holdings(index, state, seed=None, today=None, av_fetch=None, yf_fetch=None):
    """
    구성종목 비중을 돌려준다: {'asof','source','holdings':[{symbol,weight}]} 또는 None.
    순서: 상태 파일(7일 이내) → Alpha Vantage ETF_PROFILE(키 있을 때) → yfinance 상위 보유종목 → 상태 파일(오래됨) → 시드.
    새로 받은 값이 비정상(종목 5개 미만·비중 합 20% 미만)이면 버리고 이전 값을 쓴다.
    """
    cur = ((state or {}).get("holdings") or {}).get(index)
    if cur and cur.get("holdings") and _age_days(cur.get("asof"), today) <= HOLDINGS_MAX_AGE:
        return cur
    for src, fn in (("alphavantage", av_fetch), ("yfinance", yf_fetch)):
        if fn is None:
            continue
        try:
            h = clean_holdings(fn(index))
            if len(h) >= 5 and sum(x["weight"] for x in h) >= 0.2:
                new = {"asof": str(_today(today).date()), "source": src, "holdings": h}
                state.setdefault("holdings", {})[index] = new
                return new
        except Exception as e:
            print(f"⚠️ [{index}] 구성종목 조회 실패({src}, 무시): {type(e).__name__}")
    if cur and cur.get("holdings"):
        return cur
    sd = (seed or {}).get(index)
    if sd and sd.get("holdings"):
        return {"asof": sd.get("asof"), "source": "seed", "holdings": clean_holdings(sd["holdings"])}
    return None


def av_holdings(index):
    import market_context as mc
    d = mc.av_get("ETF_PROFILE", symbol=index)
    return (d or {}).get("holdings") or []


def yf_holdings(index):
    import yfinance as yf
    th = yf.Ticker(index).funds_data.top_holdings
    col = [c for c in th.columns if "percent" in str(c).lower() or "weight" in str(c).lower()][0]
    return [{"symbol": str(i), "weight": float(v)} for i, v in th[col].items()]


def overlap(h_a, h_b):
    """a 지수 안에서 b 지수 종목이 차지하는 비중(0~1)과 겹치는 종목 수."""
    try:
        b = {x["symbol"] for x in h_b}
        ov = [x for x in h_a if x["symbol"] in b]
        return float(sum(x["weight"] for x in ov)), len(ov)
    except Exception:
        return float("nan"), 0


# ------------------------------------------------------------------ 2) 기여도
def attribution(holdings, closes, lookbacks=(1, 5)):
    """
    closes: DataFrame(index=날짜 오름차순, 열=종목 종가). 반환 dict 또는 None.
      'asof'          : 마지막 종가일
      'covered'       : 가격이 있는 종목들의 비중 합
      h(1,5) 별 → 'est'(비중가중 수익률 = 지수 추정 수익률, 가격 있는 종목 비중으로 재정규화),
                  'contrib'(종목별 기여 %p, 지수 원래 비중 기준), 'up_weight'(오른 종목 비중 비율), 'up_count', 'n'
      'top5_w', 'eff_n': 쏠림(상위5 비중, 유효 종목 수 = 1/Σw²)
    가격이 있는 종목이 5개 미만이거나 비중 합이 20% 미만이면 None(엉뚱한 요약 방지).
    """
    try:
        if not holdings or closes is None or len(closes) < 2:
            return None
        w = pd.Series({h["symbol"]: h["weight"] for h in holdings}, dtype=float)
        px = closes.reindex(columns=[c for c in w.index if c in closes.columns]).apply(pd.to_numeric, errors="coerce")
        px = px.where(px > 0).dropna(how="all").sort_index()
        if px.shape[1] < 5 or len(px) < 2:
            return None
        out = {"asof": pd.Timestamp(px.index[-1]).normalize()}
        tot = float(w.sum())
        wn = w / tot if tot > 0 else w
        out["top5_w"] = float(w.sort_values(ascending=False).head(5).sum())
        out["eff_n"] = float(1.0 / (wn ** 2).sum())
        for h in lookbacks:
            if len(px) <= h:
                continue
            last, prev = px.iloc[-1], px.iloc[-1 - h]
            r = (last / prev - 1).replace([np.inf, -np.inf], np.nan).dropna()
            r = r[r.abs() < 0.9]                      # 분할·오류로 보이는 값 제외
            cov = float(w.reindex(r.index).sum())
            if len(r) < 5 or cov < 0.2:
                continue
            ww = w.reindex(r.index)
            contrib = (ww * r).sort_values()
            out[h] = {"est": float(contrib.sum() / cov), "covered": cov, "n": int(len(r)),
                      "contrib": {k: float(v) for k, v in contrib.items()},
                      "up_weight": float(ww[r > 0].sum() / cov), "up_count": int((r > 0).sum())}
        return out if any(h in out for h in lookbacks) else None
    except Exception as e:
        print(f"⚠️ 기여도 계산 실패(무시): {type(e).__name__}: {e}")
        return None


def fetch_closes(symbols, period="2mo"):
    """yfinance 로 여러 종목 종가를 한 번에. 실패하면 None."""
    try:
        import yfinance as yf
        df = yf.download(sorted(set(symbols)), period=period, progress=False, auto_adjust=True, threads=True)
        if df is None or len(df) == 0:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            lv0 = df.columns.get_level_values(0)
            close = df["Close"] if "Close" in lv0 else df.xs("Close", axis=1, level=1)
        else:
            close = df[["Close"]].rename(columns={"Close": list(set(symbols))[0]})
        close.index = pd.to_datetime(close.index).tz_localize(None) if getattr(close.index, "tz", None) is not None else pd.to_datetime(close.index)
        try:
            import common
            close = common.drop_partial_us_bar(close)      # 장중 실행이면 오늘 봉 제외
        except Exception:
            pass
        return close
    except Exception as e:
        print(f"⚠️ 구성종목 시세 조회 실패(무시): {type(e).__name__}: {e}")
        return None


# ------------------------------------------------------------------ 3) 실적 노출
def parse_earnings_csv(text, today=None):
    """Alpha Vantage EARNINGS_CALENDAR(전체) CSV → {종목: 'YYYY-MM-DD'} (오늘 이후 가장 가까운 날짜)."""
    out = {}
    try:
        if not text or "reportDate" not in text:
            return out
        df = pd.read_csv(io.StringIO(text))
        df["d"] = pd.to_datetime(df["reportDate"], errors="coerce")
        df = df.dropna(subset=["d"])
        df = df[df["d"] >= _today(today)]
        for s, g in df.groupby(df["symbol"].astype(str).str.upper()):
            out[s] = str(g["d"].min().date())
    except Exception:
        return {}
    return out


def earnings_exposure(holdings, earn_dates, today=None, window=EARN_WINDOW_BDAYS):
    """앞으로 window 거래일 안에 실적을 내는 종목: {'weight': 비중 합, 'names': [(종목, 날짜, 비중)] 비중순, 'known': 날짜를 아는 종목 비중 합}."""
    td = _today(today)
    end = td + pd.offsets.BDay(window)
    names, known = [], 0.0
    for h in holdings or []:
        d = (earn_dates or {}).get(h["symbol"])
        if not d:
            continue
        try:
            dd = pd.Timestamp(d).normalize()
        except Exception:
            continue
        known += h["weight"]
        if td <= dd <= end:
            names.append((h["symbol"], str(dd.date()), float(h["weight"])))
    names.sort(key=lambda x: -x[2])
    return {"weight": float(sum(n[2] for n in names)), "names": names, "known": float(known)}


def yf_earnings_date(symbol, today=None):
    import yfinance as yf
    cal = yf.Ticker(symbol).calendar
    ds = cal.get("Earnings Date") if isinstance(cal, dict) else None
    ds = [pd.Timestamp(x).normalize() for x in (ds or [])]
    ds = [x for x in ds if x >= _today(today)]
    return str(min(ds).date()) if ds else None


# ------------------------------------------------------------------ 4) 재무
def _row(df, *names):
    """재무제표 DataFrame(행=계정, 열=분기말 최신순)에서 이름이 맞는 첫 행을 최신순 Series 로. 없으면 None."""
    try:
        if df is None or len(df) == 0:
            return None
        idx = {str(i).strip().lower(): i for i in df.index}
        for n in names:
            if n.lower() in idx:
                s = pd.to_numeric(df.loc[idx[n.lower()]], errors="coerce")
                s.index = pd.to_datetime(s.index, errors="coerce")
                s = s[s.index.notna()].sort_index(ascending=False)
                return s if s.notna().any() else None
    except Exception:
        return None
    return None


def _at(s, k):
    try:
        v = s.iloc[k]
        return float(v) if pd.notna(v) else None
    except Exception:
        return None


def _ratio(a, b):
    return (a / b) if (a is not None and b not in (None, 0)) else None


def fundamentals_from_statements(income, balance, cashflow, info=None, eps_trend=None):
    """
    분기 재무제표(yfinance 형식) → 지표 dict. 계산할 수 없는 항목은 None.
      quarter       : 최신 분기말
      rev_yoy/qoq   : 매출 성장(전년 동기 / 전 분기). 전년 동기 값이 없으면 yoy 는 None
      gross_margin, gm_chg_yoy(%p, 없으면 전 분기 대비를 gm_chg_qoq 에)
      op_margin
      dio, dio_chg  : 재고일수 = 재고 / 매출원가 × 91, 전 분기 대비 변화(일)
      capex, capex_yoy : 설비투자(양수), 전년 동기 대비
      fcf_margin
      fwd_pe, eps_rev30 : 선행 PER, 다음 회계연도 EPS 추정치의 30일 변화율
    """
    out = {k: None for k in ("quarter", "rev_yoy", "rev_qoq", "gross_margin", "gm_chg_yoy", "gm_chg_qoq", "op_margin",
                             "dio", "dio_chg", "capex", "capex_yoy", "fcf_margin", "fwd_pe", "eps_rev30")}
    try:
        rev = _row(income, "Total Revenue", "Operating Revenue", "totalRevenue")
        gp = _row(income, "Gross Profit", "grossProfit")
        cogs = _row(income, "Cost Of Revenue", "Reconciled Cost Of Revenue", "costOfRevenue")
        op = _row(income, "Operating Income", "Total Operating Income As Reported", "operatingIncome")
        inv = _row(balance, "Inventory", "inventory")
        capex = _row(cashflow, "Capital Expenditure", "Capital Expenditure Reported", "capitalExpenditures")
        fcf = _row(cashflow, "Free Cash Flow")
        if rev is not None:
            out["quarter"] = str(rev.index[0].date())
            r0, r1, r4 = _at(rev, 0), _at(rev, 1), _at(rev, 4)
            if r0 and r0 > 0:
                out["rev_qoq"] = (r0 / r1 - 1) if r1 and r1 > 0 else None
                out["rev_yoy"] = (r0 / r4 - 1) if r4 and r4 > 0 else None
                g0 = _at(gp, 0) if gp is not None else (r0 - _at(cogs, 0) if cogs is not None and _at(cogs, 0) is not None else None)
                out["gross_margin"] = _ratio(g0, r0)
                for k, key in ((4, "gm_chg_yoy"), (1, "gm_chg_qoq")):
                    rk = _at(rev, k)
                    gk = _at(gp, k) if gp is not None else None
                    if out["gross_margin"] is not None and rk and gk is not None:
                        out[key] = out["gross_margin"] - gk / rk
                out["op_margin"] = _ratio(_at(op, 0), r0) if op is not None else None
                if fcf is not None:
                    out["fcf_margin"] = _ratio(_at(fcf, 0), r0)
        if inv is not None and cogs is not None:
            i0, c0, i1, c1 = _at(inv, 0), _at(cogs, 0), _at(inv, 1), _at(cogs, 1)
            if i0 is not None and c0 and c0 > 0:
                out["dio"] = i0 / c0 * 91.0
                if i1 is not None and c1 and c1 > 0:
                    out["dio_chg"] = out["dio"] - i1 / c1 * 91.0
        if capex is not None:
            c0, c4 = _at(capex, 0), _at(capex, 4)
            if c0 is not None:
                out["capex"] = abs(c0)
                if c4:
                    out["capex_yoy"] = abs(c0) / abs(c4) - 1
        if isinstance(info, dict):
            pe = _num(info.get("forwardPE"))
            out["fwd_pe"] = pe if pe and 0 < pe < 500 else None
        if eps_trend is not None and len(eps_trend):
            try:
                et = eps_trend.copy()
                et.columns = [str(c).lower() for c in et.columns]
                for per in ("+1y", "0y"):
                    if per in et.index and "current" in et.columns and "30daysago" in et.columns:
                        cur, old = _num(et.loc[per, "current"]), _num(et.loc[per, "30daysago"])
                        if cur is not None and old not in (None, 0):
                            out["eps_rev30"] = cur / old - 1 if old > 0 else None
                            break
            except Exception:
                pass
    except Exception as e:
        print(f"⚠️ 재무 지표 계산 실패(무시): {type(e).__name__}: {e}")
    return out


def yf_fundamentals(symbol):
    import yfinance as yf
    t = yf.Ticker(symbol)
    def g(attr):
        try:
            return getattr(t, attr)
        except Exception:
            return None
    info = None
    try:
        info = t.info
    except Exception:
        info = None
    return fundamentals_from_statements(g("quarterly_income_stmt"), g("quarterly_balance_sheet"), g("quarterly_cashflow"), info, g("eps_trend"))


def aggregate_fundamentals(holdings, fund):
    """
    비중 가중 평균으로 지수 단위 요약. fund: {종목: 지표 dict}. 항목마다 값이 있는 종목끼리만 가중(그 비중 합을 cov_* 로 함께 돌려줌).
    PER 은 이익수익률(1/PER)을 가중 평균한 뒤 뒤집는다(고PER 한 종목이 평균을 왜곡하지 않게).
    """
    out = {}
    w = {h["symbol"]: h["weight"] for h in holdings or []}
    for key in ("rev_yoy", "rev_qoq", "gross_margin", "gm_chg_yoy", "op_margin", "dio", "dio_chg", "eps_rev30", "capex_yoy"):
        num = den = 0.0
        for s, ww in w.items():
            v = (fund.get(s) or {}).get(key)
            if v is None or not np.isfinite(v) or abs(v) > 1e4:
                continue
            num += ww * v
            den += ww
        if den > 0:
            out[key] = num / den
            out[f"cov_{key}"] = den
    num = den = 0.0
    for s, ww in w.items():
        pe = (fund.get(s) or {}).get("fwd_pe")
        if pe and pe > 0:
            num += ww / pe
            den += ww
    if den > 0 and num > 0:
        out["fwd_pe"] = den / num
        out["cov_fwd_pe"] = den
    return out


def hyperscaler_capex(fund):
    """빅테크 4사 설비투자 합과 전년 동기 대비(4사 모두 값이 있을 때만 yoy)."""
    try:
        cur, prev, n = 0.0, 0.0, 0
        for s in HYPERSCALERS:
            f = fund.get(s) or {}
            c, y = f.get("capex"), f.get("capex_yoy")
            if c is None:
                continue
            cur += c
            n += 1
            if y is not None and y > -1:
                prev += c / (1 + y)
            else:
                prev = float("nan")
        if n == 0:
            return None
        return {"sum": cur, "n": n, "yoy": (cur / prev - 1) if (n == len(HYPERSCALERS) and prev and np.isfinite(prev) and prev > 0) else None}
    except Exception:
        return None


def refresh_fundamentals(state, symbols, today=None, fetch=None, per_run=FUND_PER_RUN, sleeper=time.sleep):
    """캐시가 없거나 7일 넘은 종목을 오래된 순으로 per_run 개만 갱신. 반환: 갱신한 종목 리스트. 실패한 종목은 이전 값 유지."""
    fetch = fetch or yf_fundamentals
    cache = state.setdefault("fund", {})
    need = sorted((s for s in dict.fromkeys(symbols) if _age_days((cache.get(s) or {}).get("fetched"), today) > FUND_MAX_AGE),
                  key=lambda s: -_age_days((cache.get(s) or {}).get("fetched"), today))
    done = []
    for s in need[:per_run]:
        try:
            f = fetch(s)
            if f and any(v is not None for k, v in f.items() if k != "quarter"):
                f["fetched"] = str(_today(today).date())
                cache[s] = f
                done.append(s)
            else:
                cache.setdefault(s, {})["tried"] = str(_today(today).date())
        except Exception as e:
            print(f"⚠️ [{s}] 재무 조회 실패(무시): {type(e).__name__}")
        try:
            sleeper(0.4)
        except Exception:
            pass
    return done


# ------------------------------------------------------------------ 5) 로그
def _append_csv(path, rows, keys):
    """rows 를 path 에 덧붙인다(keys 가 같은 행은 처음 값 유지). 깨진 파일은 .corrupt 로 치우고 새로 쓴다."""
    if not rows:
        return 0
    new = pd.DataFrame(rows)
    old = pd.DataFrame()
    if os.path.exists(path):
        try:
            old = pd.read_csv(path)
        except Exception:
            try:
                os.replace(path, f"{path}.corrupt_{datetime.now().strftime('%Y%m%d%H%M%S')}")
            except Exception:
                pass
    both = pd.concat([old, new], ignore_index=True) if len(old) else new
    for k in keys:
        both[k] = both[k].astype(str)
    n0 = len(old)
    both = both.drop_duplicates(keys, keep="first").reset_index(drop=True)
    if len(both) != n0:
        tmp = f"{path}.tmp_{os.getpid()}"
        both.to_csv(tmp, index=False)
        os.replace(tmp, path)
    return len(both) - n0


def log_daily(result, log_dir="logs"):
    """지수별 하루 1행: 기여도 요약. run_date 열은 가격일(push_logs 합치기 키와 맞추기 위해), 실제 기록 시각은 logged_at."""
    rows = []
    for idx, r in (result or {}).get("indexes", {}).items():
        a = r.get("attr")
        if not a or 1 not in a:
            continue
        d = str(pd.Timestamp(a["asof"]).date())
        c = a[1]["contrib"]
        top = sorted(c.items(), key=lambda kv: -abs(kv[1]))[:3]
        e = r.get("earn") or {}
        row = {"run_date": d, "ticker": idx, "horizon_days": 0, "price_date": d, "logged_at": str(datetime.now()),
               "est_ret1": a[1]["est"], "covered": a[1]["covered"], "up_weight1": a[1]["up_weight"], "up_count1": a[1]["up_count"], "n": a[1]["n"],
               "est_ret5": (a.get(5) or {}).get("est"), "up_weight5": (a.get(5) or {}).get("up_weight"),
               "top5_w": a["top5_w"], "eff_n": a["eff_n"], "earn_w10": e.get("weight"),
               "top1": top[0][0] if len(top) > 0 else None, "top1_pp": top[0][1] * 100 if len(top) > 0 else None,
               "top2": top[1][0] if len(top) > 1 else None, "top2_pp": top[1][1] * 100 if len(top) > 1 else None,
               "top3": top[2][0] if len(top) > 2 else None, "top3_pp": top[2][1] * 100 if len(top) > 2 else None}
        row.update({f"f_{k}": v for k, v in (r.get("fund_agg") or {}).items() if not k.startswith("cov_")})
        rows.append(row)
    try:
        os.makedirs(log_dir, exist_ok=True)
        return _append_csv(os.path.join(log_dir, DAILY_LOG), rows, ["price_date", "ticker"])
    except Exception as e:
        print(f"⚠️ 드라이버 로그 기록 실패(무시): {e}")
        return 0


def log_fundamentals(state, updated, log_dir="logs"):
    """새로 받은 재무 지표를 (종목, 분기) 단위로 1행씩 쌓는다. 같은 분기는 처음 값 유지(나중에 정정 여부를 볼 수 있게 logged_at 남김)."""
    rows = []
    for s in updated or []:
        f = (state.get("fund") or {}).get(s) or {}
        q = f.get("quarter")
        if not q:
            continue
        row = {"run_date": q, "ticker": s, "horizon_days": 0, "price_date": q, "logged_at": str(datetime.now())}
        row.update({k: v for k, v in f.items() if k not in ("tried",)})
        rows.append(row)
    try:
        os.makedirs(log_dir, exist_ok=True)
        return _append_csv(os.path.join(log_dir, FUND_LOG), rows, ["price_date", "ticker"])
    except Exception as e:
        print(f"⚠️ 재무 로그 기록 실패(무시): {e}")
        return 0


# ------------------------------------------------------------------ 6) 한 번에 (daily_summary 가 호출)
def collect(log_dir="logs", report_dir="reports", today=None, seed_path=SEED_FILE,
            closes_fetch=None, av_holdings_fetch=None, yf_holdings_fetch=None, earnings_text_fetch=None,
            yf_earnings_fetch=None, fund_fetch=None, sleeper=time.sleep, use_network=True):
    """세 지수의 기여도·실적 노출·재무 요약을 모아 dict 로 돌려준다. 어떤 단계가 실패해도 나머지는 채우며 예외를 밖으로 내지 않는다."""
    res = {"indexes": {}, "overlap": None, "capex": None, "notes": []}
    try:
        state = load_state(report_dir)
        seed = load_seed(seed_path)
        td = _today(today)
        if use_network:
            if av_holdings_fetch is None and os.environ.get("ALPHAVANTAGE_API_KEY"):
                av_holdings_fetch = av_holdings
            yf_holdings_fetch = yf_holdings_fetch or yf_holdings
            closes_fetch = closes_fetch or fetch_closes
            yf_earnings_fetch = yf_earnings_fetch or yf_earnings_date
            fund_fetch = fund_fetch or yf_fundamentals
            if earnings_text_fetch is None and os.environ.get("ALPHAVANTAGE_API_KEY"):
                def earnings_text_fetch():
                    import market_context as mc
                    return mc.av_get("EARNINGS_CALENDAR", as_text=True, horizon="3month")

        hold = {}
        for idx in INDEXES:
            h = get_holdings(idx, state, seed, td, av_holdings_fetch, yf_holdings_fetch)
            if h and h.get("holdings"):
                hold[idx] = h
            else:
                res["notes"].append(f"{INDEXES[idx]['name']} 구성종목 비중을 받지 못함")
        if not hold:
            return res

        # 시세(한 번에)
        price_syms = sorted({x["symbol"] for h in hold.values() for x in h["holdings"][:MAX_PRICE_NAMES]})
        closes = None
        if closes_fetch is not None:
            try:
                closes = closes_fetch(price_syms)
            except Exception as e:
                print(f"⚠️ 구성종목 시세 단계 실패(무시): {type(e).__name__}")

        # 실적일 캐시
        earn = state.setdefault("earnings", {})
        if _age_days(earn.get("asof"), td) > EARNINGS_MAX_AGE:
            dates = {}
            try:
                if earnings_text_fetch is not None:
                    dates = parse_earnings_csv(earnings_text_fetch(), td)
            except Exception:
                dates = {}
            if len(dates) < 20 and yf_earnings_fetch is not None:       # AV 가 없거나 비었으면 지수별 상위 8종목만 개별 조회
                for s in sorted({x["symbol"] for h in hold.values() for x in h["holdings"][:8]}):
                    try:
                        d = yf_earnings_fetch(s, td)
                        if d:
                            dates[s] = d
                    except Exception:
                        continue
            if dates:
                keep = {x["symbol"] for h in hold.values() for x in h["holdings"]}
                earn.clear()
                earn.update({"asof": str(td.date()), "dates": {k: v for k, v in dates.items() if k in keep}})
        earn_dates = {k: v for k, v in (earn.get("dates") or {}).items() if _age_days(v, td) <= 0}

        # 재무(돌아가며 갱신)
        fund_syms = [x["symbol"] for h in hold.values() for x in h["holdings"][:TOP_FOR_FUND]] + list(HYPERSCALERS)
        updated = []
        if fund_fetch is not None:
            updated = refresh_fundamentals(state, fund_syms, td, fund_fetch, sleeper=sleeper)
        fund = state.get("fund") or {}

        for idx, h in hold.items():
            r = {"holdings_asof": h.get("asof"), "source": h.get("source"), "n_holdings": len(h["holdings"])}
            r["attr"] = attribution(h["holdings"][:MAX_PRICE_NAMES], closes) if closes is not None else None
            r["earn"] = earnings_exposure(h["holdings"], earn_dates, td) if earn_dates else None
            top = h["holdings"][:TOP_FOR_FUND]
            r["fund_agg"] = aggregate_fundamentals(top, fund)
            r["fund_n"] = sum(1 for x in top if (fund.get(x["symbol"]) or {}).get("quarter"))
            r["fund_w"] = float(sum(x["weight"] for x in top))
            res["indexes"][idx] = r
        if "QQQ" in hold and "SOXX" in hold:
            res["overlap"] = overlap(hold["QQQ"]["holdings"], hold["SOXX"]["holdings"])
        res["capex"] = hyperscaler_capex(fund)
        res["fund_updated"] = updated
        log_daily(res, log_dir)
        log_fundamentals(state, updated, log_dir)
        save_state(state, report_dir)
    except Exception as e:
        print(f"⚠️ ETF 드라이버 수집 실패(무시): {type(e).__name__}: {e}")
    return res


def _pct(v, d=1, sign=True):
    return "-" if v is None else (f"{v * 100:+.{d}f}%" if sign else f"{v * 100:.{d}f}%")


def driver_lines(res):
    """텔레그램 섹션용 줄 리스트. 보여줄 게 없으면 []."""
    lines = []
    try:
        for idx, r in (res or {}).get("indexes", {}).items():
            meta = INDEXES[idx]
            a = r.get("attr")
            head = f"{meta['name']} ({meta['pair']})"
            if a and 1 in a:
                c = a[1]["contrib"]
                neg = [(k, v) for k, v in sorted(c.items(), key=lambda kv: kv[1]) if v < 0][:3]
                pos = [(k, v) for k, v in sorted(c.items(), key=lambda kv: -kv[1]) if v > 0][:3]
                d = pd.Timestamp(a["asof"])
                lines.append(f"{head} {d.month}/{d.day}: 지수 추정 {_pct(a[1]['est'])} · 오른 종목 비중 {a[1]['up_weight'] * 100:.0f}%"
                             + (f" · 5일 {_pct(a[5]['est'])}" if 5 in a else ""))
                parts = []
                if pos:
                    parts.append("끌어올림 " + ", ".join(f"{k} {v * 100:+.2f}%p" for k, v in pos))
                if neg:
                    parts.append("끌어내림 " + ", ".join(f"{k} {v * 100:+.2f}%p" for k, v in neg))
                if parts:
                    lines.append("  └ " + " / ".join(parts))
                lines.append(f"  └ 쏠림: 상위5 비중 {a['top5_w'] * 100:.0f}% · 유효 종목 수 {a['eff_n']:.0f}개 (계산에 쓴 비중 {a[1]['covered'] * 100:.0f}%)")
            else:
                lines.append(f"{head}: 구성종목 시세를 받지 못해 기여도 생략")
            e = r.get("earn")
            if e and e.get("names"):
                nm = ", ".join(f"{s}({pd.Timestamp(d).month}/{pd.Timestamp(d).day})" for s, d, w in e["names"][:5])
                lines.append(f"  └ 10거래일 내 실적: 비중 합 {e['weight'] * 100:.0f}% — {nm}")
            elif e:
                lines.append("  └ 10거래일 내 실적 발표 종목 없음(날짜를 아는 종목 기준)")
            f = r.get("fund_agg") or {}
            if f and r.get("fund_n", 0) >= 5:
                bits = []
                if "rev_yoy" in f:
                    bits.append(f"매출 {_pct(f['rev_yoy'], 0)}(전년비)")
                elif "rev_qoq" in f:
                    bits.append(f"매출 {_pct(f['rev_qoq'], 0)}(전분기비)")
                if "gross_margin" in f:
                    bits.append(f"매출총이익률 {f['gross_margin'] * 100:.0f}%" + (f"({f['gm_chg_yoy'] * 100:+.1f}%p)" if "gm_chg_yoy" in f else ""))
                if "dio" in f and idx == "SOXX":
                    bits.append(f"재고 {f['dio']:.0f}일" + (f"({f['dio_chg']:+.0f}일)" if "dio_chg" in f else ""))
                if "fwd_pe" in f:
                    bits.append(f"선행PER {f['fwd_pe']:.0f}배")
                if "eps_rev30" in f:
                    bits.append(f"EPS추정 30일 {_pct(f['eps_rev30'], 1)}")
                if bits:
                    lines.append(f"  └ 체력(상위 {r['fund_n']}종목 가중): " + " · ".join(bits))
        ov = (res or {}).get("overlap")
        if ov and ov[0] == ov[0] and ov[1] > 0:
            lines.append(f"겹침: 나스닥100의 {ov[0] * 100:.0f}%가 반도체지수 종목({ov[1]}개) → TQQQ와 SOXL은 상당 부분 같은 베팅")
        cx = (res or {}).get("capex")
        if cx and cx.get("yoy") is not None:
            lines.append(f"빅테크 4사 설비투자(반도체 수요): 최근 분기 합 ${cx['sum'] / 1e9:.0f}B, 전년비 {_pct(cx['yoy'], 0)}")
        for n in ((res or {}).get("notes") or [])[:3]:
            lines.append(f"ℹ️ {n}")
    except Exception:
        return lines
    return lines
