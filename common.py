"""
common.py — 레버리지/인버스 ETF 방향성 예측 시스템 공통 모듈 (v23.0)
predict_scalp.py / predict_swing.py / predict_position.py 가 공유한다.

v23.0 변경사항
------------------------------------------------
1. NaN 가격 버그 수정: fetch_display_metrics()가 ffill+dropna 가드 포함
2. 확률 보정(calibration): 교차검증 OOF 예측으로 Isotonic Regression 학습,
   "모델이 70%라 했을 때 실제로 몇 %였는지"를 보정해서 신뢰도를 높임
3. 실전 적중률 추적: logs/predictions_*.csv에 매 실행 예측을 기록하고,
   호라이즌이 지난 과거 예측은 실제 결과와 대조해서 최근 N회 적중률 계산
4. train_benchmark_model()을 공통 모듈로 이동 (3개 스크립트 중복 제거)
"""
import os
import contextlib
import re
import html
import json
import time
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from pandas.tseries.offsets import BDay
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings('ignore')

# ============================================================
# 1) 커버 종목 메타데이터
# ============================================================
LEVERAGE_UNIVERSE = {
    'TQQQ': {'multiplier': 3, 'direction': +1, 'benchmark': 'QQQ', 'family': 'Nasdaq100', 'desc': '나스닥100 +3배'},
    'SQQQ': {'multiplier': 3, 'direction': -1, 'benchmark': 'QQQ', 'family': 'Nasdaq100', 'desc': '나스닥100 -3배'},
    'SOXL': {'multiplier': 3, 'direction': +1, 'benchmark': 'SMH', 'family': 'Semiconductor', 'desc': '반도체 +3배'},
    'SOXS': {'multiplier': 3, 'direction': -1, 'benchmark': 'SMH', 'family': 'Semiconductor', 'desc': '반도체 -3배'},
    'SPXL': {'multiplier': 3, 'direction': +1, 'benchmark': 'SPY', 'family': 'SP500', 'desc': 'S&P500 +3배'},
    'SPXS': {'multiplier': 3, 'direction': -1, 'benchmark': 'SPY', 'family': 'SP500', 'desc': 'S&P500 -3배'},
}

BENCHMARK_TICKERS = ['QQQ', 'SMH', 'SPY']
REL_COMPARISON = {'QQQ': 'SPY', 'SMH': 'SPY', 'SPY': None,
                  # 개별 종목은 나스닥(QQQ) 대비 상대강도로 비교 (전부 나스닥 대형 기술주라서)
                  'AAPL': 'QQQ', 'NVDA': 'QQQ', 'TSLA': 'QQQ'}

# 개별 종목(레버리지 아님) 유니버스 — 페어 개념 없이 방향성만 그대로 사용
STOCK_TICKERS = ['AAPL', 'NVDA', 'TSLA']

# 종목별 다음 실적 발표(예정)일 — 확정 아닌 예상치 포함, 주기적으로 갱신 필요
# 날짜는 "미국 현지 발표일"(장 마감 후 발표면 한국시간으론 다음날 새벽).
# 2026-10-16 검증: TSLA 10/21 (회사 IR 확정 — days.to 및 기사 2곳 일치),
#                  NVDA 11/18 (추정, 공식 확정 전 — 캘린더 2곳 일치), AAPL 10/29 (캘린더 일치).
# 이전 값(TSLA 10/28, NVDA 8/26)은 오래되었거나 틀렸음 → 실적 경고가 엉뚱한 날 떴음.
STOCK_EARNINGS_DATES = {
    'AAPL': '2026-10-29',
    'NVDA': '2026-11-18',
    'TSLA': '2026-10-21',
}


def earnings_proximity_warning(ticker, window_days=5):
    """
    종목의 다음 실적 발표가 임박했으면 경고 문자열을 반환한다.
    가격 패턴만 학습한 정량 모델은 실적 발표 전후 신뢰도가 크게 떨어지므로,
    이 구간에서는 클로드 프로젝트/스킬(10-K, 어닝콜 트랜스크립트 기반 정성 분석)로
    가이던스 변화·리스크 요인을 따로 확인하라고 안내한다.
    반환: 경고 문자열 또는 None (임박하지 않음)
    """
    date_str = STOCK_EARNINGS_DATES.get(ticker)
    if not date_str:
        return None
    earnings_date = pd.Timestamp(date_str)
    today = pd.Timestamp.now().normalize()
    days_left = (earnings_date - today).days
    if 0 <= days_left <= window_days:
        return (f"🚨 실적 발표 D-{days_left} ({date_str}) — 이 구간은 정량 모델(가격 패턴)의 "
                f"신뢰도가 원래 낮음. 클로드 프로젝트/스킬로 가이던스·리스크 변화 등 정성 분석을 "
                f"별도로 확인 권장")
    return None
MACRO_TICKERS = {'QQQ': 'QQQ', 'SMH': 'SMH', 'SPY': 'SPY', 'VIX': '^VIX', 'TNX': '^TNX'}

# SOXX(반도체 ETF) 상위 10종목 비중 (2026-08-04 기준 스냅샷, 상위 10종목이 전체의 약 61%)
# SOXL/SOXS 모델(SMH 벤치마크)에만 구성종목 폭(breadth)/쏠림(dispersion) 피처로 사용
SOXX_TOP_HOLDINGS_WEIGHTS = {
    'AMD': 8.54, 'NVDA': 8.53, 'AVGO': 7.95, 'MU': 7.81, 'INTC': 5.41,
    'AMAT': 5.16, 'MRVL': 4.53, 'TSM': 4.40, 'KLAC': 4.32, 'LRCX': 4.24,
}


def fetch_constituent_breadth_features(start_date='2022-01-01'):
    """
    SOXX 상위 10종목의 개별 가격을 받아와 SMH 하나로는 안 보이는 두 가지 신호를 만든다.
      - ConstMomentum5 : 비중가중 평균 5일 수익률 (SMH 자체 모멘텀과 유사하지만 구성 방식이 다름)
      - ConstBreadth5  : 상위 10종목 중 5일 수익률이 양(+)인 종목의 비율 (0~1). 지수는 오르는데
                         소수 대형주가 끌고 나머지는 빠지는 "쏠림 상승"을 구분해줌
      - ConstDispersion5: 비중가중 5일 수익률의 표준편차. 종목 간 방향이 갈릴수록 커짐
                         (섹터 내부에서 의견이 갈리는 국면 = 변동성 확대 전조로 흔히 해석됨)
    반환: DatetimeIndex를 가진 DataFrame (컬럼: ConstMomentum5, ConstBreadth5, ConstDispersion5)
    """
    total_w = sum(SOXX_TOP_HOLDINGS_WEIGHTS.values())
    norm_weights = {k: v / total_w for k, v in SOXX_TOP_HOLDINGS_WEIGHTS.items()}

    ret_frames = {}
    for ticker in SOXX_TOP_HOLDINGS_WEIGHTS:
        df = fetch_ticker_data(ticker, start_date=start_date)
        if df is None:
            continue
        ret_frames[ticker] = df['Close'].pct_change(5)

    if not ret_frames:
        return pd.DataFrame(columns=['ConstMomentum5', 'ConstBreadth5', 'ConstDispersion5'])

    ret_df = pd.DataFrame(ret_frames).ffill().dropna(how='all')
    weight_series = pd.Series({k: norm_weights[k] for k in ret_df.columns})
    # 결측 종목이 있는 날짜엔 남은 종목들끼리 비중을 재정규화해서 가중평균
    valid_mask = ret_df.notna()
    effective_w = valid_mask.mul(weight_series, axis=1)
    effective_w = effective_w.div(effective_w.sum(axis=1), axis=0)

    const_momentum = (ret_df.fillna(0) * effective_w).sum(axis=1)
    const_breadth = (ret_df > 0).sum(axis=1) / valid_mask.sum(axis=1)
    const_dispersion = (ret_df.sub(const_momentum, axis=0).pow(2) * effective_w).sum(axis=1).pow(0.5)

    out = pd.DataFrame({
        'ConstMomentum5': const_momentum,
        'ConstBreadth5': const_breadth,
        'ConstDispersion5': const_dispersion,
    })
    return out.dropna()

MARKET_EVENTS = {
    "2026-07-29": {"desc": "FOMC 금리 결정 (한국시간 7/30 오전 3시 발표)",
                   "consensus": "동결 vs 25bp 인상 확률 팽팽 (동결 62~70%, 인상 25~30%)"},
    "2026-08-26": {"desc": "NVIDIA FY2027 Q2 실적 발표 (장마감 후)",
                   "consensus": "컨센서스 EPS $2.08"},
    # --- 아래는 2026-10-16 추가 (앞으로의 일정). 컨센서스는 수집하지 않음 ---
    # Tesla/Apple/NVIDIA 실적은 STOCK_EARNINGS_DATES 에서 따로 표시하므로 여기엔 넣지 않는다(중복 방지).
    "2026-10-28": {"desc": "FOMC 금리 결정 (10/27~28 회의, 한국시간 10/29 새벽 발표)",
                   "consensus": "CPI 상회 후 인상 우려 확대 — 인상 확률 보도마다 상이(약 30~46%)"},
    "2026-10-29": {"desc": "빅테크 실적 집중 (MSFT·Meta 10/28, Amazon·Intel 10/29 미국 마감 후)",
                   "consensus": "캘린더 기준, 날짜 확인 필요"},
    "2026-11-03": {"desc": "미국 중간선거",
                   "consensus": "일정 확정"},
    "2026-11-06": {"desc": "미국 10월 고용지표",
                   "consensus": "일정 보도 기준(확인 필요)"},
    "2026-11-10": {"desc": "미국 10월 CPI",
                   "consensus": "일정 보도 기준(확인 필요)"},
}


MACRO_SHOCK_STD_WINDOW = 60   # 최근 60거래일 기준으로 "평소 하루 변동폭"을 계산
MACRO_SHOCK_Z_THRESHOLD = 2.0  # 이 배수 이상 벗어나면 경고


def macro_shock_warning(macro_df, z_threshold=MACRO_SHOCK_Z_THRESHOLD, window=MACRO_SHOCK_STD_WINDOW):
    """
    당일(최근 1거래일) VIX·10년물 금리(TNX) 변동이 최근 window거래일 평소 수준 대비
    얼마나 이례적인지 z-score로 확인해서, 기술적 지표 기반 예측의 신뢰도를 낮춰봐야 하는
    날인지 경고한다.

    기존 학습 피처 VIX_chg5/TNX_chg5는 5일 변화율이라 하루짜리 급변(예: 2026-08-18
    30년물 금리 19년래 최고 + 반도체 업종 급락 같은 이벤트)이 5일 평균에 묻혀 잘 안 잡힐 수
    있다. 이 함수는 그 사각지대를 보완하는 순수 규칙 기반 경고이며, 모델 학습·피처·임계값에는
    전혀 관여하지 않는다 — 그래서 이 로직을 추가/조정해도 지금까지 쌓인 실전 적중률 로그는
    무효화되지 않는다 (검증 시계 리셋 없음).

    macro_df: fetch_macro_data()의 반환값 (컬럼: QQQ, SMH, SPY, VIX, TNX)
    반환: 경고 문자열 또는 None (평소 수준이거나 데이터 부족/결측 시 조용히 스킵)
    """
    if macro_df is None or macro_df.empty:
        return None
    if any(col not in macro_df.columns for col in ('VIX', 'TNX')):
        return None
    if len(macro_df) < window + 5:
        return None

    vix_chg1 = macro_df['VIX'].pct_change(1)
    tnx_chg1 = macro_df['TNX'].pct_change(1)

    vix_hist = vix_chg1.iloc[-(window + 1):-1]
    tnx_hist = tnx_chg1.iloc[-(window + 1):-1]
    vix_std, tnx_std = vix_hist.std(), tnx_hist.std()
    if not vix_std or not tnx_std or pd.isna(vix_std) or pd.isna(tnx_std) or vix_std == 0 or tnx_std == 0:
        return None

    vix_today, tnx_today = vix_chg1.iloc[-1], tnx_chg1.iloc[-1]
    if pd.isna(vix_today) or pd.isna(tnx_today):
        return None

    vix_z, tnx_z = vix_today / vix_std, tnx_today / tnx_std

    # VIX만 튀는 날은 흔하지만, VIX+금리가 동시에 튀거나 둘 중 하나가 아주 크게 튀는 날은
    # 드물고 임팩트가 크다 — 합성 데이터 검증 결과 평소엔 ~1.7%, 충격일엔 100% 탐지됨.
    triggered = (
        (abs(vix_z) >= z_threshold and abs(tnx_z) >= z_threshold * 0.5)
        or (abs(vix_z) >= z_threshold * 1.5)
        or (abs(tnx_z) >= z_threshold * 1.5)
    )
    if not triggered:
        return None

    return (f"🌩️ 매크로 변동성 급증 감지 (VIX 1일 변화 {vix_today*100:+.1f}%, z={vix_z:+.1f} / "
            f"10Y금리 1일 변화 {tnx_today*100:+.1f}%, z={tnx_z:+.1f}) — 학습 피처(VIX_chg5 등)는 "
            f"5일 평균이라 오늘 같은 급변은 충분히 못 잡을 수 있음. 오늘 방향 예측은 평소보다 "
            f"신뢰도 낮게 볼 것")


def check_upcoming_events(window_days=2):
    today = datetime.now().date()
    out = []
    for date_str, info in MARKET_EVENTS.items():
        ev_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        days_left = (ev_date - today).days
        if 0 <= days_left <= window_days:
            prefix = "🔴 오늘" if days_left == 0 else f"⚠️ D-{days_left}"
            out.append(f"  {prefix}: {info['desc']} | {info['consensus']}")
    return out


# ============================================================
# 2) 데이터 수집
# ============================================================
def _yf_download_with_retry(ticker, start_date, max_retries=3, base_delay=2):
    """
    야후 파이낸스가 일시적으로 요청을 막았을 때(rate limit) 바로 포기하지 않고
    잠깐 쉬었다가 다시 시도한다. 그래도 계속 실패하면 빈 DataFrame을 반환해서
    호출부가 '데이터 없음'으로 정상 처리하도록 한다.
    """
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(ticker, start=start_date, progress=False, multi_level_index=False)
            if not df.empty:
                return df
        except Exception as e:
            print(f"⚠️ [{ticker}] 조회 시도 {attempt}/{max_retries} 실패: {e}")
        if attempt < max_retries:
            wait = base_delay * attempt  # 2초, 4초로 점점 늘려가며 대기
            print(f"⏳ [{ticker}] {wait}초 대기 후 재시도...")
            time.sleep(wait)
    print(f"❌ [{ticker}] {max_retries}회 재시도 모두 실패 — 데이터 없음으로 처리")
    return pd.DataFrame()


def fetch_macro_data(start_date='2022-01-01'):
    macro_data = {}
    for name, t in MACRO_TICKERS.items():
        df_raw = _yf_download_with_retry(t, start_date)
        if not df_raw.empty:
            macro_data[name] = df_raw['Close'].squeeze()
    macro_df = pd.DataFrame(macro_data).ffill().dropna()
    return macro_df


US_EASTERN = ZoneInfo('America/New_York')


def drop_partial_us_bar(df, now_et=None):
    """
    미국 정규장이 진행 중일 때 yfinance는 "오늘" 일봉을 장중 가격으로 돌려준다.
    그 값을 종가처럼 쓰면 예측·정산이 오염되므로(2026-08-17 14:27 UTC 실행분이 실제로 그랬음),
    '오늘(미국 동부 기준) 봉'이 있고 아직 16:00 ET 전이면 마지막 행을 버린다.
    주말·공휴일엔 오늘 봉이 없으므로 아무것도 버리지 않는다.
    """
    if df is None or len(df) == 0:
        return df
    now_et = now_et or datetime.now(US_EASTERN)
    last_date = pd.Timestamp(df.index[-1]).normalize()
    if last_date == pd.Timestamp(now_et.date()) and now_et.time() < dtime(16, 0):
        return df.iloc[:-1]
    return df


def clean_price_frame(df):
    """
    가격표 위생 처리(v27, 스트레스 테스트에서 발견): 날짜 오름차순 정렬, 중복 날짜 제거(마지막 값 유지),
    종가·고가·저가가 유한하지 않거나 0 이하인 행 제거. 정상 데이터에는 아무 영향 없음(no-op).
    잘못된 시세 한 줄(0원·음수·NaN)이 inf 피처·0나누기·엉뚱한 정산으로 번지는 것을 입구에서 막는다.
    """
    if df is None or len(df) == 0:
        return df
    try:
        out = df
        if not out.index.is_monotonic_increasing:
            out = out.sort_index()
        if out.index.has_duplicates:
            out = out[~out.index.duplicated(keep='last')]
        cols = [c for c in ('High', 'Low', 'Close') if c in out.columns]
        vals = out[cols].apply(pd.to_numeric, errors='coerce')
        ok = np.isfinite(vals).all(axis=1) & (vals > 0).all(axis=1)
        if not ok.all():
            print(f"⚠️ 비정상 시세 {int((~ok).sum())}행 제외(0/음수/NaN/inf)")
            out = out[ok]
        return out
    except Exception as e:
        print(f"⚠️ 가격표 위생 처리 실패(원본 사용): {e}")
        return df


def fetch_ticker_data(ticker, start_date='2022-01-01'):
    tgt = _yf_download_with_retry(ticker, start_date)
    if tgt.empty:
        return None
    out = pd.DataFrame({
        'High': tgt['High'].squeeze(),
        'Low': tgt['Low'].squeeze(),
        'Close': tgt['Close'].squeeze(),
    })
    out = clean_price_frame(out)
    out = drop_partial_us_bar(out)
    return out if len(out) else None


def fetch_display_metrics(ticker):
    """
    v23.0 버그수정: 현재가/변동성 조회 시 NaN 방지.
    yfinance가 마지막 행에 결측치를 반환하는 경우가 있어 ffill+dropna로 방어한다.
    """
    try:
        return _fetch_display_metrics_impl(ticker)
    except Exception as e:     # v27: 한 종목 시세 문제가 스크립트 전체(=로그 기록)를 죽이지 않도록
        print(f"⚠️ [{ticker}] 표시용 지표 계산 실패(스킵): {type(e).__name__}: {e}")
        return None


def _fetch_display_metrics_impl(ticker):
    df = fetch_ticker_data(ticker)
    if df is None:
        return None
    df = df.ffill().dropna()
    if df.empty:
        return None
    price = df['Close'].iloc[-1]
    vol = df['Close'].pct_change().rolling(20).std().iloc[-1] * (252 ** 0.5)
    if pd.isna(price) or pd.isna(vol):
        return None
    close = df['Close']
    high_60d = close.tail(60).max()          # 최근 60거래일 종가 고점
    drawdown = price / high_60d - 1 if high_60d and not pd.isna(high_60d) else np.nan
    ret5 = close.iloc[-1] / close.iloc[-6] - 1 if len(close) > 5 else np.nan
    ret1 = close.iloc[-1] / close.iloc[-2] - 1 if len(close) > 1 else np.nan
    return {'price': price, 'vol': vol, 'df': df,
            'price_date': pd.Timestamp(df.index[-1]).normalize(),   # 이 가격이 속한 거래일(종가일)
            'high_60d': float(high_60d), 'drawdown': float(drawdown),
            'ret1': float(ret1), 'ret5': float(ret5)}


def fetch_analyst_momentum_features(ticker, start_date='2022-01-01'):
    """
    가격 패턴과 완전히 독립적인 정보원: 애널리스트 등급 변경(상향/하향) 이력.
    yfinance의 upgrades_downgrades는 실제 등급 변경이 일어난 날짜가 찍혀있어서
    (다른 구성종목 폭 피처와 달리) 과거로 거슬러 학습용 시계열을 만들 수 있다.

    - AnalystNet30     : 최근 30일 (상향 - 하향) 건수 — 양수면 애널리스트들이 낙관 전환 중
    - AnalystActivity30: 최근 30일 등급 변경 총 건수 — 그 종목에 대한 애널리스트 관심도/논쟁 강도

    ETF(QQQ/SMH/SPY)는 애널리스트 등급 자체가 없으므로 개별 종목(AAPL/NVDA/TSLA 등)에만 사용한다.
    yfinance API 응답 형식이 바뀌거나 데이터가 없으면 조용히 None을 반환 — 호출부에서 이미
    "None이면 이 피처 없이 진행"하도록 방어적으로 짜여 있어야 한다.
    """
    try:
        raw = yf.Ticker(ticker).upgrades_downgrades
    except Exception as e:
        print(f"⚠️ [{ticker}] 애널리스트 데이터 조회 실패: {e}")
        return None
    if raw is None or raw.empty:
        print(f"⚠️ [{ticker}] 애널리스트 데이터 없음 (upgrades_downgrades 비어있음)")
        return None

    ud = raw.reset_index()
    date_col = ud.columns[0]  # 보통 'GradeDate'라는 이름의 인덱스가 첫 컬럼으로 옴
    if 'Action' not in ud.columns:
        print(f"⚠️ [{ticker}] 'Action' 컬럼 없음 — yfinance 응답 형식 확인 필요. 실제 컬럼: {list(ud.columns)}")
        return None

    ud[date_col] = pd.to_datetime(ud[date_col], errors='coerce')
    ud = ud.dropna(subset=[date_col])
    ud = ud[ud[date_col] >= pd.Timestamp(start_date)]

    print(f"📋 [{ticker}] 애널리스트 등급변경 원본 {len(raw)}건 중 {start_date} 이후 {len(ud)}건 사용")

    # 이벤트가 너무 적으면(예: 몇 건) 롤링 30일 피처가 거의 항상 0이다가 어쩌다 한 번 튀는
    # 형태가 되어 트리 모델이 그 한 번의 스파이크에 과적합하기 쉽다. 최소 건수 미달이면
    # 이 피처 자체를 신뢰할 수 없다고 보고 사용하지 않는다.
    MIN_EVENTS = 10
    if len(ud) < MIN_EVENTS:
        print(f"⚠️ [{ticker}] 애널리스트 이벤트 {len(ud)}건 < 최소 기준 {MIN_EVENTS}건 — 이 피처 사용 안 함 (데이터 부실 의심)")
        return None

    action = ud['Action'].astype(str).str.lower()
    ud['is_up'] = action.eq('up').astype(int)
    ud['is_down'] = action.eq('down').astype(int)
    ud = ud.set_index(date_col).sort_index()

    daily_up = ud['is_up'].resample('D').sum()
    daily_down = ud['is_down'].resample('D').sum()
    daily_total = ud['Action'].resample('D').count()

    full_idx = pd.date_range(daily_up.index.min(), pd.Timestamp.now().normalize(), freq='D')
    daily_up = daily_up.reindex(full_idx, fill_value=0)
    daily_down = daily_down.reindex(full_idx, fill_value=0)
    daily_total = daily_total.reindex(full_idx, fill_value=0)

    net30 = (daily_up - daily_down).rolling(30, min_periods=1).sum()
    activity30 = daily_total.rolling(30, min_periods=1).sum()

    return pd.DataFrame({'AnalystNet30': net30, 'AnalystActivity30': activity30})


# ============================================================
# 3) 기술적 지표 / 피처 엔지니어링
# ============================================================
def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + (gain / loss)))


def _bollinger(series, period=20, std_mult=2.0):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return ma + std_mult*std, ma, ma - std_mult*std


def _atr(df, period=14):
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def _stochastic_k(df, period=14):
    low_min = df['Low'].rolling(period).min()
    high_max = df['High'].rolling(period).max()
    return ((df['Close'] - low_min) / (high_max - low_min).replace(0, np.nan)) * 100


def build_features(df, benchmark_col):
    df['RSI'] = _rsi(df['Close'])
    macd_line = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Hist'] = macd_line - macd_line.ewm(span=9, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['Price_to_EMA20'] = (df['Close'] / df['EMA_20']) - 1

    bb_upper, bb_ma, bb_lower = _bollinger(df['Close'])
    df['BB_Pct'] = ((df['Close'] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)) * 100

    rolling_max_252 = df['Close'].rolling(252, min_periods=60).max()
    df['Momentum_High52w'] = (df['Close'] >= rolling_max_252 * 0.995).astype(int).rolling(20).mean()

    short_slope = df['Close'].pct_change(20)
    long_slope = df['Close'].pct_change(120) / 6
    df['TrendSlope'] = short_slope - long_slope
    df['Momentum_120'] = df['Close'].pct_change(120)
    df['ROC10'] = df['Close'].pct_change(10)

    df['ATR14'] = _atr(df) / df['Close']
    df['StochK'] = _stochastic_k(df)
    df['HistVol20'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)

    if benchmark_col and benchmark_col in df.columns:
        df['RelBenchmark20'] = df['Close'].pct_change(20) - df[benchmark_col].pct_change(20)
    else:
        df['RelBenchmark20'] = 0

    if 'VIX' in df.columns:
        df['VIX_level'] = df['VIX']
        df['VIX_chg5'] = df['VIX'].pct_change(5)
    if 'TNX' in df.columns:
        df['TNX_chg5'] = df['TNX'].pct_change(5)

    # SOXX 구성종목 폭/쏠림 피처 — train_benchmark_model에서 SMH일 때만 미리 join해줌.
    # join이 안 됐으면(다른 벤치마크/종목) 0으로 채워 피처 컬럼 개수를 항상 동일하게 유지.
    for col in ('ConstMomentum5', 'ConstBreadth5', 'ConstDispersion5'):
        if col not in df.columns:
            df[col] = 0

    # 애널리스트 등급변경 피처도 동일한 패턴: 개별 종목에서만 미리 join됨, 아니면 0
    for col in ('AnalystNet30', 'AnalystActivity30'):
        if col not in df.columns:
            df[col] = 0

    return df


FEATURE_COLUMNS = [
    'RSI', 'MACD_Hist', 'Price_to_EMA20', 'BB_Pct',
    'Momentum_High52w', 'TrendSlope', 'Momentum_120', 'ROC10',
    'ATR14', 'StochK', 'HistVol20', 'RelBenchmark20',
    'VIX_level', 'VIX_chg5', 'TNX_chg5',
    'ConstMomentum5', 'ConstBreadth5', 'ConstDispersion5',
    'AnalystNet30', 'AnalystActivity30',
]


# ============================================================
# 4) 타겟 생성 — 대칭 3구간(상승/하락/횡보), 횡보는 학습 제외
# ============================================================
def build_target(df, horizon_days, threshold_cap):
    rolling_median_ret = df['Close'].pct_change().abs().rolling(window=60, min_periods=20).median()
    dyn_threshold = np.clip(rolling_median_ret * 0.5 * np.sqrt(horizon_days), 0.003, threshold_cap)
    df['Target_Ret'] = df['Close'].pct_change(horizon_days).shift(-horizon_days)

    target = np.full(len(df), np.nan)
    up_mask = (df['Target_Ret'] > dyn_threshold).values
    down_mask = (df['Target_Ret'] < -dyn_threshold).values
    target[up_mask] = 1
    target[down_mask] = 0
    df['Target'] = target
    df['DeadZone_Ratio'] = 1 - (up_mask.sum() + down_mask.sum()) / len(df)
    return df


# ============================================================
# 5) 벤치마크 모델 학습 (공통화) + 확률 보정(calibration)
# ============================================================
def _train_benchmark_model_impl(benchmark, macro_df, horizon_days, threshold_cap,
                           benchmark_multiplier=3, min_train_rows=300, model_params=None,
                           extra_features_df=None, extra_feature_cols=None):
    """
    벤치마크 하나에 대해 XGB+LGBM 앙상블을 학습하고,
    교차검증 중 나온 out-of-fold 예측으로 Isotonic Regression 보정기를 만들어
    최종 확률에 적용한다.

    extra_features_df / extra_feature_cols: SMH의 경우 fetch_constituent_breadth()로
    만든 구성종목 체감폭/쏠림도 데이터를 넘기면 기본 15개 피처에 추가로 합쳐서 학습한다.
    다른 벤치마크(QQQ/SPY/개별종목)는 기존과 동일하게 None으로 두면 된다.

    반환: dict(up_prob, down_prob, raw_up_prob, f1, prec, rec, dead_zone) 또는 None
    """
    if model_params is None:
        model_params = {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05}

    df = fetch_ticker_data(benchmark)
    if df is None:
        return None
    df = df.join(macro_df, how='left')
    if extra_features_df is not None:
        df = df.join(extra_features_df, how='left')
    df = df.ffill().dropna()
    df = build_features(df, benchmark_col=REL_COMPARISON[benchmark])
    df = build_target(df, horizon_days, threshold_cap / benchmark_multiplier)
    df = df.replace([np.inf, -np.inf], np.nan)   # v27: 0원/결측 가격이 만든 inf 피처가 학습을 죽이지 않도록

    feature_cols = FEATURE_COLUMNS + (extra_feature_cols or [])

    df_train = df.dropna(subset=['Target'] + feature_cols).copy()
    if len(df_train) < min_train_rows:
        return None

    X, y = df_train[feature_cols], df_train['Target']
    if y.nunique() < 2:      # v27: 상승/하락 라벨 중 한쪽만 있으면 학습·보정이 의미 없음
        print(f"⚠️ [{benchmark}] 학습 라벨이 한 클래스뿐 — 이번 회차 스킵")
        return None

    tscv = TimeSeriesSplit(n_splits=3)
    xgb = XGBClassifier(eval_metric='logloss', random_state=42, **model_params)
    lgbm = LGBMClassifier(verbose=-1, random_state=42, **model_params)

    f1 = prec = rec = 0
    oof_probs, oof_labels = [], []
    n_ok_folds = 0
    for tr_idx, te_idx in tscv.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        if y_tr.nunique() < 2:
            continue          # v27: 학습 구간 라벨이 한 클래스뿐인 폴드는 건너뜀(XGBoost가 ValueError로 죽음)
        n_ok_folds += 1
        scaler = StandardScaler()
        X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)

        xgb.fit(X_tr_s, y_tr)
        lgbm.fit(X_tr_s, y_tr)
        probs = (xgb.predict_proba(X_te_s) + lgbm.predict_proba(X_te_s)) / 2
        preds = (probs[:, 1] >= 0.5).astype(int)

        f1 += f1_score(y_te, preds, average='macro', zero_division=0)
        prec += precision_score(y_te, preds, zero_division=0)
        rec += recall_score(y_te, preds, zero_division=0)

        oof_probs.extend(probs[:, 1].tolist())
        oof_labels.extend(y_te.tolist())
    if n_ok_folds == 0 or len(set(oof_labels)) < 2:
        print(f"⚠️ [{benchmark}] 유효한 교차검증 폴드가 없거나 OOF 라벨이 한 클래스 — 이번 회차 스킵")
        return None
    f1, prec, rec = f1/n_ok_folds, prec/n_ok_folds, rec/n_ok_folds   # 폴드가 3개 다 유효하면 기존(/3)과 동일

    # 확률 보정기: OOF (원시확률, 실제라벨) 쌍으로 Isotonic Regression 학습
    calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    calibrator.fit(np.array(oof_probs), np.array(oof_labels))

    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    xgb.fit(X_scaled, y)
    lgbm.fit(X_scaled, y)

    # 피처 중요도: XGB/LGBM 각각을 0~1로 정규화한 뒤 평균 — 두 모델의 스케일이 달라서
    # (XGB는 gain 기반, LGBM은 split 기반) 그냥 더하면 한쪽에 치우치므로 정규화 후 평균낸다.
    xgb_imp = xgb.feature_importances_
    lgbm_imp = lgbm.feature_importances_
    xgb_imp_norm = xgb_imp / xgb_imp.sum() if xgb_imp.sum() > 0 else xgb_imp
    lgbm_imp_norm = lgbm_imp / lgbm_imp.sum() if lgbm_imp.sum() > 0 else lgbm_imp
    avg_importance = (xgb_imp_norm + lgbm_imp_norm) / 2
    importance_pairs = sorted(zip(feature_cols, avg_importance), key=lambda x: -x[1])
    top_features = importance_pairs[:5]

    latest = scaler_final.transform(df[feature_cols].iloc[[-1]])
    final_probs = (xgb.predict_proba(latest) + lgbm.predict_proba(latest)) / 2
    raw_up_prob = float(final_probs[0][1])
    calibrated_up_prob = float(calibrator.predict([raw_up_prob])[0])
    # v27: 등온회귀는 표본이 얇은 구간에서 0%/100%를 내놓을 수 있다(실제 로그에도 0.0% 존재). '확률 0%'는
    # 표본 수십 건으로는 정당화될 수 없으므로 1~99%로 제한한다. 50% 경계는 그대로라 예측 방향(라벨)은 불변이고,
    # 하락확률 = 1 - 상승확률 이므로 롱/인버스 페어 합 100% 대칭도 유지된다.
    if not np.isfinite(calibrated_up_prob):
        print(f"⚠️ [{benchmark}] 보정 확률이 유한하지 않음 — 이번 회차 스킵")
        return None
    _unclamped = calibrated_up_prob
    calibrated_up_prob = min(max(calibrated_up_prob, PROB_FLOOR), 1 - PROB_FLOOR)
    clamped = bool(abs(_unclamped - calibrated_up_prob) > 1e-12)
    if clamped:
        print(f"ℹ️ [{benchmark}] 보정 확률 {_unclamped*100:.1f}% → {calibrated_up_prob*100:.1f}%로 제한(등온회귀 극단값; 과신 신호일 수 있음)")

    # 기준선(base rate): 횡보 제외 학습 샘플 중 실제 상승 비율.
    # 모델의 진짜 엣지 = 보정확률 - 기준선. 이 값이 0에 가까우면
    # 모델이 "그 시점 특유의 정보"가 아니라 그냥 과거 평균(강세장 편향 등)을
    # 재현하고 있을 가능성이 높다는 뜻.
    base_rate = float(y.mean() * 100)

    return {
        'up_prob': calibrated_up_prob * 100,
        'down_prob': (1 - calibrated_up_prob) * 100,
        'raw_up_prob': raw_up_prob * 100,
        'base_rate': base_rate,
        'edge': calibrated_up_prob * 100 - base_rate,
        'f1': f1, 'prec': prec, 'rec': rec,
        'dead_zone': df['DeadZone_Ratio'].iloc[-1],
        'top_features': top_features,
        'calib_unclamped': _unclamped * 100, 'calib_clamped': clamped,
    }


PROB_FLOOR = 0.01   # 보정 확률의 하한(상한은 1 - 하한)
LAST_TRAIN_ERRORS = {}   # {벤치마크: 마지막 학습 실패 사유} — 리포트/디버깅용


def train_benchmark_model(benchmark, macro_df, horizon_days, threshold_cap, *args, **kwargs):
    """
    v27: 학습 구현(_train_benchmark_model_impl)을 예외로부터 격리하는 얇은 래퍼.
    한 벤치마크의 학습이 어떤 이유로든 죽어도 None을 돌려줘서, 호출 스크립트가 나머지 벤치마크와
    그 전에 쌓인 예측 기록을 계속 처리하게 한다(이전엔 예외 하나가 스크립트 전체와 그날 로그 기록을 날렸음).
    """
    try:
        LAST_TRAIN_ERRORS.pop(benchmark, None)
        return _train_benchmark_model_impl(benchmark, macro_df, horizon_days, threshold_cap, *args, **kwargs)
    except Exception as e:
        import traceback as _tb
        LAST_TRAIN_ERRORS[benchmark] = f"{type(e).__name__}: {e}"
        print(f"❌ [{benchmark}] 학습 중 예외(이 벤치마크만 스킵): {type(e).__name__}: {e}")
        print(_tb.format_exc()[-1500:])
        return None


# ============================================================
# 6) 페어 확률 변환
# ============================================================
def apply_direction(benchmark_up_prob, benchmark_down_prob, direction):
    if direction == +1:
        return benchmark_up_prob, benchmark_down_prob
    else:
        return benchmark_down_prob, benchmark_up_prob


# ============================================================
# 7) 실전 적중률 추적 (예측 로그)
# ============================================================
LOG_DIR = "logs"
LOG_COLUMNS = ['run_date', 'benchmark', 'ticker', 'horizon_days', 'predicted_label',
               'predicted_up_prob', 'raw_up_prob', 'price_at_prediction', 'target_date',
               'actual_price', 'actual_return', 'actual_label', 'correct', 'near_earnings',
               'price_date']   # v25: 예측에 쓴 마지막 종가일(구 로그엔 없음 → 아래 폴백으로 보강)


def _log_path(name):
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, f"predictions_{name}.csv")


LOG_NUMERIC_COLS = ['horizon_days', 'predicted_label', 'predicted_up_prob', 'raw_up_prob', 'price_at_prediction',
                    'actual_price', 'actual_return', 'actual_label', 'correct']


def _normalize_log(df):
    """로그 DataFrame을 표준 형태로: 누락 컬럼 보강, 날짜·숫자 강제 변환, run_date/ticker가 없는 행 제거. (정제 후 df, 제거한 행 수)"""
    for col in LOG_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df['run_date'] = pd.to_datetime(df['run_date'], errors='coerce')
    df['target_date'] = pd.to_datetime(df['target_date'], errors='coerce')
    df['price_date'] = pd.to_datetime(df['price_date'], errors='coerce')
    for col in LOG_NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    bad = df['run_date'].isna() | df['ticker'].isna() | (df['ticker'].astype(str).str.strip() == '')
    n_bad = int(bad.sum())
    if n_bad:
        df = df[~bad].reset_index(drop=True)
    return df, n_bad


def _preserve_corrupt(path, move=False):
    """읽기에 문제가 있던 원본을 덮어쓰기 전에 보존한다(이후 저장이 기록을 날리는 사고 방지)."""
    try:
        import shutil
        dst = f"{path}.corrupt_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        (shutil.move if move else shutil.copy2)(path, dst)
        return dst
    except Exception:
        return None


def load_log(name):
    """
    v27: 로그가 깨져 있어도 죽지 않는다. 정상 파일은 기존과 동일하게 읽는다.
    - 일부 행만 문제(날짜/티커 없음): 그 행만 제외하고 원본은 .corrupt_* 로 보존, 정제본을 저장.
    - 파일 구조 자체가 깨짐(잘린 줄·바이너리 등): 원본 보존 후 읽히는 줄만 살려서(on_bad_lines='skip') 계속.
    - 하나도 못 살리면 원본을 옮겨두고 빈 로그로 시작(다음 실행이 같은 파일에서 또 죽지 않도록).
    """
    path = _log_path(name)
    if not os.path.exists(path):
        return pd.DataFrame(columns=LOG_COLUMNS)
    try:
        df = pd.read_csv(path)
        if 'run_date' not in df.columns or 'ticker' not in df.columns:
            raise ValueError("필수 컬럼(run_date/ticker) 없음")
        df, n_bad = _normalize_log(df)
        if n_bad:
            dst = _preserve_corrupt(path)
            print(f"⚠️ [{name}] 로그에서 읽을 수 없는 행 {n_bad}건 제외(원본 보존: {dst})")
            save_log(name, df)
        return df
    except Exception as e:
        print(f"⚠️ [{name}] 예측 로그 읽기 실패({type(e).__name__}: {e}) — 원본 보존 후 복구 시도")
        dst = _preserve_corrupt(path)
        try:
            df = pd.read_csv(path, engine='python', on_bad_lines='skip')
            if 'run_date' in df.columns and 'ticker' in df.columns:
                df, _ = _normalize_log(df)
                if len(df):
                    print(f"ℹ️ [{name}] 읽히는 {len(df)}행만 살려 계속합니다(원본: {dst})")
                    save_log(name, df)
                    return df
        except Exception as e2:
            print(f"⚠️ [{name}] 복구 실패: {e2}")
        _preserve_corrupt(path, move=True)
        print(f"⚠️ [{name}] 살릴 행이 없어 빈 로그로 시작합니다(원본은 .corrupt_* 로 보존됨)")
        return pd.DataFrame(columns=LOG_COLUMNS)


def atomic_to_csv(df, path):
    """임시 파일에 완전히 쓴 뒤 os.replace로 교체한다(v27). 쓰는 도중 프로세스가 끊겨도 원본이 반쯤 잘리지 않는다."""
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


def save_log(name, df):
    atomic_to_csv(df, _log_path(name))


def _fallback_price_date(run_date):
    """
    price_date 기록이 없는 과거 행용 폴백: 실행 시점에 확보된 마지막 종가일 = 실행일 직전 영업일.
    (화~금 실행 → 전날 종가, 토·일·월 실행 → 직전 금요일 종가. 공휴일은 정산 때 searchsorted가 보정)
    """
    return pd.Timestamp(run_date).normalize() - BDay(1)


def compute_aligned_outcome(price_df, price_date, horizon_days, logged_entry_price=None, entry_tol=0.015):
    """
    모델이 학습한 목표(종가 t → t+h '거래일' 뒤 종가)와 똑같은 기준으로 실제 결과를 계산한다.
    v24 이전 정산은 '실행일 + h영업일 이후 첫 종가'를 써서, 화~금 실행분이 h+1일 수익률로 채점됐다.

    price_df: 'Close' 컬럼, 날짜 오름차순. price_date: 예측에 쓴 마지막 종가일.
    아직 h거래일이 지나지 않았으면 None.
    entry 가격은 같은 조정주가 시리즈의 price_date 종가를 쓰되(배당 조정 편향 방지),
    로그 기록가와 1.5% 넘게 어긋나면(날짜 불일치 의심) 로그 기록가를 쓴다(구 방식).
    """
    try:
        h = int(horizon_days)
    except Exception:
        return None
    if h < 1 or pd.isna(price_date):
        return None
    close = pd.to_numeric(price_df['Close'], errors='coerce')
    close = close[np.isfinite(close) & (close > 0)]          # v27: 0/음수/NaN 종가는 거래 봉으로 취급하지 않음
    if not close.index.is_monotonic_increasing:
        close = close.sort_index()
    if close.index.has_duplicates:
        close = close[~close.index.duplicated(keep='last')]
    pos = close.index.searchsorted(pd.Timestamp(price_date), side='right') - 1
    if pos < 0:
        return None
    tpos = pos + h
    if tpos >= len(close):
        return None
    entry = float(close.iloc[pos])
    logged = pd.to_numeric(logged_entry_price, errors='coerce') if logged_entry_price is not None else np.nan
    if np.isfinite(logged) and logged > 0:
        if abs(entry / float(logged) - 1) > entry_tol:
            entry = float(logged)
    actual_price = float(close.iloc[tpos])
    actual_ret = (actual_price - entry) / entry
    return {'actual_price': actual_price, 'actual_return': actual_ret,
            'actual_label': 1 if actual_ret > 0 else 0,
            'target_used': pd.Timestamp(close.index[tpos]).normalize()}


def settle_predictions(name):
    """
    호라이즌이 지난 과거 예측의 실제 결과를 확인해서 채워넣는다 (v25: 모델 목표와 같은 거래일 기준).
    v27: 행 단위로 격리한다. 오염된 한 행(predicted_label/horizon NaN, 진입가 0 등)이 예외를 일으켜도
    그 행만 건너뛰고 나머지는 정상 정산한다(이전엔 한 행 때문에 이후 모든 실행이 영구히 죽었음).
    """
    df = load_log(name)
    if df.empty:
        return df
    today = pd.Timestamp.now().normalize()
    poison = []     # 정산 불가(오염) 행의 인덱스

    # 1) 미정산 행: price_date 보강 + target_date를 '가격일 + h영업일'로 교정
    pending_idx = list(df.index[df['actual_label'].isna()])
    changed = False
    for idx in pending_idx:
        try:
            pdv = df.at[idx, 'price_date']
            if pd.isna(pdv):
                pdv = _fallback_price_date(df.at[idx, 'run_date'])
            tdv = pd.Timestamp(pdv) + BDay(int(df.at[idx, 'horizon_days']))
        except Exception:
            poison.append(idx)
            continue
        if not (pd.notna(df.at[idx, 'price_date']) and df.at[idx, 'price_date'] == pdv
                and pd.notna(df.at[idx, 'target_date']) and df.at[idx, 'target_date'] == tdv):
            changed = True
        df.at[idx, 'price_date'] = pdv
        df.at[idx, 'target_date'] = tdv

    mask_unresolved = df['actual_label'].isna() & (df['target_date'] <= today)
    if poison:
        mask_unresolved.loc[poison] = False
    for ticker in df.loc[mask_unresolved, 'ticker'].unique():
        try:
            price_df = fetch_ticker_data(ticker)
        except Exception as e:
            print(f"⚠️ [{ticker}] 정산용 시세 조회 실패(보류): {e}")
            continue
        if price_df is None:
            continue
        price_df = price_df.ffill().dropna()

        rows = df[(df['ticker'] == ticker) & mask_unresolved]
        for idx, row in rows.iterrows():
            try:
                if pd.isna(row['predicted_label']):
                    raise ValueError("predicted_label 없음")
                res = compute_aligned_outcome(price_df, row['price_date'], row['horizon_days'],
                                              row['price_at_prediction'])
                if res is None:
                    continue
                df.loc[idx, 'actual_price'] = res['actual_price']
                df.loc[idx, 'actual_return'] = res['actual_return']
                df.loc[idx, 'actual_label'] = res['actual_label']
                df.loc[idx, 'correct'] = int(res['actual_label'] == int(row['predicted_label']))
                changed = True
            except Exception as e:
                poison.append(idx)
                print(f"⚠️ [{name}] 정산 불가 행 격리: {ticker} run_date={row.get('run_date')} ({type(e).__name__}: {e})")
    if poison:
        print(f"⚠️ [{name}] 오염된 행 {len(set(poison))}건은 정산에서 제외했습니다(로그 파일은 그대로 두었으니 확인해 주세요).")

    if changed:
        save_log(name, df)
    return df


@contextlib.contextmanager
def _log_lock(name):
    """같은 기계에서 두 프로세스가 같은 로그를 동시에 '읽고→쓰기' 하다 한쪽 기록이 덮어써지는 것을 막는다(v28).
    잠금 파일은 logs/ 밖(임시 폴더)에 둬서 git 커밋에 섞이지 않는다. 잠금을 못 얻거나 fcntl이 없으면 조용히 건너뛴다."""
    fh = None
    try:
        import fcntl, hashlib, tempfile
        key = hashlib.md5(os.path.abspath(_log_path(name)).encode()).hexdigest()[:12]
        fh = open(os.path.join(tempfile.gettempdir(), f"battery_{key}.lock"), 'w')
        fcntl.flock(fh, fcntl.LOCK_EX)
    except Exception:
        fh = None
    try:
        yield
    finally:
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass


def _drop_invalid_records(new_df, name=""):
    """
    기록 전 검증(v27): 상승확률이 유한하고 0~100이며, 진입가가 유한한 양수이고, 호라이즌이 양의 정수이며,
    run_date·ticker가 있는 행만 로그에 남긴다. 나머지는 그 행만 거부하고 이유를 출력한다.
    (이전엔 NaN 확률이 '하락 예측'으로, 130%·음수 가격이 그대로 로그에 기록됐다.)
    """
    num = lambda c: pd.to_numeric(new_df[c], errors='coerce') if c in new_df.columns else pd.Series(np.nan, index=new_df.index)
    up, price, hz = num('predicted_up_prob'), num('price_at_prediction'), num('horizon_days')
    reasons = pd.Series('', index=new_df.index)
    reasons[~np.isfinite(up) | (up < 0) | (up > 100)] += '상승확률 범위/NaN '
    reasons[~np.isfinite(price) | (price <= 0)] += '진입가 비정상 '
    reasons[~np.isfinite(hz) | (hz < 1)] += '호라이즌 비정상 '
    reasons[new_df['run_date'].isna()] += 'run_date 없음 '
    tk = new_df['ticker'] if 'ticker' in new_df.columns else pd.Series(np.nan, index=new_df.index)
    reasons[tk.isna() | (tk.astype(str).str.strip() == '')] += 'ticker 없음 '
    bad = reasons != ''
    for i in new_df.index[bad]:
        print(f"⚠️ [{name}] 기록 거부: {new_df.at[i, 'ticker'] if 'ticker' in new_df.columns else '?'} — {reasons[i].strip()}")
    return new_df[~bad]


def append_predictions(name, rows):
    """
    실전 적중률 로그에 새 예측을 추가한다.

    안전장치 1 (신규 행 스킵): 같은 날짜(run_date의 날짜 부분)에 동일한
    (ticker, horizon_days) 조합의 기록이 이미 로그에 있으면 그 행은 조용히 스킵한다.
    -> 워크플로우가 하루에 여러 번 실행돼도(예약 실행 후 수동 재실행 포함) 로그는
       하루 1건으로 유지되고, 실전 적중률 표본이 같은 날 재실행분으로 부풀려지지 않는다.

    안전장치 2 (저장 직전 최종 정리): 두 실행이 정말로 겹쳐서(레이스 컨디션) 안전장치 1을
    각자 통과해버린 경우에 대비해, 최종 저장 직전에 (날짜, ticker, horizon_days) 기준으로
    한 번 더 그룹화해서 run_date가 가장 이른 행만 남긴다. dedup_logs.py의 정리 규칙과 동일.

    -> 이전 버전(v23.1)에 있던 안전장치 1이 이후 커밋에서 유실되어 있었고, 그 자리를
       메우려 했던 GitHub Actions concurrency 설정은 "겹치는 실행"만 막을 뿐 "예약 실행이
       끝난 뒤 별개로 들어오는 수동 재실행"은 막지 못해 중복이 계속 쌓였다(로그 전체 행의
       약 30%). 이번에 안전장치 1을 복원하고 안전장치 2를 추가로 더했다.
    -> 로직/피처/임계값과는 무관한 "로깅 안전장치"이며 예측 모델 자체는 전혀 바뀌지 않는다.
    """
    with _log_lock(name):
        _append_predictions_impl(name, rows)


def _tk(series):
    """티커 비교용 정규화(공백 제거·대문자). 로그 원본 값은 바꾸지 않고 '키 비교'에만 쓴다(v28)."""
    return series.astype(str).str.strip().str.upper()


def _append_predictions_impl(name, rows):
    df = load_log(name)
    new_df = pd.DataFrame(rows)
    if new_df.empty:
        return

    new_df['run_date'] = pd.to_datetime(new_df['run_date'], errors='coerce')
    new_df = _drop_invalid_records(new_df, name)
    if new_df.empty:
        return
    new_df['ticker'] = _tk(new_df['ticker'])

    if 'price_date' not in new_df.columns:
        new_df['price_date'] = pd.NaT
    new_df['price_date'] = pd.to_datetime(new_df['price_date'], errors='coerce')

    # v28: 한 번에 들어온 배치 안의 중복도 같은 규칙으로 제거(같은 실행일 또는 같은 종가 기준 → 가장 이른 run_date 1건)
    new_df = new_df.sort_values('run_date', kind='stable')
    eff_pd = new_df['price_date'].copy()
    _m = eff_pd.isna()
    if _m.any():
        eff_pd[_m] = new_df.loc[_m, 'run_date'].apply(_fallback_price_date)
    _dup = (new_df.assign(_d=new_df['run_date'].dt.date).duplicated(['_d', 'ticker', 'horizon_days'])
            | pd.DataFrame({'_p': eff_pd.dt.date, 't': new_df['ticker'], 'h': new_df['horizon_days']}).duplicated())
    if _dup.any():
        print(f"ℹ️ [{name}] 한 배치 안의 중복 {int(_dup.sum())}건 제거")
        new_df = new_df[~_dup.values]

    if not df.empty:
        df_tk = _tk(df['ticker'])
        existing_keys = set(
            zip(df['run_date'].dt.date, df_tk, df['horizon_days'])
        )
        # v25: '같은 종가 기준' 예측도 중복으로 본다(토요일 실행과 월요일 실행은 둘 다 금요일 종가 기준 =
        # 같은 정보). 구 로그 행은 price_date가 없어 폴백(직전 영업일)으로 계산.
        df_pd = df['price_date'].copy()
        miss = df_pd.isna()
        if miss.any():
            df_pd[miss] = df.loc[miss, 'run_date'].apply(_fallback_price_date)
        existing_pd_keys = set(zip(df_pd.dt.date, df_tk, df['horizon_days']))
        dup_mask = new_df.apply(
            lambda r: ((r['run_date'].date(), r['ticker'], r['horizon_days']) in existing_keys) or
                      (pd.notna(r['price_date']) and
                       (r['price_date'].date(), r['ticker'], r['horizon_days']) in existing_pd_keys),
            axis=1
        )
        n_skipped = int(dup_mask.sum())
        if n_skipped > 0:
            print(f"ℹ️ [{name}] 오늘 이미 기록된 예측 {n_skipped}건 스킵 (같은 날 재실행 방지)")
        new_df = new_df[~dup_mask]

    if new_df.empty:
        return

    combined = pd.concat([df, new_df], ignore_index=True)

    # 안전장치 2: 저장 직전 최종 중복 제거 (레이스 컨디션 대비)
    combined['_run_date_parsed'] = pd.to_datetime(combined['run_date'])
    combined['_run_date_only'] = combined['_run_date_parsed'].dt.date
    combined = combined.sort_values('_run_date_parsed')
    before_n = len(combined)
    combined = combined.groupby(
        ['_run_date_only', 'ticker', 'horizon_days'], as_index=False
    ).head(1)
    after_n = len(combined)
    if before_n != after_n:
        print(f"⚠️ [{name}] 저장 직전 레이스 컨디션 중복 {before_n - after_n}건 추가 제거")
    combined = combined.drop(columns=['_run_date_parsed', '_run_date_only'])
    combined = combined.sort_values('run_date').reset_index(drop=True)

    save_log(name, combined)


def wilson_ci(k, n, z=1.96):
    """
    이항비율(적중률)의 Wilson score 신뢰구간. 표본이 적을수록(n이 작을수록) 구간이
    넓어져서 "숫자만 보면 그럴싸해도 사실 못 믿는 수준"이라는 걸 그대로 드러낸다.
    (일반 정규근사 신뢰구간보다 n이 작을 때 더 안정적이라 이항비율엔 Wilson을 표준으로 씀)

    k: 적중 횟수, n: 전체 표본 수, z: 신뢰수준 z-value (기본 1.96 = 95%)
    반환: (하한%, 상한%). n==0이면 (0.0, 100.0) — "아무것도 모른다"를 표현.
    """
    if n <= 0:
        return 0.0, 100.0
    k = min(max(k, 0), n)      # v27: 비정상 입력(k>n, k<0)에서도 0~100% 구간을 돌려줌
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)) / denom
    return max(0.0, (center - margin) * 100), min(100.0, (center + margin) * 100)


def rolling_accuracy(df, ticker, window=20, baseline=None):
    """
    baseline(0~100, 보통 그 벤치마크의 base_rate)을 넘겨주면, 실전 적중률의 95% 신뢰구간
    하한이 baseline을 넘는지("beats_baseline")까지 같이 계산한다. 이게 아니면 "적중률이
    기준선보다 높아 보인다"는 게 표본이 적어서 생긴 우연인지 진짜 우위인지 구분이 안 된다.
    """
    sub = df[(df['ticker'] == ticker) & df['correct'].notna()].sort_values('run_date').tail(window)
    if sub.empty:
        return None
    n = len(sub)
    k = int(sub['correct'].sum())
    accuracy = sub['correct'].mean() * 100
    ci_low, ci_high = wilson_ci(k, n)
    out = {'n': n, 'accuracy': accuracy, 'ci_low': ci_low, 'ci_high': ci_high}
    if baseline is not None:
        out['beats_baseline'] = ci_low > baseline
    return out


def rolling_accuracy_by_earnings(df, ticker, window=20, min_n=3):
    """
    실적 발표 임박 구간(near_earnings=True)과 평상시(False)를 나눠서 적중률을 비교한다.
    각 그룹에 최소 min_n건 이상 쌓여야 결과를 반환한다 (그 전엔 통계적으로 의미 없음).
    """
    sub = df[(df['ticker'] == ticker) & df['correct'].notna()].sort_values('run_date')
    near = sub[sub['near_earnings'] == True].tail(window)
    normal = sub[sub['near_earnings'] != True].tail(window)
    result = {}
    if len(near) >= min_n:
        result['near_earnings'] = {'n': len(near), 'accuracy': near['correct'].mean() * 100}
    if len(normal) >= min_n:
        result['normal'] = {'n': len(normal), 'accuracy': normal['correct'].mean() * 100}
    return result


def evaluate_signal(df, ticker, min_eff=15):
    """
    "이 신호를 믿어도 되는가"를 엄격하게 판정한다 (v25, 표시 전용 — 예측 모델과 무관).

    기존 rolling_accuracy 의 문제 두 가지를 바로잡는다.
      1) 겹치는 표본: 5일·20일 예측은 연속된 행끼리 거의 같은 구간을 공유한다.
         독립 관측 수는 n 이 아니라 n // 호라이즌 이다 (20일 예측 20행 ≈ 독립 1건).
      2) 비교 기준: 학습 기준선이 아니라 "그 기간 한쪽만 계속 찍었을 때의 최고 적중률"
         (naive_best)과 비교한다. 항상 상승만 찍어도 나오는 점수를 넘어야 실력이다.

    status: 'pass'  독립 관측 n_eff >= min_eff 이고 Wilson 95% 하한 > naive_best
            'fail'  표본은 충분한데 통과 못 함
            'insufficient'  독립 관측이 min_eff 미만 (아직 판정 불가)
    정산 완료 행이 없으면 None.
    """
    sub = df[(df['ticker'] == ticker) & df['correct'].notna()].copy()
    sub['correct'] = pd.to_numeric(sub['correct'], errors='coerce')     # v27: 오염된 값(문자 등)에 죽지 않도록
    sub = sub[sub['correct'].notna()]
    if sub.empty:
        return None
    n = len(sub)
    hv = pd.to_numeric(sub['horizon_days'], errors='coerce').dropna()
    h = max(int(hv.median()) if len(hv) else 1, 1)
    acc = float(sub['correct'].mean())
    up_rate = float((sub['actual_label'] == 1).mean())
    naive_best = max(up_rate, 1 - up_rate)
    n_eff = max(n // h, 1)
    k_eff = int(round(acc * n_eff))
    lo, hi = wilson_ci(k_eff, n_eff)
    pred_up_share = float((sub['predicted_label'] == 1).mean())
    if n_eff < min_eff:
        status = 'insufficient'
    else:
        status = 'pass' if lo / 100 > naive_best else 'fail'
    return {'n': n, 'n_eff': n_eff, 'horizon': h, 'accuracy': acc * 100,
            'up_rate': up_rate * 100, 'naive_best': naive_best * 100,
            'edge_vs_naive': (acc - naive_best) * 100,
            'ci_low': lo, 'ci_high': hi,
            'pred_up_share': pred_up_share * 100,
            'one_sided': bool(pred_up_share >= 0.9 or pred_up_share <= 0.1),
            'status': status}


# ---- 요약 리포트용 수집기 (6개 예측 스크립트가 각자 채우고, daily_summary.py 가 읽는다) ----
REPORT_DIR = "reports"
_SUMMARY_ITEMS = []


def add_summary_item(**kwargs):
    """각 스크립트가 종목별 결과를 한 줄씩 등록한다. 실패해도 예측·로그 기록엔 영향 없도록 조용히 무시."""
    try:
        _SUMMARY_ITEMS.append(kwargs)
    except Exception as e:
        print(f"⚠️ 요약 항목 등록 실패(무시): {e}")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return str(o)


def finish_report(log_name, horizon_days, report_text):
    """
    상세 리포트는 reports/detail_<이름>.txt 로 저장하고(텔레그램은 SEND_DETAIL=1 일 때만 전송),
    요약용 구조화 결과는 reports/summary_<이름>.json 으로 저장한다.
    기존엔 6개 메시지(약 14,000자)가 따로 날아와 읽기 어려웠음 → 매일 1개 요약(daily_summary.py)로 대체.
    """
    try:
        os.makedirs(REPORT_DIR, exist_ok=True)
        with open(os.path.join(REPORT_DIR, f"detail_{log_name}.txt"), 'w', encoding='utf-8') as f:
            f.write(report_text)
        payload = {'log': log_name, 'horizon_days': horizon_days,
                   'generated_at': datetime.now().isoformat(), 'items': list(_SUMMARY_ITEMS)}
        with open(os.path.join(REPORT_DIR, f"summary_{log_name}.json"), 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=1, default=_json_default)
    except Exception as e:
        print(f"⚠️ 리포트 저장 실패(무시): {e}")
    if os.environ.get('SEND_DETAIL') == '1':
        send_telegram(report_text)


def make_prediction_record(run_date, benchmark, ticker, horizon_days, up_prob, raw_up_prob, price,
                            near_earnings=False, price_date=None):
    predicted_label = (1 if up_prob >= 50 else 0) if (up_prob is not None and np.isfinite(up_prob)) else np.nan   # v27: NaN 확률을 '하락'으로 기록하지 않음
    # v25: 목표일은 '예측에 쓴 마지막 종가일 + h영업일'. (이전엔 실행일 기준이라 하루씩 밀려 채점됐음)
    if price_date is not None and not pd.isna(price_date):
        pdate = pd.Timestamp(price_date).normalize()
    else:
        pdate = _fallback_price_date(run_date)
    target_date = pdate + BDay(horizon_days)
    return {'price_date': pdate,
        'run_date': run_date, 'benchmark': benchmark, 'ticker': ticker,
        'horizon_days': horizon_days, 'predicted_label': predicted_label,
        'predicted_up_prob': up_prob, 'raw_up_prob': raw_up_prob,
        'price_at_prediction': price, 'target_date': target_date,
        'actual_price': np.nan, 'actual_return': np.nan,
        'actual_label': np.nan, 'correct': np.nan,
        'near_earnings': near_earnings,
    }


# ============================================================
# ============================================================
# 8) 텔레그램 알림
# ============================================================
_TG_TAG = re.compile(r"</?(?:b|i|u|s|em|strong|code|pre|a)(?:\s[^>]*)?>", re.I)
TG_MAX = 4000   # 텔레그램 한도(4096)보다 여유


def _tg_plain(msg):
    """HTML 모드 실패 시 재전송용: 서식 태그만 제거(traceback의 <module> 같은 꺾쇠는 보존)하고 엔티티를 되돌린다."""
    return html.unescape(_TG_TAG.sub('', str(msg)))


def _tg_split(msg, limit=TG_MAX):
    """한도를 넘는 메시지를 줄 경계에서 나눈다. 청크마다 열린 <b>/<i>/<code> 태그를 닫고 다음 청크에서 다시 연다."""
    msg = str(msg)
    if len(msg) <= limit:
        return [msg]
    parts, cur = [], ''
    for line in msg.split('\n'):
        while len(line) > limit:
            if cur:
                parts.append(cur); cur = ''
            parts.append(line[:limit]); line = line[limit:]
        add = line if not cur else '\n' + line
        if cur and len(cur) + len(add) > limit:
            parts.append(cur); cur = line
        else:
            cur += add
    if cur:
        parts.append(cur)
    out, carry = [], []
    for part in parts:
        text = ''.join(f'<{t}>' for t in carry) + part
        stack = list(carry)
        for m in re.finditer(r'<(/?)(b|i|u|s|code|pre)>', part):
            if m.group(1):
                if stack and stack[-1] == m.group(2):
                    stack.pop()
            else:
                stack.append(m.group(2))
        carry = stack
        out.append(text + ''.join(f'</{t}>' for t in reversed(stack)))
    return out


def send_telegram(message):
    """
    v27: 응답 코드를 확인한다. HTML 파싱 실패(400) 등으로 거부되면 태그를 제거한 평문으로 한 번 더 보낸다
    (이전엔 400이어도 조용히 유실됐고, 에러 알림에 traceback의 <module> 같은 꺾쇠가 있으면 알림 자체가 사라질 수 있었음).
    반환: 모든 조각이 전달되면 True.
    """
    token = os.environ.get('TELEGRAM_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        print("⚠️ TELEGRAM_TOKEN / TELEGRAM_CHAT_ID 미설정 — 콘솔에만 출력합니다.")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    def _post(text, html_mode):
        data = {"chat_id": chat_id, "text": text}
        if html_mode:
            data["parse_mode"] = "HTML"
        r = requests.post(url, data=data, timeout=10)
        code = getattr(r, 'status_code', 200)
        return code == 200, code, getattr(r, 'text', '')

    all_ok = True
    for chunk in _tg_split(message):
        try:
            ok, code, body = _post(chunk, True)
            if not ok:
                print(f"⚠️ 텔레그램 HTML 전송 거부({code}): {str(body)[:160]} — 평문으로 재전송")
                ok, code, body = _post(_tg_plain(chunk), False)
                if not ok:
                    print(f"❌ 텔레그램 평문 재전송도 실패({code}): {str(body)[:160]}")
            all_ok &= ok
        except Exception as e:
            print(f"⚠️ 텔레그램 전송 실패: {e}")
            all_ok = False
    return all_ok


def direction_label(up_prob, down_prob):
    if up_prob >= 65:
        return "🟢 강한 상승 우세"
    elif up_prob >= 55:
        return "🟡 약한 상승 우세"
    elif down_prob >= 65:
        return "🔴 강한 하락 우세"
    elif down_prob >= 55:
        return "🟠 약한 하락 우세"
    else:
        return "⚪ 방향성 불분명 (50%대 근접)"


def edge_label(edge):
    """
    edge = 보정확률 - 기준선(base rate).
    0에 가까우면 모델이 그 시점 정보가 아니라 과거 평균(강세장 편향 등)을
    재현하고 있을 가능성이 높다는 뜻.
    """
    if abs(edge) < 3:
        return "⚠️ 엣지 거의 없음 (기준선과 거의 동일 — 그냥 과거 평균을 재현 중일 수 있음)"
    elif abs(edge) < 7:
        return "△ 약한 엣지"
    else:
        return "✅ 뚜렷한 엣지"
