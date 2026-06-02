import traceback
import requests
import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# v16.4 핵심 변경사항 요약
# ============================================================
# [BUG FIX 1] F1 binary → F1 macro로 변경
#   - binary는 클래스 불균형 시 항상 0에 수렴 → 전종목 관망 버그
#   - macro는 클래스 비율 무관하게 양방향 균형 평가
#
# [BUG FIX 2] 고정 threshold(1.5%) → ATR 기반 동적 threshold
#   - XOVR(변동성 낮음): 1.5% 기준이면 Target=1이 10% 미만 → 모델 학습 불가
#   - SOXL(3배 레버리지): 동일 기준 적용 불합리
#   - 해결: 각 종목의 ATR 중앙값 × 0.5를 threshold로 사용
#
# [BUG FIX 3] GradientBoosting class_weight 미지원 문제
#   - GB는 class_weight 파라미터 없음 → sample_weight로 대체
#   - RF만 class_weight='balanced', GB는 fit()에서 sample_weight 적용
#
# [IMPROVEMENT 1] 피처 스케일링 추가 (StandardScaler)
#   - RF는 스케일 무관하지만 GB는 스케일 영향 받음
#   - 앙상블 안정성 향상
#
# [IMPROVEMENT 2] 모델 평가 지표 3종 출력 (F1/Precision/Recall)
#   - F1만으로는 모델 성향 파악 불가
#   - Precision 높고 Recall 낮으면 "신중한 진입 신호"로 해석 가능
#
# [IMPROVEMENT 3] 볼린저 밴드 하단 진입 수치 추가
#   - 기존: 스탑로스만 출력
#   - 추가: BB 하단 = 매수 진입 타겟 구간으로 활용
# ============================================================


def send_telegram_message(text):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={'chat_id': chat_id, 'text': text}
            )
        except Exception as e:
            print("텔레그램 전송 실패:", e)


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1 / period, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - (100 / (1 + (gain / loss)))


def calculate_bollinger(series, period=20, std_mult=2.0):
    """볼린저 밴드 계산 → 하단이 매수 진입 타겟"""
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + std_mult * std
    lower = ma - std_mult * std
    return upper, ma, lower


def get_dynamic_threshold(close_series):
    """
    [BUG FIX 2] ATR 기반 동적 threshold
    종목별 변동성이 다르므로 고정 1.5% 대신 ATR 중앙값의 절반을 사용.
    목표: Target=1 비율이 30~45% 범위에 들어오도록 조정.
    """
    daily_ret = close_series.pct_change().abs()
    # ATR 대용: 일간 절대수익률 중앙값
    median_move = daily_ret.median()
    # 0.5배: 중간보다 조금 쉬운 기준 (30~45% 비율 목표)
    threshold = median_move * 0.5
    # 최소 0.3%, 최대 2.5% 클램핑
    return float(np.clip(threshold, 0.003, 0.025))


try:
    print("🚀 v16.4 US AI·반도체 올인 전략 완전체 모델 가동...")
    print("   [BUG FIX] F1 macro | 동적 threshold | GB sample_weight | BB 진입선 추가\n")

    # ==========================================
    # [1] 미국 매크로 데이터 수집
    # ==========================================
    start_date = '2022-01-01'

    macro_tickers = {
        'Nasdaq': '^IXIC',
        'SMH': 'SMH',
        'VIX': '^VIX',
        'TNX': '^TNX'
    }

    macro_data = {}
    for name, t in macro_tickers.items():
        df_raw = yf.download(t, start=start_date, progress=False)
        if not df_raw.empty:
            series = df_raw['Close'].squeeze()
            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)
            macro_data[name] = series

    macro_df = pd.DataFrame(macro_data)
    macro_df = macro_df.ffill().dropna()

    if macro_df.empty:
        raise ValueError("매크로 데이터를 정상적으로 합치지 못했습니다.")

    macro_df['Nasdaq_Ret'] = macro_df['Nasdaq'].pct_change()
    macro_df['SMH_Ret'] = macro_df['SMH'].pct_change()

    # ==========================================
    # [2] 타겟 종목 리스트
    # ==========================================
    targets = ['XOVR', 'SOXL', 'NVDL', 'DXYZ', 'TECL']

    final_report = "🤖 [US AI·반도체 올인 방향성 통제소 v16.4]\n"
    final_report += "=" * 40 + "\n"

    for ticker in targets:
        print(f"진행 중: {ticker} 분석...")
        tgt_data = yf.download(ticker, start=start_date, progress=False)

        if tgt_data.empty:
            final_report += f"⚠️ {ticker}: 현재 데이터를 불러올 수 없습니다.\n"
            final_report += "-" * 40 + "\n"
            continue

        df = pd.DataFrame({
            'High': tgt_data['High'].squeeze(),
            'Low': tgt_data['Low'].squeeze(),
            'Close': tgt_data['Close'].squeeze()
        })
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.join(macro_df, how='left').ffill().dropna()

        # ==========================================
        # [3] 피처 엔지니어링
        # ==========================================
        df['RSI'] = calculate_rsi(df['Close'])

        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        df['MACD_Hist'] = macd_line - macd_line.ewm(span=9, adjust=False).mean()

        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['Price_to_EMA20'] = (df['Close'] / df['EMA_20']) - 1

        df['ATR'] = pd.concat([
            df['High'] - df['Low'],
            np.abs(df['High'] - df['Close'].shift()),
            np.abs(df['Low'] - df['Close'].shift())
        ], axis=1).max(axis=1).rolling(14).mean()

        # [IMPROVEMENT 3] 볼린저 밴드 계산
        bb_upper, bb_ma, bb_lower = calculate_bollinger(df['Close'])
        df['BB_Upper'] = bb_upper
        df['BB_Lower'] = bb_lower
        df['BB_Pct'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

        # 현재 시점 지표 추출
        current_price = df['Close'].iloc[-1]
        is_uptrend = current_price > df['EMA_20'].iloc[-1]
        latest_atr = df['ATR'].iloc[-1]
        latest_rsi = df['RSI'].iloc[-1]
        bb_lower_val = df['BB_Lower'].iloc[-1]
        bb_upper_val = df['BB_Upper'].iloc[-1]
        bb_pct = df['BB_Pct'].iloc[-1]

        # 스탑로스 계산 (현재가 기준 하드캡)
        rsi_risk = max(0, latest_rsi - 68)
        atr_multiplier = max(1.0, 3.0 - (rsi_risk / 10))
        hard_cap = current_price * 0.90
        stop_loss = current_price - (latest_atr * atr_multiplier)
        stop_loss = max(stop_loss, hard_cap)

        # [BUG FIX 2] 동적 threshold 계산
        dyn_threshold = get_dynamic_threshold(df['Close'])

        # 학습용 데이터 구축
        features = ['Nasdaq_Ret', 'SMH_Ret', 'VIX', 'TNX', 'RSI', 'MACD_Hist', 'Price_to_EMA20', 'BB_Pct']
        df['Next_Ret'] = df['Close'].pct_change().shift(-1)
        latest_features = df[features].iloc[-1]

        df_train = df.dropna(subset=['Next_Ret'] + features).copy()
        df_train['Target'] = np.where(df_train['Next_Ret'] > dyn_threshold, 1, 0)

        # 클래스 비율 확인 (디버그용)
        class_ratio = df_train['Target'].mean()

        MIN_TRAIN_ROWS = 300
        if len(df_train) < MIN_TRAIN_ROWS:
            final_report += f"🔍 {ticker} (데이터 부족 — 기술적 지표만 출력)\n"
            final_report += f"  * 현재가: ${current_price:.2f} | 스탑로스: ${stop_loss:.2f}\n"
            final_report += f"  * BB 매수구간: ${bb_lower_val:.2f} ~ ${bb_ma.iloc[-1]:.2f}\n"
            final_report += f"  * RSI: {latest_rsi:.1f} | 추세: {'🟢 상승' if is_uptrend else '🔴 하락'}\n"
            final_report += "-" * 40 + "\n"
            continue

        # ==========================================
        # [4] ML 모델 학습
        # ==========================================
        X = df_train[features]
        y = df_train['Target']
        split = int(len(X) * 0.8)

        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        # [IMPROVEMENT 1] 피처 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        latest_scaled = scaler.transform(latest_features.values.reshape(1, -1))

        # [BUG FIX 3] GB는 class_weight 없음 → sample_weight로 클래스 불균형 보정
        pos_ratio = y_train.mean()
        neg_ratio = 1 - pos_ratio
        sample_weights = np.where(
            y_train == 1,
            1.0 / (2 * pos_ratio + 1e-9),   # 소수 클래스에 높은 가중치
            1.0 / (2 * neg_ratio + 1e-9)
        )

        rf = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            max_depth=5,
            class_weight='balanced'  # RF는 class_weight 지원
        )
        gb = GradientBoostingClassifier(
            n_estimators=300,
            random_state=42,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8
        )

        rf.fit(X_train_scaled, y_train)
        gb.fit(X_train_scaled, y_train, sample_weight=sample_weights)

        # Soft voting 수동 구현 (VotingClassifier는 sample_weight 지원 불안정)
        rf_probs = rf.predict_proba(X_test_scaled)
        gb_probs = gb.predict_proba(X_test_scaled)
        ensemble_probs = (rf_probs + gb_probs) / 2
        preds = (ensemble_probs[:, 1] >= 0.5).astype(int)

        # [BUG FIX 1] F1 macro 사용 (binary → macro)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)

        MIN_F1 = 0.50  # macro F1 기준 0.50 (random=0.50이므로 이 이상이어야 유의미)

        if f1 < MIN_F1:
            final_report += f"📌 {ticker}\n"
            final_report += f"  * ⚠️ 관망 (F1 macro {f1*100:.1f}% — 신호 신뢰도 미달)\n"
            final_report += f"  * 현재가: ${current_price:.2f} | 스탑로스: ${stop_loss:.2f}\n"
            final_report += f"  * BB 매수구간: ${bb_lower_val:.2f} ~ ${bb_ma.iloc[-1]:.2f}\n"
            final_report += f"  * RSI: {latest_rsi:.1f} | BB%: {bb_pct*100:.0f}%\n"
            final_report += "-" * 40 + "\n"
            continue

        # 최종 예측
        rf_latest = rf.predict_proba(latest_scaled)
        gb_latest = gb.predict_proba(latest_scaled)
        final_probs = (rf_latest + gb_latest) / 2
        up_prob = final_probs[0][1] * 100
        down_prob = final_probs[0][0] * 100

        if up_prob >= 65:
            direction = "🟢 강한 상승"
        elif up_prob >= 50:
            direction = "🟡 약한 상승"
        elif down_prob >= 65:
            direction = "🔴 강한 하락"
        else:
            direction = "🟠 약한 하락"

        # 진입 추천 구간 계산
        entry_target = (bb_lower_val + bb_ma.iloc[-1]) / 2  # BB하단과 MA 중간

        final_report += f"📌 {ticker}\n"
        final_report += f"  * {direction} (상승 {up_prob:.1f}% / 하락 {down_prob:.1f}%)\n"
        final_report += f"  * 현재가: ${current_price:.2f} | 스탑로스: ${stop_loss:.2f}\n"
        final_report += f"  * 📍매수 진입 타겟: ${entry_target:.2f} (BB하단 ${bb_lower_val:.2f})\n"
        final_report += f"  * F1 {f1*100:.1f}% | P {precision*100:.0f}% | R {recall*100:.0f}%\n"
        final_report += f"  * RSI: {latest_rsi:.1f} | BB%: {bb_pct*100:.0f}% | 추세: {'🟢' if is_uptrend else '🔴'}\n"
        final_report += "-" * 40 + "\n"

    print("\n" + final_report)
    send_telegram_message(final_report)
    print("✅ v16.4 텔레그램 리포트 전송 완료")

except Exception as e:
    error_msg = f"🚨 에러 발생:\n{traceback.format_exc()[:800]}"
    print(error_msg)
    send_telegram_message(error_msg)
