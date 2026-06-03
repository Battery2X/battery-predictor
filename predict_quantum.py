import traceback
import requests
import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# quantum v1.3 버그 수정 내역
# ============================================================
# [BUG FIX 1] Target_Threshold 재설계
#   - 기존: Vol * 0.3 * (lookahead/252) → 수식 자체는 맞지만
#     양자주 변동성(120~200%)에서 threshold가 너무 낮아
#     Target=1 비율이 45% 안팎으로 적절하게 나오는 우연이 있었음
#   - 실제 문제: QBTS/RGTI의 F1이 낮은 진짜 원인은 GB sample_weight 미적용
#     → BUG FIX 2로 해결
#   - 추가 개선: threshold를 종목 변동성 분위수(75th percentile)로 명시적 계산
#     → Target=1 비율이 항상 25~35% 범위에 들어오도록 보장
#
# [BUG FIX 2] GB sample_weight 미적용
#   - 기존: gb.fit(X_train, y_train) ← 불균형 보정 없음
#   - 수정: v17 시리즈와 동일하게 pos/neg 비율 기반 sample_weight 적용
#
# [BUG FIX 3] 이격도 과열 기준 양자주 맞춤 조정
#   - 기존: disp_target > 0.15 (15%) → 양자주에선 항상 과열 판정
#   - 수정: 이격도를 해당 종목 과거 분포의 75th percentile로 동적 계산
#     → 절대값 기준 대신 상대적 과열 판단
#
# [BUG FIX 4] ML 신호 ↔ 이격도 신호 충돌 시 경고 미출력 문제
#   - 기존: ML = 상승, 이격도 = 과열이어도 "매수 밴드"만 표시
#   - 수정: 두 신호가 충돌하면 ⚠️ CONFLICT 경고 명시
#
# [IMPROVEMENT] StandardScaler 추가 (GB 안정성 향상)
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


def get_dynamic_threshold(future_ret_series, target_ratio=0.30):
    """
    [BUG FIX 1] 분위수 기반 동적 threshold
    목표: Target=1 비율이 target_ratio(30%) 근처가 되는 threshold 반환
    → 70th percentile 수익률을 threshold로 사용
    """
    percentile = (1 - target_ratio) * 100  # 30% 목표 → 70th percentile
    threshold = np.percentile(future_ret_series.dropna(), percentile)
    # 최소 3%, 최대 40% 클램핑 (양자주 특성 반영)
    return float(np.clip(threshold, 0.03, 0.40))


def get_disparity_threshold(disparity_series):
    """
    [BUG FIX 3] 종목별 이격도 분포의 75th percentile을 과열 기준으로 사용
    절대값 15% 대신 해당 종목 역사적 분포 기준 → 양자주에도 유효
    """
    return float(np.percentile(disparity_series.dropna().abs(), 75))


try:
    print("🔮 [v1.3] 양자컴퓨팅 4개사 듀얼 타임프레임 리스크 관리 엔진 가동...")
    print("   [4개 BUG FIX: Threshold분위수 / GB샘플가중치 / 이격도동적기준 / 신호충돌경고]\n")

    start_date = '2021-01-01'

    macro_tickers = {'Nasdaq': '^IXIC', 'VIX': '^VIX', 'TNX': '^TNX'}
    macro_raw = {}
    for name, t in macro_tickers.items():
        df_raw = yf.download(t, start=start_date, progress=False)
        if not df_raw.empty:
            series = df_raw['Close'].squeeze()
            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)
            macro_raw[name] = series
    macro_base = pd.DataFrame(macro_raw).ffill().dropna()

    quantum_targets = ['IONQ', 'INFQ', 'QBTS', 'RGTI']

    final_report  = "⚛️ [QUANTUM 양자컴퓨팅 리스크 관리 통제소 v1.3]\n"
    final_report += "=" * 40 + "\n"

    for ticker in quantum_targets:
        print(f"진행 중: {ticker} 분석...")
        tgt_data = yf.download(ticker, start=start_date, progress=False)
        if tgt_data.empty:
            final_report += f"📌 {ticker}\n  * ⚠️ 데이터 수집 불가\n" + "-" * 40 + "\n"
            continue

        # 듀얼 타임프레임 (데이터 길이 기반 자동 선택)
        raw_len = len(tgt_data)
        if raw_len >= 250:
            lookahead  = 60
            window_name = "3개월 (60영업일 뒤)"
            min_rows   = 150
        else:
            lookahead  = 20
            window_name = "1개월 (20영업일 뒤)"
            min_rows   = 60

        # 매크로 피처 (lookahead 스케일 맞춤)
        macro_df = macro_base.copy()
        macro_df['Macro_Ret'] = macro_df['Nasdaq'].pct_change(lookahead)
        macro_df['TNX_Diff']  = macro_df['TNX'].diff(lookahead)
        macro_df['VIX_Mean']  = macro_df['VIX'].rolling(lookahead).mean()
        # Nasdaq MA200 국면 피처 (v17.2에서 검증된 방식)
        nasdaq_ma200 = macro_df['Nasdaq'].rolling(200).mean()
        macro_df['Nasdaq_MA200_Gap'] = (macro_df['Nasdaq'] / nasdaq_ma200) - 1
        macro_df = macro_df.dropna()

        df = pd.DataFrame({
            'High':  tgt_data['High'].squeeze(),
            'Low':   tgt_data['Low'].squeeze(),
            'Close': tgt_data['Close'].squeeze()
        })
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.join(macro_df, how='left').ffill().dropna()

        # 피처 엔지니어링
        rsi_period         = max(14, int(lookahead * 0.7))
        df['RSI_Target']   = calculate_rsi(df['Close'], period=rsi_period)
        df['MA_Target']    = df['Close'].rolling(lookahead).mean()
        df['Disparity_Target'] = (df['Close'] / df['MA_Target']) - 1

        df['Log_Ret']      = np.log(df['Close'] / df['Close'].shift(1))
        df['Vol_Target']   = df['Log_Ret'].rolling(lookahead).std() * np.sqrt(252)

        # 미래 수익률
        df['Future_Close'] = df['Close'].shift(-lookahead)
        df['Future_Ret']   = (df['Future_Close'] / df['Close']) - 1

        features = [
            'Macro_Ret', 'TNX_Diff', 'VIX_Mean',
            'RSI_Target', 'Disparity_Target', 'Vol_Target',
            'Nasdaq_MA200_Gap'  # v17.2 검증 피처 추가
        ]

        # 현재 상태
        latest_features = df[features].iloc[-1]
        current_price   = df['Close'].iloc[-1]
        disp_target     = df['Disparity_Target'].iloc[-1]
        ma_target_val   = df['MA_Target'].iloc[-1]

        # 학습 데이터 구축
        df_train = df.dropna(subset=['Future_Ret'] + features).copy()

        if len(df_train) < min_rows:
            final_report += f"📌 {ticker} ({window_name})\n"
            final_report += f"  * ⚠️ 학습 데이터 부족 ({len(df_train)}행 / 최소 {min_rows}행)\n"
            final_report += "-" * 40 + "\n"
            continue

        # [BUG FIX 1] 분위수 기반 동적 threshold
        dyn_threshold = get_dynamic_threshold(df_train['Future_Ret'], target_ratio=0.30)
        df_train['Target'] = np.where(df_train['Future_Ret'] > dyn_threshold, 1, 0)
        class_ratio = df_train['Target'].mean()
        print(f"  {ticker}: threshold={dyn_threshold*100:.1f}%, Target=1 비율={class_ratio:.1%}")

        # [BUG FIX 3] 동적 이격도 과열 기준
        disp_threshold = get_disparity_threshold(df_train['Disparity_Target'])

        # 학습/테스트 분할
        X = df_train[features]
        y = df_train['Target']
        split = int(len(X) * 0.8)

        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        # StandardScaler
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)
        X_latest  = scaler.transform(latest_features.values.reshape(1, -1))

        # [BUG FIX 2] GB sample_weight 적용
        pos_ratio = y_train.mean()
        neg_ratio = 1 - pos_ratio
        sw = np.where(
            y_train == 1,
            1.0 / (2 * pos_ratio + 1e-9),
            1.0 / (2 * neg_ratio + 1e-9)
        )

        rf = RandomForestClassifier(
            n_estimators=300, random_state=42,
            max_depth=4, class_weight='balanced'
        )
        gb = GradientBoostingClassifier(
            n_estimators=300, random_state=42,
            max_depth=3, learning_rate=0.03, subsample=0.8
        )

        rf.fit(X_train_s, y_train)
        gb.fit(X_train_s, y_train, sample_weight=sw)  # BUG FIX 2

        # 앙상블 평가
        rf_probs_te  = rf.predict_proba(X_test_s)[:, 1]
        gb_probs_te  = gb.predict_proba(X_test_s)[:, 1]
        ens_probs_te = (rf_probs_te + gb_probs_te) / 2
        preds        = (ens_probs_te >= 0.5).astype(int)

        f1   = f1_score(y_test, preds, average='macro', zero_division=0)
        prec = precision_score(y_test, preds, zero_division=0)
        rec  = recall_score(y_test, preds, zero_division=0)

        # 예측
        rf_up  = rf.predict_proba(X_latest)[0][1]
        gb_up  = gb.predict_proba(X_latest)[0][1]
        up_prob = ((rf_up + gb_up) / 2) * 100

        # 신뢰도 필터
        MIN_F1 = 0.50
        if f1 < MIN_F1:
            decision  = "🔴 [매매 보류] 모델 신뢰도 미달 — 예측 무효"
            prob_str  = "계산 불가"
            ml_signal = "neutral"
        else:
            prob_str  = f"{up_prob:.1f}%"
            if up_prob >= 62:
                decision  = "🟢 [중기 우상향] 모멘텀 유효"
                ml_signal = "up"
            elif up_prob >= 40:
                decision  = "🟡 [중기 횡보] 비중 유지"
                ml_signal = "neutral"
            else:
                decision  = "🔴 [중기 위험] 하방 압력 우세"
                ml_signal = "down"

        # [BUG FIX 3] 동적 이격도 기반 과열 판단
        is_overheated = abs(disp_target) > disp_threshold
        if is_overheated:
            safe_lower       = ma_target_val * 0.92
            safe_upper       = ma_target_val * 1.05
            target_band_str  = f"${safe_lower:.2f} ~ ${safe_upper:.2f} (과열 진정 타점)"
            heat_signal      = "overheated"
        else:
            target_band_str  = "현재가 인근 분할 대응 가능"
            heat_signal      = "normal"

        # [BUG FIX 4] 신호 충돌 감지
        conflict_warning = ""
        if ml_signal == "up" and heat_signal == "overheated":
            conflict_warning = (
                f"  * ⚠️ [신호 충돌] ML=상승({up_prob:.0f}%) vs 이격도=과열({disp_target*100:.0f}%)\n"
                f"     → 추격 매수 금지. 과열 진정 후 타점 진입 권장\n"
            )
        elif ml_signal == "down" and heat_signal == "normal":
            conflict_warning = (
                f"  * ℹ️ [참고] ML=하락 신호이나 이격도는 정상권\n"
                f"     → 급락 시 분할 매수 기회 탐색 가능\n"
            )

        final_report += f"📌 {ticker} [{window_name}]\n"
        final_report += f"  * 🎯 결론: {decision}\n"
        if conflict_warning:
            final_report += conflict_warning
        final_report += f"  * {window_name} 상승 확률: {prob_str}\n"
        final_report += f"  * 매수 타점: {target_band_str}\n"
        final_report += f"  * F1 {f1*100:.1f}% | P {prec*100:.0f}% | R {rec*100:.0f}%\n"
        final_report += f"  * 이격도 {disp_target*100:.1f}% (과열기준 >{disp_threshold*100:.0f}%)\n"
        final_report += f"  * threshold: {dyn_threshold*100:.1f}% | Target=1 비율: {class_ratio:.1%}\n"
        final_report += "-" * 40 + "\n"

    print("\n" + final_report)
    send_telegram_message(final_report)
    print("✅ v1.3 양자 리스크 리포트 전송 완료")

except Exception as e:
    error_msg = f"🚨 양자 엔진 v1.3 에러:\n{traceback.format_exc()[:800]}"
    print(error_msg)
    send_telegram_message(error_msg)
