import traceback
import requests
import os
import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 텔레그램 전송을 위한 함수 분리
def send_telegram_message(text):
    # GitHub Secrets에서 주입한 환경변수 가져오기
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        print("❌ 환경변수에 TELEGRAM_TOKEN 또는 TELEGRAM_CHAT_ID가 없습니다.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': text}
    requests.post(url, data=payload)

# 메인 로직 시작
try:
    print("🚀 예측 모델 실행 시작...")
    
    # ==========================================
    # 여기에 기존 v4.0 모델 코드를 그대로 넣으세요.
    # (데이터 수집, 피처 엔지니어링, 모델 학습, 예측 등)
    # ==========================================
    
    # --- [이 아래는 예측 완료 후 결과 메시지를 만드는 부분 예시] ---
    # if prediction[0] == 1:
    #     result_msg = f"📈 상승 예상 (+0.3% 돌파 확률: {prob[1]*100:.1f}%)"
    # else:
    #     result_msg = f"📉 하락/횡보 예상 (확률: {prob[0]*100:.1f}%)"
    #
    # message = f"🤖 [TIGER 2차전지 TOP10 AI 예측]\n{result_msg}"

    # 정상 완료 시 텔레그램 전송
    send_telegram_message(message)
    print("✅ 실행 및 텔레그램 전송 완료!")

except Exception as e:
    # 에러가 발생하면 상세 로그를 텍스트로 변환
    error_msg = traceback.format_exc()
    print("❌ 치명적인 에러 발생:\n", error_msg)
    
    # 텔레그램으로 에러 내용 바로 쏘기 (너무 길면 잘릴 수 있으니 1000자까지만)
    send_telegram_message(f"🚨 [긴급] 예측 봇 실행 실패!\n\n{error_msg[:1000]}")
