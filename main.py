name: AI Semiconductor Market Analysis

on:
  # 매일 자정(UTC 기준)에 자동 실행 (필요에 따라 주기 변경 가능)
  schedule:
    - cron: '0 0 * * *'
  # GitHub 페이지에서 수동으로 실행할 수 있는 버튼 활성화
  workflow_dispatch:

jobs:
  run-analysis:
    runs-on: ubuntu-latest
    steps:
      # 1. 코드 체크아웃: v4에서 v7으로 업데이트 (Node.js 24 지원 및 보안 강화)
      - name: Checkout Repository
        uses: actions/checkout@v7

      # 2. 파이썬 환경 설정: v5에서 v6으로 업데이트 (Node.js 24 지원)
      - name: Set up Python
        uses: actions/setup-python@v6
        with:
          python-version: '3.13' # 사용하시는 파이썬 버전에 맞게 수정 가능
          
      # 3. 파이썬 의존성 패키지 설치
      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install yfinance pandas numpy scikit-learn requests

      # 4. 분석 스크립트 실행
      - name: Run Analysis Script
        run: |
          python main.py
