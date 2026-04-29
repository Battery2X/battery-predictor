import os
import FinanceDataReader as fdr
import yfinance as yf
...

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
USE_TELEGRAM     = True
