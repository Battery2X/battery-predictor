import os
import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
import warnings
warnings.filterwarnings('ignore')

TELEGRAM_TOKEN   = os.environ.get("8738275971:AAF-cUJOYFtLRFSyF_fOs-T61BYquYRgV_4", "")
TELEGRAM_CHAT_ID = os.environ.get("8731055974", "")
USE_TELEGRAM     = True
