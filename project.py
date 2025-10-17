import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from scipy import stats

from googleapiclient.discovery import build 
import time 
import csv 
from datetime import dateime, timedelta 

API_KEY = "AIzaSyA86bBmTuMc304g-T2BQaqLbEUQloVsvUg"
youtube = build("youtube", "v3", developerKey= API_KEY)

daily_limit = 10000
safe_buffer = 500
quota_used = 0

with open("travel_vlogs_2017_2019.csv", "w", newline= "utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Title", "Channel", "Publish Date", "Video URL", "View Count"])

    next_page_token = None
    total_videos = 0

    while True:
        #stop before exceeding quota
        if quota_used + 150 > daily_limit - safe_buffer:
            print("quota limit reached. pausing until tomorrow...")
            tomorrow = datetime.now() + timedelta(days=1)
