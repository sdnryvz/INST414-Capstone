
# To push to git use these three steps:
# git add .
# git commit -m "Describe what you changed"
# git push origin main 

# Pulling API

from googleapiclient.discovery import build
import pandas as pd
import os
import time
import csv
from datetime import date, datetime, timedelta
import re

from inst414_capstone.config import API_KEY

OUTPUT_FILE = "travel_vlogs_2017_2019_v2.csv"
LOG_FILE = "dataset_log.txt"
PROGRESS_FILE = "progress.txt"

DAILY_QUOTA_LIMIT = 10000   # stop before full limit
SAFE_BUFFER = 500
quota_used = 0

queries = [ 
    "travel vlog",
    "travel vlog europe",
    "travel vlog asia",
    "solo travel vlog",
    "adventure vlog"
]

date_ranges = [
    ("2017-01-01T00:00:00Z", "2017-06-30T23:59:59Z"),
    ("2017-07-01T00:00:00Z", "2017-12-31T23:59:59Z"),
    ("2018-01-01T00:00:00Z", "2018-06-30T23:59:59Z"),
    ("2018-07-01T00:00:00Z", "2018-12-31T23:59:59Z"),
    ("2019-01-01T00:00:00Z", "2019-06-30T23:59:59Z"),
    ("2019-07-01T00:00:00Z", "2019-12-31T23:59:59Z")
]

# ------------------------------
# INITIAL SETUP
# ------------------------------
youtube = build("youtube", "v3", developerKey=API_KEY)

# Load progress (nextPageToken)
try:
    with open(PROGRESS_FILE, "r") as f:
        next_page_token = f.read().strip()
        if next_page_token == "":
            next_page_token = None
        print(f"🔁 Resuming from saved page token: {next_page_token}")
except FileNotFoundError:
    next_page_token = None
    print("🚀 Starting fresh search.")

# Load existing data to prevent duplicates
seen_videos = set()
if os.path.exists(OUTPUT_FILE):
    existing_data = pd.read_csv(OUTPUT_FILE)
    seen_videos = set(existing_data["Video URL"].tolist())
    print(f"🧠 Loaded {len(seen_videos)} previously collected videos.")
else:
    # Create new CSV file with headers if it doesn't exist
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Title", "Channel", "Publish Date", "Video URL", "View Count", "Likes", "Comments", "Duration"])

total_videos = len(seen_videos)

# ------------------------------
# MAIN LOOP
# ------------------------------
for query in queries:
    for start_date, end_date in date_ranges:
        print(f"\n🔍 Searching for '{query}' from {start_date[:10]} to {end_date[:10]}...")
        next_page_token = None
    while True:
        # Check quota before each batch
        if quota_used + 150 > DAILY_QUOTA_LIMIT - SAFE_BUFFER:
            print("⚠️ Quota limit reached. Pausing until tomorrow...")
            tomorrow = datetime.now() + timedelta(days=1)
            midnight = datetime.combine(tomorrow, datetime.min.time())
            sleep_seconds = (midnight - datetime.now()).seconds + 60
            print(f"🕒 Sleeping for {sleep_seconds/3600:.2f} hours...")
            time.sleep(sleep_seconds)
            quota_used = 0  # reset after midnight

        # Search request (100 units)
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            order="date",
            maxResults=50,
            publishedAfter=start_date,
            publishedBefore=end_date,
            pageToken=next_page_token
        )
        response = request.execute()
        quota_used += 100

        # Collect video IDs for stats call
        video_ids = [item["id"]["videoId"] for item in response["items"]]
        if not video_ids:
            print("✅ No more videos found.")
            break

        # Get statistics (1 per video)
        stats_request = youtube.videos().list(
            part="statistics",
            id=",".join(video_ids)
        )
        stats_response = stats_request.execute()
        quota_used += len(video_ids)

        def parse_duration(duration):
            match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
            if not match:
                return 0
            hours, minutes, seconds = match.groups()
            return int(hours or 0) * 3600 + int(minutes or 0) * 60 + int(seconds or 0)

        stats_dict = {}
        for item in stats_response["items"]:
            vid = item["id"]
            stats = item.get("statistics", {})
            details = item.get("contentDetails", {})

            stats_dict[vid] = {
                "views": stats.get("viewCount", "0"),
                "likes": stats.get("likeCount", "0"),
                "comments": stats.get("commentCount", "0"),
                "duration": parse_duration(details.get("duration", "PT0S"))
            }

        # Write to CSV, skip duplicates
        with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for item in response["items"]:
                title = item["snippet"]["title"]
                channel = item["snippet"]["channelTitle"]
                publish_date = item["snippet"]["publishedAt"]
                video_id = item["id"]["videoId"]
                url = f"https://www.youtube.com/watch?v={video_id}"

                if url in seen_videos:
                    continue  # skip duplicates
                video_stats = stats_dict.get(video_id, {})
                view_count = video_stats.get("views", "0")
                like_count = video_stats.get("likes", "0")
                comment_count = video_stats.get("comments", "0")
                duration_sec = video_stats.get("duration", "0")

                writer.writerow([title, channel, publish_date, url, view_count, like_count, comment_count, duration_sec])

                seen_videos.add(url)
                total_videos += 1

                # Progress counter (every 100 videos)
                if total_videos % 100 == 0:
                    print(f"📈 {total_videos} total unique videos collected so far")

    # Save progress
    next_page_token = response.get("nextPageToken")
    with open(PROGRESS_FILE, "w") as f:
        f.write(next_page_token if next_page_token else "")

    if not next_page_token:
        print("✅ All pages retrieved.")
        break

    print(f"✅ Batch complete. Quota used: {quota_used}, Total videos: {total_videos}")
    time.sleep(1)

# ------------------------------
# FINAL STATS + DAILY LOG
# ------------------------------
print(f"\n🎬 Total unique videos in dataset: {len(seen_videos)}")

# Log the total for the day
with open(LOG_FILE, "a") as log:
    log.write(f"{date.today()}: {len(seen_videos)} total videos\n")

print(f"🗓️ Logged {len(seen_videos)} videos on {date.today()}")
print(f"📄 Log file: {LOG_FILE}")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from scipy import stats
import re

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
    writer.writerow(["Title", "Channel", "Publish Date", "Video URL", "View Count", "Likes", "Comments", "Duration (sec)"])

    next_page_token = None
    total_videos = 0

    while True:
        #stop before exceeding quota
        if quota_used + 150 > daily_limit - safe_buffer:
            print("quota limit reached. pausing until tomorrow...")
            tomorrow = datetime.now() + timedelta(days=1)

