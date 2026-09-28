import re
from youtube_comment_downloader import YoutubeCommentDownloader
from google_play_scraper import reviews, Sort

OUTPUT_FILE = "marathi_clean_dataset.txt"
TARGET_COUNT = 8000

DEV_RANGE = r'[\u0900-\u097F]'
VOWELS = set("अआइईउऊएऐओऔािीुूेैोौंः")

def basic_cleanup(text):
    text = text.strip()

    # remove emojis and junk but keep Devanagari + space + basic punctuation
    text = re.sub(r'[^\u0900-\u097F\s।?!,.]', ' ', text)

    # normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def is_clean_marathi(text):
    if not text:
        return False

    words = text.split()

    # relaxed length
    if len(words) < 3 or len(words) > 25:
        return False

    # allow some noise (70% devanagari instead of 95%)
    total = len(text)
    dev = len(re.findall(DEV_RANGE, text))
    if total == 0 or (dev / total) < 0.7:
        return False

    # must contain at least 1 vowel
    if not any(ch in VOWELS for ch in text):
        return False

    # reject extreme repetition only (4+ instead of 3+)
    if re.search(r'(.)\1{3,}', text):
        return False

    return True

seen = set()
data = []

def add_sentence(s):
    s = basic_cleanup(s)
    if not s:
        return
    if s in seen:
        return
    if is_clean_marathi(s):
        seen.add(s)
        data.append(s)

def collect_youtube(video_urls, limit=5000):
    downloader = YoutubeCommentDownloader()
    count = 0

    for url in video_urls:
        comments = downloader.get_comments_from_url(url)
        for c in comments:
            add_sentence(c['text'])
            count += 1
            if len(data) >= TARGET_COUNT or count >= limit:
                return

def collect_playstore(app_ids, per_app=2000):
    for app in app_ids:
        result, _ = reviews(
            app,
            lang='mr',
            country='in',
            sort=Sort.NEWEST,
            count=per_app
        )
        for r in result:
            add_sentence(r['content'])
            if len(data) >= TARGET_COUNT:
                return

youtube_videos = [
    "https://www.youtube.com/watch?v=VIDEO_ID_1",
    "https://www.youtube.com/watch?v=VIDEO_ID_2"
]

apps = [
    "com.whatsapp",
    "com.instagram.android"
]

print("Collecting YouTube...")
collect_youtube(youtube_videos)

print("Collecting Play Store...")
collect_playstore(apps)

print("Collected:", len(data))

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for s in data:
        f.write(s + "\n")

print("Saved.")