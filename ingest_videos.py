from pathlib import Path
import shutil
from datetime import datetime

INBOX = Path("videos_inbox")
VIDEOS = Path("videos")
INBOX.mkdir(exist_ok=True)
VIDEOS.mkdir(exist_ok=True)

EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}

def unique_name(ext: str) -> str:
    # SRC_20260221_141233_0001.mp4
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i in range(1, 10000):
        name = f"SRC_{stamp}_{i:04d}{ext}"
        if not (VIDEOS / name).exists():
            return name
    raise RuntimeError("Could not find unique filename")

files = [p for p in INBOX.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
if not files:
    print("No videos found in videos_inbox/")
    raise SystemExit(0)

moved = 0
for p in sorted(files):
    new_name = unique_name(p.suffix.lower())
    dst = VIDEOS / new_name
    shutil.move(str(p), str(dst))
    print(f"Moved: {p.name} -> {dst.name}")
    moved += 1

print(f"Done. {moved} video(s) ingested into videos/")