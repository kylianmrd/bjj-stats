from pathlib import Path
from datetime import datetime

ROOT = Path("dataset")
PREFIXES_OK = ("IMG_", "VID_", "MANUAL_")
EXT = ".jpg"

date_tag = datetime.now().strftime("%Y%m%d")  # ex: 20260220

for label_dir in ROOT.iterdir():
    if not label_dir.is_dir():
        continue

    label = label_dir.name
    files = sorted([p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() == EXT])

    counter = 1
    for p in files:
        if p.name.startswith(PREFIXES_OK):
            continue  # déjà bien nommé

        new_name = f"MANUAL_{date_tag}_{counter:06d}.jpg"
        new_path = label_dir / new_name

        while new_path.exists():
            counter += 1
            new_name = f"MANUAL_{date_tag}_{counter:06d}.jpg"
            new_path = label_dir / new_name

        print(f"[{label}] {p.name} -> {new_name}")
        p.rename(new_path)
        counter += 1

print("Renommage MANUAL terminé.")