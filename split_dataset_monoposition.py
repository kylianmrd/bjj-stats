import os
import random
from collections import defaultdict
from pathlib import Path

DATASET_DIR = Path("dataset")
SPLITS_DIR = Path("splits")
SPLITS_DIR.mkdir(exist_ok=True)

# ratios par "source" (vidéo) dans chaque classe
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

SEED = 42
random.seed(SEED)

CLASSES = [d.name for d in DATASET_DIR.iterdir() if d.is_dir()]
print("Classes:", CLASSES)

def get_source_id(filename: str) -> str:
    # attend des noms type VID_0001_000123.jpg
    # source_id = VID_0001
    base = Path(filename).stem
    parts = base.split("_")
    return "_".join(parts[:2])  # VID + 0001

# regroupe les images par classe puis par source
class_to_source_to_files = defaultdict(lambda: defaultdict(list))

total_images = 0
for cls in CLASSES:
    cls_dir = DATASET_DIR / cls
    for f in cls_dir.iterdir():
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        source_id = get_source_id(f.name)
        class_to_source_to_files[cls][source_id].append(str(f))
        total_images += 1

print(f"Images trouvées: {total_images}")

train_files, val_files, test_files = [], [], []
split_sources_summary = {"train": defaultdict(int), "val": defaultdict(int), "test": defaultdict(int)}

for cls in CLASSES:
    sources = list(class_to_source_to_files[cls].keys())
    random.shuffle(sources)

    n = len(sources)
    n_train = max(1, int(n * TRAIN_RATIO))
    n_val = max(1, int(n * VAL_RATIO))
    n_test = max(1, n - n_train - n_val)  # le reste

    # ajuste si trop peu de sources
    if n < 3:
        # tout en train si pas assez de vidéos pour val/test
        train_src = sources
        val_src, test_src = [], []
    else:
        train_src = sources[:n_train]
        val_src = sources[n_train:n_train + n_val]
        test_src = sources[n_train + n_val:n_train + n_val + n_test]

    for s in train_src:
        files = class_to_source_to_files[cls][s]
        train_files.extend(files)
        split_sources_summary["train"][cls] += len(files)

    for s in val_src:
        files = class_to_source_to_files[cls][s]
        val_files.extend(files)
        split_sources_summary["val"][cls] += len(files)

    for s in test_src:
        files = class_to_source_to_files[cls][s]
        test_files.extend(files)
        split_sources_summary["test"][cls] += len(files)

def write_list(path: Path, items: list[str]):
    path.write_text("\n".join(items) + ("\n" if items else ""), encoding="utf-8")

write_list(SPLITS_DIR / "train.txt", train_files)
write_list(SPLITS_DIR / "val.txt", val_files)
write_list(SPLITS_DIR / "test.txt", test_files)

print("\n--- SPLIT SUMMARY (mono-position stratified by class) ---")
print(f"TRAIN images: {len(train_files)} | labels: {dict(split_sources_summary['train'])}")
print(f"VAL images:   {len(val_files)} | labels: {dict(split_sources_summary['val'])}")
print(f"TEST images:  {len(test_files)} | labels: {dict(split_sources_summary['test'])}")

print("\nFichiers écrits:")
print("splits/train.txt")
print("splits/val.txt")
print("splits/test.txt")