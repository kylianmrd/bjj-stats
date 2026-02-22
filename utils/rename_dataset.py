from pathlib import Path

ROOT = Path("dataset")  # ton dataset est directement dataset/
EXT = ".jpg"

def rename_label_folder(label_dir: Path):
    label = label_dir.name
    files = sorted([p for p in label_dir.iterdir() if p.is_file() and p.suffix == EXT])

    for p in files:
        if not p.name.startswith("IMG_"):
            continue  # on laisse aug_*.jpg et les "9.jpg" tranquilles

        # ex: IMG_5265_00018.jpg -> source=IMG_5265, frame=00018
        parts = p.stem.split("_")
        if len(parts) < 3:
            continue

        source = "_".join(parts[:2])     # IMG_5265
        frame  = parts[2]               # 00018
        new_name = f"{source}_{label}_{frame}.jpg"
        new_path = label_dir / new_name

        if new_path.exists():
            print(f"SKIP (exists) {new_name}")
            continue

        print(f"{p.name} -> {new_name}")
        p.rename(new_path)

for d in ROOT.iterdir():
    if d.is_dir():
        rename_label_folder(d)

print("Done.")