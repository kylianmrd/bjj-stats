from pathlib import Path
import os
import random
from collections import Counter, defaultdict

def is_valid_frame(path: Path) -> bool:
    name = path.name
    return name.startswith("IMG_") or name.startswith("VID_") or name.startswith("MANUAL_")

SEED = 42
random.seed(SEED)

DATASET_ROOT = Path("dataset")   # dataset/close_guard/*.jpg etc.
SPLITS_DIR = Path("splits")
SPLITS_DIR.mkdir(exist_ok=True)

# Ratios (par source)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

EXT = ".jpg"

def get_label(path: Path) -> str:
    # label = dossier parent (close_guard, mount, side_control, back_control)
    return path.parent.name

def get_source_id(path: Path) -> str:
    """
    source_id = préfixe avant le 2e underscore si c'est du style IMG_5265_00018.jpg
    - IMG_5265_00018.jpg        -> IMG_5265
    - IMG_5265_close_guard_00018.jpg -> IMG_5265 (ça marche aussi)
    fallback: tout avant le 1er underscore
    """
    stem = path.stem  # sans .jpg
    parts = stem.split("_")
    if len(parts) >= 2:
        return parts[0] + "_" + parts[1]  # IMG_5265
    return parts[0]

def is_augmented(path: Path) -> bool:
    return path.name.startswith("aug_")

# 1) Scan images
all_images = sorted([p for p in DATASET_ROOT.rglob(f"*{EXT}") if p.is_file() and is_valid_frame(p)])
if not all_images:
    raise SystemExit(f"Aucune image trouvée sous {DATASET_ROOT.resolve()}")

# 2) Grouper par source_id
by_source = defaultdict(list)
for p in all_images:
    src = get_source_id(p)
    by_source[src].append(p)

sources = sorted(by_source.keys())
print(f"Sources trouvées: {len(sources)}")
print(f"Images trouvées: {len(all_images)}")

# 3) Construire un "label dominant" par source (pour garder un peu d'équilibre)
source_dominant = {}
for src, paths in by_source.items():
    labels = [get_label(p) for p in paths if not is_augmented(p)]  # on ignore aug_ pour estimer
    if not labels:  # si une source n'a que des aug_ (rare), on prend tout
        labels = [get_label(p) for p in paths]
    dominant = Counter(labels).most_common(1)[0][0]
    source_dominant[src] = dominant

# 4) Split stratifié approximatif par dominant_label (au niveau source)
# -> on répartit les sources par classe dominante, puis on mélange dans chaque classe
sources_by_dom = defaultdict(list)
for src in sources:
    sources_by_dom[source_dominant[src]].append(src)

for k in sources_by_dom:
    random.shuffle(sources_by_dom[k])

train_sources, val_sources, test_sources = [], [], []

def take_counts(n):
    n_train = int(round(TRAIN_RATIO * n))
    n_val   = int(round(VAL_RATIO * n))
    n_test  = n - n_train - n_val
    return n_train, n_val, n_test

for dom_label, srcs in sources_by_dom.items():
    n = len(srcs)
    n_train, n_val, n_test = take_counts(n)
    train_sources += srcs[:n_train]
    val_sources   += srcs[n_train:n_train+n_val]
    test_sources  += srcs[n_train+n_val:]
# --- Fallback si val/test vides (peu de sources) ---
if len(sources) >= 2 and len(val_sources) == 0:
    # prend 1 source du train et l'envoie en val
    val_sources = [train_sources.pop()]

if len(sources) >= 3 and len(test_sources) == 0:
    # prend 1 source du train et l'envoie en test
    test_sources = [train_sources.pop()]
    
# Mélange final
random.shuffle(train_sources)
random.shuffle(val_sources)
random.shuffle(test_sources)

# 5) Expand vers images
train_paths, val_paths, test_paths = [], [], []

for src in train_sources:
    # train: on prend tout (y compris aug_)
    train_paths += by_source[src]

for src in val_sources:
    # val/test: on exclut aug_*
    val_paths += [p for p in by_source[src] if not is_augmented(p)]

for src in test_sources:
    test_paths += [p for p in by_source[src] if not is_augmented(p)]

# 6) Écrire splits (chemins relatifs)
def write_list(name, paths):
    out = SPLITS_DIR / f"{name}.txt"
    with out.open("w") as f:
        for p in sorted(paths):
            f.write(str(p.as_posix()) + "\n")
    return out

train_file = write_list("train", train_paths)
val_file   = write_list("val", val_paths)
test_file  = write_list("test", test_paths)

# 7) Résumé
def summarize(name, paths):
    labels = [get_label(Path(p)) for p in paths]
    print(f"\n{name}: {len(paths)} images")
    print("labels:", dict(Counter(labels)))

print("\n--- SPLIT SUMMARY ---")
print(f"train sources: {len(train_sources)} | val sources: {len(val_sources)} | test sources: {len(test_sources)}")
summarize("TRAIN", [str(p) for p in train_paths])
summarize("VAL",   [str(p) for p in val_paths])
summarize("TEST",  [str(p) for p in test_paths])

print("\nFichiers écrits:")
print(train_file, val_file, test_file, sep="\n")