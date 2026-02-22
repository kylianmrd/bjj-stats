from pathlib import Path

ROOT = Path("bjj_stats/dataset/back_control")

for i, p in enumerate(ROOT.rglob("*.jpg")):
    print(p.resolve())
    if i == 9:
        break
