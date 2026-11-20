
from pathlib import Path
from PIL import Image
import imagehash
from itertools import combinations
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "processed" / "PPE"

SPLITS = ["train", "valid", "test"]
THRESHOLD = 5
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = []

print("FINAL THREE-WAY SIMILARITY CHECK")

for split in SPLITS:
    folder = DATASET_ROOT / split / "images"

    for path in folder.iterdir():
        if path.suffix.lower() not in EXTENSIONS:
            continue

        with Image.open(path) as img:
            phash = imagehash.phash(img)

        images.append({
            "split": split,
            "name": path.name,
            "hash": phash
        })

print("Images processed:", len(images))

parent = list(range(len(images)))

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

similar_pairs = 0

for i, j in combinations(range(len(images)), 2):
    if images[i]["hash"] - images[j]["hash"] <= THRESHOLD:
        similar_pairs += 1
        union(i, j)

groups = defaultdict(list)

for i, item in enumerate(images):
    groups[find(i)].append(item)

duplicate_groups = [
    group for group in groups.values()
    if len(group) > 1
]

cross_split_groups = [
    group for group in duplicate_groups
    if len({item["split"] for item in group}) > 1
]

print("\nSimilar pairs:", similar_pairs)
print("Groups containing similar images:", len(duplicate_groups))
print("Groups crossing dataset splits:", len(cross_split_groups))

for number, group in enumerate(cross_split_groups, 1):
    print(f"\nCross-split group {number}:")
    for item in group:
        print(f"{item['split']}: {item['name']}")

print("\nFINAL SIMILARITY CHECK FINISHED")
