
from pathlib import Path
from PIL import Image
import imagehash
from itertools import combinations
from collections import defaultdict
import shutil
import random
import csv
import yaml

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT_ROOT / "dataset" / "raw" / "PPE"
OUTPUT = PROJECT_ROOT / "dataset" / "processed" / "PPE"

SEED = 42
THRESHOLD = 5

RATIOS = {
    "train": 0.70,
    "valid": 0.15,
    "test": 0.15
}

CLASS_NAMES = {
    0: "Helmet",
    1: "NoHelmet",
    2: "NoVest",
    3: "Vest"
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(SEED)

# Prevent accidental overwrite
if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Rename or remove it only after reviewing it."
    )

# Collect all images and matching labels
images = []

for split in ["train", "valid"]:
    image_dir = SOURCE / split / "images"
    label_dir = SOURCE / split / "labels"

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Missing folders in {split}")

    for image_path in image_dir.iterdir():
        if image_path.suffix.lower() not in EXTENSIONS:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            raise FileNotFoundError(f"Missing label: {label_path}")

        with Image.open(image_path) as img:
            phash = imagehash.phash(img)

        images.append({
            "name": image_path.name,
            "source_split": split,
            "image_path": image_path,
            "label_path": label_path,
            "hash": phash
        })

print("Images loaded:", len(images))

# Group similar images using perceptual hashes
parent = list(range(len(images)))

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    root_a = find(a)
    root_b = find(b)
    if root_a != root_b:
        parent[root_b] = root_a

print("Grouping similar images...")

for i, j in combinations(range(len(images)), 2):
    if images[i]["hash"] - images[j]["hash"] <= THRESHOLD:
        union(i, j)

groups = defaultdict(list)

for i, item in enumerate(images):
    groups[find(i)].append(item)

groups = list(groups.values())

# Assign complete groups to splits
random.shuffle(groups)
groups.sort(key=len, reverse=True)

targets = {
    name: len(images) * ratio
    for name, ratio in RATIOS.items()
}

assigned = {name: [] for name in RATIOS}
counts = {name: 0 for name in RATIOS}

for group in groups:
    # Choose the split with the largest remaining relative capacity
    selected = max(
        RATIOS,
        key=lambda name: (targets[name] - counts[name]) / targets[name]
    )

    assigned[selected].append(group)
    counts[selected] += len(group)

print("\nPlanned split sizes:")
for name in RATIOS:
    print(f"{name}: {counts[name]} images")

# Create output only after planning
for split in RATIOS:
    (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)

# Copy images and labels; original data remains untouched
for split, split_groups in assigned.items():
    for group in split_groups:
        for item in group:
            shutil.copy2(
                item["image_path"],
                OUTPUT / split / "images" / item["name"]
            )
            shutil.copy2(
                item["label_path"],
                OUTPUT / split / "labels" / f"{Path(item['name']).stem}.txt"
            )

# Save group assignment report
with open(OUTPUT / "split_groups.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["Group_ID", "New_Split", "Image_Name", "Original_Split"])

    for split, split_groups in assigned.items():
        for group_id, group in enumerate(split_groups, start=1):
            for item in group:
                writer.writerow([
                    f"{split}_{group_id}",
                    split,
                    item["name"],
                    item["source_split"]
                ])

# Create YOLO dataset configuration
data_config = {
    "path": ".",
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 4,
    "names": CLASS_NAMES
}

with open(OUTPUT / "data.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(
        data_config,
        f,
        sort_keys=False,
        allow_unicode=True
    )

print("\nDataset split created successfully!")
print("Output:", OUTPUT)
print("Original dataset was not modified.")
