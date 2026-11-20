
from pathlib import Path
from PIL import Image
import imagehash
from itertools import combinations
import csv

# Locate project and dataset automatically
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "raw" / "PPE"

SPLITS = ["train", "valid"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
THRESHOLD = 5

images = []

print("PPE DATASET SIMILARITY CHECK")
print("Dataset:", DATASET_ROOT)
print("\nCalculating perceptual hashes...")

for split in SPLITS:
    folder = DATASET_ROOT / split / "images"

    for path in folder.iterdir():
        if path.suffix.lower() not in EXTENSIONS:
            continue

        try:
            with Image.open(path) as img:
                phash = imagehash.phash(img)

            images.append({
                "split": split,
                "name": path.name,
                "hash": phash
            })

        except Exception as error:
            print(f"Could not read {path.name}: {error}")

print("Images processed:", len(images))
print("Comparing images...")

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

similar_pairs = []

for i, j in combinations(range(len(images)), 2):
    distance = images[i]["hash"] - images[j]["hash"]

    if distance <= THRESHOLD:
        similar_pairs.append((i, j, distance))
        union(i, j)

groups = {}

for i, item in enumerate(images):
    groups.setdefault(find(i), []).append(item)

duplicate_groups = [
    group for group in groups.values()
    if len(group) > 1
]

cross_split_groups = [
    group for group in duplicate_groups
    if len({item["split"] for item in group}) > 1
]

report_path = PROJECT_ROOT / "similarity_report.csv"

with open(report_path, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["Group_ID", "Split", "Image_Name"])

    for group_id, group in enumerate(duplicate_groups, start=1):
        for item in group:
            writer.writerow([
                group_id,
                item["split"],
                item["name"]
            ])

print("\n" + "=" * 45)
print("SIMILARITY RESULTS")
print("=" * 45)
print("Total images:", len(images))
print("Similar pairs:", len(similar_pairs))
print("Groups with similar images:", len(duplicate_groups))
print("Groups crossing Train and Valid:", len(cross_split_groups))

for group_id, group in enumerate(cross_split_groups, start=1):
    print(f"\nGroup {group_id}:")
    for item in group:
        print(f"{item['split']}: {item['name']}")

print("\nReport saved:", report_path)
print("No images or labels were modified.")
