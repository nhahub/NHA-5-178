
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "raw" / "PPE"

FILES_TO_MOVE = [
    "00472_jpg.rf.d60afe6757fa20700be65d043ed72bfd.jpg",
    "00721_jpg.rf.c55bede6add835754b3ae1758d9e4c21.jpg",
    "02476_jpg.rf.b7bc819fdd857d5f8232dce3a4404846.jpg",
    "02969_jpg.rf.5e7edd065aa71cb17281df847afcb219.jpg",
]

valid_images = DATASET_ROOT / "valid" / "images"
valid_labels = DATASET_ROOT / "valid" / "labels"
train_images = DATASET_ROOT / "train" / "images"
train_labels = DATASET_ROOT / "train" / "labels"

backup = DATASET_ROOT / "backup_before_split_fix"

operations = []

for filename in FILES_TO_MOVE:
    image_src = valid_images / filename
    label_src = valid_labels / (Path(filename).stem + ".txt")

    image_dst = train_images / filename
    label_dst = train_labels / (Path(filename).stem + ".txt")

    if not image_src.exists() or not label_src.exists():
        raise FileNotFoundError(f"Missing pair: {filename}")

    if image_dst.exists() or label_dst.exists():
        raise FileExistsError(f"Already exists in train: {filename}")

    operations.append((image_src, label_src, image_dst, label_dst))

# Create backup directories
backup_images = backup / "valid" / "images"
backup_labels = backup / "valid" / "labels"

backup_images.mkdir(parents=True, exist_ok=True)
backup_labels.mkdir(parents=True, exist_ok=True)

# Backup before moving anything
for image_src, label_src, _, _ in operations:
    shutil.copy2(image_src, backup_images / image_src.name)
    shutil.copy2(label_src, backup_labels / label_src.name)

# Verify backup
for image_src, label_src, _, _ in operations:
    assert (backup_images / image_src.name).exists()
    assert (backup_labels / label_src.name).exists()

print("Backup completed and verified.")

# Move image-label pairs
for image_src, label_src, image_dst, label_dst in operations:
    shutil.move(str(image_src), str(image_dst))
    shutil.move(str(label_src), str(label_dst))
    print("Moved:", image_src.name)

print("\nSplit correction completed.")
print("Backup location:", backup)
