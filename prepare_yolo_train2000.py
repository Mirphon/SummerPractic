
import json
import os
import random
from pathlib import Path
from collections import defaultdict

SURVEILLANCE_CLASSES = [
    "person", "bicycle", "car", "motorcycle",
    "bus", "truck", "backpack", "handbag", "suitcase",
]

ANN_TRAIN        = "data/processed/annotations_train.json"
IMAGES_DIR_TRAIN = "data/raw/coco/train2017"
LABELS_OUT       = "data/processed/labels/train2000"
LIST_OUT         = "data/processed/train2000_images.txt"
MAX_IMAGES       = 2000
RANDOM_SEED      = 42


def coco_bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    return cx, cy, nw, nh


def main():
    print("Загружаем аннотации train...")
    with open(ANN_TRAIN, "r") as f:
        data = json.load(f)

    coco_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}
    name_to_local_id = {name: idx for idx, name in enumerate(SURVEILLANCE_CLASSES)}

    anns_by_image = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    valid_images = []
    for img_info in data["images"]:
        img_path = os.path.join(IMAGES_DIR_TRAIN, img_info["file_name"])
        if os.path.exists(img_path):
            valid_images.append(img_info)

    print(f"Найдено изображений на диске: {len(valid_images)}")

    random.seed(RANDOM_SEED)
    random.shuffle(valid_images)
    selected = valid_images[:MAX_IMAGES]
    print(f"Отобрано: {len(selected)} изображений")

    labels_dir = Path(LABELS_OUT)
    labels_dir.mkdir(parents=True, exist_ok=True)

    junction_img_dir = Path.cwd() / "data" / "dataset" / "images" / "train2000"

    total_boxes = 0
    with open(LIST_OUT, "w") as list_file:
        for img_info in selected:
            img_id   = img_info["id"]
            img_w    = img_info["width"]
            img_h    = img_info["height"]
            filename = img_info["file_name"]

            abs_path = junction_img_dir / filename
            list_file.write(str(abs_path) + "\n")

            label_lines = []
            for ann in anns_by_image.get(img_id, []):
                if ann.get("iscrowd", 0) == 1:
                    continue
                cat_name = coco_id_to_name.get(ann["category_id"])
                if cat_name not in name_to_local_id:
                    continue
                x, y, w, h = ann["bbox"]
                if w * h < 32 * 32 or w < 10 or h < 10:
                    continue
                local_id = name_to_local_id[cat_name]
                cx, cy, nw, nh = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
                label_lines.append(f"{local_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                total_boxes += 1

            stem = Path(filename).stem
            with open(labels_dir / f"{stem}.txt", "w") as lf:
                lf.write("\n".join(label_lines))

    print(f"\nГотово!")
    print(f"  Labels:          {LABELS_OUT}  ({len(selected)} файлов)")
    print(f"  Список путей:    {LIST_OUT}")
    print(f"  Всего рамок:     {total_boxes}")
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()