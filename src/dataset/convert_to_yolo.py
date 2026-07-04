import json
import os
from pathlib import Path
from collections import defaultdict

SURVEILLANCE_CLASSES = [
    "person",       
    "bicycle",      
    "car",          
    "motorcycle",   
    "bus",          
    "truck",       
    "backpack",    
    "handbag",      
    "suitcase",     
]

PROCESSED_DIR  = "data/processed"
IMAGES_VAL_DIR = "data/raw/coco/val2017"

SPLITS = {
    "val":  f"{PROCESSED_DIR}/annotations_val.json",
    "test": f"{PROCESSED_DIR}/annotations_test.json",
}

TRAIN_ANN = f"{PROCESSED_DIR}/annotations_train.json"
TRAIN_IMAGES_DIR = "data/raw/coco/train2017"
if os.path.exists(TRAIN_ANN):
    SPLITS["train"] = TRAIN_ANN



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


def convert_split(ann_file, split_name):
    print(f"\n{'='*50}")
    print(f"Конвертируем: {split_name}  ({ann_file})")

    if not os.path.exists(ann_file):
        print(f"  Файл не найден, пропускаем: {ann_file}")
        return

    with open(ann_file, "r") as f:
        data = json.load(f)

    coco_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}
    name_to_local_id = {name: idx for idx, name in enumerate(SURVEILLANCE_CLASSES)}

    anns_by_image = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    id_to_img = {img["id"]: img for img in data["images"]}

    labels_dir = Path(f"data/processed/labels/{split_name}")
    labels_dir.mkdir(parents=True, exist_ok=True)

    images_list_path = Path(f"data/processed/{split_name}_images.txt")

    if split_name == "train":
        images_dir = TRAIN_IMAGES_DIR
    else:
        images_dir = IMAGES_VAL_DIR

    converted   = 0
    skipped     = 0
    total_boxes = 0

    with open(images_list_path, "w") as img_list_file:
        for img_info in data["images"]:
            img_id   = img_info["id"]
            img_w    = img_info["width"]
            img_h    = img_info["height"]
            filename = img_info["file_name"]

            img_path = os.path.join(images_dir, filename)
            if not os.path.exists(img_path):
                skipped += 1
                continue

            abs_path = str(Path(img_path).resolve())
            img_list_file.write(abs_path + "\n")

            label_lines = []
            for ann in anns_by_image.get(img_id, []):
                cat_name = coco_id_to_name.get(ann["category_id"])
                if cat_name not in name_to_local_id:
                    continue

                local_id = name_to_local_id[cat_name]
                cx, cy, nw, nh = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
                label_lines.append(f"{local_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                total_boxes += 1

            stem = Path(filename).stem  # имя без расширения
            label_path = labels_dir / f"{stem}.txt"
            with open(label_path, "w") as lf:
                lf.write("\n".join(label_lines))

            converted += 1

    print(f"  Изображений сконвертировано: {converted}")
    print(f"  Изображений пропущено:       {skipped}")
    print(f"  Всего рамок записано:        {total_boxes}")
    print(f"  Labels → {labels_dir}")
    print(f"  Список изображений → {images_list_path}")


def create_data_yaml():

    yaml_content = f"""# data.yaml — конфигурация датасета для YOLOv8
# Ultralytics автоматически читает этот файл при обучении

path: .  # корень проекта

# Списки изображений
val:  data/processed/val_images.txt
test: data/processed/test_images.txt

# train — раскомментируйте когда скачаете train2017
# train: data/processed/train_images.txt

# Количество классов
nc: {len(SURVEILLANCE_CLASSES)}

# Названия классов (порядок совпадает с id в .txt файлах)
names:
"""
    for idx, name in enumerate(SURVEILLANCE_CLASSES):
        yaml_content += f"  {idx}: {name}\n"

    yaml_path = "configs/data.yaml"
    os.makedirs("configs", exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\nСоздан: {yaml_path}")
    print("Содержимое:")
    print(yaml_content)



if __name__ == "__main__":
    print("▶ Конвертация аннотаций COCO → YOLO формат")

    for split_name, ann_file in SPLITS.items():
        convert_split(ann_file, split_name)

    print("\n▶ Создание data.yaml")
    create_data_yaml()

    print("\n" + "="*50)
    print("Готово! Структура после конвертации:")
    print("  data/processed/")
    print("    labels/")
    print("      val/    ← .txt файлы с рамками для val")
    print("      test/   ← .txt файлы с рамками для test")
    print("    val_images.txt   ← список путей к картинкам")
    print("    test_images.txt")
    print("  configs/data.yaml  ← конфиг датасета для YOLO")
    print("="*50)
    print("\nСледующий шаг: запустите python train_yolo.py")