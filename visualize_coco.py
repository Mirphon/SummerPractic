import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ANNOTATIONS_FILE = "data/raw/coco/annotations/instances_val2017.json"
IMAGES_DIR       = "data/raw/coco/val2017"
OUTPUT_DIR       = "results/plots"

SURVEILLANCE_CLASSES = [
    "person", "bicycle", "car", "motorcycle",
    "bus", "truck", "backpack", "handbag", "suitcase"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_annotations(ann_file):
    print(f"Загружаем аннотации: {ann_file}")
    with open(ann_file, "r") as f:
        data = json.load(f)
    print(f"  Изображений:  {len(data['images'])}")
    print(f"  Аннотаций:    {len(data['annotations'])}")
    print(f"  Категорий:    {len(data['categories'])}")
    return data


def analyze_dataset(data):
    print("\n=== АНАЛИЗ ДАТАСЕТА ===")
    id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}

    class_counts = {}
    for ann in data["annotations"]:
        name = id_to_name[ann["category_id"]]
        class_counts[name] = class_counts.get(name, 0) + 1

    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    print("\nТоп-15 классов по числу объектов:")
    for name, count in sorted_counts[:15]:
        bar = "█" * (count // 500)
        print(f"  {name:<20} {count:>6}  {bar}")

    return class_counts, id_to_name


def plot_class_distribution(class_counts, output_dir):
    surveillance_counts = {
        k: v for k, v in class_counts.items()
        if k in SURVEILLANCE_CLASSES
    }
    sorted_items = sorted(surveillance_counts.items(), key=lambda x: x[1], reverse=True)
    names  = [x[0] for x in sorted_items]
    counts = [x[1] for x in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))
    bars = ax.bar(names, counts, color=colors, edgecolor="white", linewidth=0.8)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{count:,}", ha="center", va="bottom", fontsize=9
        )

    ax.set_title(
        "Распределение объектов по классам (COCO val2017)\n"
        "Классы, релевантные для видеонаблюдения",
        fontsize=13, pad=15
    )
    ax.set_xlabel("Класс объекта")
    ax.set_ylabel("Количество аннотаций")
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"График сохранён: {path}")


def plot_bbox_sizes(data, id_to_name, output_dir):
    widths, heights = [], []
    for ann in data["annotations"]:
        if id_to_name[ann["category_id"]] in SURVEILLANCE_CLASSES:
            _, _, w, h = ann["bbox"]
            widths.append(w)
            heights.append(h)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(widths,  bins=50, color="#3A86FF", edgecolor="white", linewidth=0.5)
    axes[0].set_title("Распределение ширины рамок (bbox)")
    axes[0].set_xlabel("Ширина (пиксели)")
    axes[0].set_ylabel("Количество")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    axes[1].hist(heights, bins=50, color="#FF6B6B", edgecolor="white", linewidth=0.5)
    axes[1].set_title("Распределение высоты рамок (bbox)")
    axes[1].set_xlabel("Высота (пиксели)")
    axes[1].set_ylabel("Количество")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Анализ размеров объектов — классы видеонаблюдения", fontsize=13)
    fig.tight_layout()

    path = os.path.join(output_dir, "bbox_sizes.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"График сохранён: {path}")



def visualize_samples(data, id_to_name, images_dir, output_dir, n=6):
    surveillance_ids = {
        cat["id"] for cat in data["categories"]
        if cat["name"] in SURVEILLANCE_CLASSES
    }

    valid_image_ids = {
        ann["image_id"] for ann in data["annotations"]
        if ann["category_id"] in surveillance_ids
    }

    img_to_anns = {}
    for ann in data["annotations"]:
        if ann["image_id"] in valid_image_ids:
            img_to_anns.setdefault(ann["image_id"], []).append(ann)

    id_to_img = {img["id"]: img for img in data["images"]}
    sample_ids = random.sample(list(valid_image_ids), min(n, len(valid_image_ids)))

    color_map = {
        "person":     "#FF6B6B",
        "car":        "#3A86FF",
        "bicycle":    "#06D6A0",
        "motorcycle": "#FFD166",
        "bus":        "#A855F7",
        "truck":      "#FB923C",
        "backpack":   "#F472B6",
        "handbag":    "#34D399",
        "suitcase":   "#60A5FA",
    }

    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()

    for idx, image_id in enumerate(sample_ids):
        img_info = id_to_img[image_id]
        img_path = os.path.join(images_dir, img_info["file_name"])

        if not os.path.exists(img_path):
            print(f"  Файл не найден: {img_path}")
            continue

        img = plt.imread(img_path)
        ax  = axes[idx]
        ax.imshow(img)
        ax.axis("off")

        obj_count = 0
        for ann in img_to_anns.get(image_id, []):
            cat_name = id_to_name[ann["category_id"]]
            if cat_name not in SURVEILLANCE_CLASSES:
                continue

            x, y, w, h = ann["bbox"]
            color = color_map.get(cat_name, "#FFFFFF")

            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                x, y - 4, cat_name,
                fontsize=8, color="white",
                bbox=dict(facecolor=color, alpha=0.85, pad=1, edgecolor="none")
            )
            obj_count += 1

        ax.set_title(
            f"{img_info['file_name']}\n"
            f"{img_info['width']}×{img_info['height']}px  |  объектов: {obj_count}",
            fontsize=9
        )

    for i in range(len(sample_ids), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "Примеры изображений COCO val2017 с аннотациями\n(классы видеонаблюдения)",
        fontsize=14, y=1.01
    )
    fig.tight_layout()

    path = os.path.join(output_dir, "sample_images.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Визуализация сохранена: {path}")



if __name__ == "__main__":
    data = load_annotations(ANNOTATIONS_FILE)
    class_counts, id_to_name = analyze_dataset(data)
    plot_class_distribution(class_counts, OUTPUT_DIR)
    plot_bbox_sizes(data, id_to_name, OUTPUT_DIR)
    visualize_samples(data, id_to_name, IMAGES_DIR, OUTPUT_DIR, n=6)
    print("\n=== Готово! Графики сохранены в results/plots/ ===")