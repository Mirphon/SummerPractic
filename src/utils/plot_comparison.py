
import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "results/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


models = ["Faster R-CNN", "EfficientDet-D0", "DETR", "SSD300", "YOLOv8s"]
f1_scores = [0.7231, 0.6640, 0.6215, 0.6130, 0.3460]
types = ["Two-stage", "One-stage", "Transformer", "One-stage", "One-stage"]


type_colors = {
    "Two-stage":   "#2563EB",
    "One-stage":   "#10B981",
    "Transformer": "#F59E0B",
}
colors = [type_colors[t] for t in types]


def plot_f1_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(models, f1_scores, color=colors, edgecolor="white", linewidth=1.2, width=0.6)

    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{score:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title("Сравнение моделей по метрике F1-score\n(базовая конфигурация, 2000 изображений)",
                 fontsize=14, pad=15)
    ax.set_ylabel("F1-score", fontsize=12)
    ax.set_xlabel("Модель", fontsize=12)
    ax.set_ylim(0, 0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xticklabels(models, rotation=15, ha="right")

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=c, label=t) for t, c in type_colors.items()]
    ax.legend(handles=legend_elems, title="Тип архитектуры", loc="upper right")

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "f1_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Сохранён: {path}")



def plot_experiments_comparison():
    exp_models = ["Faster R-CNN", "EfficientDet", "DETR", "SSD300", "YOLOv8s"]
    baseline = [0.7231, 0.6433, 0.5996, 0.6130, 0.3460]
    exp2     = [0.6099, 0.6640, 0.6215, 0.5616, 0.1820]

    x = np.arange(len(exp_models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width/2, baseline, width, label="Базовый эксперимент",
                   color="#3B82F6", edgecolor="white", linewidth=1)
    bars2 = ax.bar(x + width/2, exp2, width, label="Эксперимент 2",
                   color="#F97316", edgecolor="white", linewidth=1)

    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Влияние гиперпараметров на качество моделей", fontsize=14, pad=15)
    ax.set_ylabel("F1-score", fontsize=12)
    ax.set_xlabel("Модель", fontsize=12)
    ax.set_ylim(0, 0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_models, rotation=15, ha="right")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "experiments_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Сохранён: {path}")


if __name__ == "__main__":
    print("Построение сравнительных графиков...")
    plot_f1_comparison()
    plot_experiments_comparison()
