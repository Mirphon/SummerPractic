# Детектирование и классификация объектов в системах видеонаблюдения

Проект в рамках учебной практики (технологическая). Сравнительный анализ пяти
современных архитектур глубокого обучения для задачи детектирования и
классификации объектов в сценах видеонаблюдения.

## Описание

Реализован полный ML-пайплайн (данные → предобработка → обучение → оценка →
анализ) и проведено сравнение пяти моделей детектирования объектов:

- **YOLOv8s** — одноэтапный детектор (Ultralytics)
- **Faster R-CNN** ResNet50-FPN — двухэтапный детектор (torchvision)
- **SSD300** VGG16 — одноэтапный детектор (torchvision)
- **EfficientDet-D0** — одноэтапный детектор с BiFPN (effdet)
- **DETR** ResNet-50 — трансформерный детектор (HuggingFace)

В качестве данных использовано подмножество датасета **COCO 2017** (9 классов,
релевантных для видеонаблюдения: person, bicycle, car, motorcycle, bus, truck,
backpack, handbag, suitcase). Все модели обучены на едином подмножестве из
2000 изображений для обеспечения сопоставимости результатов.

## Результаты

Сравнение моделей по метрике F1-score (валидационная выборка, 2000 изображений):

| Модель | Тип | F1-score |
|---|---|---|
| Faster R-CNN | Two-stage | **0.7231** |
| EfficientDet-D0 | One-stage | 0.6640 |
| DETR | Transformer | 0.6215 |
| SSD300 | One-stage | 0.6130 |
| YOLOv8s | One-stage | 0.3460 |

Наилучший результат показала модель Faster R-CNN. Подробный анализ — в отчёте.

## Структура проекта

```
CV_MTUCI/
├── configs/
│   └── data.yaml              # конфигурация датасета для YOLO
├── src/
│   ├── dataset/               # подготовка и конвертация данных
│   │   ├── prepare_data.py            # фильтрация, разбивка train/val/test
│   │   ├── convert_to_yolo.py         # конвертация COCO → YOLO формат
│   │   └── prepare_yolo_train2000.py  # формирование подмножества 2000 изобр.
│   ├── training/              # обучение моделей (по 2 эксперимента на модель)
│   │   ├── train_yolo.py / train_yolo_exp2.py
│   │   ├── train_faster_rcnn.py / train_faster_rcnn_exp2.py
│   │   ├── train_ssd.py / train_ssd_exp2.py
│   │   ├── train_efficientdet.py / _exp2.py / _exp3.py
│   │   └── train_detr.py / train_detr_exp2.py
│   └── utils/                 # вспомогательные скрипты
│       ├── visualize_coco.py          # визуализация датасета (EDA)
│       ├── plot_comparison.py         # сравнительные графики моделей
│       └── check_gpu.py               # проверка доступности GPU
├── results/                   # графики обучения и сравнения
├── requirements.txt
└── README.md
```

Примечание: вычисление метрик (Precision, Recall, F1) и загрузка моделей
реализованы внутри соответствующих скриптов обучения в `src/training/`.

## Требования

- Python 3.10–3.12
- GPU с поддержкой CUDA (проект тестировался на NVIDIA RTX 4060 Laptop, 8 ГБ)
- Зависимости из `requirements.txt`

## Установка

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

## Подготовка данных

1. Скачать датасет COCO 2017 (аннотации и изображения):
   - Аннотации: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   - Train: http://images.cocodataset.org/zips/train2017.zip
   - Val: http://images.cocodataset.org/zips/val2017.zip
2. Распаковать в `data/raw/coco/`
3. Запустить подготовку данных:

```bash
python src/dataset/prepare_data.py
python src/dataset/convert_to_yolo.py
```

## Запуск обучения

Все скрипты запускаются **из корня проекта**:

```bash
python src/training/train_yolo.py
python src/training/train_faster_rcnn.py
python src/training/train_ssd.py
python src/training/train_efficientdet.py
python src/training/train_detr.py
```

Результаты (веса, графики, метрики) сохраняются в `results/` и `runs/`.

## Визуализация результатов

```bash
python src/utils/visualize_coco.py     # анализ и визуализация датасета
python src/utils/plot_comparison.py    # сравнительные графики моделей
```

## Примечания

- Обучение проводилось на подмножестве из 2000 изображений в связи с
  ограниченными вычислительными ресурсами; на полном датасете COCO показатели
  моделей существенно выше.
- Оценка выполнялась на валидационной выборке val2017, так как аннотации
  официальной тестовой выборки COCO закрыты (стандартная практика в литературе).
- Файлы обученных весов не включены в репозиторий (воспроизводятся запуском
  обучения); при необходимости могут быть предоставлены отдельно.
