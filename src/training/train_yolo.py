
import os
import torch
from ultralytics import YOLO
from pathlib import Path


MODEL_SIZE   = "yolov8s.pt"  
DATA_YAML    = "configs/data.yaml"
PROJECT_DIR  = "results/yolov8l"
EXPERIMENT   = "baseline"


EPOCHS     = 20       
IMG_SIZE   = 640     
BATCH_SIZE = 4       
WORKERS    = 0        
LR0        = 0.01     
LRF        = 0.01     
MOMENTUM   = 0.937    
OPTIMIZER  = "SGD"   


def check_setup():
    print("="*50)
    print("ПРОВЕРКА ПЕРЕД ОБУЧЕНИЕМ")
    print("="*50)

    # GPU
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU:   {gpu}")
        print(f"  VRAM:  {vram:.1f} GB")
    else:
        print("  ВНИМАНИЕ: GPU не найден, обучение будет на CPU")


    checks = {
        "configs/data.yaml":                   "конфиг датасета",
        "data/processed/val_images.txt":       "список val изображений",
        "data/processed/test_images.txt":      "список test изображений",
        "data/processed/labels/val":           "labels для val",
    }
    all_ok = True
    for path, desc in checks.items():
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {status}  {desc:35} {path}")
        if not exists:
            all_ok = False

    if not all_ok:
        print("\nНЕ ВСЕ ФАЙЛЫ НАЙДЕНЫ!")
        print("Убедитесь, что вы запустили:")
        print("  1. python prepare_data.py")
        print("  2. python convert_to_yolo.py")
        return False

    print("\nВсё готово к обучению!")
    return True


def train():

    print("\n" + "="*50)
    print(f"ОБУЧЕНИЕ {MODEL_SIZE.upper()}")
    print("="*50)
    print(f"  Эпох:       {EPOCHS}")
    print(f"  Размер:     {IMG_SIZE}px")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Optimizer:  {OPTIMIZER}, lr={LR0}")
    print("="*50)

    model = YOLO(MODEL_SIZE)

    results = model.train(
        data       = DATA_YAML,
        epochs     = EPOCHS,
        imgsz      = IMG_SIZE,
        batch      = BATCH_SIZE,
        workers    = WORKERS,
        lr0        = LR0,
        lrf        = LRF,
        momentum   = MOMENTUM,
        optimizer  = OPTIMIZER,
        project    = PROJECT_DIR,
        name       = EXPERIMENT,
        exist_ok   = True,

        flipud     = 0.0, 
        fraction   = 1.0,  
        fliplr     = 0.5,    
        mosaic     = 0.0,    
        hsv_h      = 0.015,  
        hsv_s      = 0.7,    
        hsv_v      = 0.4,  

        plots      = True,   
        save       = True,   
        verbose    = True,   
        device     = 0,      

  
        save_period = 10,   
    )

    print("\n" + "="*50)
    print("Обучение завершено!")
    print(f"Результаты сохранены: {PROJECT_DIR}/{EXPERIMENT}/")
    print("="*50)
    return results


def evaluate():

    best_weights = f"runs/detect/{PROJECT_DIR}/{EXPERIMENT}/weights/best.pt"

    if not os.path.exists(best_weights):
        print(f"Веса не найдены: {best_weights}")
        return

    print("\n" + "="*50)
    print("ОЦЕНКА МОДЕЛИ")
    print("="*50)

    model = YOLO(best_weights)

    print("\n► Оценка на VAL наборе:")
    val_results = model.val(
        data    = DATA_YAML,
        split   = "val",
        imgsz   = IMG_SIZE,
        batch   = BATCH_SIZE,
        device  = 0,
        plots   = True,
        project = PROJECT_DIR,
        name    = f"{EXPERIMENT}_eval_val",
        exist_ok = True,
    )
    print_metrics(val_results, "VAL")

    


def print_metrics(results, split_name):
    try:
        box = results.box
        print(f"\n  Метрики на {split_name}:")
        print(f"    mAP50:      {box.map50:.4f}   (mAP при IoU=0.50)")
        print(f"    mAP50-95:   {box.map:.4f}   (mAP при IoU=0.50..0.95)")
        print(f"    Precision:  {box.mp:.4f}")
        print(f"    Recall:     {box.mr:.4f}")
    except Exception as e:
        print(f"  Не удалось вывести метрики: {e}")



if __name__ == "__main__":
    if check_setup():
        train()
        evaluate()