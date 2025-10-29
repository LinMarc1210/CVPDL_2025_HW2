from ultralytics import YOLO
import torch
import os

DATA_YAML_PATH = 'yolo_drone.yaml'
MODEL_ARCH = 'yolo11s.yaml'
IMG_SIZE = 1920
BATCH_SIZE = 4
EPOCHS = 200


def train_yolo_from_scratch():
    print(f"[INFO] using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    if not os.path.exists(DATA_YAML_PATH):
        print(f"[ERROR] cannot find {DATA_YAML_PATH}")
        return

    print(f"[INFO] building {MODEL_ARCH} model (From Scratch)...")
    model = YOLO(MODEL_ARCH)

    print(f"[INFO] start training...")
    print(f"[INFO] current working directory: {os.getcwd()}")
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        pretrained=False,
        rect=True,
        device=0,
        lr0=0.009,
        weight_decay=0.0003,
        mosaic=1.0,
        name=f'{MODEL_ARCH.split(".")[0]}_{IMG_SIZE}_scratch',
        project=os.path.join(os.getcwd(), 'runs', 'detect'),
    )

    print(f"[INFO] training completed!")
    print(f"[INFO] best model saved at: {results.save_dir}")


if __name__ == '__main__':
    train_yolo_from_scratch()
