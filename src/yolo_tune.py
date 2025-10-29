from ultralytics import YOLO
import os

DATA_YAML_PATH = 'yolo_drone.yaml'
MODEL_ARCH = 'yolo11s.yaml'
IMG_SIZE = 1920
EPOCHS = 20

def tune_yolo():
    os.chdir(os.getcwd())
    model = YOLO(MODEL_ARCH)
    space = {
        'lr0': (1e-4, 1e-1),
        'weight_decay': (0.0, 0.001),
        'scale': (0.0, 0.9),
    }

    print("[INFO] start tuning...")
    results = model.tune(
        data=DATA_YAML_PATH,
        imgsz=IMG_SIZE,
        rect=False,
        batch=-1,
        epochs=EPOCHS,
        iterations=10,
        space=space,
        device=0
    )
    
if __name__ == '__main__':
    tune_yolo()
