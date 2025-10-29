from ultralytics import YOLO
import torch
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

MODEL_PATH = 'runs/detect/yolo11s_1920_scratch2/weights/best.pt'
TEST_IMG_DIR = "test/"
CONF_THRESHOLD = 0.0
NMS_IOU_THRESHOLD = 0.7
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 1920
BATCH_SIZE = 1
USE_HALF_PRECISION = True


def inference_and_submit_yolo(model_path, test_dir, conf_threshold, device, imgsz, batch_size, use_half):
    set_seed(42)
    try:
        print(f"[INFO] Loading YOLO model from: {model_path}")
        model = YOLO(model_path)
        model.to(device)
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return

    try:
        test_img_files = sorted([
            os.path.join(test_dir, f)
            for f in os.listdir(test_dir)
            if f.endswith('.png')
        ])
        if not test_img_files:
            print(f"[ERROR] No .png images found in {test_dir}")
            return
        print(f"[INFO] Found {len(test_img_files)} test images.")
        # for test_img_file in test_img_files:
        #     with Image.open(test_img_file) as img:
        #         w, h = img.size
        #         print(f"-> Test image {test_img_file}, size: {w}x{h}")
    except FileNotFoundError:
        print(f"[ERROR] Test image directory not found at {test_dir}")
        return

    predictions = []
    print(f"[INFO] Starting inference with imgsz={imgsz}, conf={conf_threshold}...")

    for i in tqdm(range(0, len(test_img_files), BATCH_SIZE)):
        batch_paths = test_img_files[i: i + BATCH_SIZE]
        results_list = model.predict(
            source=batch_paths,
            imgsz=imgsz,
            conf=conf_threshold,
            device=device,
            half=use_half,
            iou=NMS_IOU_THRESHOLD,
            batch=BATCH_SIZE,
            rect=True,
            verbose=False
        )
        for img_path, results in zip(batch_paths, results_list):
            img_id = int(os.path.splitext(os.path.basename(img_path))[0][3:])
            boxes = results.boxes
            parts = []
            if boxes is not None and len(boxes) > 0:
                for box_idx in range(len(boxes)):
                    score = boxes.conf[box_idx].cpu().item()
                    x_min, y_min, x_max, y_max = boxes.xyxy[box_idx].cpu().tolist()
                    w = x_max - x_min
                    h = y_max - y_min
                    class_id_csv = int(boxes.cls[box_idx].cpu().item())

                    parts.append(
                        f"{score:.6f} {x_min:.2f} {y_min:.2f} {w:.2f} {h:.2f} {class_id_csv}"
                    )
            if not parts:
                pred_str = "0.0 0 0 1 1 0"  # Default prediction if no parts are detected
            else:
                pred_str = " ".join(parts)
            predictions.append([img_id, pred_str])

    submission = pd.DataFrame(predictions, columns=['Image_ID', 'PredictionString'])
    submission = submission.sort_values(by='Image_ID')
    submission.to_csv('answer.csv', index=False, sep=',')
    print("[INFO] Submission file saved as: answer.csv")

if __name__ == '__main__':
    print(f"[INFO] Using device: {DEVICE}")
    inference_and_submit_yolo(MODEL_PATH, TEST_IMG_DIR, CONF_THRESHOLD, 
                              DEVICE, IMG_SIZE, BATCH_SIZE, USE_HALF_PRECISION)
