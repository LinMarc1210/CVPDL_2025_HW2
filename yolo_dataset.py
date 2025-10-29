import os
import shutil
from PIL import Image, ImageDraw
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

ORIGINAL_IMG_DIR = "train/"
ORIGINAL_GT_DIR = "train/"

YOLO_DATASET_DIR = "drone_yolo_dataset"
TRAIN_IMG_PATH = os.path.join(YOLO_DATASET_DIR, "yolo_train", "images")
TRAIN_LABEL_PATH = os.path.join(YOLO_DATASET_DIR, "yolo_train", "labels")
VAL_IMG_PATH = os.path.join(YOLO_DATASET_DIR, "yolo_val", "images")
VAL_LABEL_PATH = os.path.join(YOLO_DATASET_DIR, "yolo_val", "labels")
VAL_SPLIT_RATIO = 0.2


def convert_coordinates(img_size, box):
    # box = [x_min, y_min, w, h]
    W, H = img_size
    x, y, w, h = box

    x_center = (x + w / 2.0) / W
    y_center = (y + h / 2.0) / H
    w_norm = w / W
    h_norm = h / H

    return x_center, y_center, w_norm, h_norm

def visualize_one_image():
    try:
        gt_files = [f for f in os.listdir(
            ORIGINAL_GT_DIR) if f.endswith('.txt')]
        if not gt_files:
            print(f"[ERROR] cannot find any .txt annotation files in {ORIGINAL_GT_DIR}")
            return
        txt_name = random.choice(gt_files)
        base_name = os.path.splitext(txt_name)[0]
        img_name = f"{base_name}.png"
        img_path = os.path.join(ORIGINAL_IMG_DIR, img_name)
        txt_path = os.path.join(ORIGINAL_GT_DIR, txt_name)
        if not os.path.exists(img_path) or not os.path.exists(txt_path):
            print(f"[ERROR] cannot find corresponding file {img_name} or {txt_name}")
            return
    except Exception as e:
        print(f"[ERROR] Error reading file list: {e}")
        return

    try:
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            try:
                class_id, x, y, w, h = map(float, line.strip().split(","))
                class_id = int(class_id)
                x_min = x
                y_min = y
                x_max = x + w
                y_max = y + h
                draw.rectangle([x_min, y_min, x_max, y_max],
                               outline="red", width=2)
                draw.text((x_min, y_min - 10), str(class_id), fill="red")
            except Exception as e:
                print(f"[ERROR] Error parsing line in {txt_name}: {e}")
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.title(f"Ground Truth: {img_name}")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"[ERROR] Error processing image {img_path}: {e}")


def process_dataset():
    os.makedirs(TRAIN_IMG_PATH, exist_ok=True)
    os.makedirs(TRAIN_LABEL_PATH, exist_ok=True)
    os.makedirs(VAL_IMG_PATH, exist_ok=True)
    os.makedirs(VAL_LABEL_PATH, exist_ok=True)
    
    gt_files = [f for f in os.listdir(ORIGINAL_GT_DIR) if f.endswith('.txt')]
    random.shuffle(gt_files)
    split_index = int(len(gt_files) * (1 - VAL_SPLIT_RATIO))
    train_files = gt_files[:split_index]
    val_files = gt_files[split_index:]
    
    print(f"[INFO] LABELED files: {len(gt_files)}")
    print(f"[INFO] TRAIN set files: {len(train_files)}")
    print(f"[INFO] VAL set files: {len(val_files)}")

    for file_list, img_out_path, label_out_path in [
        (train_files, TRAIN_IMG_PATH, TRAIN_LABEL_PATH),
        (val_files, VAL_IMG_PATH, VAL_LABEL_PATH)
    ]:
        print(f"\n[INFO] processing {label_out_path}...")
        for txt_name in tqdm(file_list):
            base_name = os.path.splitext(txt_name)[0]
            img_name = f"{base_name}.png"
            img_path_in = os.path.join(ORIGINAL_IMG_DIR, img_name)
            txt_path_in = os.path.join(ORIGINAL_GT_DIR, txt_name)
            if not os.path.exists(img_path_in) or not os.path.exists(txt_path_in):
                print(f"[ERROR] cannot find {img_name} or {txt_name}, skipping.")
                continue
            try:
                with Image.open(img_path_in) as img:
                    W, H = img.size
            except Exception as e:
                print(f"[ERROR] cannot read image {img_path_in}: {e}")
                continue
            shutil.copy(img_path_in, os.path.join(img_out_path, img_name))
            
            yolo_labels = []
            with open(txt_path_in, 'r') as f_in:
                lines = f_in.readlines()
                if not lines:
                    continue
                for line in lines:
                    try:
                        class_id, x, y, w, h = map(
                            float, line.strip().split(","))

                        class_id = int(class_id)
                        xc_norm, yc_norm, w_norm, h_norm = convert_coordinates(
                            (W, H), (x, y, w, h))
                        yolo_labels.append(
                            f"{class_id} {xc_norm} {yc_norm} {w_norm} {h_norm}")
                    except Exception as e:
                        print(f"[ERROR] Error processing {txt_name}: {e}")
            if yolo_labels:
                with open(os.path.join(label_out_path, txt_name), 'w') as f_out:
                    f_out.write("\n".join(yolo_labels))
    print("\n[INFO] Dataset conversion completed!")


if __name__ == "__main__":
    process_dataset()
    # check ground truth
    # visualize_one_image()
