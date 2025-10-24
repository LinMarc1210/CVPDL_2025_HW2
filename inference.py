import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T  # 【修改】匯入 v2
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import os
import random
import numpy as np
import pandas as pd
import glob


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


TEST_IMG_DIR = "test/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONF_THRESHOLD = 0.0
CONF_LIMIT = 1.0


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return [], []
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


class TestDataset(Dataset):
    """
    用於推論的 Dataset，只讀取圖片並提取 frame ID。
    """

    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith('.png')])
        self.frames = [int(os.path.splitext(f)[0][3:]) for f in self.img_files]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        img_path = os.path.join(
            self.img_dir, f"img{str(frame).zfill(4)}.png")

        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"警告：找不到圖片 {img_path}")
            return None

        if self.transform:
            img = self.transform(img)

        return img, frame


def load_inference_model(model_path, device, num_classes=5):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"模型權重已從 {model_path} 成功加載。")
    else:
        raise FileNotFoundError(f"錯誤：找不到模型檔案 {model_path}")

    model.to(device)
    model.eval()
    return model


def inference_and_submit(model, test_dir, conf_threshold, conf_limit, device):
    set_seed(42)

    test_transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = TestDataset(
        img_dir=test_dir, transform=test_transform)

    if len(test_dataset) == 0:
        print(f"錯誤：在 {test_dir} 中找不到任何 .png 圖片。")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    predictions = []

    with torch.no_grad():
        for imgs, frames in test_loader:
            imgs = [img.to(device) for img in imgs]
            preds = model(imgs)

            for pred, frame_id in zip(preds, frames):
                img_id = frame_id
                parts = []

                for score, box, label in zip(pred['scores'], pred['boxes'], pred['labels']):
                    if score < conf_threshold or score > conf_limit:
                        continue
                    x_min, y_min, x_max, y_max = box.cpu().tolist()
                    w = x_max - x_min
                    h = y_max - y_min

                    class_id_csv = label.cpu().item() - 1

                    parts.append(
                        f"{score:.6f} {x_min:.2f} {y_min:.2f} {w:.2f} {h:.2f} {class_id_csv}")
                if parts:
                    pred_str = " ".join(parts)
                else:
                    pred_str = "0.000000 0.00 0.00 0.00 0.00 0"
                predictions.append([int(img_id), pred_str])

    submission = pd.DataFrame(predictions, columns=[
                              'Image_ID', 'PredictionString'])
    submission = submission.sort_values(by='Image_ID')

    submission.to_csv('answer.csv', index=False, sep=',')
    print("✅ Submission 檔案已儲存為：answer.csv")


if __name__ == '__main__':
    try:
        latest_checkpoint = max(
            glob.glob("checkpoint/*.pth"),
            key=os.path.getctime
        )
        print(f"-> 找到最新模型: {latest_checkpoint}")
    except ValueError:
        print("錯誤：在 checkpoint/ 資料夾中找不到任何 .pth 模型檔案！")
        exit()

    trained_model = load_inference_model(
        latest_checkpoint, DEVICE, num_classes=5)

    inference_and_submit(trained_model, TEST_IMG_DIR, CONF_THRESHOLD, CONF_LIMIT, DEVICE)
