from torchvision.tv_tensors import BoundingBoxes
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.tv_tensors import BoundingBoxes
import torchvision.transforms.v2 as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchmetrics.detection import MeanAveragePrecision

import os
from datetime import datetime
from PIL import Image
import argparse
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return [], []
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

class DroneDataset(Dataset):
    def __init__(self, img_dir, gt_dir, transform=None, is_test=False):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.is_test = is_test
        self.gt_dict = {}    # {frame: {class_id:[[x, y, w, h], ...], ...}}
        self.img_files = [
            os.path.join(img_dir, f"img{str(i).zfill(4)}.png") for i in range(1, 951)]
        self.gt_files = [
            os.path.join(gt_dir, f"img{str(i).zfill(4)}.txt") for i in range(1, 951)]
        
        valid_frames = []
        for frame, file in enumerate(self.gt_files, 1):
            if os.path.exists(file):
                self.gt_dict[frame] = {class_id: []
                                       for class_id in range(0, 4)}
                with open(file, "r") as f:
                    lines = f.readlines()
                    if not lines:
                        continue
                    for line in lines:
                        class_id, x, y, w, h = map(
                            float, line.strip().split(","))
                        self.gt_dict[frame][int(class_id)].append([x, y, w, h])
                if any(self.gt_dict[frame].values()):
                    valid_frames.append(frame)
        self.frames = sorted(valid_frames)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        img_path = os.path.join(self.img_dir, f"img{str(frame).zfill(4)}.png")
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"警告：找不到圖片檔案 {img_path}，跳過此幀。")
            return None
        W, H = img.size
        boxes = []
        classes_list = []
        for class_id, box_list in self.gt_dict.get(frame, {}).items():
            for (x, y, w, h) in box_list:
                x1 = max(0, min(W-1, x))
                y1 = max(0, min(H-1, y))
                x2 = max(0, min(W-1, x + w))
                y2 = max(0, min(H-1, y + h))

                if (x2 - x1) > 1.0 and (y2 - y1) > 1.0:
                    boxes.append([x1, y1, x2, y2])
                    classes_list.append(class_id)

        target = {}
        if len(boxes) == 0:
            target["boxes"] = BoundingBoxes(
                torch.empty((0, 4), dtype=torch.float32),
                format="XYXY",
                canvas_size=(H, W)
            )
            target["labels"] = torch.empty((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(classes_list, dtype=torch.int64) + 1
            target["boxes"] = BoundingBoxes(
                boxes_tensor,
                format="XYXY",
                canvas_size=(H, W)
            )
            target["labels"] = labels_tensor
        target["image_id"] = torch.tensor([frame])
        if self.transform:
            img, target = self.transform(img, target)

        return img, target

def get_args():
    parser = argparse.ArgumentParser(
        description="Train on Drone Dataset")
    parser.add_argument('--img_dir', type=str,
                        default="train/", help='Path to image directory')
    parser.add_argument('--gt_dir', type=str,
                        default="train/", help='Path to ground truth directory')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loader workers')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

def train_one_epoch(model, dataloader, optimizer, device, scaler):
    model.train()
    total_loss = 0
    for imgs, targets in dataloader:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.amp.autocast(device_type='cuda'):
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, device):
    model.train()
    total_loss = 0
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]
            with torch.amp.autocast('cuda'):
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())
            total_loss += loss.item()
    return total_loss / len(dataloader)


def calculate_map(model, dataloader, device):
    metric = MeanAveragePrecision(iou_type="bbox").to(device)
    model.eval()
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]
            predictions = model(imgs)
            metric.update(predictions, targets)
    try:
        mAP_dict = metric.compute()
        val_map = mAP_dict['map'].cpu().item()
        return val_map
    except Exception as e:
        print(f"警告：計算 mAP 時出錯（可能是驗證集上沒有任何預測或標註）：{e}")
        return 0.0


def main():
    args = get_args()
    set_seed(args.seed)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {DEVICE}")

    # train & validate
    transform = T.Compose([
        T.ToImage(),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomShortestSize(min_size=(800, 800),
                             max_size=1200, antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    full_dataset = DroneDataset(
        img_dir=args.img_dir, gt_dir=args.gt_dir, transform=transform)

    N_TOTAL = len(full_dataset)
    N_VAL = int(0.2 * N_TOTAL)
    N_TRAIN = N_TOTAL - N_VAL

    train_dataset, val_dataset = random_split(full_dataset, [N_TRAIN, N_VAL])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    print(f"數據集大小: 總計 {N_TOTAL} 幀 | 訓練 {N_TRAIN} 幀 | 驗證 {N_VAL} 幀")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    MODEL_BASE_NAME = f"stage1_drone_{timestamp}"
    anchor_sizes = ((64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=None,
    )
    model.rpn.anchor_generator = rpn_anchor_generator

    num_classes = 5
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    print("-> 實施 MobileNetV3-FPN 微調策略：BackBone 所有層都參與微調。")
    for param in model.parameters():
        param.requires_grad = True

    model.to(DEVICE)

    # optimizer & learning rate scheduler & GPU gradient scaler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2)
    scaler = torch.amp.GradScaler('cuda')

    log_dir = f"runs/{MODEL_BASE_NAME}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"-> TensorBoard 日誌將儲存於: {log_dir}")

    epochs = args.epochs
    PATIENCE = args.patience
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print("-> 開始訓練...")
    os.makedirs("checkpoint", exist_ok=True)

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, DEVICE, scaler)
        val_loss = validate_one_epoch(model, val_loader, DEVICE)
        scheduler.step(val_loss)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar(
            'LearningRate', optimizer.param_groups[0]['lr'], epoch)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.to("cpu")
            best_model_path = f"checkpoint/{MODEL_BASE_NAME}_best.pth"
            torch.save(model_to_save.state_dict(), best_model_path)
            print(f"*** 驗證損失改善，儲存最佳模型於 {best_model_path} ***")
            model.to(DEVICE)
        else:
            epochs_no_improve += 1
            print(f"驗證損失未改善，已連續 {epochs_no_improve} 個 Epoch。")

        if epochs_no_improve >= PATIENCE:
            print(f"\n📢 早期停止：驗證損失連續 {PATIENCE} 個 Epoch 未改善。停止訓練。")
            break

    writer.close()
    print("所有訓練已完成。")

if __name__ == '__main__':
    main()
