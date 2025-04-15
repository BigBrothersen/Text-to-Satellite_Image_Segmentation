import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from UNET import UNet
from dataset import SatelliteDataset

# Config
lr = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16
num_epochs = 75
num_workers = min(4, os.cpu_count())
height = 256 # 2448 originally
width = 256 # 2448 originally
pin_memory = True
load_model = False

# Directories
train_image_dir = "data/training_data/images"
train_mask_dir = "data/training_data/masks"
test_image_dir = "data/test_data/images"
test_mask_dir = "data/test_data/masks"

# ImageNet mean/std for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = A.Compose([
    A.Resize(height, width),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Resize(height, width),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

train_dataset = SatelliteDataset(
    image_dir=train_image_dir,
    mask_dir=train_mask_dir,
    transform=train_transform
)

test_dataset = SatelliteDataset(
    image_dir=test_image_dir,
    mask_dir=test_mask_dir,
    transform=test_transform
)

# train_dataset.summarize()
# train_dataset.display_sample(1)

from tqdm import tqdm

def calculate_mean_iou(preds, targets, num_classes=7):
    """Calculates the mean Intersection over Union (IoU) for multi-class segmentation."""
    iou = 0
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union > 0:
            iou += intersection / union
    
    return  iou / num_classes

def calculate_per_class_iou(preds, targets, num_classes=7):
    ious = [0] * num_classes
    counts = [0] * num_classes
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union > 0:
            ious[cls] += intersection / union
            counts[cls] += 1
    return [ious[cls] / counts[cls] if counts[cls] > 0 else 0 for cls in range(num_classes)]

def train_epoch(loader, model, optimizer, loss_fn, scaler, num_classes=7):
    model.train()
    loop = tqdm(loader, desc="Training")
    running_loss = 0.0
    class_ious = [0.0] * num_classes
    class_counts = [0] * num_classes

    for batch_idx, (data, target) in enumerate(loop):
        data = data.to(device=device)
        target = target.long().to(device=device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())
        running_loss += loss.item()

        # Calculate per-class IoU for this batch
        preds = torch.argmax(predictions, dim=1)
        batch_ious = calculate_per_class_iou(preds, target, num_classes)
        for cls in range(num_classes):
            class_ious[cls] += batch_ious[cls]
            if batch_ious[cls] > 0:
                class_counts[cls] += 1

    avg_class_ious = [class_ious[cls] / class_counts[cls] if class_counts[cls] > 0 else 0 for cls in range(num_classes)]
    return running_loss / len(loader), avg_class_ious

def eval_model(loader, model, loss_fn, device, num_classes=7):
    model.eval()
    loop = tqdm(loader, desc="Evaluating")
    total_loss = 0
    class_ious = [0.0] * num_classes
    class_counts = [0] * num_classes
    
    with torch.no_grad():
        for data, target in loop:  # Fixed enumeration
            data = data.to(device)
            target = target.long().to(device)
            
            predictions = model(data)
            loss = loss_fn(predictions, target)
            total_loss += loss.item()
            
            # Calculate IoUs
            preds = torch.argmax(predictions, dim=1)
            batch_ious = calculate_per_class_iou(preds, target, num_classes)
            for cls in range(num_classes):
                if batch_ious[cls] > 0:  # Only count if class appears
                    class_ious[cls] += batch_ious[cls]
                    class_counts[cls] += 1
    
    avg_loss = total_loss / len(loader)
    avg_class_ious = [class_ious[cls]/class_counts[cls] if class_counts[cls]>0 else 0 
                     for cls in range(num_classes)]
    mean_iou = sum(avg_class_ious) / num_classes
    
    return avg_loss, mean_iou, avg_class_ious

from torch.utils.data import DataLoader
from torch import optim
from torch.cuda.amp import GradScaler
import torch.nn as nn


def main():

    torch.cuda.empty_cache()

    LABELS_MAPPING = {
        0: 'Urban',
        1: 'Agriculture',
        2: 'Rangeland',
        3: 'Forest',
        4: 'Water',
        5: 'Barren',
        6: 'Unknown'
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    model = UNet(
        in_channels=3,
        num_classes=7,
        features=[64, 128, 256, 512]
    ).to(device=device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    num_epochs = 75
    best_miou = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss, train_class_ious = train_epoch(train_loader, model, optimizer, loss_fn, scaler, num_classes=7)
        val_loss, val_miou, val_class_ious = eval_model(test_loader, model, loss_fn, device)

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}")

        # Update performance history in the model
        model.update_performance_history(val_miou, val_class_ious, epoch+1)

        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved!")

    print("Training complete.")
    model.plot_performance()
    model.plot_class_IoU(LABELS_MAPPING)

if __name__ == '__main__':
    main()