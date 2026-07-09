import os
import glob
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from DATAUTLIS import genSpoof_list, Dataset_ASVspoof2019_train
from MODELDEFINITION import Model

print("=" * 60)
print("       AASIST3 KAN-UPGRADE TRAINING ENGINE STARTING")
print("=" * 60)

# 1. HARDWARE & DEVICE CHECK
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")
USE_AMP = device.type == "cuda"

# 2. HYPERPARAMETERS
BATCH_SIZE = 32
NUM_EPOCHS = 25
BASE_LR = 1e-4
MIN_LR = 5e-6
WEIGHT_DECAY = 1e-4
CLASS_WEIGHTS = [0.1, 0.9]  # index 0 = spoof, index 1 = bonafide (see genSpoof_list)
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 3. DATASET PATH AUTO-DETECT
# Kaggle mounts datasets at /kaggle/input/<dataset-slug>/... — the exact slug
# depends on what you named it when you added it, so we search for it instead
# of hardcoding a path that will only work for one specific username/slug.
def find_dataset_root():
    candidates = glob.glob("/kaggle/input/**/ASVspoof2019_LA_train", recursive=True)
    if candidates:
        return os.path.dirname(candidates[0])
    if os.path.exists("./LA/ASVspoof2019_LA_train"):
        return "./LA"
    return None

DATASET_ROOT = find_dataset_root()
if DATASET_ROOT is None:
    print("\nCould not auto-locate the LA dataset under /kaggle/input.")
    print("Available inputs:", os.listdir("/kaggle/input") if os.path.exists("/kaggle/input") else "none")
    raise FileNotFoundError(
        "Set DATASET_ROOT manually to the folder that contains "
        "ASVspoof2019_LA_train / ASVspoof2019_LA_dev / ASVspoof2019_LA_cm_protocols"
    )
print(f"Using dataset root: {DATASET_ROOT}")

TRAIN_AUDIO_DIR = os.path.join(DATASET_ROOT, "ASVspoof2019_LA_train", "flac")
DEV_AUDIO_DIR = os.path.join(DATASET_ROOT, "ASVspoof2019_LA_dev", "flac")

TRAIN_PROTOCOL = os.path.join(DATASET_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt")
DEV_PROTOCOL = os.path.join(DATASET_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.dev.trl.txt")

# 4. DATA LOADERS
print("\n[1/4] Parsing dataset protocols...")
train_meta, train_files = genSpoof_list(TRAIN_PROTOCOL, is_train=True)
dev_meta, dev_files = genSpoof_list(DEV_PROTOCOL, is_train=False)

train_dataset = Dataset_ASVspoof2019_train(train_files, train_meta, TRAIN_AUDIO_DIR, is_train=True)
dev_dataset = Dataset_ASVspoof2019_train(dev_files, dev_meta, DEV_AUDIO_DIR, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"Data Loaded! Train Steps/Epoch: {len(train_loader)} | Dev Steps: {len(dev_loader)}")

# 5. MODEL CONFIG
d_args = {
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
    "gat_dims": [24, 32],
    "pool_ratios": [0.4, 0.5, 0.7, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}

print("\n[2/4] Initializing AASIST3 model with KAN modules...")
model = Model(d_args).to(device)

# 6. OPTIMIZER, CRITERION, SCHEDULER
criterion = nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

total_steps = NUM_EPOCHS * len(train_loader)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=MIN_LR)
scaler = torch.amp.GradScaler(enabled=USE_AMP)


def run_validation():
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch_x, batch_y in dev_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=USE_AMP):
                _, batch_out = model(batch_x, Freq_aug=False)
                loss = criterion(batch_out, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(batch_out, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    return total_loss / total, correct / total


# 7. TRAINING LOOP
print("\n[3/4] Starting Training Execution Loop...")
best_dev_acc = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    epoch_start = time.time()

    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=USE_AMP):
            _, batch_out = model(batch_x, Freq_aug=True)
            loss = criterion(batch_out, batch_y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * batch_x.size(0)

    epoch_time = time.time() - epoch_start
    avg_loss = running_loss / len(train_dataset)

    dev_loss, dev_acc = run_validation()

    print(
        f"Epoch [{epoch+1:03d}/{NUM_EPOCHS:03d}] | "
        f"Train Loss: {avg_loss:.5f} | Dev Loss: {dev_loss:.5f} | Dev Acc: {dev_acc:.4f} | "
        f"Time: {int(epoch_time)}s | LR: {optimizer.param_groups[0]['lr']:.2e}"
    )

    # checkpoint every epoch (small models; adjust if disk space is tight)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "last.pt"))
    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best.pt"))
        print(f"  -> New best dev accuracy ({dev_acc:.4f}), saved best.pt")

print(f"\n[4/4] Training Complete! Best dev accuracy: {best_dev_acc:.4f}")
print(f"Checkpoints saved in: {CHECKPOINT_DIR}")
