import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Teri unique repository files se modules import kiye
from DATAUTLIS import genSpoof_list, ASVspoofLADataset
from MODELDEFINITION import Model  # Tera complete KAN AASIST3 Model

print("="*60)
print("       AASIST3 KAN-UPGRADE TRAINING ENGINE STARTING")
print("="*60)

# 1. HARDWARE & DEVICE CHECK
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# 2. HYPERPARAMETERS & CONFIGS (Article & Notebook standard match)
BATCH_SIZE = 32
NUM_EPOCHS = 25
BASE_LR = 1e-4
MIN_LR = 5e-6
WEIGHT_DECAY = 1e-4
CLASS_WEIGHTS = [0.1, 0.9]

# Dataset location setup (Auto-picks Kaggle dataset path)
DATASET_ROOT = "/kaggle/input/datasets/mahimyadav2006/ladataset/LA"
if not os.path.exists(DATASET_ROOT):
    # Fallback local path agar tu local machine par test kare
    DATASET_ROOT = "./LA"

TRAIN_AUDIO_DIR = os.path.join(DATASET_ROOT, "ASVspoof2019_LA_train/flac")
DEV_AUDIO_DIR = os.path.join(DATASET_ROOT, "ASVspoof2019_LA_dev/flac")

TRAIN_PROTOCOL = os.path.join(DATASET_ROOT, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
DEV_PROTOCOL = os.path.join(DATASET_ROOT, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt")

# 3. PREPARE DATA LOADERS
print("\n[1/4] Parsing dataset protocols...")
train_meta, train_files = genSpoof_list(TRAIN_PROTOCOL, is_train=True)
dev_meta, dev_files = genSpoof_list(DEV_PROTOCOL, is_train=False)

train_dataset = ASVspoofLADataset(train_files, train_meta, TRAIN_AUDIO_DIR, training=True)
dev_dataset = ASVspoofLADataset(dev_files, dev_meta, DEV_AUDIO_DIR, training=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"Data Loaded! Train Steps/Epoch: {len(train_loader)} | Dev Steps: {len(dev_loader)}")

# 4. INITIALIZE AASIST3 CONFIG ARGS
d_args = {
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
    "gat_dims": [24, 32],
    "pool_ratios": [0.4, 0.5, 0.7, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0]
}

print("\n[2/4] Initializing AASIST3 model with KAN modules...")
model = Model(d_args).to(device)

# 5. OPTIMIZER, CRITERION & COSINE SCHEDULER
criterion = nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

total_steps = NUM_EPOCHS * len(train_loader)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=MIN_LR)
scaler = torch.amp.GradScaler(enabled=True)

# 6. EXECUTING TRAINING LOOP
print("\n[3/4] Starting Training Execution Loop...")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    epoch_start = time.time()
    
    for batch_idx, (batch_x, batch_y, _, _, _) in enumerate(train_loader):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed Precision forward pass for fast execution
        with torch.amp.autocast(device_type="cuda", enabled=True):
            _, batch_out = model(batch_x, Freq_aug=True)
            loss = criterion(batch_out, batch_y)
            
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        running_loss += loss.item() * batch_x.size(0)
        
    epoch_time = time.time() - epoch_start
    avg_loss = running_loss / len(train_dataset)
    
    print(f"Epoch [{epoch+1:03d}/{NUM_EPOCHS:03d}] | Train Loss: {avg_loss:.5f} | Time: {int(epoch_time)}s | LR: {optimizer.param_groups[0]['lr']:.2e}")

print("\n[4/4] Training Complete! Model state frozen successfully.")
