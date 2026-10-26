"""Train the KAN-based ASVspoof model on ASVspoof2019 LA."""
import os
import glob
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve

from DATAUTLIS import genSpoof_list, Dataset_ASVspoof2019_train
from MODELDEFINITION import Model

SEED = int(os.environ.get("SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda" and os.environ.get("USE_AMP", "0") == "1"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "25"))
BASE_LR = float(os.environ.get("BASE_LR", "1e-4"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
# Quick-test knobs: 0 means "use everything". A seeded random subset keeps both classes.
MAX_TRAIN_SAMPLES = int(os.environ.get("MAX_TRAIN_SAMPLES", "0"))
MAX_DEV_SAMPLES = int(os.environ.get("MAX_DEV_SAMPLES", "0"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "200"))
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "./checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"Device={DEVICE} | AMP={USE_AMP} | batch={BATCH_SIZE} | epochs={NUM_EPOCHS}")
if DEVICE.type == "cuda" and not USE_AMP:
    print("AMP is disabled by default for maximum KAN numerical compatibility; set USE_AMP=1 after a successful baseline run.")

def find_dataset_root():
    candidates = glob.glob("/kaggle/input/**/ASVspoof2019_LA_train", recursive=True)
    # The official ZIP may contain an extra parent directory, so search only
    # likely local data roots rather than recursively scanning the whole repo.
    for base in ("./LA", "./data/LA", "./data"):
        if os.path.isdir(base):
            candidates.extend(glob.glob(os.path.join(base, "**", "ASVspoof2019_LA_train"), recursive=True))
    if os.path.isdir("./ASVspoof2019_LA_train"):
        candidates.append("./ASVspoof2019_LA_train")
    for candidate in candidates:
        root = os.path.dirname(candidate)
        if (os.path.isdir(os.path.join(root, "ASVspoof2019_LA_dev"))
                and os.path.isdir(os.path.join(root, "ASVspoof2019_LA_cm_protocols"))):
            return root
    return None

root = os.environ.get("DATASET_ROOT") or find_dataset_root()
if not root:
    raise FileNotFoundError(
        "ASVspoof LA dataset not found. Set DATASET_ROOT to the folder containing "
        "ASVspoof2019_LA_train, ASVspoof2019_LA_dev and ASVspoof2019_LA_cm_protocols."
    )

train_audio = os.path.join(root, "ASVspoof2019_LA_train", "flac")
dev_audio = os.path.join(root, "ASVspoof2019_LA_dev", "flac")
protocol_dir = os.path.join(root, "ASVspoof2019_LA_cm_protocols")
train_protocol = os.path.join(protocol_dir, "ASVspoof2019.LA.cm.train.trn.txt")
dev_protocol = os.path.join(protocol_dir, "ASVspoof2019.LA.cm.dev.trl.txt")
for required in (train_audio, dev_audio, train_protocol, dev_protocol):
    if not os.path.exists(required):
        raise FileNotFoundError(f"Required dataset path is missing: {required}")

train_meta, train_ids = genSpoof_list(train_protocol, is_train=True)
dev_meta, dev_ids = genSpoof_list(dev_protocol, is_train=False)
if not train_ids or not dev_ids:
    raise RuntimeError("Train or dev protocol contains no examples.")
if MAX_TRAIN_SAMPLES > 0:
    random.Random(SEED).shuffle(train_ids)
    train_ids = train_ids[:MAX_TRAIN_SAMPLES]
if MAX_DEV_SAMPLES > 0:
    random.Random(SEED).shuffle(dev_ids)
    dev_ids = dev_ids[:MAX_DEV_SAMPLES]
for name, ids, meta in (("train", train_ids, train_meta), ("dev", dev_ids, dev_meta)):
    n_bona = sum(meta[i] for i in ids)
    print(f"{name}: {len(ids)} clips | bonafide={n_bona} spoof={len(ids) - n_bona}")
    if n_bona == 0 or n_bona == len(ids):
        raise RuntimeError(f"The {name} subset has only one class; EER is undefined. Use a larger subset.")
train_ds = Dataset_ASVspoof2019_train(train_ids, train_meta, train_audio, is_train=True)
dev_ds = Dataset_ASVspoof2019_train(dev_ids, dev_meta, dev_audio, is_train=False)
loader_args = dict(num_workers=NUM_WORKERS, pin_memory=USE_AMP, persistent_workers=NUM_WORKERS > 0)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, **loader_args)
dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_args)
print(f"Dataset root: {root}\nTrain examples: {len(train_ds)} | Dev examples: {len(dev_ds)}")

d_args = {
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
    "gat_dims": [24, 32],
    "pool_ratios": [0.4, 0.5, 0.7, 0.5],
    "temperatures": [2.0, 2.0, 1.0, 1.0],
}
model = Model(d_args).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9], dtype=torch.float32, device=DEVICE))
optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, NUM_EPOCHS * len(train_loader)), eta_min=5e-6)
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

def autocast_context():
    return torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP)

def evaluate():
    model.eval()
    total_loss, total, correct = 0.0, 0, 0
    labels_all, scores_all = [], []
    with torch.no_grad():
        for waves, labels in dev_loader:
            waves = waves.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            with autocast_context():
                _, logits = model(waves)
                loss = criterion(logits, labels)
            total_loss += loss.item() * waves.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
            labels_all.extend(labels.cpu().tolist())
            # Higher score means more likely bonafide (label 1).
            scores_all.extend((logits[:, 1] - logits[:, 0]).float().cpu().tolist())
    fpr, tpr, _ = roc_curve(labels_all, scores_all, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fnr[idx] + fpr[idx]) / 2.0)
    return total_loss / max(total, 1), correct / max(total, 1), eer

def save_checkpoint(path, epoch, best_eer, dev_loss, dev_acc):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_dev_eer": best_eer,
        "dev_loss": dev_loss,
        "dev_accuracy": dev_acc,
        "d_args": d_args,
        "seed": SEED,
    }, path)

best_eer = float("inf")
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    started = time.time()
    for step, (waves, labels) in enumerate(train_loader, start=1):
        waves = waves.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            _, logits = model(waves)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}, step {step}: {loss.item()}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        epoch_loss += loss.item() * waves.size(0)
        if LOG_EVERY > 0 and step % LOG_EVERY == 0:
            print(f"  epoch {epoch} step {step}/{len(train_loader)} | loss={loss.item():.4f} | {time.time() - started:.0f}s elapsed", flush=True)

    dev_loss, dev_acc, dev_eer = evaluate()
    elapsed = int(time.time() - started)
    avg_train_loss = epoch_loss / len(train_ds)
    save_checkpoint(os.path.join(CHECKPOINT_DIR, "last.pt"), epoch, best_eer, dev_loss, dev_acc)
    print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | train_loss={avg_train_loss:.5f} | dev_loss={dev_loss:.5f} | dev_acc={dev_acc:.4f} | dev_EER={dev_eer:.4f} | {elapsed}s")
    if dev_eer < best_eer:
        best_eer = dev_eer
        save_checkpoint(os.path.join(CHECKPOINT_DIR, "best.pt"), epoch, best_eer, dev_loss, dev_acc)
        print(f"  Saved new best checkpoint (dev EER={best_eer:.4f})")

print(f"Training complete. Best dev EER={best_eer:.4f}. Checkpoints: {CHECKPOINT_DIR}")
