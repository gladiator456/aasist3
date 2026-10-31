"""Score an ASVspoof 2019 LA split with a trained checkpoint and report EER / ROC-AUC.

Use this on the held-out *eval* split for the number you report: TRAIN.py picks
best.pt by dev EER, so the dev EER it prints is slightly optimistic.

    DATASET_ROOT=/path/to/LA CHECKPOINT=checkpoints/best.pt python EVAL.py
    SPLIT=dev ...          # score the dev split instead
    MAX_SAMPLES=2000 ...   # quick check on a random subset
    SCORES_OUT=scores.txt  # also write "utterance_id score label" per line
"""
import os
import glob
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score

from DATAUTLIS import genSpoof_list, Dataset_ASVspoof2019_train
from MODELDEFINITION import Model, DEFAULT_D_ARGS

SPLIT = os.environ.get("SPLIT", "eval").lower()
if SPLIT not in {"eval", "dev"}:
    raise ValueError("SPLIT must be 'eval' or 'dev'")
CHECKPOINT = os.environ.get("CHECKPOINT") or os.path.join(os.environ.get("CHECKPOINT_DIR", "./checkpoints"), "best.pt")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0"))
SCORES_OUT = os.environ.get("SCORES_OUT", "")
SEED = int(os.environ.get("SEED", "42"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

audio_name = f"ASVspoof2019_LA_{SPLIT}"
protocol_name = f"ASVspoof2019.LA.cm.{SPLIT}.trl.txt"


def find_root():
    candidates = glob.glob(f"/kaggle/input/**/{audio_name}", recursive=True)
    for base in ("./LA", "./data/LA", "./data"):
        if os.path.isdir(base):
            candidates.extend(glob.glob(os.path.join(base, "**", audio_name), recursive=True))
    for candidate in candidates:
        root = os.path.dirname(candidate)
        if os.path.isdir(os.path.join(root, "ASVspoof2019_LA_cm_protocols")):
            return root
    return None


root = os.environ.get("DATASET_ROOT") or find_root()
if not root:
    raise FileNotFoundError(f"Could not find {audio_name}; set DATASET_ROOT to the folder that contains it.")
audio_dir = os.path.join(root, audio_name, "flac")
protocol = os.path.join(root, "ASVspoof2019_LA_cm_protocols", protocol_name)
for required in (audio_dir, protocol, CHECKPOINT):
    if not os.path.exists(required):
        raise FileNotFoundError(f"Required path is missing: {required}")

labels, ids = genSpoof_list(protocol, is_train=False)
if MAX_SAMPLES > 0:
    random.Random(SEED).shuffle(ids)
    ids = ids[:MAX_SAMPLES]
n_bona = sum(labels[i] for i in ids)
print(f"{SPLIT}: {len(ids)} clips | bonafide={n_bona} spoof={len(ids) - n_bona} | device={DEVICE}")
if n_bona == 0 or n_bona == len(ids):
    raise RuntimeError("Only one class in the selected clips; EER is undefined. Use a larger MAX_SAMPLES.")

ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
if ckpt.get("d_args") != DEFAULT_D_ARGS:
    raise RuntimeError(f"{CHECKPOINT} was trained with different model settings; it cannot be loaded into the current model.")
model = Model(DEFAULT_D_ARGS).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded {CHECKPOINT} (epoch {ckpt.get('epoch')}, recorded best dev EER {ckpt.get('best_dev_eer', float('nan')):.4f})")

dataset = Dataset_ASVspoof2019_train(ids, labels, audio_dir, is_train=False)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                    pin_memory=DEVICE.type == "cuda")
scores, truth = [], []
with torch.no_grad():
    for step, (waves, y) in enumerate(loader, start=1):
        _, logits = model(waves.to(DEVICE, non_blocking=True))
        scores.extend((logits[:, 1] - logits[:, 0]).float().cpu().tolist())  # higher = more bonafide
        truth.extend(y.tolist())
        if step % 200 == 0:
            print(f"  scored {len(scores)}/{len(ids)}", flush=True)

fpr, tpr, _ = roc_curve(truth, scores, pos_label=1)
fnr = 1.0 - tpr
idx = int(np.nanargmin(np.abs(fnr - fpr)))
eer = float((fnr[idx] + fpr[idx]) / 2.0)
auc = float(roc_auc_score(truth, scores))
print(f"{SPLIT} EER = {eer * 100:.2f}% | ROC-AUC = {auc:.4f} | clips = {len(scores)}")
print("Note: only the first ~4 s of each clip is scored, and min t-DCF is not computed (it needs the ASV score files).")

if SCORES_OUT:
    with open(SCORES_OUT, "w") as f:
        for key, score, y in zip(ids, scores, truth):
            f.write(f"{key} {score:.6f} {y}\n")
    print(f"Wrote scores to {SCORES_OUT}")
