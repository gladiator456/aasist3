# Full training on Kaggle

## Before starting

- Use a GPU accelerator in Kaggle (T4 x2 or P100). Turn **Internet on** in the notebook settings so the repo and `efficient-kan` can be installed.
- Attach the ASVspoof2019 LA dataset (the `awsaf49/asvpoof-2019-dataset` input has worked before); check the exact `LA/LA` path under `/kaggle/input`.
- **Start a fresh run.** The model architecture changed in the stability update (abs() in the front end, readout LayerNorm, wider classifier grid), so checkpoints from earlier trials, including the old 2-epoch trial's `best.pt`, cannot be loaded with `INIT_FROM` or `RESUME_FROM`. The trainer stops with a clear error if you try.

## Setup (first notebook cell)

```bash
!git clone https://github.com/gladiator456/aasist3.git /kaggle/working/aasist3
%cd /kaggle/working/aasist3
!pip install -q -r requirements.txt
!ls /kaggle/input   # find the dataset folder name
```

## Short trial first (a few minutes)

```bash
!DATASET_ROOT="/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA" \
CHECKPOINT_DIR="/kaggle/working/checkpoints_trial" \
MAX_TRAIN_SAMPLES=2000 MAX_DEV_SAMPLES=1000 NUM_EPOCHS=2 BATCH_SIZE=4 LOG_EVERY=50 WARMUP_STEPS=50 \
python -u TRAIN.py
```

Check that it runs, note GPU memory (`!nvidia-smi`), and confirm `last.pt` / `best.pt` appear. Raise `BATCH_SIZE` (default 8) only if memory allows; lower it if you hit out-of-memory.

## Full run

```bash
%cd /kaggle/working/aasist3

!DATASET_ROOT="/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA" \
CHECKPOINT_DIR="/kaggle/working/checkpoints_full" \
NUM_EPOCHS=25 \
BATCH_SIZE=8 \
BASE_LR=1e-4 \
LOG_EVERY=100 \
NUM_WORKERS=2 \
python -u TRAIN.py
```

Do not set `MAX_TRAIN_SAMPLES` or `MAX_DEV_SAMPLES` for full-dataset training. Their default is 0, meaning use all examples. If `BATCH_SIZE=8` runs out of memory, use 4 or 2.

## What the trainer reports

Each epoch prints training loss, development loss, accuracy, EER, ROC-AUC, a confusion matrix, and recall for both spoof and bonafide classes. Do not use accuracy alone: the dataset is imbalanced. The first epochs use a linear LR warm-up (`WARMUP_STEPS`, default 500) before cosine decay.

- `checkpoints_full/best.pt`: checkpoint with the lowest development EER seen in this run.
- `checkpoints_full/last.pt`: most recent completed epoch.

Kaggle sessions have time limits. Download or "Save Version" the checkpoint files from `/kaggle/working` before the session ends.

## Resuming an interrupted full run

Set `RESUME_FROM` to the full run's `last.pt` (or `auto`) and keep `NUM_EPOCHS`, dataset paths, batch size, and subset settings consistent with the original run. Resume restores the optimizer, scheduler, and scaler state. Do not use `INIT_FROM` and `RESUME_FROM` together.

```bash
%cd /kaggle/working/aasist3

!DATASET_ROOT="/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA" \
RESUME_FROM=auto \
CHECKPOINT_DIR="/kaggle/working/checkpoints_full" \
NUM_EPOCHS=25 \
BATCH_SIZE=8 \
BASE_LR=1e-4 \
LOG_EVERY=100 \
NUM_WORKERS=2 \
python -u TRAIN.py
```

If resuming in a fresh Kaggle session, first restore the saved checkpoint files into the same `CHECKPOINT_DIR` (for example by attaching your saved output as a dataset and copying `last.pt` over).
